#!/usr/bin/env python3
"""
Pairwise Image Distribution Comparison Tool
============================================
Computes pairwise FID, KID, and/or LPIPS between up to 5 datasets:
    - nuscenes     (CAM_* flat dirs, ALL images including night)
    - nuscenes_day (same path as --nuscenes, night scenes EXCLUDED via token filter)
    - seed4d       (spawn_point structure, *_rgb.png images)
    - flux2        (spawn_point structure, *_rgb.png images)
    - secogan      (spawn_point structure, *_rgb.png images)

Metrics
-------
  fid   — Fréchet Inception Distance (distributional, any pair)
  kid   — Kernel Inception Distance  (distributional, any pair)
  lpips — TRUE PAIRED LPIPS (only between sensor datasets: seed4d / flux2 / secogan)
          Images are matched by (town, spawn_point_N, cam_idx) — identical viewpoint.
          Only keys present in BOTH datasets are evaluated; sub-sampled to
          --sample-size if the intersection is larger.

LPIPS pairing key
-----------------
  .../TownXX/.../spawn_point_N/.../sensors/C_rgb.png
   → key = (TownXX, N, C)

Usage
-----
python fid_comparison.py \\
    --nuscenes           /data/nuscenes/samples \\
    --seed4d             /data/seed4d \\
    --flux2              /data/flux2 \\
    --secogan            /data/secogan \\
    --output-dir         /results/fid \\
    --metrics fid kid lpips ssim encoder_dist \\
    --night-trainval-txt /data/night_trainval.txt \\
    --night-test-txt     /data/night_test.txt \\
    --nuscenes-dataroot  /data/nuscenes \\
    --pixelsplat-src     /app/felix/code/pixelsplat_Sim2Real/src \\
    --ckpt-seed4d        /checkpoints/seed4d_best.ckpt \\
    --ckpt-flux2         /checkpoints/flux2_best.ckpt \\
    --ckpt-secogan       /checkpoints/secogan_best.ckpt \\
    [--sample-size 5000] \\
    [--resize 256x256] \\
    [--gpu 0]

Dependencies
------------
    pip install torch torchvision torch-fidelity lpips opencv-python pandas Pillow
    pip install nuscenes-devkit   # required for night-scene filtering
"""

import argparse
import itertools
import random
import shutil
import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

# ------------------------------------------------------------------ #
#  Metric registry                                                     #
# ------------------------------------------------------------------ #

AVAILABLE_METRICS = ['fid', 'kid', 'lpips', 'ssim', 'encoder_dist']
SENSOR_DATASETS   = {'seed4d', 'flux2', 'secogan'}

# Checkpoint names — used as suffixes for encoder_dist columns
ENCODER_CKPT_NAMES = ['seed4d', 'flux2', 'secogan']

METRIC_LABELS = {
    'fid':                      'Fréchet Inception Distance (FID)',
    'kid_mean':                 'Kernel Inception Distance (KID)',
    'lpips_mean':               'Mean Paired LPIPS',
    'ssim_mean':                'Mean Paired SSIM',
    'encoder_dist_seed4d_mean': 'Encoder Distance (SEED4D feature space)',
    'encoder_dist_flux2_mean':  'Encoder Distance (Flux2 feature space)',
    'encoder_dist_secogan_mean':'Encoder Distance (SecoGAN feature space)',
}

# Metrics that require paired sensor datasets (seed4d / flux2 / secogan)
PAIRED_METRICS = {'lpips', 'ssim', 'encoder_dist'}


# ------------------------------------------------------------------ #
#  Night-scene helpers                                                 #
# ------------------------------------------------------------------ #

def load_night_scene_filenames(trainval_txt, test_txt, nuscenes_dataroot):
    """Return set of image basenames that belong to night scenes."""
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        raise ImportError(
            "nuscenes-devkit is required for night filtering. "
            "Install with: pip install nuscenes-devkit"
        )

    night_sample_tokens = set()
    for txt_path in [trainval_txt, test_txt]:
        txt_path = Path(txt_path)
        if not txt_path.exists():
            print(f"  Warning: night scenes file not found: {txt_path} – skipping")
            continue
        with open(txt_path) as f:
            tokens = {line.strip() for line in f if line.strip()}
        print(f"  Loaded {len(tokens)} tokens from {txt_path.name}")
        night_sample_tokens |= tokens

    print(f"  Total merged night sample tokens: {len(night_sample_tokens)}")
    if not night_sample_tokens:
        return set()

    night_filenames = set()
    for version in ('v1.0-trainval', 'v1.0-test'):
        try:
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version=version, dataroot=nuscenes_dataroot, verbose=False)
        except Exception as e:
            print(f"  Warning: could not load NuScenes {version}: {e} – skipping")
            continue
        for sd in nusc.sample_data:
            if sd['sensor_modality'] != 'camera':
                continue
            if sd['sample_token'] in night_sample_tokens:
                night_filenames.add(Path(sd['filename']).name)

    print(f"  Total night image filenames resolved: {len(night_filenames)}")
    return night_filenames


# ------------------------------------------------------------------ #
#  Sampling – flat layout (nuScenes)                                   #
# ------------------------------------------------------------------ #

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

CAMERA_MAPPING = {
    0: 'CAM_FRONT',
    1: 'CAM_FRONT_RIGHT',
    2: 'CAM_FRONT_LEFT',
    3: 'CAM_BACK',
    4: 'CAM_BACK_LEFT',
    5: 'CAM_BACK_RIGHT',
}


def sample_flat_dataset(root: Path, sample_size: int,
                        night_filenames: set = None) -> list[Path]:
    """
    Random sample from a flat CAM_* directory layout (nuScenes style).
    Mirrors process_directory() in the nuScenes analyzer.
    """
    all_files: list[Path] = []
    cam_dirs = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith('CAM_')])
    if not cam_dirs:
        cam_dirs = [root]

    for cam_dir in cam_dirs:
        files = [f for f in cam_dir.glob('*') if f.suffix.lower() in IMAGE_EXTENSIONS]
        if night_filenames:
            before = len(files)
            files = [f for f in files if f.name not in night_filenames]
            removed = before - len(files)
            if removed:
                print(f"    Night filter: removed {removed} from {cam_dir.name}")
        all_files.extend(files)

    print(f"  Total images available after filtering: {len(all_files)}")
    if not all_files:
        return []

    random.seed(42)
    if len(all_files) <= sample_size:
        print(f"  Using all {len(all_files)} images")
        return all_files
    sampled = random.sample(all_files, sample_size)
    print(f"  Randomly sampled {sample_size} / {len(all_files)} images")
    return sampled


# ------------------------------------------------------------------ #
#  Sampling / indexing – sensor layout (SEED4D / SecoGAN / Flux)      #
# ------------------------------------------------------------------ #

def _parse_sensor_key(path: Path):
    """
    Extract (town, spawn_point_num, cam_idx) from a sensor image path.

    Expected path fragment:
      .../TownXX/.../spawn_point_N/step_0/ego_vehicle/nuscenes_invisible/sensors/C_rgb.png
    Returns None if the key cannot be parsed.
    """
    parts = path.parts
    town  = next((p for p in parts if p.startswith('Town')), None)
    spawn = next((p for p in parts if p.startswith('spawn_point_')), None)
    if town is None or spawn is None:
        return None
    try:
        spawn_num = int(spawn.split('_')[-1])
        cam_idx   = int(path.stem.split('_')[0])   # "0_rgb" → 0
        return (town, spawn_num, cam_idx)
    except (ValueError, IndexError):
        return None


def index_sensor_dataset(root: Path) -> dict:
    """
    Walk the sensor tree and return
      {(town, spawn_point_num, cam_idx): Path}
    for every image found.
    """
    pattern = '**/vehicle.audi.tt/spawn_point_*/step_0/ego_vehicle/nuscenes_invisible/sensors'
    index = {}
    for sensors_dir in root.glob(pattern):
        if not sensors_dir.is_dir():
            continue
        for cam_idx in CAMERA_MAPPING:
            img = sensors_dir / f'{cam_idx}_rgb.png'
            if img.exists():
                key = _parse_sensor_key(img)
                if key is not None:
                    index[key] = img
    return index


def sample_sensor_dataset(root: Path, sample_size: int) -> list[Path]:
    """Random sample from the sensor layout (used for FID/KID staging)."""
    index     = index_sensor_dataset(root)
    all_files = list(index.values())
    print(f"  Found {len(all_files)} images across all spawn_points/cameras")
    if not all_files:
        return []
    random.seed(42)
    if len(all_files) <= sample_size:
        print(f"  Using all {len(all_files)} images")
        return all_files
    sampled = random.sample(all_files, sample_size)
    print(f"  Randomly sampled {sample_size} / {len(all_files)} images")
    return sampled


# ------------------------------------------------------------------ #
#  Staging helper (FID / KID)                                         #
# ------------------------------------------------------------------ #

def stage_images(image_paths: list[Path], stage_dir: Path,
                 resize=None) -> None:
    """
    Populate a flat staging directory for torch-fidelity.
    If resize=(W,H) every image is resized with INTER_AREA and written as PNG.
    Otherwise symlinks (→ hardlinks → copy) are used to avoid data duplication.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(image_paths):
        if resize is not None:
            dst = stage_dir / f'{i:06d}.png'
            img = cv2.imread(str(src))
            if img is None:
                print(f"  Warning: could not read {src}, skipping")
                continue
            h, w = img.shape[:2]
            if (w, h) != resize:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img)
        else:
            dst = stage_dir / f'{i:06d}{src.suffix}'
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                try:
                    dst.hardlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)


# ------------------------------------------------------------------ #
#  FID / KID                                                           #
# ------------------------------------------------------------------ #

def compute_fid_kid(name_a: str, stage_a: Path,
                    name_b: str, stage_b: Path,
                    compute_fid: bool, compute_kid: bool,
                    cuda: bool) -> dict:
    """Compute FID and/or KID between two staged directories."""
    import torch_fidelity

    active_labels = []
    if compute_fid: active_labels.append('FID')
    if compute_kid: active_labels.append('KID')
    print(f"\n  [{'/'.join(active_labels)}]  {name_a}  ↔  {name_b}")
    t0 = time.time()

    n_a = len(list(stage_a.iterdir()))
    n_b = len(list(stage_b.iterdir()))

    raw = torch_fidelity.calculate_metrics(
        input1=str(stage_a),
        input2=str(stage_b),
        cuda=cuda,
        fid=compute_fid,
        kid=compute_kid,
        kid_subset_size=min(1000, n_a, n_b) if compute_kid else 1000,
        verbose=False,
    )

    result = {'elapsed_s': time.time() - t0}
    if compute_fid:
        result['fid'] = raw.get('frechet_inception_distance', float('nan'))
        print(f"    FID  = {result['fid']:.4f}")
    if compute_kid:
        result['kid_mean'] = raw.get('kernel_inception_distance_mean', float('nan'))
        result['kid_std']  = raw.get('kernel_inception_distance_std',  float('nan'))
        print(f"    KID  = {result['kid_mean']:.6f}  (±{result['kid_std']:.6f})")
    print(f"    Time = {result['elapsed_s']:.1f}s")
    return result


# ------------------------------------------------------------------ #
#  LPIPS – true paired computation for sensor datasets                 #
# ------------------------------------------------------------------ #

def _load_tensor(path: Path, resize, device: torch.device):
    """Load image as (1,3,H,W) float32 tensor in [-1,1]."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        h, w = img.shape[:2]
        if (w, h) != resize:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def compute_lpips_paired(name_a: str, root_a: Path,
                         name_b: str, root_b: Path,
                         sample_size: int,
                         resize,
                         cuda: bool) -> dict:
    """
    Compute mean LPIPS over structurally matched pairs.
    Match key = (town, spawn_point_num, cam_idx) — same viewpoint, different renderer.
    """
    import lpips as lpips_lib

    print(f"\n  [LPIPS paired]  {name_a}  ↔  {name_b}")
    t0 = time.time()

    print(f"    Indexing {name_a}…")
    index_a = index_sensor_dataset(root_a)
    print(f"    Indexing {name_b}…")
    index_b = index_sensor_dataset(root_b)

    common_keys = sorted(set(index_a) & set(index_b))
    print(f"    Common (town, spawn_point, cam) pairs: {len(common_keys)}")

    if not common_keys:
        print(f"    WARNING: no matching keys found!")
        return {'lpips_mean': float('nan'), 'lpips_std': float('nan'),
                'lpips_n_pairs': 0, 'elapsed_s': time.time() - t0}

    random.seed(42)
    if len(common_keys) > sample_size:
        common_keys = random.sample(common_keys, sample_size)
        print(f"    Sub-sampled to {sample_size} pairs")

    device  = torch.device('cuda' if cuda else 'cpu')
    loss_fn = lpips_lib.LPIPS(net='alex').to(device)
    loss_fn.eval()

    scores = []
    for i, key in enumerate(common_keys):
        img_a = _load_tensor(index_a[key], resize, device)
        img_b = _load_tensor(index_b[key], resize, device)
        if img_a is None or img_b is None:
            continue
        with torch.no_grad():
            scores.append(loss_fn(img_a, img_b).item())
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(common_keys)} pairs…")

    lpips_mean = float(np.mean(scores)) if scores else float('nan')
    lpips_std  = float(np.std(scores))  if scores else float('nan')
    elapsed    = time.time() - t0

    print(f"    LPIPS = {lpips_mean:.6f}  (±{lpips_std:.6f})  over {len(scores)} pairs")
    print(f"    Time  = {elapsed:.1f}s")

    return {'lpips_mean': lpips_mean, 'lpips_std': lpips_std,
            'lpips_n_pairs': len(scores), 'elapsed_s': elapsed}


# ------------------------------------------------------------------ #
#  SSIM – paired computation for sensor datasets                       #
# ------------------------------------------------------------------ #

def _load_numpy_rgb(path: Path, resize) -> np.ndarray | None:
    """Load image as (H, W, 3) uint8 numpy array, optionally resized."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        h, w = img.shape[:2]
        if (w, h) != resize:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    return img


def _ssim_pair(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Compute mean SSIM between two (H, W, 3) uint8 images.
    Uses a 11×11 Gaussian window, data_range=255, multichannel.
    """
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(img_a, img_b, data_range=255, channel_axis=2,
                      win_size=11, gaussian_weights=True))


def compute_ssim_paired(name_a: str, root_a: Path,
                        name_b: str, root_b: Path,
                        sample_size: int,
                        resize) -> dict:
    """
    Compute mean SSIM over structurally matched pairs.
    Match key = (town, spawn_point_num, cam_idx).
    """
    print(f"\n  [SSIM paired]  {name_a}  ↔  {name_b}")
    t0 = time.time()

    print(f"    Indexing {name_a}…")
    index_a = index_sensor_dataset(root_a)
    print(f"    Indexing {name_b}…")
    index_b = index_sensor_dataset(root_b)

    common_keys = sorted(set(index_a) & set(index_b))
    print(f"    Common (town, spawn_point, cam) pairs: {len(common_keys)}")

    if not common_keys:
        print("    WARNING: no matching keys found!")
        return {'ssim_mean': float('nan'), 'ssim_std': float('nan'),
                'ssim_n_pairs': 0, 'elapsed_s': time.time() - t0}

    random.seed(42)
    if len(common_keys) > sample_size:
        common_keys = random.sample(common_keys, sample_size)
        print(f"    Sub-sampled to {sample_size} pairs")

    scores = []
    for i, key in enumerate(common_keys):
        img_a = _load_numpy_rgb(index_a[key], resize)
        img_b = _load_numpy_rgb(index_b[key], resize)
        if img_a is None or img_b is None:
            continue
        scores.append(_ssim_pair(img_a, img_b))
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(common_keys)} pairs…")

    ssim_mean = float(np.mean(scores)) if scores else float('nan')
    ssim_std  = float(np.std(scores))  if scores else float('nan')
    elapsed   = time.time() - t0

    # SSIM is a similarity (higher = more similar), print clearly
    print(f"    SSIM  = {ssim_mean:.6f}  (±{ssim_std:.6f})  over {len(scores)} pairs")
    print(f"    Note: SSIM is a SIMILARITY score (1.0 = identical, higher is better)")
    print(f"    Time  = {elapsed:.1f}s")

    return {'ssim_mean': ssim_mean, 'ssim_std': ssim_std,
            'ssim_n_pairs': len(scores), 'elapsed_s': elapsed}


# ------------------------------------------------------------------ #
#  PixelSplat encoder feature distance – paired, sensor datasets only  #
# ------------------------------------------------------------------ #

def load_pixelsplat_backbone(checkpoint_path: Path,
                             pixelsplat_src: Path,
                             device: torch.device):
    """
    Load the backbone + backbone_projection from a PixelSplat .ckpt file.

    The checkpoint is a PyTorch Lightning checkpoint whose state_dict contains:
      encoder.backbone.*
      encoder.backbone_projection.*

    We reconstruct the backbone by inspecting the state dict to determine
    which backbone type was used (resnet vs dino), then load weights.

    Returns a callable that takes a (B, V, 3, H, W) tensor and returns
    projected features (B, V, d_feature, H, W).

    Parameters
    ----------
    checkpoint_path : Path
        Path to the .ckpt file.
    pixelsplat_src : Path
        Path to the pixelsplat src/ directory (the one containing model/).
        This directory is added to sys.path so the backbone modules can be imported.
    device : torch.device
    """
    import sys as _sys
    import types as _types
    import importlib.util as _ilu
    import re as _re

    def _mock(name: str, **attrs):
        """Register a stub module and all its parent packages in sys.modules."""
        parts = name.split('.')
        for i in range(1, len(parts) + 1):
            dotted = '.'.join(parts[:i])
            if dotted not in _sys.modules:
                m = _types.ModuleType(dotted)
                m.__package__ = dotted
                m.__path__    = []
                _sys.modules[dotted] = m
        for k, v in attrs.items():
            setattr(_sys.modules[name], k, v)
        return _sys.modules[name]

    def _load_module_patched(dotted_name: str, path: Path, package: str):
        """
        Load a .py file as `dotted_name`, rewriting any relative imports that
        reference packages outside our controlled hierarchy into absolute imports
        that hit our pre-registered stubs.

        Specifically:  from ....dataset.types import X
                  →    from dataset.types import X
        and similarly for any other deep relative imports.
        """
        if dotted_name in _sys.modules:
            return _sys.modules[dotted_name]

        src_text = path.read_text()

        # Replace "from ....X" (any number of leading dots ≥ 2) where X is a
        # known stub package (dataset, etc.) with a plain absolute import.
        # This handles backbone.py's `from ....dataset.types import BatchedViews`.
        src_text = _re.sub(
            r'from \.{2,}(dataset\S*)',
            r'from \1',
            src_text,
        )
        # Also neutralise any other deep relative imports that would fail
        # (e.g. from ....geometry, from ....model) by replacing with a stub import.
        # We do this by turning them into: from _stub import *  (harmless)
        src_text = _re.sub(
            r'from \.{2,}(\w+)',
            lambda m: f'from _stub import _  # was: from ....{m.group(1)}',
            src_text,
        )

        code = compile(src_text, str(path), 'exec')
        mod  = _types.ModuleType(dotted_name)
        mod.__package__ = package
        mod.__name__    = dotted_name
        mod.__file__    = str(path)
        _sys.modules[dotted_name] = mod
        exec(code, mod.__dict__)
        return mod

    # ── stubs for every external symbol the backbone files reference ──────
    _mock('dataset.types',    BatchedViews=dict, DataShim=None)
    _mock('_stub',            _=None)   # catch-all for neutralised imports

    # Parent packages must exist in sys.modules for dotted names to resolve
    backbone_pkg = 'model.encoder.backbone'
    _mock('model')
    _mock('model.encoder')
    _mock(backbone_pkg)

    backbone_dir = pixelsplat_src / 'model' / 'encoder' / 'backbone'

    bb_base   = _load_module_patched(f'{backbone_pkg}.backbone',
                                     backbone_dir / 'backbone.py',        backbone_pkg)
    bb_resnet = _load_module_patched(f'{backbone_pkg}.backbone_resnet',
                                     backbone_dir / 'backbone_resnet.py', backbone_pkg)
    bb_dino   = _load_module_patched(f'{backbone_pkg}.backbone_dino',
                                     backbone_dir / 'backbone_dino.py',   backbone_pkg)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    # ── detect backbone type from key names ──────────────────────────
    backbone_keys = [k for k in state if k.startswith('encoder.backbone.')]
    if not backbone_keys:
        raise ValueError(
            "No 'encoder.backbone.*' keys found in checkpoint. "
            "Make sure this is a PixelSplat EncoderEpipolar checkpoint."
        )

    # dino backbone has a 'dino' submodule; resnet does not
    is_dino = any('encoder.backbone.dino.' in k for k in backbone_keys)

    # ── infer d_out from the backbone projection weight ───────────────
    proj_w = state.get('encoder.backbone_projection.1.weight')   # Linear weight
    if proj_w is None:
        raise ValueError("Cannot find 'encoder.backbone_projection.1.weight' in checkpoint.")
    d_feature = proj_w.shape[0]   # out_features of the Linear

    # ── infer backbone d_out from the projection input size ───────────
    d_backbone_out = proj_w.shape[1]   # in_features

    print(f"    Backbone type : {'dino' if is_dino else 'resnet'}")
    print(f"    d_backbone_out: {d_backbone_out}")
    print(f"    d_feature     : {d_feature}")

    # ── build backbone ────────────────────────────────────────────────
    if is_dino:
        BackboneDino    = bb_dino.BackboneDino
        BackboneDinoCfg = bb_dino.BackboneDinoCfg
        # Derive patch size from pos_embed shape:
        #   tokens = (H/ps) * (W/ps) + 1  where H=W=224 (DINO default)
        #   so n_patches = n_tokens - 1,  ps = 224 / sqrt(n_patches)
        pos_embed_w = next((v for k, v in state.items()
                            if 'encoder.backbone.dino.pos_embed' in k), None)
        if pos_embed_w is not None:
            n_tokens  = pos_embed_w.shape[1]          # e.g. 785 or 197
            n_patches = n_tokens - 1                  # subtract CLS token
            import math as _math
            ps = int(224 / _math.sqrt(n_patches))     # 8 or 16
        else:
            ps = 16   # fallback
        embed_dim  = pos_embed_w.shape[-1] if pos_embed_w is not None else 768
        size       = 's' if embed_dim < 600 else 'b'
        model_name = f'dino_vit{size}{ps}'            # e.g. 'dino_vitb8'
        print(f"    DINO variant  : {model_name}  (patch_size={ps}, embed_dim={embed_dim})")
        cfg      = BackboneDinoCfg(name='dino', model=model_name, d_out=d_backbone_out)
        backbone = BackboneDino(cfg, 3)
    else:
        BackboneResnet    = bb_resnet.BackboneResnet
        BackboneResnetCfg = bb_resnet.BackboneResnetCfg
        # detect resnet variant from layer keys
        has_layer4 = any('encoder.backbone.model.layer4' in k for k in backbone_keys)
        has_layer3 = any('encoder.backbone.model.layer3' in k for k in backbone_keys)
        # count projection layers to get num_layers
        proj_layers = {k.split('.')[3] for k in backbone_keys if 'projections' in k}
        num_layers  = len(proj_layers)
        # detect model size from first conv out_channels
        conv1_w = next((v for k, v in state.items()
                        if 'encoder.backbone.model.conv1.weight' in k), None)
        # resnet18/34 → 64 channels, resnet50+ → 64 too but deeper blocks differ
        # safest: use resnet18 as default, actual weights are loaded anyway
        cfg = BackboneResnetCfg(
            name='resnet', model='resnet18',
            num_layers=num_layers, use_first_pool=False,
            d_out=d_backbone_out,
        )
        backbone = BackboneResnet(cfg, 3)

    # ── strip prefix and load backbone weights ────────────────────────
    backbone_state = {
        k.replace('encoder.backbone.', ''): v
        for k, v in state.items()
        if k.startswith('encoder.backbone.') and not k.startswith('encoder.backbone_projection.')
    }
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"    Backbone missing keys  : {len(missing)} "
              f"(first: {missing[0]})")
    if unexpected:
        print(f"    Backbone unexpected keys: {len(unexpected)}")

    # ── build backbone_projection  (ReLU + Linear) ───────────────────
    import torch.nn as nn
    proj = nn.Sequential(nn.ReLU(), nn.Linear(d_backbone_out, d_feature))
    proj_state = {
        k.replace('encoder.backbone_projection.', ''): v
        for k, v in state.items()
        if k.startswith('encoder.backbone_projection.')
    }
    proj.load_state_dict(proj_state, strict=True)

    backbone  = backbone.to(device).eval()
    proj      = proj.to(device).eval()

    class _PixelSplatEncoder(torch.nn.Module):
        """Thin wrapper: image (B,V,3,H,W) → projected features (B,V,d_feature,H,W)."""
        def __init__(self, bb, pr):
            super().__init__()
            self.bb = bb
            self.pr = pr

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # images: (B, V, 3, H, W), already normalised to [-1, 1] or [0, 1]
            from einops import rearrange
            feats = self.bb({'image': images})                          # (B,V,d_bb,H,W)
            feats = rearrange(feats, 'b v c h w -> b v h w c')
            feats = self.pr(feats)                                      # (B,V,H,W,d_feat)
            feats = rearrange(feats, 'b v h w c -> b v c h w')
            return feats

    return _PixelSplatEncoder(backbone, proj)


def compute_encoder_dist_paired(name_a: str, root_a: Path,
                                name_b: str, root_b: Path,
                                encoder_model,
                                ckpt_name: str,
                                sample_size: int,
                                resize,
                                device: torch.device,
                                batch_size: int = 16) -> dict:
    """
    Compute mean cosine distance between PixelSplat encoder features of matched pairs.

    Features are extracted per image (V=1), spatially global-average-pooled to a
    1-D vector, then cosine distance = 1 - cosine_similarity is computed.
    Lower = more similar in the feature space the reconstruction model uses.

    ckpt_name: one of 'seed4d' | 'flux2' | 'secogan' — used as column suffix so
               results from different checkpoints don't overwrite each other.
    """
    col = f'encoder_dist_{ckpt_name}'   # e.g. 'encoder_dist_seed4d'

    print(f"\n  [Encoder dist / {ckpt_name} ckpt]  {name_a}  ↔  {name_b}")
    t0 = time.time()

    print(f"    Indexing {name_a}…")
    index_a = index_sensor_dataset(root_a)
    print(f"    Indexing {name_b}…")
    index_b = index_sensor_dataset(root_b)

    common_keys = sorted(set(index_a) & set(index_b))
    print(f"    Common (town, spawn_point, cam) pairs: {len(common_keys)}")

    if not common_keys:
        print("    WARNING: no matching keys found!")
        return {f'{col}_mean': float('nan'), f'{col}_std': float('nan'),
                f'{col}_n_pairs': 0, 'elapsed_s': time.time() - t0}

    random.seed(42)
    if len(common_keys) > sample_size:
        common_keys = random.sample(common_keys, sample_size)
        print(f"    Sub-sampled to {sample_size} pairs")

    cos_sim = torch.nn.CosineSimilarity(dim=1)

    def _extract_features_batch(paths: list[Path]) -> torch.Tensor:
        """Load a batch of images and return GAP feature vectors (N, d_feature)."""
        imgs = []
        for p in paths:
            t = _load_tensor(p, resize, device)   # (1, 3, H, W) in [-1,1]
            if t is None:
                # placeholder zeros — filtered out after
                imgs.append(None)
            else:
                imgs.append(t)

        valid_mask = [x is not None for x in imgs]
        valid_imgs = [x for x in imgs if x is not None]
        if not valid_imgs:
            return None, valid_mask

        # Stack as (N, 3, H, W), wrap in (N, V=1, 3, H, W)
        batch = torch.cat(valid_imgs, dim=0).unsqueeze(1)   # (N, 1, 3, H, W)
        with torch.no_grad():
            feats = encoder_model(batch)                    # (N, 1, d, H, W)
        feats = feats[:, 0]                                 # (N, d, H, W)
        feats = feats.mean(dim=[2, 3])                      # GAP → (N, d)
        return feats, valid_mask

    scores = []
    keys_a_paths = [index_a[k] for k in common_keys]
    keys_b_paths = [index_b[k] for k in common_keys]

    for start in range(0, len(common_keys), batch_size):
        batch_paths_a = keys_a_paths[start:start + batch_size]
        batch_paths_b = keys_b_paths[start:start + batch_size]

        feats_a, mask_a = _extract_features_batch(batch_paths_a)
        feats_b, mask_b = _extract_features_batch(batch_paths_b)

        # only keep positions valid in both
        valid = [a and b for a, b in zip(mask_a, mask_b)]
        if feats_a is None or feats_b is None:
            continue

        # filter to jointly valid
        fa_idx = [i for i, (m, v) in enumerate(zip(mask_a, valid)) if m and v]
        fb_idx = [i for i, (m, v) in enumerate(zip(mask_b, valid)) if m and v]
        if not fa_idx:
            continue

        fa = feats_a[fa_idx]
        fb = feats_b[fb_idx]

        dist = 1.0 - cos_sim(fa, fb)   # cosine distance ∈ [0, 2]
        scores.extend(dist.cpu().tolist())

        done = start + len(batch_paths_a)
        if done % 500 < batch_size:
            print(f"    {done}/{len(common_keys)} pairs…")

    enc_mean = float(np.mean(scores)) if scores else float('nan')
    enc_std  = float(np.std(scores))  if scores else float('nan')
    elapsed  = time.time() - t0

    print(f"    Encoder dist ({ckpt_name}) = {enc_mean:.6f}  (±{enc_std:.6f})  over {len(scores)} pairs")
    print(f"    (cosine distance in {ckpt_name} feature space; lower = more similar)")
    print(f"    Time = {elapsed:.1f}s")

    return {f'{col}_mean': enc_mean, f'{col}_std': enc_std,
            f'{col}_n_pairs': len(scores), 'elapsed_s': elapsed}


# ------------------------------------------------------------------ #
#  Output helpers                                                      #
# ------------------------------------------------------------------ #

def print_matrix(results: list, dataset_names: list, col: str) -> None:
    """Print a symmetric matrix. Skips silently if col not in any result."""
    if not any(col in r for r in results):
        return

    n     = len(dataset_names)
    col_w = max(18, max(len(n) for n in dataset_names) + 2)
    header = f"{'':>{col_w}}" + ''.join(f"{n:>{col_w}}" for n in dataset_names)
    print(header)
    print('-' * len(header))

    lookup = {}
    for r in results:
        if col in r:
            lookup[(r['dataset_a'], r['dataset_b'])] = r[col]
            lookup[(r['dataset_b'], r['dataset_a'])] = r[col]

    for row in dataset_names:
        line = f"{row:>{col_w}}"
        for c in dataset_names:
            if row == c:
                line += f"{'—':>{col_w}}"
            else:
                val = lookup.get((row, c), float('nan'))
                line += f"{val:>{col_w}.4f}" if not np.isnan(val) else f"{'n/a':>{col_w}}"
        print(line)


def save_matrix_plot(results: list, dataset_names: list,
                     output_path: Path, col: str, title: str) -> None:
    """Save a heatmap for one metric column."""
    if not any(col in r for r in results):
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plot")
        return

    n      = len(dataset_names)
    matrix = np.full((n, n), float('nan'))
    idx    = {name: i for i, name in enumerate(dataset_names)}

    for r in results:
        if col in r:
            i, j = idx[r['dataset_a']], idx[r['dataset_b']]
            matrix[i, j] = r[col]
            matrix[j, i] = r[col]
    np.fill_diagonal(matrix, 0)

    finite = matrix[np.isfinite(matrix)]
    vmax   = float(finite.max()) if len(finite) else 1.0

    fig, ax = plt.subplots(figsize=(max(6, n * 1.8), max(5, n * 1.6)))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label=col)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(dataset_names, rotation=35, ha='right', fontsize=10)
    ax.set_yticklabels(dataset_names, fontsize=10)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if i == j:
                text = '—'
            elif np.isnan(val):
                text = 'n/a'
            else:
                text = f'{val:.4f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                    color='white' if (not np.isnan(val) and vmax > 0 and val > vmax * 0.6)
                    else 'black')

    similarity_metrics = {'ssim_mean'}
    direction = '(higher = more similar)' if col in similarity_metrics else '(lower = more similar)'
    ax.set_title(f'Pairwise {title}\n{direction}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Pairwise FID / KID / LPIPS comparison between up to 5 datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset paths
    parser.add_argument('--nuscenes', type=str, default=None,
                        help='Path to nuScenes samples root (CAM_* dirs). '
                             'Produces "nuscenes" (all) and "nuscenes_day" (filtered) variants.')
    parser.add_argument('--seed4d',  type=str, default=None,
                        help='Root dir of SEED4D dataset (vehicle.audi.tt subtree)')
    parser.add_argument('--flux2',   type=str, default=None,
                        help='Root dir of Flux2 dataset (vehicle.audi.tt subtree)')
    parser.add_argument('--secogan', type=str, default=None,
                        help='Root dir of SecoGAN dataset (vehicle.audi.tt subtree)')

    # Night filtering
    parser.add_argument('--night-trainval-txt', type=str, default=None)
    parser.add_argument('--night-test-txt',     type=str, default=None)
    parser.add_argument('--nuscenes-dataroot',  type=str, default=None)

    # General
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for results CSV and plots')
    parser.add_argument('--metrics', nargs='+', default=['fid', 'kid'],
                        choices=AVAILABLE_METRICS,
                        help='Metrics to compute (default: fid kid). '
                             'ssim / lpips / encoder_dist are paired and only available '
                             'between sensor datasets (seed4d / flux2 / secogan), matched '
                             'by (town, spawn_point, camera_index). '
                             'Note: SSIM is a similarity score (higher = more similar); '
                             'all others are distance scores (lower = more similar).')
    parser.add_argument('--ckpt-seed4d', type=str, default=None,
                        help='PixelSplat .ckpt trained on SEED4D. Used for encoder_dist '
                             'computed in the SEED4D feature space.')
    parser.add_argument('--ckpt-flux2', type=str, default=None,
                        help='PixelSplat .ckpt trained on Flux2. Used for encoder_dist '
                             'computed in the Flux2 feature space.')
    parser.add_argument('--ckpt-secogan', type=str, default=None,
                        help='PixelSplat .ckpt trained on SecoGAN. Used for encoder_dist '
                             'computed in the SecoGAN feature space.')
    parser.add_argument('--pixelsplat-src', type=str, default=None,
                        help='Path to the PixelSplat src/ directory '
                             '(e.g. /app/felix/code/pixelsplat_Sim2Real/src). '
                             'Added to sys.path so backbone modules can be imported. '
                             'Required when --metrics includes encoder_dist.')
    parser.add_argument('--encoder-batch-size', type=int, default=16,
                        help='Batch size for encoder feature extraction (default: 16). '
                             'Reduce if you hit OOM during encoder_dist computation.')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Images per dataset for FID/KID, max pairs for LPIPS (default: 5000)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU')
    parser.add_argument('--resize', type=str, default=None,
                        help='Resize all images to WxH, e.g. 256x256. '
                             'Required when datasets have different native resolutions.')
    parser.add_argument('--keep-staging', action='store_true',
                        help='Keep temporary staging dirs after completion')

    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = list(dict.fromkeys(args.metrics))   # deduplicate, preserve order
    print(f"Metrics: {metrics}")

    cuda = not args.no_gpu and torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(args.gpu)
        print(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("CPU mode (will be slow for large sample sizes)")

    # ── resize ──────────────────────────────────────────────────────
    resize = None
    if args.resize:
        try:
            rw, rh = map(int, args.resize.lower().split('x'))
            if rw <= 0 or rh <= 0:
                raise ValueError
            resize = (rw, rh)
            print(f"Resize: {rw}×{rh} applied to all datasets")
        except (ValueError, AttributeError):
            print(f"Error: invalid --resize '{args.resize}'. Use WxH, e.g. 256x256.")
            sys.exit(1)
    else:
        print("No --resize – images used at original resolution.")

    # ── night filter ─────────────────────────────────────────────────
    night_filenames: set = set()
    if args.nuscenes:
        if args.night_trainval_txt and args.night_test_txt and args.nuscenes_dataroot:
            print("\nLoading night scene filter…")
            night_filenames = load_night_scene_filenames(
                args.night_trainval_txt, args.night_test_txt, args.nuscenes_dataroot
            )
            print(f"Night filter: {len(night_filenames)} images excluded from nuscenes_day.\n")
        else:
            print("WARNING: --nuscenes given but night-filter args missing. "
                  "Both nuScenes variants will include ALL images.\n")

    # ── active dataset list ──────────────────────────────────────────
    dataset_specs = []
    if args.nuscenes:
        dataset_specs.append(('nuscenes',     args.nuscenes, 'flat'))
        dataset_specs.append(('nuscenes_day', args.nuscenes, 'flat'))
    if args.seed4d:
        dataset_specs.append(('seed4d',  args.seed4d,  'sensor'))
    if args.flux2:
        dataset_specs.append(('flux2',   args.flux2,   'sensor'))
    if args.secogan:
        dataset_specs.append(('secogan', args.secogan, 'sensor'))

    active      = [(name, Path(path), dtype) for name, path, dtype in dataset_specs]
    all_names   = [name for name, _, _ in active]
    root_lookup = {name: path  for name, path, _ in active}
    type_lookup = {name: dtype for name, _, dtype in active}

    if len(active) < 2:
        print("Error: provide at least 2 dataset paths.")
        sys.exit(1)

    print(f"\nActive datasets ({len(active)}):")
    for name, path, dtype in active:
        print(f"  {name:20s} [{dtype:6s}]  {path}")

    # Validate paired metrics availability
    paired_requested = [m for m in metrics if m in PAIRED_METRICS]
    if paired_requested:
        sensor_names = [n for n in all_names if type_lookup[n] == 'sensor']
        if len(sensor_names) < 2:
            print(f"Error: metrics {paired_requested} require at least 2 sensor datasets "
                  "(seed4d / flux2 / secogan).")
            sys.exit(1)

    # Validate encoder_dist requirements
    if 'encoder_dist' in metrics:
        if not args.pixelsplat_src:
            print("Error: --metrics encoder_dist requires --pixelsplat-src.")
            sys.exit(1)
        src_path = Path(args.pixelsplat_src)
        if not src_path.exists():
            print(f"Error: pixelsplat src dir not found: {src_path}")
            sys.exit(1)
        # At least one checkpoint must be provided
        ckpt_map = {}
        for name, attr in [('seed4d', args.ckpt_seed4d),
                            ('flux2',  args.ckpt_flux2),
                            ('secogan',args.ckpt_secogan)]:
            if attr:
                p = Path(attr)
                if not p.exists():
                    print(f"Error: checkpoint not found: {p}")
                    sys.exit(1)
                ckpt_map[name] = p
        if not ckpt_map:
            print("Error: --metrics encoder_dist requires at least one of "
                  "--ckpt-seed4d / --ckpt-flux2 / --ckpt-secogan.")
            sys.exit(1)

    # ── sample + stage for FID/KID ───────────────────────────────────
    need_staging  = 'fid' in metrics or 'kid' in metrics
    staging_dirs: dict = {}
    cleanup       = False

    if need_staging:
        print(f"\nSampling {args.sample_size} images per dataset for FID/KID…")
        print("=" * 60)

        sampled: dict = {}
        for name, path, dtype in active:
            print(f"\n[{name}]")
            if not path.exists():
                print(f"  ERROR: path does not exist: {path}")
                sys.exit(1)
            if dtype == 'flat':
                nf    = night_filenames if name == 'nuscenes_day' else set()
                files = sample_flat_dataset(path, args.sample_size, night_filenames=nf)
            else:
                files = sample_sensor_dataset(path, args.sample_size)
            if not files:
                print(f"  ERROR: no images found for '{name}'")
                sys.exit(1)
            sampled[name] = files
            print(f"  → {len(files)} images selected")

        if args.keep_staging:
            tmp_root = output_dir / '_staging'
            tmp_root.mkdir(exist_ok=True)
            staging_dirs = {n: tmp_root / n for n in sampled}
            cleanup = False
        else:
            tmp_root     = Path(tempfile.mkdtemp(prefix='fid_staging_'))
            staging_dirs = {n: tmp_root / n for n in sampled}
            cleanup      = True

        print(f"\nStaging images…")
        for name, files in sampled.items():
            info = f", resizing to {resize[0]}×{resize[1]}" if resize else ""
            print(f"  [{name}] {len(files)} images{info}")
            stage_images(files, staging_dirs[name], resize=resize)

    # ── load PixelSplat encoders (one per provided checkpoint) ──────────────
    # encoder_models: {ckpt_name: model}  e.g. {'seed4d': <model>, 'flux2': <model>}
    encoder_models: dict = {}
    if 'encoder_dist' in metrics:
        device = torch.device('cuda' if cuda else 'cpu')
        src_path = Path(args.pixelsplat_src)
        for ckpt_name, ckpt_path in ckpt_map.items():
            print(f"\nLoading PixelSplat encoder [{ckpt_name}] from {ckpt_path}…")
            encoder_models[ckpt_name] = load_pixelsplat_backbone(
                checkpoint_path=ckpt_path,
                pixelsplat_src=src_path,
                device=device,
            )
            print(f"    [{ckpt_name}] encoder loaded and ready.")

    # ── pairwise computation ─────────────────────────────────────────
    pairs = list(itertools.combinations(all_names, 2))
    print(f"\nComputing metrics for {len(pairs)} pairs…")
    print("=" * 60)

    results: list = []
    t_total = time.time()

    for name_a, name_b in pairs:
        result = {'dataset_a': name_a, 'dataset_b': name_b}

        # FID / KID
        if need_staging:
            r = compute_fid_kid(
                name_a, staging_dirs[name_a],
                name_b, staging_dirs[name_b],
                compute_fid='fid' in metrics,
                compute_kid='kid' in metrics,
                cuda=cuda,
            )
            result.update({k: v for k, v in r.items() if k != 'elapsed_s'})

        # LPIPS – only between sensor datasets
        if 'lpips' in metrics:
            if type_lookup[name_a] == 'sensor' and type_lookup[name_b] == 'sensor':
                r = compute_lpips_paired(
                    name_a, root_lookup[name_a],
                    name_b, root_lookup[name_b],
                    sample_size=args.sample_size,
                    resize=resize,
                    cuda=cuda,
                )
                result.update({k: v for k, v in r.items() if k != 'elapsed_s'})
            else:
                print(f"\n  [LPIPS] Skipping {name_a} ↔ {name_b} "
                      f"(paired LPIPS only between sensor datasets)")
                result.update({'lpips_mean': float('nan'), 'lpips_std': float('nan'),
                               'lpips_n_pairs': 0})

        # SSIM – only between sensor datasets
        if 'ssim' in metrics:
            if type_lookup[name_a] == 'sensor' and type_lookup[name_b] == 'sensor':
                r = compute_ssim_paired(
                    name_a, root_lookup[name_a],
                    name_b, root_lookup[name_b],
                    sample_size=args.sample_size,
                    resize=resize,
                )
                result.update({k: v for k, v in r.items() if k != 'elapsed_s'})
            else:
                print(f"\n  [SSIM] Skipping {name_a} ↔ {name_b} "
                      f"(paired SSIM only between sensor datasets)")
                result.update({'ssim_mean': float('nan'), 'ssim_std': float('nan'),
                               'ssim_n_pairs': 0})

        # Encoder distance – only between sensor datasets, one pass per checkpoint
        if 'encoder_dist' in metrics:
            if type_lookup[name_a] == 'sensor' and type_lookup[name_b] == 'sensor':
                for ckpt_name, enc_model in encoder_models.items():
                    r = compute_encoder_dist_paired(
                        name_a, root_lookup[name_a],
                        name_b, root_lookup[name_b],
                        encoder_model=enc_model,
                        ckpt_name=ckpt_name,
                        sample_size=args.sample_size,
                        resize=resize,
                        device=torch.device('cuda' if cuda else 'cpu'),
                        batch_size=args.encoder_batch_size,
                    )
                    result.update({k: v for k, v in r.items() if k != 'elapsed_s'})
            else:
                print(f"\n  [Encoder dist] Skipping {name_a} ↔ {name_b} "
                      f"(encoder_dist only between sensor datasets)")
                for ckpt_name in encoder_models:
                    col = f'encoder_dist_{ckpt_name}'
                    result.update({f'{col}_mean': float('nan'),
                                   f'{col}_std':  float('nan'),
                                   f'{col}_n_pairs': 0})

        results.append(result)

    print(f"\nAll done in {time.time() - t_total:.1f}s")

    # ── save results ─────────────────────────────────────────────────
    import pandas as pd

    results_df = pd.DataFrame(results)
    csv_path   = output_dir / 'comparison_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print + plot each metric
    metric_outputs = []
    if 'fid'          in metrics: metric_outputs.append(('fid',               'FID',            METRIC_LABELS['fid']))
    if 'kid'          in metrics: metric_outputs.append(('kid_mean',          'KID',            METRIC_LABELS['kid_mean']))
    if 'lpips'        in metrics: metric_outputs.append(('lpips_mean',        'LPIPS (paired)', METRIC_LABELS['lpips_mean']))
    if 'ssim'         in metrics: metric_outputs.append(('ssim_mean',         'SSIM (paired)',  METRIC_LABELS['ssim_mean']))
    if 'encoder_dist' in metrics:
        for ckpt_name in encoder_models:
            col   = f'encoder_dist_{ckpt_name}_mean'
            label = f'Encoder dist ({ckpt_name})'
            title = METRIC_LABELS.get(col, f'Encoder Distance ({ckpt_name} feature space)')
            metric_outputs.append((col, label, title))

    for col, label, title in metric_outputs:
        similarity = col in {'ssim_mean'}
        direction  = 'higher = more similar' if similarity else 'lower = more similar'
        print(f"\n{label} Matrix ({direction}):")
        print_matrix(results, all_names, col)
        save_matrix_plot(results, all_names,
                         output_dir / f'{col}_matrix.png', col, title)

    # ── cleanup ──────────────────────────────────────────────────────
    if cleanup and need_staging:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print("\nTemporary staging directories cleaned up.")

    print("\nDone!")


if __name__ == '__main__':
    main()