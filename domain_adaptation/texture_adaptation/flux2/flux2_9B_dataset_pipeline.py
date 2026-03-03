import os
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict
import json
from multiprocessing import Process
import torch.multiprocessing as mp
from datetime import datetime

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================
BASE_DATA_DIR = "/app/felix/data/seed4d/data/data_diverse_1600x900_2poses_flux2_9B_2/static"
FLUX_MODEL_PATH = "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B/snapshots/main"
OUTPUT_BASE_DIR = "/app/felix/data/neuralremaster/flux2_9B_domain_translation"
TEMP_POSTER_DIR = "/app/felix/data/neuralremaster/flux2_9B_domain_translation/temp_posters"

# Processing parameters
IMAGES_PER_POSTER = 16  # 4x4 grid
POSTER_SIZE = (1024, 1024)
SLICE_SIZE = 256  # Each slice will be 256x256
NUM_INFERENCE_STEPS = 50
RANDOM_SEED = 0

# Flux prompt
FLUX_PROMPT = "photorealistic street scene with very smooth cloudy conditions, clear sight, high quality photograph, stay very close to the input image contentwise, keep the grid structure of the image"

# GPU settings
GPU_IDS = [0, 1]  # Use GPU 0 and GPU 1
DTYPE = torch.bfloat16

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def natural_sort_key(s):
    """
    Sort strings containing numbers in natural order.
    Example: ['0_rgb.png', '1_rgb.png', ..., '9_rgb.png', '10_rgb.png', '11_rgb.png']
    Instead of: ['0_rgb.png', '10_rgb.png', '11_rgb.png', ..., '1_rgb.png', '2_rgb.png']
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def format_time_elapsed(start_time):
    """Format elapsed time nicely."""
    elapsed = datetime.now() - start_time
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def create_poster(image_paths: List[str], output_path: str, 
                 poster_target_size: Tuple[int, int] = (1024, 1024)) -> Dict:
    """
    Create a poster image from a list of image paths.
    Images are arranged in natural order: 0,1,2,3... (left-to-right, top-to-bottom)
    Returns metadata about the poster including image positions.
    """
    if not image_paths:
        print("No images found!")
        return None
    
    # Target size for each individual image
    individual_img_size = (800, 600)
    
    # Load and resize all images IN ORDER
    images = []
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        img_resized = img.resize(individual_img_size, Image.LANCZOS)
        images.append(img_resized)
    
    img_width, img_height = individual_img_size
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = cols
    
    # Fill remaining spots with black images
    total_slots = rows * cols
    num_black_images = total_slots - num_images
    
    if num_black_images > 0:
        black_image = Image.new('RGB', individual_img_size, color='black')
        images.extend([black_image] * num_black_images)
    
    # Create poster canvas
    poster_width = cols * img_width
    poster_height = rows * img_height
    poster = Image.new('RGB', (poster_width, poster_height), color='black')
    
    # Paste images in natural order: 0,1,2,3... (left-to-right, top-to-bottom)
    image_positions = []
    for idx in range(len(images)):
        row = idx // cols  # Integer division gives row
        col = idx % cols   # Modulo gives column
        x = col * img_width
        y = row * img_height
        poster.paste(images[idx], (x, y))
        
        # Only store positions for actual images (not black fill)
        if idx < num_images:
            image_positions.append({
                'index': idx,  # Natural counting: 0,1,2,3...
                'original_path': image_paths[idx],
                'grid_position': (row, col),
                'pixel_position': (x, y)
            })
    
    # Resize poster to target size
    poster = poster.resize(poster_target_size, Image.LANCZOS)
    
    # Save poster
    poster.save(output_path)
    
    metadata = {
        'num_real_images': num_images,
        'num_black_fill': num_black_images,
        'grid_size': (rows, cols),
        'original_size': (poster_width, poster_height),
        'final_size': poster_target_size,
        'image_positions': image_positions
    }
    
    return metadata


def slice_image_to_grid(image_path: str, slice_size: int = 256) -> List[Tuple[Image.Image, int, int]]:
    """
    Slice an image into a grid of smaller images in natural order.
    Order: left-to-right, top-to-bottom (0,1,2,3...)
    Returns list of (image, row, col) tuples.
    """
    img = Image.open(image_path)
    width, height = img.size
    
    cols = width // slice_size
    rows = height // slice_size
    
    slices = []
    # Natural order: iterate rows first (top-to-bottom), then cols (left-to-right)
    for row in range(rows):
        for col in range(cols):
            left = col * slice_size
            upper = row * slice_size
            right = left + slice_size
            lower = upper + slice_size
            
            slice_img = img.crop((left, upper, right, lower))
            slices.append((slice_img, row, col))
    
    return slices


def find_rgb_images(base_path: str) -> List[str]:
    """
    Find all RGB images in nuscenes_invisible and sphere_invisible directories.
    Returns naturally sorted list (0, 1, 2, ..., 9, 10, 11, ...).
    """
    rgb_images = []
    
    # Check nuscenes_invisible
    nuscenes_dir = os.path.join(base_path, "nuscenes_invisible/sensors")
    if os.path.exists(nuscenes_dir):
        filenames = [f for f in os.listdir(nuscenes_dir) 
                    if '_rgb' in f and f.endswith('.png')]
        filenames.sort(key=natural_sort_key)  # Natural sorting
        for filename in filenames:
            rgb_images.append(os.path.join(nuscenes_dir, filename))
    
    # Check sphere_invisible
    sphere_dir = os.path.join(base_path, "sphere_invisible/sensors")
    if os.path.exists(sphere_dir):
        filenames = [f for f in os.listdir(sphere_dir) 
                    if '_rgb' in f and f.endswith('.png')]
        filenames.sort(key=natural_sort_key)  # Natural sorting
        for filename in filenames:
            rgb_images.append(os.path.join(sphere_dir, filename))
    
    return rgb_images


def get_all_spawnpoints(base_data_dir: str) -> List[Tuple[str, str, str, str, str]]:
    """
    Find all spawn points across all towns.
    Returns list of (town, weather, vehicle, spawn_point, step) tuples.
    """
    spawnpoints = []
    
    if not os.path.exists(base_data_dir):
        print(f"Base directory does not exist: {base_data_dir}")
        return spawnpoints
    
    # Iterate through directory structure
    for town in sorted(os.listdir(base_data_dir), key=natural_sort_key):
        town_path = os.path.join(base_data_dir, town)
        if not os.path.isdir(town_path):
            continue
        
        for weather in sorted(os.listdir(town_path), key=natural_sort_key):
            weather_path = os.path.join(town_path, weather)
            if not os.path.isdir(weather_path):
                continue
            
            for vehicle in sorted(os.listdir(weather_path), key=natural_sort_key):
                vehicle_path = os.path.join(weather_path, vehicle)
                if not os.path.isdir(vehicle_path):
                    continue
                
                for spawn_point in sorted(os.listdir(vehicle_path), key=natural_sort_key):
                    spawn_path = os.path.join(vehicle_path, spawn_point)
                    if not os.path.isdir(spawn_path):
                        continue
                    
                    for step in sorted(os.listdir(spawn_path), key=natural_sort_key):
                        step_path = os.path.join(spawn_path, step)
                        if not os.path.isdir(step_path):
                            continue
                        
                        ego_path = os.path.join(step_path, "ego_vehicle")
                        if os.path.isdir(ego_path):
                            spawnpoints.append((town, weather, vehicle, spawn_point, step))
    
    return spawnpoints


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_spawnpoint(town: str, weather: str, vehicle: str, spawn_point: str, 
                       step: str, pipe: Flux2KleinPipeline, gpu_id: int,
                       sp_idx: int, total_sp: int, start_time: datetime):
    """
    Process a single spawn point: create poster, run through Flux, slice and replace originals.
    """
    elapsed = format_time_elapsed(start_time)
    progress_pct = (sp_idx / total_sp) * 100
    
    print(f"\n{'='*80}")
    print(f"[GPU {gpu_id}] [{elapsed}] Progress: {sp_idx}/{total_sp} ({progress_pct:.1f}%)")
    print(f"[GPU {gpu_id}] Processing: {town}/{weather}/{vehicle}/{spawn_point}/{step}")
    print(f"{'='*80}")
    
    # Build base path
    base_path = os.path.join(BASE_DATA_DIR, town, weather, vehicle, spawn_point, step, "ego_vehicle")
    
    # Find all RGB images (naturally sorted)
    rgb_images = find_rgb_images(base_path)
    
    if not rgb_images:
        print(f"[GPU {gpu_id}] No RGB images found for this spawn point. Skipping.")
        return
    
    print(f"[GPU {gpu_id}] Found {len(rgb_images)} RGB images")
    
    # Create output directories
    poster_output_dir = os.path.join(TEMP_POSTER_DIR, town, weather, vehicle, spawn_point, step)
    os.makedirs(poster_output_dir, exist_ok=True)
    
    flux_output_dir = os.path.join(OUTPUT_BASE_DIR, "flux_outputs", town, weather, vehicle, spawn_point, step)
    os.makedirs(flux_output_dir, exist_ok=True)
    
    # Calculate number of posters needed
    num_posters = math.ceil(len(rgb_images) / IMAGES_PER_POSTER)
    print(f"[GPU {gpu_id}] Will create {num_posters} poster(s)")
    
    # Process each poster
    for poster_idx in range(num_posters):
        poster_start = datetime.now()
        print(f"\n[GPU {gpu_id}] --- Poster {poster_idx + 1}/{num_posters} ---")
        
        start_idx = poster_idx * IMAGES_PER_POSTER
        end_idx = min(start_idx + IMAGES_PER_POSTER, len(rgb_images))
        poster_images = rgb_images[start_idx:end_idx]
        
        # Create poster
        print(f"[GPU {gpu_id}] Creating poster from {len(poster_images)} images (indices {start_idx}-{end_idx-1})...")
        print(f"[GPU {gpu_id}] First image: {os.path.basename(poster_images[0])}")
        print(f"[GPU {gpu_id}] Last image: {os.path.basename(poster_images[-1])}")
        poster_filename = f"poster_{poster_idx:03d}.png"
        poster_path = os.path.join(poster_output_dir, poster_filename)
        
        metadata = create_poster(poster_images, poster_path, POSTER_SIZE)
        
        if metadata is None:
            continue
        
        print(f"[GPU {gpu_id}] ✓ Poster created: {poster_filename}")
        
        # Save metadata
        metadata_path = os.path.join(poster_output_dir, f"poster_{poster_idx:03d}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Run through Flux
        print(f"[GPU {gpu_id}] Running Flux2 inference ({NUM_INFERENCE_STEPS} steps)...")
        input_img = Image.open(poster_path)
        
        device = f"cuda:{gpu_id}"
        flux_start = datetime.now()
        with torch.no_grad():
            flux_output = pipe(
                prompt=FLUX_PROMPT,
                image=[input_img],
                height=POSTER_SIZE[1],
                width=POSTER_SIZE[0],
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=torch.Generator(device=device).manual_seed(RANDOM_SEED)
            ).images[0]
        
        flux_duration = (datetime.now() - flux_start).total_seconds()
        print(f"[GPU {gpu_id}] ✓ Flux inference completed in {flux_duration:.1f}s")
        
        # Save Flux output
        flux_output_filename = f"flux_poster_{poster_idx:03d}.png"
        flux_output_path = os.path.join(flux_output_dir, flux_output_filename)
        flux_output.save(flux_output_path)
        
        # Slice Flux output (slices are in natural order: 0,1,2,3...)
        print(f"[GPU {gpu_id}] Slicing output into {SLICE_SIZE}x{SLICE_SIZE} pieces...")
        slices = slice_image_to_grid(flux_output_path, SLICE_SIZE)
        print(f"[GPU {gpu_id}] ✓ Created {len(slices)} slices in natural order")
        
        # Map slices back to original images
        grid_rows, grid_cols = metadata['grid_size']
        slices_per_row = POSTER_SIZE[0] // SLICE_SIZE
        slices_per_col = POSTER_SIZE[1] // SLICE_SIZE
        
        # Calculate how many slices per original image
        slices_per_image_row = slices_per_row // grid_cols
        slices_per_image_col = slices_per_col // grid_rows
        
        # Replace original images with corresponding slices
        print(f"[GPU {gpu_id}] Overwriting originals with {SLICE_SIZE}x{SLICE_SIZE} slices...")
        replaced_count = 0
        
        # Process in natural order: 0,1,2,3...
        for img_info in sorted(metadata['image_positions'], key=lambda x: x['index']):
            img_idx = img_info['index']
            original_path = img_info['original_path']
            grid_row, grid_col = img_info['grid_position']
            
            # Calculate which slice corresponds to this image
            # Each image in the grid gets one slice from the corresponding position
            start_slice_row = grid_row * slices_per_image_col
            start_slice_col = grid_col * slices_per_image_row
            
            # Get the first (top-left) slice for this image position
            first_slice_idx = start_slice_row * slices_per_row + start_slice_col
            
            if first_slice_idx < len(slices):
                first_slice_img, _, _ = slices[first_slice_idx]
                
                # Overwrite the original image with this 256x256 slice
                first_slice_img.save(original_path)
                replaced_count += 1
        
        poster_duration = (datetime.now() - poster_start).total_seconds()
        print(f"[GPU {gpu_id}] ✓ Overwrote {replaced_count} original images with 256x256 slices (poster time: {poster_duration:.1f}s)")


def worker_process(gpu_id: int, spawnpoints: List[Tuple[str, str, str, str, str]]):
    """
    Worker process that handles a subset of spawn points on a specific GPU.
    """
    start_time = datetime.now()
    
    print(f"\n{'#'*80}")
    print(f"[GPU {gpu_id}] Starting worker at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[GPU {gpu_id}] Processing {len(spawnpoints)} spawn points")
    print(f"{'#'*80}\n")
    
    # Set the GPU for this process
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    # Load Flux pipeline for this GPU - load to CPU first, then move
    print(f"[GPU {gpu_id}] Loading Flux2 pipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        FLUX_MODEL_PATH,
        torch_dtype=DTYPE
    )
    # Manually move each component to the target GPU
    pipe.text_encoder = pipe.text_encoder.to(device)
    pipe.transformer = pipe.transformer.to(device)
    pipe.vae = pipe.vae.to(device)
    
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print(f"[GPU {gpu_id}] ✓ Pipeline loaded successfully!\n")
    
    # Process assigned spawn points
    total_sp = len(spawnpoints)
    for idx, (town, weather, vehicle, spawn_point, step) in enumerate(spawnpoints, 1):
        try:
            process_spawnpoint(town, weather, vehicle, spawn_point, step, pipe, gpu_id, 
                             idx, total_sp, start_time)
        except Exception as e:
            print(f"[GPU {gpu_id}] ✗ ERROR processing {town}/{weather}/{vehicle}/{spawn_point}/{step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed = format_time_elapsed(start_time)
    print(f"\n{'#'*80}")
    print(f"[GPU {gpu_id}] Worker completed! Total time: {elapsed}")
    print(f"[GPU {gpu_id}] Processed {total_sp} spawn points")
    print(f"{'#'*80}\n")


def main():
    """
    Main pipeline execution with parallel processing.
    """
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    pipeline_start = datetime.now()
    
    print("="*80)
    print("CARLA Flux2 Domain Translation Pipeline (Parallel)")
    print(f"Started at: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Base data directory: {BASE_DATA_DIR}")
    print(f"Flux model path: {FLUX_MODEL_PATH}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Temporary poster directory: {TEMP_POSTER_DIR}")
    print(f"GPUs to use: {GPU_IDS}")
    print(f"Output: Overwrite originals with first 256x256 slice (NATURAL SORT)")
    print("="*80)
    
    # Find all spawn points
    print("\nScanning for spawn points...")
    spawnpoints = get_all_spawnpoints(BASE_DATA_DIR)
    print(f"✓ Found {len(spawnpoints)} spawn point(s) to process\n")
    
    if not spawnpoints:
        print("✗ No spawn points found. Please check your BASE_DATA_DIR path.")
        return
    
    # Split spawn points across GPUs
    num_gpus = len(GPU_IDS)
    spawnpoints_per_gpu = [[] for _ in range(num_gpus)]
    
    for idx, spawnpoint in enumerate(spawnpoints):
        gpu_idx = idx % num_gpus
        spawnpoints_per_gpu[gpu_idx].append(spawnpoint)
    
    print("Work distribution:")
    for gpu_idx, gpu_id in enumerate(GPU_IDS):
        print(f"  GPU {gpu_id}: {len(spawnpoints_per_gpu[gpu_idx])} spawn points")
    print()
    
    # Create and start processes
    processes = []
    for gpu_idx, gpu_id in enumerate(GPU_IDS):
        if len(spawnpoints_per_gpu[gpu_idx]) > 0:
            p = Process(target=worker_process, args=(gpu_id, spawnpoints_per_gpu[gpu_idx]))
            p.start()
            processes.append(p)
    
    print(f"✓ Started {len(processes)} worker process(es)\n")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    pipeline_duration = format_time_elapsed(pipeline_start)
    
    print("\n" + "="*80)
    print("Pipeline completed!")
    print(f"Total time: {pipeline_duration}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()