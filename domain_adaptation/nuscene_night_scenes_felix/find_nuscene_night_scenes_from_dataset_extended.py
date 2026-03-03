#!/usr/bin/env python3
"""Classify NuScenes samples into day and night scenes based on image brightness."""

import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from PIL import Image

def analyze_image_lighting(image_path):
    """
    Comprehensive lighting analysis of an image.
    Returns multiple metrics for robust day/night classification.
    """
    try:
        image = Image.open(str(image_path)).convert('RGB')
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert RGB to grayscale manually
    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    
    # Convert RGB to HSV manually
    img_normalized = img_array / 255.0
    r, g, b = img_normalized[:, :, 0], img_normalized[:, :, 1], img_normalized[:, :, 2]
    
    max_rgb = np.max(img_normalized, axis=2)
    min_rgb = np.min(img_normalized, axis=2)
    diff = max_rgb - min_rgb
    
    # V channel is just the max
    v_channel = max_rgb * 255
    
    metrics = {}
    
    # 1. Mean brightness (original method)
    metrics['mean_brightness'] = np.mean(gray)
    
    # 2. Median brightness (more robust to outliers like streetlights)
    metrics['median_brightness'] = np.median(gray)
    
    # 3. Percentage of dark pixels (pixels below threshold)
    dark_pixel_threshold = 50
    metrics['dark_pixel_ratio'] = np.sum(gray < dark_pixel_threshold) / gray.size
    
    # 4. HSV Value channel (represents brightness in HSV space)
    metrics['hsv_value_mean'] = np.mean(v_channel)
    
    # 5. Standard deviation (night scenes often have high variance due to artificial lights)
    metrics['brightness_std'] = np.std(gray)
    
    # 6. Histogram analysis - check distribution
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize
    
    # Check if most pixels are in lower brightness range (0-85)
    metrics['low_brightness_mass'] = np.sum(hist[:85])
    
    # 7. Upper percentile brightness (90th percentile - helps ignore bright outliers)
    metrics['percentile_90'] = np.percentile(gray, 90)
    
    return metrics

def is_night_scene_robust(image_path, thresholds=None):
    """
    Robust night scene detection using multiple metrics.
    
    Default thresholds tuned for NuScenes (VERY STRICT to catch all night scenes):
    - mean_brightness < 95 (RAISED - catches brighter night scenes)
    - median_brightness < 85 (RAISED - allows for bright artificial lights)
    - dark_pixel_ratio > 0.55 (LOWERED - less strict, catches scenes with more bright areas)
    - hsv_value_mean < 90 (RAISED - more permissive)
    - low_brightness_mass > 0.55 (LOWERED - less strict)
    - percentile_90 < 140 (RAISED - allows bright headlights and structures)
    """
    if thresholds is None:
        thresholds = {
            'mean_brightness': 87,      
            'median_brightness': 75,    
            'dark_pixel_ratio': 0.55,   
            'hsv_value_mean': 85,       
            'low_brightness_mass': 0.55,  
            'percentile_90': 120        
        }
    
    metrics = analyze_image_lighting(image_path)
    if metrics is None:
        return False, None
    
    # Voting system: count how many metrics indicate night
    votes = 0
    total_checks = 0
    
    # Check each metric
    if metrics['mean_brightness'] < thresholds['mean_brightness']:
        votes += 1
    total_checks += 1
    
    if metrics['median_brightness'] < thresholds['median_brightness']:
        votes += 1
    total_checks += 1
    
    if metrics['dark_pixel_ratio'] > thresholds['dark_pixel_ratio']:
        votes += 1
    total_checks += 1
    
    if metrics['hsv_value_mean'] < thresholds['hsv_value_mean']:
        votes += 1
    total_checks += 1
    
    if metrics['low_brightness_mass'] > thresholds['low_brightness_mass']:
        votes += 1
    total_checks += 1
    
    if metrics['percentile_90'] < thresholds['percentile_90']:
        votes += 1
    total_checks += 1
    
    # Require at least 3 out of 6 metrics to indicate night (aggressive to catch all night scenes)
    # This ensures scenes with bright artificial lights are still classified as night
    is_night = votes >= 3
    
    return is_night, metrics

def resize_and_save_image(src_path, dst_path, scale=0.2):
    """Resize image to specified scale and save."""
    try:
        image = Image.open(str(src_path))
        
        # Calculate new dimensions (20% of original)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # Resize image using LANCZOS for high quality downsampling
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Save resized image
        resized.save(str(dst_path), quality=85)
        return True
    except Exception as e:
        print(f"Error resizing {src_path}: {e}")
        return False

# Configuration
nuscenes_root = "/app/datasets/nuscenes_full/"
RESIZE_SCALE = 0.2  # Save images at 20% of original resolution

# Choose detection method: 'simple' or 'robust'
DETECTION_METHOD = 'robust'  # Change to 'simple' for original method

# Simple method threshold
BRIGHTNESS_THRESHOLD = 85

# Create output directory
output_dir = Path("./night_samples_robust")
output_dir.mkdir(exist_ok=True)

# Process both versions
versions = ['v1.0-trainval', 'v1.0-test']

for version in versions:
    print(f"\n{'='*60}")
    print(f"Processing {version}")
    print('='*60)
    
    print("Initializing NuScenes...")
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
    
    # Dictionary to track samples and their metrics
    sample_metrics = {}  # {sample_token: metrics_dict}
    sample_is_night = {}  # {sample_token: bool}
    sample_file_paths = {}  # {sample_token: {cam_name: filepath}} - CACHE FILE PATHS
    
    # Camera names to check
    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Check lighting of all samples - if ANY camera detects night, the whole sample is night
    print(f"Scanning {len(nusc.sample)} samples for lighting conditions...")
    print(f"Using {DETECTION_METHOD} detection method")
    print(f"Using scene-level propagation: if ANY camera detects night, entire sample is marked as night")
    
    for sample in tqdm(nusc.sample):
        # Initialize storage for this sample
        sample_token = sample['token']
        sample_metrics[sample_token] = {}
        sample_file_paths[sample_token] = {}
        
        # Check all cameras for this sample
        is_sample_night = False
        
        for cam_name in cam_names:
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            filepath = cam_data['filename']
            
            # CACHE THE FILE PATH
            sample_file_paths[sample_token][cam_name] = filepath
            
            # Build full path for analysis
            image_path = Path(nuscenes_root) / filepath
            
            if DETECTION_METHOD == 'robust':
                is_night, metrics = is_night_scene_robust(image_path)
                if metrics is not None:
                    sample_metrics[sample_token][cam_name] = metrics
                    # If ANY camera detects night, mark the whole sample as night
                    if is_night:
                        is_sample_night = True
            else:
                # Original simple method
                try:
                    image = Image.open(str(image_path)).convert('L')  # Convert to grayscale
                    img_array = np.array(image)
                    mean_brightness = np.mean(img_array)
                    sample_metrics[sample_token][cam_name] = {'mean_brightness': mean_brightness}
                    if mean_brightness < BRIGHTNESS_THRESHOLD:
                        is_sample_night = True
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        
        # Store the final decision for this sample
        sample_is_night[sample_token] = is_sample_night
    
    # Classify samples
    night_sample_tokens = {token for token, is_night in sample_is_night.items() if is_night}
    day_sample_tokens = {token for token, is_night in sample_is_night.items() if not is_night}
    
    print(f"\nTotal samples: {len(nusc.sample)}")
    print(f"Night samples: {len(night_sample_tokens)}")
    print(f"Day samples: {len(day_sample_tokens)}")
    
    # Print some example metrics for night scenes
    if night_sample_tokens and DETECTION_METHOD == 'robust':
        print("\nExample night scene metrics (first sample, CAM_FRONT):")
        example_token = sorted(night_sample_tokens)[0]
        if 'CAM_FRONT' in sample_metrics[example_token]:
            example_metrics = sample_metrics[example_token]['CAM_FRONT']
            for key, value in example_metrics.items():
                print(f"  {key}: {value:.2f}")
    
    # ========== WRITE SAMPLE TOKENS ==========
    # Write night sample tokens to file
    night_tokens_file = output_dir / f'nuscenes_{version}_night_scenes_{DETECTION_METHOD}.txt'
    with open(night_tokens_file, 'w') as f:
        for token in sorted(night_sample_tokens):
            f.write(f"{token}\n")
    print(f"\nWrote {night_tokens_file}")
    
    # Write day sample tokens to file
    day_tokens_file = output_dir / f'nuscenes_{version}_day_scenes_{DETECTION_METHOD}.txt'
    with open(day_tokens_file, 'w') as f:
        for token in sorted(day_sample_tokens):
            f.write(f"{token}\n")
    print(f"Wrote {day_tokens_file}")
    # =========================================
    
    # ========== WRITE FILE PATHS (USING CACHE) ==========
    # Collect all file paths for night scenes - NO MORE nusc.get() calls!
    night_paths = []
    print(f"\nCollecting file paths for {len(night_sample_tokens)} night samples...")
    for sample_token in sorted(night_sample_tokens):
        for cam_name in cam_names:
            filepath = sample_file_paths[sample_token][cam_name]
            night_paths.append(filepath)
    
    # Collect all file paths for day scenes
    day_paths = []
    print(f"Collecting file paths for {len(day_sample_tokens)} day samples...")
    for sample_token in sorted(day_sample_tokens):
        for cam_name in cam_names:
            filepath = sample_file_paths[sample_token][cam_name]
            day_paths.append(filepath)
    
    # Write night file paths
    night_paths_file = output_dir / f'nuscenes_{version}_night_filepaths_{DETECTION_METHOD}.txt'
    with open(night_paths_file, 'w') as f:
        for path in sorted(night_paths):
            f.write(f"{path}\n")
    print(f"Wrote {len(night_paths)} night file paths to {night_paths_file}")
    
    # Write day file paths
    day_paths_file = output_dir / f'nuscenes_{version}_day_filepaths_{DETECTION_METHOD}.txt'
    with open(day_paths_file, 'w') as f:
        for path in sorted(day_paths):
            f.write(f"{path}\n")
    print(f"Wrote {len(day_paths)} day file paths to {day_paths_file}")
    # ==========================================
    
    # ========== SAVE COMPARISON SAMPLES (USING CACHE) ==========
    # Save a few samples from each category for visual inspection
    print(f"\nSaving comparison samples...")
    comparison_dir = output_dir / f"comparison_{DETECTION_METHOD}_{version}"
    comparison_dir.mkdir(exist_ok=True)
    
    # Save 5 night samples
    night_comparison_dir = comparison_dir / "night"
    night_comparison_dir.mkdir(exist_ok=True)
    
    for idx, sample_token in enumerate(sorted(night_sample_tokens)[:5]):
        filepath = sample_file_paths[sample_token]['CAM_FRONT']
        src = Path(nuscenes_root) / filepath
        dst = night_comparison_dir / f"night_{idx}_{sample_token[:8]}.jpg"
        
        if src.exists():
            shutil.copy2(src, dst)
    
    # Save 5 day samples
    day_comparison_dir = comparison_dir / "day"
    day_comparison_dir.mkdir(exist_ok=True)
    
    for idx, sample_token in enumerate(sorted(day_sample_tokens)[:5]):
        filepath = sample_file_paths[sample_token]['CAM_FRONT']
        src = Path(nuscenes_root) / filepath
        dst = day_comparison_dir / f"day_{idx}_{sample_token[:8]}.jpg"
        
        if src.exists():
            shutil.copy2(src, dst)
    
    print(f"Saved comparison samples to {comparison_dir}")
    # =============================================
    
    # ========== SAVE ALL NIGHT SCENE IMAGES AT 20% RESOLUTION (USING CACHE) ==========
    if night_sample_tokens:
        sorted_night_tokens = sorted(night_sample_tokens)
        
        print(f"\nSaving images from ALL {len(sorted_night_tokens)} night scenes at {int(RESIZE_SCALE*100)}% resolution")
        
        # Create parent directory for this version inside night_samples
        parent_dir = output_dir / f"night_images_{version}_{DETECTION_METHOD}"
        parent_dir.mkdir(exist_ok=True)
        
        total_images = 0
        for idx, sample_token in enumerate(tqdm(sorted_night_tokens, desc=f"Saving {version} night images")):
            # Create output directory inside parent directory
            sample_output_dir = parent_dir / f"sample_{idx:04d}_{sample_token[:8]}"
            sample_output_dir.mkdir(exist_ok=True)
            
            for cam_name in cam_names:
                filepath = sample_file_paths[sample_token][cam_name]
                
                # Source and destination paths
                src = Path(nuscenes_root) / filepath
                
                # Extract timestamp from the cached filepath (filename format: {timestamp}.jpg)
                timestamp = Path(filepath).stem
                dst = sample_output_dir / f"{cam_name}_{timestamp}.jpg"
                
                if src.exists():
                    # Resize and save instead of copy
                    if resize_and_save_image(src, dst, scale=RESIZE_SCALE):
                        total_images += 1
        
        print(f"Saved {total_images} resized images ({int(RESIZE_SCALE*100)}% resolution) from {len(sorted_night_tokens)} night samples to {parent_dir}")
    else:
        print("\nNo night scenes found!")
    # =================================================

print(f"\n{'='*60}")
print("Processing complete!")
print('='*60)