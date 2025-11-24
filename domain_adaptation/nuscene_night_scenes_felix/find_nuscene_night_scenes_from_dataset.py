#!/usr/bin/env python3
"""Classify NuScenes samples into day and night scenes based on image brightness."""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

def is_image_dark(image_path, brightness_threshold):
    """Calculates mean brightness and determines if the image is dark."""
    image = cv2.imread(str(image_path))
    if image is None:
        return False, None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_image)
    return mean_brightness < brightness_threshold, mean_brightness

# Configuration
nuscenes_root = "/app/datasets/nuscenes_full/"
BRIGHTNESS_THRESHOLD = 82  # Tuned visually by checking images for every value between 90-10; Images below this mean are considered 'dark'

# Process both versions
versions = ['v1.0-trainval', 'v1.0-test']

for version in versions:
    print(f"\n{'='*60}")
    print(f"Processing {version}")
    print('='*60)
    
    print("Initializing NuScenes...")
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
    
    # Dictionary to track night samples and their brightness values
    sample_brightness = {}  # {sample_token: brightness_value}
    
    # Check brightness of all samples
    print(f"Scanning {len(nusc.sample)} samples for brightness...")
    
    for sample in tqdm(nusc.sample):
        # Check CAM_FRONT for brightness (representative camera)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        
        image_path = Path(nuscenes_root) / cam_data['filename']
        
        is_dark, brightness = is_image_dark(image_path, BRIGHTNESS_THRESHOLD)
        if brightness is not None:
            sample_brightness[sample['token']] = brightness
    
    # Classify night samples based on main threshold
    night_sample_tokens = {token for token, brightness in sample_brightness.items() 
                          if brightness < BRIGHTNESS_THRESHOLD}
    
    print(f"\nTotal samples: {len(nusc.sample)}")
    print(f"Night samples (threshold={BRIGHTNESS_THRESHOLD}): {len(night_sample_tokens)}")
    print(f"Day samples: {len(nusc.sample) - len(night_sample_tokens)}")
    
    # Write night sample tokens to file
    output_file = f'nuscenes_{version}_night_scenes.txt'
    with open(output_file, 'w') as f:
        for token in sorted(night_sample_tokens):
            f.write(f"{token}\n")
    
    print(f"Wrote {output_file}")
    
    # ========== THRESHOLD TESTING ==========
    print(f"\nCreating threshold test samples...")
    threshold_parent_dir = Path("./find_threshold")
    threshold_parent_dir.mkdir(exist_ok=True)
    
    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # For each threshold from 10 to 89, save one sample
    for threshold in range(10, 90):
        # Find samples with brightness just below this threshold
        # (between threshold-5 and threshold to get representative samples)
        candidates = [token for token, brightness in sample_brightness.items() 
                     if threshold - 5 <= brightness < threshold]
        
        if candidates:
            # Take the first candidate
            sample_token = candidates[0]
            sample = nusc.get('sample', sample_token)
            brightness_val = sample_brightness[sample_token]
            
            # Create directory for this threshold
            threshold_dir = threshold_parent_dir / f"threshold={threshold}"
            threshold_dir.mkdir(exist_ok=True)
            
            # Save all 6 camera images
            output_dir = threshold_dir / f"sample_{sample_token[:8]}_brightness_{brightness_val:.1f}"
            output_dir.mkdir(exist_ok=True)
            
            for cam_name in cam_names:
                cam_token = sample['data'][cam_name]
                cam_data = nusc.get('sample_data', cam_token)
                
                src = Path(nuscenes_root) / cam_data['filename']
                dst = output_dir / f"{cam_name}.jpg"
                
                if src.exists():
                    shutil.copy2(src, dst)
            
            print(f"  Threshold {threshold}: saved sample with brightness {brightness_val:.1f}")
    
    print(f"\nThreshold test samples saved to {threshold_parent_dir}")
    # ========================================
    
    # Save images from every 300th night scene sample
    if night_sample_tokens:
        sorted_night_tokens = sorted(night_sample_tokens)
        samples_to_save = sorted_night_tokens[::300]  # Every 300th sample
        
        print(f"\nSaving images from {len(samples_to_save)} night scenes (every 300th)")
        
        # Create parent directory for this version
        parent_dir = Path(f"./night_samples_{version}")
        parent_dir.mkdir(exist_ok=True)
        
        for idx, sample_token in enumerate(samples_to_save):
            brightness_val = sample_brightness.get(sample_token, 0)
            print(f"\n  Sample {idx*300}: {sample_token} (brightness: {brightness_val:.1f})")
            
            # Get the sample
            sample = nusc.get('sample', sample_token)
            
            # Create output directory inside parent directory
            output_dir = parent_dir / f"sample_{idx*300:04d}_{sample_token[:8]}"
            output_dir.mkdir(exist_ok=True)
            
            img_count = 0
            for cam_name in cam_names:
                cam_token = sample['data'][cam_name]
                cam_data = nusc.get('sample_data', cam_token)
                
                # Source and destination paths
                src = Path(nuscenes_root) / cam_data['filename']
                dst = output_dir / f"{cam_name}_{cam_data['timestamp']}.jpg"
                
                if src.exists():
                    shutil.copy2(src, dst)
                    img_count += 1
            
            print(f"    Saved {img_count} images to {output_dir}")
    else:
        print("\nNo night scenes found!")

print(f"\n{'='*60}")
print("Processing complete!")
print('='*60)