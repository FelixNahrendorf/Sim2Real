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
BRIGHTNESS_THRESHOLD = 97 #82  # Tuned visually by checking images for every value between 90-10; Images below this mean are considered 'dark'

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
        cam_token = sample['data']['CAM_FRONT_RIGHT']
        cam_data = nusc.get('sample_data', cam_token)
        
        image_path = Path(nuscenes_root) / cam_data['filename']
        
        is_dark, brightness = is_image_dark(image_path, BRIGHTNESS_THRESHOLD)
        if brightness is not None:
            sample_brightness[sample['token']] = brightness
    
    # Classify night samples based on main threshold
    night_sample_tokens = {token for token, brightness in sample_brightness.items() 
                          if brightness < BRIGHTNESS_THRESHOLD}
    day_sample_tokens = {token for token, brightness in sample_brightness.items() 
                        if brightness >= BRIGHTNESS_THRESHOLD}
    
    print(f"\nTotal samples: {len(nusc.sample)}")
    print(f"Night samples (threshold={BRIGHTNESS_THRESHOLD}): {len(night_sample_tokens)}")
    print(f"Day samples: {len(day_sample_tokens)}")
    
    # Camera names
    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Write night sample tokens to file
    output_file = f'nuscenes_{version}_night_scenes_for_secogan.txt'
    with open(output_file, 'w') as f:
        for token in sorted(night_sample_tokens):
            f.write(f"{token}\n")
    print(f"Wrote {output_file}")
    
    # Write day sample tokens to file
    day_output_file = f'nuscenes_{version}_day_scenes_for_secogan.txt'
    with open(day_output_file, 'w') as f:
        for token in sorted(day_sample_tokens):
            f.write(f"{token}\n")
    print(f"Wrote {day_output_file}")
    
    # ========== WRITE FILE PATHS ==========
    # Collect all file paths for night scenes
    night_paths = []
    print(f"\nCollecting file paths for {len(night_sample_tokens)} night samples...")
    for sample_token in tqdm(sorted(night_sample_tokens)):
        sample = nusc.get('sample', sample_token)
        for cam_name in cam_names:
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            file_path = os.path.join(nuscenes_root, cam_data['filename'])
            night_paths.append(file_path)
    
    # Collect all file paths for day scenes
    day_paths = []
    print(f"Collecting file paths for {len(day_sample_tokens)} day samples...")
    for sample_token in tqdm(sorted(day_sample_tokens)):
        sample = nusc.get('sample', sample_token)
        for cam_name in cam_names:
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            file_path = os.path.join(nuscenes_root, cam_data['filename'])
            day_paths.append(file_path)
    
    # Write night file paths
    night_paths_file = f'nuscenes_{version}_night_filepaths_for_secogan.txt'
    with open(night_paths_file, 'w') as f:
        for path in sorted(night_paths):
            f.write(f"{path}\n")
    print(f"Wrote {len(night_paths)} night file paths to {night_paths_file}")
    
    # Write day file paths
    day_paths_file = f'nuscenes_{version}_day_filepaths_for_secogan.txt'
    with open(day_paths_file, 'w') as f:
        for path in sorted(day_paths):
            f.write(f"{path}\n")
    print(f"Wrote {len(day_paths)} day file paths to {day_paths_file}")
    # ==========================================
    
print(f"\n{'='*60}")
print("Processing complete!")
print('='*60)