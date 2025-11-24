#!/usr/bin/env python3
"""Classify NuScenes samples into bright and dark scenes based on input list."""

import shutil
from pathlib import Path
from nuscenes.nuscenes import NuScenes

# Paths
input_file = "/app/code/Sim2Real/domain_adaptation/CAM_FRONT_non_dark.txt"  # Input file with bright scene paths
nuscenes_root = "/app/datasets/nuscenes_full/"

# Process both versions
versions = ['v1.0-trainval', 'v1.0-test']

for version in versions:
    print(f"\n{'='*60}")
    print(f"Processing {version}")
    print('='*60)
    
    print("Initializing NuScenes...")
    # Initialize NuScenes (like in dataset_seed4d.py)
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
    
    # Read input file and extract timestamps
    print(f"Reading {input_file}...")
    bright_timestamps = set()
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract timestamp from path
                timestamp = int(line.split('__')[-1].replace('.jpg', ''))
                bright_timestamps.add(timestamp)
    
    print(f"Found {len(bright_timestamps)} bright scene timestamps")
    
    # Find sample_tokens for bright timestamps
    print("Mapping timestamps to sample_tokens...")
    bright_sample_tokens = set()
    timestamp_to_sample = {}
    
    for sample_data in nusc.sample_data:
        timestamp = sample_data['timestamp']
        sample_token = sample_data['sample_token']
        timestamp_to_sample[timestamp] = sample_token
        
        if timestamp in bright_timestamps:
            bright_sample_tokens.add(sample_token)
    
    print(f"Found {len(bright_sample_tokens)} unique bright sample_tokens")
    
    # Get all sample_tokens from nuScenes
    all_sample_tokens = set()
    for sample in nusc.sample:
        all_sample_tokens.add(sample['token'])
    
    # Dark scenes are all samples not in bright scenes
    dark_sample_tokens = all_sample_tokens - bright_sample_tokens
    
    print(f"Total samples: {len(all_sample_tokens)}")
    print(f"Bright samples: {len(bright_sample_tokens)}")
    print(f"Dark samples: {len(dark_sample_tokens)}")
    
    # Version-specific output filenames
    version_suffix = version.replace('.', '_')
    bright_output = f'nuscenes_bright_scenes_{version_suffix}.txt'
    dark_output = f'nuscenes_dark_scenes_{version_suffix}.txt'
    
    # Write bright sample tokens to file
    with open(bright_output, 'w') as f:
        for token in sorted(bright_sample_tokens):
            f.write(f"{token}\n")
    
    # Write dark sample tokens to file
    with open(dark_output, 'w') as f:
        for token in sorted(dark_sample_tokens):
            f.write(f"{token}\n")
    
    print(f"\nWrote {bright_output}")
    print(f"Wrote {dark_output}")
    
    # Save images from every 500th dark scene sample
    if dark_sample_tokens:
        sorted_dark_tokens = sorted(dark_sample_tokens)
        samples_to_save = sorted_dark_tokens[::500]  # Every 100th sample
        
        print(f"\nSaving images from {len(samples_to_save)} dark scenes (every 100th)")
        
        # Create parent directory for this version
        parent_dir = Path(f"./dark_samples_{version}")
        parent_dir.mkdir(exist_ok=True)
        
        # Get all camera data for this sample
        cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        for idx, sample_token in enumerate(samples_to_save):
            print(f"\n  Sample {idx*500}: {sample_token}")
            
            # Get the sample
            sample = nusc.get('sample', sample_token)
            
            # Create output directory inside parent directory
            output_dir = parent_dir / f"sample_{idx*500:04d}_{sample_token[:8]}"
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
        print("\nNo dark scenes found!")

print(f"\n{'='*60}")
print("Processing complete!")
print('='*60)