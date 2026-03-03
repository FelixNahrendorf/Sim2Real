#!/usr/bin/env python3
"""Classify images from a path list into day and night scenes based on brightness."""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Try different image loading methods
try:
    import cv2
    USE_CV2 = hasattr(cv2, 'imread')
except ImportError:
    USE_CV2 = False

if not USE_CV2:
    try:
        from PIL import Image
        USE_PIL = True
    except ImportError:
        USE_PIL = False
        print("Warning: Neither cv2 nor PIL available for image loading")

def is_image_dark(image_path, brightness_threshold):
    """Calculates mean brightness and determines if the image is dark."""
    try:
        if USE_CV2:
            image = cv2.imread(str(image_path))
            if image is None:
                return False, None
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_image)
        elif USE_PIL:
            image = Image.open(image_path)
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            mean_brightness = np.mean(np.array(gray_image))
        else:
            return False, None
            
        return mean_brightness < brightness_threshold, mean_brightness
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(
        description='Classify images into day and night scenes based on brightness.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to text file containing image paths (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='night_scenes.txt',
        help='Output file for night scene paths (default: night_scenes.txt)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=82,
        help='Brightness threshold (default: 82, lower values = darker images)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    BRIGHTNESS_THRESHOLD = args.threshold
    input_file = Path(args.input_file)
    output_file = Path(args.output)
    
    print(f"{'='*60}")
    print(f"Image loading method: {'OpenCV' if USE_CV2 else 'PIL' if USE_PIL else 'None'}")
    print(f"Processing images from {input_file}")
    print(f"Brightness threshold: {BRIGHTNESS_THRESHOLD}")
    print('='*60)
    
    # Read input file
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    with open(input_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(image_paths)} image paths in input file")
    
    # Track brightness and night scenes
    night_scenes = []
    brightness_values = []
    skipped_count = 0
    error_count = 0
    
    # Process each image
    print("\nScanning images for brightness...")
    for image_path_str in tqdm(image_paths):
        image_path = Path(image_path_str)
        
        if not image_path.exists():
            skipped_count += 1
            continue
        
        is_dark, brightness = is_image_dark(image_path, BRIGHTNESS_THRESHOLD)
        
        if brightness is not None:
            brightness_values.append(brightness)
            if is_dark:
                night_scenes.append(image_path_str)
        else:
            error_count += 1
    
    # Statistics
    print(f"\n{'='*60}")
    print("Results:")
    print('='*60)
    print(f"Total images processed: {len(brightness_values)}")
    print(f"Skipped (not found): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Night scenes (threshold={BRIGHTNESS_THRESHOLD}): {len(night_scenes)}")
    print(f"Day scenes: {len(brightness_values) - len(night_scenes)}")
    
    if brightness_values:
        print(f"\nBrightness statistics:")
        print(f"  Min: {min(brightness_values):.1f}")
        print(f"  Max: {max(brightness_values):.1f}")
        print(f"  Mean: {np.mean(brightness_values):.1f}")
        print(f"  Median: {np.median(brightness_values):.1f}")
    
    # Write night scenes to output file
    if night_scenes:
        with open(output_file, 'w') as f:
            for path in night_scenes:
                f.write(f"{path}\n")
        
        print(f"\nWrote {len(night_scenes)} night scene paths to {output_file}")
    else:
        print("\nNo night scenes found!")
    
    print('='*60)

if __name__ == "__main__":
    main()

'''
# Basic usage with default threshold (82)
python find_nuscene_night_scenes_from_path_txt.py all_cameras_daytime_paths_nuscenes_test.txt --output night_scenes_found_in_day_scenes_test.txt

# Specify custom output file
python find_nuscene_night_scenes_from_path_txt.py input_paths.txt --output my_night_scenes.txt

# Use different brightness threshold
python find_nuscene_night_scenes_from_path_txt.py input_paths.txt --threshold 75 --output night_scenes_t75.txt
'''