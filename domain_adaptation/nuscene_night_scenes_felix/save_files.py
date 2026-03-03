#!/usr/bin/env python3
"""Copy images from a path list to an output directory."""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description='Copy images from a text file list to an output directory.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to text file containing image paths (one per line)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_images',
        help='Output directory for copied images (default: output_images)'
    )
    parser.add_argument(
        '--preserve-structure',
        action='store_true',
        help='Preserve original directory structure (default: flatten to single directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be copied without actually copying'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    print(f"{'='*60}")
    print(f"Copying images from {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Preserve structure: {args.preserve_structure}")
    print(f"Dry run: {args.dry_run}")
    print('='*60)
    
    # Read input file
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    with open(input_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(image_paths)} image paths in input file\n")
    
    # Create output directory
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    copied_count = 0
    skipped_count = 0
    name_collision_count = 0
    name_collision_map = {}  # Track how many times each filename appears
    
    # Process each image
    print("Processing images...")
    for image_path_str in tqdm(image_paths):
        source_path = Path(image_path_str)
        
        # Check if source exists
        if not source_path.exists():
            skipped_count += 1
            continue
        
        # Determine destination path
        if args.preserve_structure:
            # Preserve directory structure relative to root
            # Try to find common root (e.g., /app/datasets/nuscenes_full/)
            try:
                # Find the part after the common root
                parts = source_path.parts
                # Look for a recognizable root marker (adjust as needed)
                if 'samples' in parts:
                    idx = parts.index('samples')
                    relative_parts = parts[idx:]
                    dest_path = output_dir / Path(*relative_parts)
                else:
                    # Fallback: use last 3 parts of path
                    relative_parts = parts[-3:]
                    dest_path = output_dir / Path(*relative_parts)
            except:
                dest_path = output_dir / source_path.name
        else:
            # Flatten: all images go to output_dir root
            base_name = source_path.name
            
            # Handle name collisions by adding a counter
            if base_name in name_collision_map:
                name_collision_count += 1
                name_collision_map[base_name] += 1
                stem = source_path.stem
                suffix = source_path.suffix
                base_name = f"{stem}_{name_collision_map[base_name]:03d}{suffix}"
            else:
                name_collision_map[base_name] = 0
            
            dest_path = output_dir / base_name
        
        # Create parent directory if needed
        if not args.dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        try:
            if args.dry_run:
                print(f"Would copy: {source_path} -> {dest_path}")
            else:
                shutil.copy2(source_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"\nError copying {source_path}: {e}")
            skipped_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print('='*60)
    print(f"Total paths in input file: {len(image_paths)}")
    print(f"Successfully copied: {copied_count}")
    print(f"Skipped (not found/errors): {skipped_count}")
    if not args.preserve_structure and name_collision_count > 0:
        print(f"Name collisions resolved: {name_collision_count}")
    
    if not args.dry_run:
        print(f"\nImages saved to: {output_dir.absolute()}")
    print('='*60)

if __name__ == "__main__":
    main()


    
'''
# Basic usage - flatten all images to one directory
python copy_images_from_list.py night_scenes.txt

# Specify custom output directory
python copy_images_from_list.py night_scenes.txt --output-dir my_night_images

# Preserve directory structure
python copy_images_from_list.py night_scenes.txt --output-dir structured_output --preserve-structure

# Dry run to see what would be copied
python copy_images_from_list.py night_scenes.txt --dry-run

# Full example with all options
python copy_images_from_list.py all_cameras_daytime_paths_nuscenes_test.txt \
    --output-dir nuscenes_daytime_images \
    --preserve-structure

'''