#!/usr/bin/env python3
"""Create 50x50 grids from day and night scene filepaths."""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

def read_filepaths(filepath_txt):
    """
    Read image filepaths from a text file.
    Returns list of relative paths.
    """
    filepaths = []
    with open(filepath_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                filepaths.append(line)
    return filepaths

def create_grid(image_paths, nuscenes_root, output_path, grid_size=(50, 50), target_size=(64, 64)):
    """
    Create a grid of images.
    
    Args:
        image_paths: List of relative paths to images
        nuscenes_root: Root directory of nuscenes dataset
        output_path: Where to save the grid
        grid_size: Tuple of (rows, cols)
        target_size: Size to resize each image to
    """
    rows, cols = grid_size
    total_slots = rows * cols
    
    print(f"Creating grid with {len(image_paths)} images (max {total_slots} slots)...")
    
    # Create list of images, padding with white images if needed
    images = []
    
    # Load actual images
    for rel_path in tqdm(image_paths, desc="Loading images"):
        try:
            # Construct full path
            full_path = Path(nuscenes_root) / rel_path
            
            img = Image.open(full_path)
            # Resize to target size
            img_resized = img.resize(target_size, Image.LANCZOS)
            images.append(img_resized)
        except Exception as e:
            print(f"Error loading {rel_path}: {e}")
            # Add white image as fallback
            white_img = Image.new('RGB', target_size, color='white')
            images.append(white_img)
    
    # Fill remaining slots with white images
    if len(images) < total_slots:
        print(f"Padding with {total_slots - len(images)} white images...")
        for _ in range(total_slots - len(images)):
            white_img = Image.new('RGB', target_size, color='white')
            images.append(white_img)
    
    # Create the grid
    print("Assembling grid...")
    grid_width = cols * target_size[0]
    grid_height = rows * target_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images into grid
    for idx, img in enumerate(tqdm(images, desc="Creating grid")):
        row = idx // cols
        col = idx % cols
        x = col * target_size[0]
        y = row * target_size[1]
        grid_image.paste(img, (x, y))
    
    # Save the grid
    print(f"Saving grid to {output_path}...")
    grid_image.save(output_path, quality=95)
    print(f"Grid saved successfully! ({len(image_paths)} images used, {total_slots} total slots)")

def create_multiple_grids(image_paths, nuscenes_root, output_dir, prefix, grid_size=(50, 50), target_size=(64, 64)):
    """
    Create multiple grids to show all images.
    
    Args:
        image_paths: List of all relative image paths
        nuscenes_root: Root directory of nuscenes dataset
        output_dir: Directory to save grids
        prefix: Name prefix for grid files (e.g., 'day' or 'night')
        grid_size: Tuple of (rows, cols)
        target_size: Size to resize each image to
    """
    rows, cols = grid_size
    images_per_grid = rows * cols
    
    total_images = len(image_paths)
    num_grids = (total_images + images_per_grid - 1) // images_per_grid  # Ceiling division
    
    print(f"\nTotal images: {total_images}")
    print(f"Images per grid: {images_per_grid}")
    print(f"Creating {num_grids} grid(s) for {prefix}...")
    
    for grid_idx in range(num_grids):
        start_idx = grid_idx * images_per_grid
        end_idx = min(start_idx + images_per_grid, total_images)
        
        grid_images = image_paths[start_idx:end_idx]
        
        print(f"\n--- {prefix.capitalize()} Grid {grid_idx + 1}/{num_grids} ---")
        print(f"Images {start_idx + 1} to {end_idx} (total: {len(grid_images)})")
        
        # Create output path
        if num_grids == 1:
            output_path = output_dir / f"{prefix}_grid_{grid_size[0]}x{grid_size[1]}.jpg"
        else:
            output_path = output_dir / f"{prefix}_grid_{grid_size[0]}x{grid_size[1]}_part{grid_idx + 1:02d}.jpg"
        
        # Create the grid
        create_grid(grid_images, nuscenes_root, output_path, grid_size=grid_size, target_size=target_size)

def main():
    # Configuration
    nuscenes_root = "/app/datasets/nuscenes_full/"
    base_dir = Path("./night_samples_for_secogan")
    
    # Files to process
    filepath_files = {
        'day': base_dir / "nuscenes_v1.0-trainval_day_filepaths_for_secogan.txt", 
        'night': base_dir / "nuscenes_v1.0-trainval_night_filepaths_for_secogan.txt"
    }
    
    grid_size = (50, 50)  # 50x50 grid (2500 images total)
    target_size = (64, 64)  # Size of each thumbnail in the grid
    
    for category, filepath_file in filepath_files.items():
        print(f"\n{'='*60}")
        print(f"Processing {category} scenes from {filepath_file.name}")
        print('='*60)
        
        if not filepath_file.exists():
            print(f"Skipping {category} - file not found: {filepath_file}")
            continue
        
        # Read filepaths from text file
        print(f"Reading filepaths from {filepath_file}...")
        image_paths = read_filepaths(filepath_file)
        
        if not image_paths:
            print(f"No filepaths found in {filepath_file}")
            continue
        
        print(f"Found {len(image_paths)} image paths")
        
        # Create output directory for this category
        output_dir = base_dir / f"trainval_{category}_grids"
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create multiple grids to show all images
        create_multiple_grids(image_paths, nuscenes_root, output_dir, category,
                            grid_size=grid_size, target_size=target_size)
    
    print(f"\n{'='*60}")
    print("All grids created successfully!")
    print('='*60)

if __name__ == "__main__":
    main()