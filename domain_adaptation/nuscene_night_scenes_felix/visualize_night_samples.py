#!/usr/bin/env python3
"""Create 20x20 grids from night scene samples."""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

def get_first_image_from_sample_dirs(parent_dir):
    """
    Get the first image from each sample directory.
    Returns list of image paths sorted by sample directory name.
    """
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        print(f"Directory not found: {parent_dir}")
        return []
    
    # Get all sample directories (those starting with 'sample_')
    sample_dirs = sorted([d for d in parent_path.iterdir() 
                         if d.is_dir() and d.name.startswith('sample_')])
    
    image_paths = []
    for sample_dir in sample_dirs:
        # Get all image files in the directory
        images = sorted([f for f in sample_dir.iterdir() 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if images:
            # Take the first image
            image_paths.append(images[0])
    
    return image_paths

def create_grid(image_paths, output_path, grid_size=(20, 20), target_size=(128, 128)):
    """
    Create a grid of images.
    
    Args:
        image_paths: List of paths to images (should be <= grid_size[0] * grid_size[1])
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
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(img_path)
            # Resize to target size
            img_resized = img.resize(target_size, Image.LANCZOS)
            images.append(img_resized)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
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
    print(f"Grid saved successfully! ({len(image_paths)} samples used, {total_slots} total slots)")

def create_multiple_grids(image_paths, output_dir, dataset_name, grid_size=(20, 20), target_size=(128, 128)):
    """
    Create multiple grids to show all images.
    
    Args:
        image_paths: List of all image paths
        output_dir: Directory to save grids
        dataset_name: Name prefix for grid files
        grid_size: Tuple of (rows, cols)
        target_size: Size to resize each image to
    """
    rows, cols = grid_size
    images_per_grid = rows * cols
    
    total_images = len(image_paths)
    num_grids = (total_images + images_per_grid - 1) // images_per_grid  # Ceiling division
    
    print(f"\nTotal samples: {total_images}")
    print(f"Images per grid: {images_per_grid}")
    print(f"Creating {num_grids} grid(s)...")
    
    for grid_idx in range(num_grids):
        start_idx = grid_idx * images_per_grid
        end_idx = min(start_idx + images_per_grid, total_images)
        
        grid_images = image_paths[start_idx:end_idx]
        
        print(f"\n--- Grid {grid_idx + 1}/{num_grids} ---")
        print(f"Images {start_idx + 1} to {end_idx} (total: {len(grid_images)})")
        
        # Create output path
        if num_grids == 1:
            output_path = output_dir / f"grid_{grid_size[0]}x{grid_size[1]}.jpg"
        else:
            output_path = output_dir / f"grid_{grid_size[0]}x{grid_size[1]}_part{grid_idx + 1:02d}.jpg"
        
        # Create the grid
        create_grid(grid_images, output_path, grid_size=grid_size, target_size=target_size)

def main():
    # Configuration
    base_dir = Path("./night_samples")
    
    datasets = [
        "night_images_v1.0-test_robust",
        "night_images_v1.0-trainval_robust"
    ]
    
    grid_size = (20, 20)  # 20x20 grid (400 images total)
    target_size = (128, 128)  # Size of each thumbnail in the grid
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset}")
        print('='*60)
        
        # Get input directory
        input_dir = base_dir / dataset
        
        if not input_dir.exists():
            print(f"Skipping {dataset} - directory not found")
            continue
        
        # Get first image from each sample
        image_paths = get_first_image_from_sample_dirs(input_dir)
        
        if not image_paths:
            print(f"No images found in {input_dir}")
            continue
        
        print(f"Found {len(image_paths)} sample directories with images")
        
        # Create multiple grids to show all samples
        create_multiple_grids(image_paths, input_dir, dataset, 
                            grid_size=grid_size, target_size=target_size)
    
    print(f"\n{'='*60}")
    print("All grids created successfully!")
    print('='*60)

if __name__ == "__main__":
    main()