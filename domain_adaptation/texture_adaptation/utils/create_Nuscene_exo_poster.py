import os
from PIL import Image
import math

def create_poster(image_paths, output_path, poster_target_size=None):
    """
    Create a poster image from a list of image paths.
    
    Args:
        image_paths: List of paths to images
        output_path: Path to save the poster
        poster_target_size: Tuple (width, height) to resize final poster, or None for original size
    """
    if not image_paths:
        print("No images found!")
        return
    
    # Target size for each individual image
    individual_img_size = (128, 128)
    
    # Load and resize all images to 128x128
    images = []
    for path in image_paths:
        img = Image.open(path)
        img_resized = img.resize(individual_img_size, Image.LANCZOS)
        images.append(img_resized)
    
    # Get dimensions of a single image
    img_width, img_height = individual_img_size
    
    # Calculate grid dimensions (make it square)
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = cols  # Make it square
    
    # Fill remaining spots with black images
    total_slots = rows * cols
    num_black_images = total_slots - num_images
    
    if num_black_images > 0:
        black_image = Image.new('RGB', individual_img_size, color='black')
        images.extend([black_image] * num_black_images)
    
    # Create poster canvas at original size (no gaps)
    poster_width = cols * img_width
    poster_height = rows * img_height
    poster = Image.new('RGB', (poster_width, poster_height), color='black')
    
    # Paste images into poster without gaps
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        poster.paste(img, (x, y))
    
    # Resize poster if target size specified
    if poster_target_size:
        poster = poster.resize(poster_target_size, Image.LANCZOS)
        print(f"Saved poster: {output_path}")
        print(f"  - Images: {num_images} (+ {num_black_images} black fill)")
        print(f"  - Each image: {img_width}x{img_height}px")
        print(f"  - Grid: {rows}x{cols}")
        print(f"  - Original size: {poster_width}x{poster_height}px")
        print(f"  - Resized to: {poster_target_size[0]}x{poster_target_size[1]}px")
    else:
        print(f"Saved poster: {output_path}")
        print(f"  - Images: {num_images} (+ {num_black_images} black fill)")
        print(f"  - Each image: {img_width}x{img_height}px")
        print(f"  - Grid: {rows}x{cols}")
        print(f"  - Size: {poster_width}x{poster_height}px")
    
    # Save poster
    poster.save(output_path)

def create_small_posters(image_paths, output_dir, images_per_poster=16, poster_size=(1024, 1024)):
    """
    Create multiple small posters with a fixed number of images each.
    
    Args:
        image_paths: List of paths to images
        output_dir: Directory to save posters
        images_per_poster: Number of images per poster (default 16 for 4x4 grid)
        poster_size: Final size of each poster
    """
    os.makedirs(output_dir, exist_ok=True)
    total_images = len(image_paths)
    num_posters = math.ceil(total_images / images_per_poster)
    
    print(f"\nCreating {num_posters} small posters with up to {images_per_poster} images each:")
    
    for poster_idx in range(num_posters):
        start_idx = poster_idx * images_per_poster
        end_idx = min(start_idx + images_per_poster, total_images)
        
        # Get subset of images for this poster
        poster_images = image_paths[start_idx:end_idx]
        
        # Create output filename
        output_filename = f"poster_small_{poster_idx+1:02d}_of_{num_posters:02d}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create poster
        print(f"\nPoster {poster_idx+1}/{num_posters}:")
        create_poster(poster_images, output_path, poster_target_size=poster_size)

# ============================================================================
# CONFIGURATION SECTION - Adjust these parameters as needed
# ============================================================================

# Define paths
nuscenes_ego = "/app/felix/data/pixelsplat_Sim2Real/outputs/pixelsplat_new_training_Diverse_Ego-Exo-Mixed-eval_ego-exo_Nuscenes_experiment17/seed/Carla_0/reference"
nuscenes_exo = "/app/felix/data/pixelsplat_Sim2Real/outputs/pixelsplat_new_training_Diverse_Ego-Exo-Mixed-eval_ego-exo_Nuscenes_experiment17/seed/Carla_0/color"

# Number of images to use from each directory (None = use all)
num_ego_images = 6 #None  # Set to a number like 50 to limit, or None for all
num_exo_images = 3 #None  # Set to a number like 50 to limit, or None for all

# Images per poster for small posters
images_per_poster = 9 #16  # Default: 16 images per poster (4x4 grid)

# Poster size for small posters
poster_size = (1024, 1024)  # Default: 1024x1024 pixels

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================

# Collect all RGB images
rgb_images = []

# From nuscenes_ego
if os.path.exists(nuscenes_ego):
    ego_files = sorted([f for f in os.listdir(nuscenes_ego) if f.endswith('.png')])
    if num_ego_images is not None:
        ego_files = ego_files[:num_ego_images]
    for filename in ego_files:
        rgb_images.append(os.path.join(nuscenes_ego, filename))
    print(f"Using {len(ego_files)} images from nuscenes_ego")

# From nuscenes_exo
if os.path.exists(nuscenes_exo):
    exo_files = sorted([f for f in os.listdir(nuscenes_exo) if f.endswith('.png')])
    if num_exo_images is not None:
        exo_files = exo_files[:num_exo_images]
    for filename in exo_files:
        rgb_images.append(os.path.join(nuscenes_exo, filename))
    print(f"Using {len(exo_files)} images from nuscenes_exo")

print(f"\nTotal: {len(rgb_images)} RGB images")

# Create output directory
output_dir = "/app/felix/data/neuralremaster/flux1.1/input/poster/Nuscene_exo_Carla_0_6ego_3exo"
os.makedirs(output_dir, exist_ok=True)

# Create poster with original size
create_poster(
    rgb_images,
    os.path.join(output_dir, "poster_original.png"),
    poster_target_size=None
)

# Create poster resized to 3000x3000px total
'''create_poster(
    rgb_images,
    os.path.join(output_dir, "poster_3008x3008.png"),
    poster_target_size=(3008, 3008)
)'''

# Create poster resized to 328x312px total (all images)
'''create_poster(
    rgb_images,
    os.path.join(output_dir, "poster_1024x1024.png"),
    poster_target_size=(1024, 1024)
)'''

# Create multiple small 1024x1024 posters with 16 images each
create_small_posters(
    rgb_images,
    os.path.join(output_dir, "poster"),
    images_per_poster=images_per_poster,
    poster_size=poster_size
)

print("\nDone! All posters saved to:", output_dir)