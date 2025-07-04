import os
import shutil
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image

# Configuration
NUSCENE_DATA_DIR = "/app/datasets/nuscenes_full/"
OUTPUT_DIR = "/app/outputs/ego_exo_nuscene_on_carla_ckpt_not_changed_transform_debug"

# Camera names mapping
NUSCENES_CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

# Output filename mapping (camera name -> output filename)
CAMERA_TO_FILENAME = {
    'CAM_FRONT': 'CAMERA_FRONT.png',
    'CAM_FRONT_RIGHT': 'CAMERA_FRONT_RIGHT.png', 
    'CAM_FRONT_LEFT': 'CAMERA_FRONT_LEFT.png',
    'CAM_BACK': 'CAMERA_BACK.png',
    'CAM_BACK_LEFT': 'CAMERA_BACK_LEFT.png',
    'CAM_BACK_RIGHT': 'CAMERA_BACK_RIGHT.png'
}

def extract_nuscenes_camera_images(num_samples=5, version='v1.0-mini'):
    """
    Extract camera images from NuScenes dataset and save with specified naming convention.
    
    Args:
        num_samples (int): Number of samples to process
        version (str): NuScenes version to use
    """
    
    # Initialize NuScenes
    print(f"Initializing NuScenes dataset (version: {version})...")
    nusc = NuScenes(version=version, dataroot=NUSCENE_DATA_DIR)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}")
    
    # Get scenes for the dataset
    all_splits = create_splits_scenes()
    if version == 'v1.0-mini':
        scene_names = all_splits['mini_train'][:3]  # Use first 3 scenes for demo
    elif version == 'v1.0-trainval':
        scene_names = all_splits['train'][:3]
    else:
        raise NotImplementedError(f"Version {version} not supported")
    
    # Get samples from selected scenes
    samples = []
    for scene in nusc.scene:
        if scene["name"] in scene_names:
            sample_token = scene["first_sample_token"]
            count = 0
            while sample_token != "" and count < num_samples:
                samples.append(sample_token)
                sample = nusc.get("sample", sample_token)
                sample_token = sample["next"]
                count += 1
    
    print(f"Found {len(samples)} samples to process")
    
    # Process each sample
    for sample_idx, sample_token in enumerate(samples):
        print(f"\nProcessing sample {sample_idx + 1}/{len(samples)}: {sample_token}")
        
        # Create sample-specific directory
        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:03d}_{sample_token[:8]}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get sample data
        sample = nusc.get("sample", sample_token)
        
        # Process each camera
        for camera_name in NUSCENES_CAMERAS:
            if camera_name in sample["data"]:
                # Get camera data
                camera_token = sample["data"][camera_name]
                camera_data = nusc.get("sample_data", camera_token)
                
                # Source image path
                source_path = os.path.join(NUSCENE_DATA_DIR, camera_data["filename"])
                
                # Output filename
                output_filename = CAMERA_TO_FILENAME[camera_name]
                output_path = os.path.join(sample_dir, output_filename)
                
                # Copy and convert image
                if os.path.exists(source_path):
                    try:
                        # Load image and convert to PNG
                        with Image.open(source_path) as img:
                            # Convert to RGB if needed (in case of RGBA or other formats)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Save as PNG
                            img.save(output_path, 'PNG')
                        
                        print(f"  ✅ {camera_name} -> {output_filename}")
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {camera_name}: {e}")
                        
                else:
                    print(f"  ⚠️  Source image not found: {source_path}")
            else:
                print(f"  ⚠️  Camera {camera_name} not available in this sample")
        
        # Create a summary file for the sample
        summary_path = os.path.join(sample_dir, "sample_info.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Sample Token: {sample_token}\n")
            f.write(f"Scene: {sample['scene_token']}\n")
            f.write(f"Timestamp: {sample['timestamp']}\n")
            f.write(f"Available Cameras:\n")
            for camera_name in NUSCENES_CAMERAS:
                if camera_name in sample["data"]:
                    camera_token = sample["data"][camera_name]
                    camera_data = nusc.get("sample_data", camera_token)
                    f.write(f"  {camera_name}: {camera_data['filename']}\n")
                else:
                    f.write(f"  {camera_name}: NOT AVAILABLE\n")
    
    print(f"\n🎉 Extraction complete!")
    print(f"📁 Images saved to: {OUTPUT_DIR}")
    print(f"📊 Processed {len(samples)} samples")
    
    # Create overall summary
    summary_file = os.path.join(OUTPUT_DIR, "extraction_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("NuScenes Camera Image Extraction Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset Version: {version}\n")
        f.write(f"Source Directory: {NUSCENE_DATA_DIR}\n")
        f.write(f"Output Directory: {OUTPUT_DIR}\n")
        f.write(f"Total Samples Processed: {len(samples)}\n")
        f.write(f"Camera Names: {', '.join(NUSCENES_CAMERAS)}\n")
        f.write(f"Output Naming Convention:\n")
        for camera, filename in CAMERA_TO_FILENAME.items():
            f.write(f"  {camera} -> {filename}\n")

def main():
    """Main function to run the extraction."""
    try:
        # Check if source directory exists
        if not os.path.exists(NUSCENE_DATA_DIR):
            print(f"❌ Error: NuScenes data directory not found: {NUSCENE_DATA_DIR}")
            print("Please update NUSCENE_DATA_DIR to point to your NuScenes dataset location.")
            return
        
        # Run extraction
        extract_nuscenes_camera_images(num_samples=3, version='v1.0-mini')
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
