import os
import shutil
from pathlib import Path

def copy_transform_files():
    """
    Copy transform files from theo_Town02 to seed4d directory structure
    for all spawn points 1-101
    """
    
    # Define source and destination base paths
    source_base = "/mnt/share/felix/data/theo_Town02/Town02/ClearNoon/vehicle.tesla.invisible"
    dest_base = "/mnt/share/felix/data/seed4d/static/Town02/ClearNoon/vehicle.audi.tt"
    
    # Files to copy
    files_to_copy = [
        "transforms_ego_test.json",
        "transforms_ego_train.json", 
        "transforms_test.json",
        "transforms_train.json"
    ]
    
    # Counter for successful copies
    successful_copies = 0
    failed_copies = 0
    
    # Loop through spawn points 1-101
    for spawn_point in range(1, 102):
        print(f"Processing spawn_point_{spawn_point}...")
        
        # Define source and destination directories
        source_dir = Path(source_base) / f"spawn_point_{spawn_point}" / "step_0" / "sphere" / "transforms"
        dest_dir = Path(dest_base) / f"spawn_point_{spawn_point}" / "step_0" / "ego_vehicle" / "sphere" / "transforms"
        
        # Check if destination directory exists
        if not dest_dir.exists():
            print(f"  ✗ Destination directory does not exist: {dest_dir}")
            failed_copies += 1
            continue
        
        # Copy each file
        spawn_point_success = True
        for filename in files_to_copy:
            source_file = source_dir / filename
            dest_file = dest_dir / filename
            
            try:
                if source_file.exists():
                    shutil.copy2(source_file, dest_file)
                    print(f"  ✓ Copied {filename}")
                else:
                    print(f"  ✗ Source file not found: {source_file}")
                    spawn_point_success = False
            except Exception as e:
                print(f"  ✗ Error copying {filename}: {e}")
                spawn_point_success = False
        
        if spawn_point_success:
            successful_copies += 1
        else:
            failed_copies += 1
        
        print()  # Empty line for readability
    
    # Summary
    print("=" * 50)
    print("COPY OPERATION SUMMARY")
    print("=" * 50)
    print(f"Total spawn points processed: 101")
    print(f"Successful copies: {successful_copies}")
    print(f"Failed copies: {failed_copies}")
    print(f"Success rate: {(successful_copies/101)*100:.1f}%")

if __name__ == "__main__":
    print("Starting transform files copy operation...")
    print("Source: /app/data/theo_Town02/Town02/ClearNoon/vehicle.tesla.invisible")
    print("Target: /app/data/seed4d/static/Town02/ClearNoon/vehicle.audi.tt")
    print("Spawn points: 1-101")
    print("Files: transforms_ego_test.json, transforms_ego_train.json, transforms_test.json, transforms_train.json")
    print()
    
    try:
        copy_transform_files()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
