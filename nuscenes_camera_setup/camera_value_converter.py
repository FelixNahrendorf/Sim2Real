import json
import math
import argparse
import os
import glob
from typing import Dict, List, Tuple, Any

def load_json_file(file_path: str) -> Any:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        return None

def save_json_file(data: Any, file_path: str) -> bool:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"Processed data saved to '{file_path}'")
        return True
    except Exception as e:
        print(f"Error saving file '{file_path}': {e}")
        return False

def quaternion_to_euler(q: List[float]) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
    
    Args:
        q: Quaternion as [w, x, y, z] or [x, y, z, w]
    
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Handle both quaternion formats
    if len(q) == 4:
        # Assume [w, x, y, z] format first, but check if it makes sense
        w, x, y, z = q
        
        # Check if this might be [x, y, z, w] format
        # A simple heuristic: w should typically be the largest component for unit quaternions
        if abs(q[3]) > abs(q[0]):
            x, y, z, w = q
    else:
        raise ValueError("Quaternion must have 4 components")
    
    # Normalize quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return 0.0, 0.0, 0.0
    
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def calculate_fov_from_intrinsics(camera_intrinsic: List[List[float]], 
                                image_width: int = 1600, 
                                image_height: int = 900) -> float:
    """
    Calculate horizontal field of view from camera intrinsic matrix.
    
    Args:
        camera_intrinsic: 3x3 camera intrinsic matrix
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Horizontal field of view in degrees
    """
    # Extract focal length in x direction
    fx = camera_intrinsic[0][0]
    
    # Calculate horizontal FOV
    fov_rad = 2 * math.atan(image_width / (2 * fx))
    fov_deg = math.degrees(fov_rad)
    
    return fov_deg

def process_camera_calibration(calibration_data: List[Dict]) -> Dict[str, List[float]]:
    """
    Process camera calibration data and extract coordinates, pitch, yaw, and fov.
    
    Args:
        calibration_data: List of calibration dictionaries
    
    Returns:
        Dictionary with coordinates, pitchs, yaws, and fov lists
    """
    coordinates = []
    pitchs = []
    yaws = []
    fovs = []
    
    for entry in calibration_data:
        if not isinstance(entry, dict):
            continue
            
        # Extract translation (coordinates)
        translation = entry.get("translation", [])
        if len(translation) >= 3:
            coordinates.append([
                float(translation[0]),
                float(translation[1]),
                float(translation[2])
            ])
        else:
            print(f"Warning: Invalid translation data: {translation}")
            continue
        
        # Extract rotation and convert to pitch/yaw
        rotation = entry.get("rotation", [])
        if len(rotation) >= 4:
            try:
                roll, pitch, yaw = quaternion_to_euler(rotation)
                pitchs.append(pitch)
                yaws.append(yaw)
            except Exception as e:
                print(f"Warning: Error processing rotation {rotation}: {e}")
                pitchs.append(0.0)
                yaws.append(0.0)
        else:
            print(f"Warning: Invalid rotation data: {rotation}")
            pitchs.append(0.0)
            yaws.append(0.0)
        
        # Extract camera intrinsics and calculate FOV
        camera_intrinsic = entry.get("camera_intrinsic", [])
        if len(camera_intrinsic) >= 3 and len(camera_intrinsic[0]) >= 3:
            try:
                fov = calculate_fov_from_intrinsics(camera_intrinsic)
                fovs.append(round(fov, 1))  # Round to 1 decimal place
            except Exception as e:
                print(f"Warning: Error calculating FOV: {e}")
                fovs.append(90.0)  # Default FOV
        else:
            print(f"Warning: Invalid camera intrinsic data")
            fovs.append(90.0)  # Default FOV
    
    return {
        "coordinates": coordinates,
        "pitchs": pitchs,
        "yaws": yaws,
        "fov": fovs
    }

def process_single_camera_file(file_path: str, output_dir: str) -> bool:
    """
    Process a single camera JSON file.
    
    Args:
        file_path: Path to the camera JSON file
        output_dir: Output directory for processed files
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing: {file_path}")
    
    # Load camera data
    camera_data = load_json_file(file_path)
    if camera_data is None:
        return False
    
    # Ensure we have a list
    if not isinstance(camera_data, list):
        print(f"Error: Expected list format in {file_path}")
        return False
    
    print(f"  Found {len(camera_data)} calibration entries")
    
    # Process the data
    processed_data = process_camera_calibration(camera_data)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_converted.json")
    
    # Save processed data
    success = save_json_file(processed_data, output_file)
    
    if success:
        print(f"  ✓ Processed {len(processed_data['coordinates'])} entries")
        print(f"  ✓ Coordinates: {len(processed_data['coordinates'])}")
        print(f"  ✓ Pitch values: {len(processed_data['pitchs'])}")
        print(f"  ✓ Yaw values: {len(processed_data['yaws'])}")
        print(f"  ✓ FOV values: {len(processed_data['fov'])}")
    
    return success

def process_camera_files(input_pattern: str, output_dir: str = ".") -> None:
    """
    Process multiple camera JSON files matching the input pattern.
    
    Args:
        input_pattern: File pattern to match (e.g., "CAM_*.json" or specific file)
        output_dir: Output directory for processed files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching files
    if os.path.isfile(input_pattern):
        # Single file specified
        files = [input_pattern]
    else:
        # Pattern matching
        files = glob.glob(input_pattern)
    
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(files)} file(s) to process:")
    for file in files:
        print(f"  - {file}")
    
    # Process each file
    successful = 0
    for file_path in files:
        if process_single_camera_file(file_path, output_dir):
            successful += 1
    
    print(f"\n" + "="*60)
    print(f"Processing Summary:")
    print(f"  Total files: {len(files)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {len(files) - successful}")
    print(f"  Output directory: {output_dir}")
    print("="*60)

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process camera calibration JSON files and extract coordinates, pitch, yaw, and FOV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_processor.py CAM_FRONT.json
  python camera_processor.py "CAM_*.json" --output-dir ./processed
  python camera_processor.py CAM_FRONT.json CAM_BACK.json --output-dir ./output
        """
    )
    
    parser.add_argument("input_files", nargs="+", 
                       help="Input camera JSON file(s) or pattern (e.g., 'CAM_*.json')")
    parser.add_argument("--output-dir", "-o", default=".", 
                       help="Output directory for processed files (default: current directory)")
    parser.add_argument("--image-width", type=int, default=1600,
                       help="Image width in pixels for FOV calculation (default: 1600)")
    parser.add_argument("--image-height", type=int, default=900,
                       help="Image height in pixels for FOV calculation (default: 900)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Camera Calibration Data Processor")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Image dimensions: {args.image_width}x{args.image_height}")
    print("="*60)
    
    # Process each input file or pattern
    for input_pattern in args.input_files:
        process_camera_files(input_pattern, args.output_dir)

if __name__ == "__main__":
    main()
