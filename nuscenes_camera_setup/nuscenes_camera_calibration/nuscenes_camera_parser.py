import json
import argparse
import os
from typing import Dict, List, Any

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
        print(f"Data saved to '{file_path}'")
        return True
    except Exception as e:
        print(f"Error saving file '{file_path}': {e}")
        return False

def filter_camera_sensors(sensor_data: List[Dict]) -> Dict[str, str]:
    """
    Filter sensors for camera modality and extract token-channel mapping.
    
    Args:
        sensor_data: List of sensor dictionaries
        
    Returns:
        Dictionary mapping sensor tokens to their channels
    """
    camera_tokens = {}
    
    for sensor in sensor_data:
        if isinstance(sensor, dict) and sensor.get("modality") == "camera":
            token = sensor.get("token")
            channel = sensor.get("channel")
            
            if token and channel:
                camera_tokens[token] = channel
                print(f"Found camera sensor - Token: {token[:8]}..., Channel: {channel}")
    
    return camera_tokens

def filter_calibration_data(calibration_data: List[Dict], target_tokens: set) -> Dict[str, List[Dict]]:
    """
    Filter calibration data for specific sensor tokens.
    
    Args:
        calibration_data: List of calibration dictionaries
        target_tokens: Set of sensor tokens to filter for
        
    Returns:
        Dictionary mapping sensor tokens to their calibration data
    """
    filtered_data = {}
    
    for token in target_tokens:
        filtered_data[token] = []
    
    for calibration in calibration_data:
        if isinstance(calibration, dict):
            sensor_token = calibration.get("sensor_token")
            
            if sensor_token in target_tokens:
                filtered_data[sensor_token].append(calibration)
    
    return filtered_data

def process_camera_data(sensor_file: str, calibration_file: str, output_dir: str = ".") -> None:
    """
    Main processing function to filter and save camera calibration data.
    
    Args:
        sensor_file: Path to the sensor JSON file
        calibration_file: Path to the calibration JSON file
        output_dir: Directory to save output files
    """
    print("Loading sensor data...")
    sensor_data = load_json_file(sensor_file)
    if sensor_data is None:
        return
    
    print("Loading calibration data...")
    calibration_data = load_json_file(calibration_file)
    if calibration_data is None:
        return
    
    # Handle both list and dict formats
    if isinstance(sensor_data, dict):
        # If it's a dict, look for a key that contains the sensor list
        sensor_list = None
        for key, value in sensor_data.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if the first item looks like a sensor
                if isinstance(value[0], dict) and "modality" in value[0]:
                    sensor_list = value
                    break
        if sensor_list is None:
            print("Error: Could not find sensor list in the sensor file")
            return
    else:
        sensor_list = sensor_data
    
    if isinstance(calibration_data, dict):
        # If it's a dict, look for a key that contains the calibration list
        calibration_list = None
        for key, value in calibration_data.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if the first item looks like calibration data
                if isinstance(value[0], dict) and "sensor_token" in value[0]:
                    calibration_list = value
                    break
        if calibration_list is None:
            print("Error: Could not find calibration list in the calibration file")
            return
    else:
        calibration_list = calibration_data
    
    print(f"\nProcessing {len(sensor_list)} sensors...")
    
    # Step 1: Filter camera sensors and get token-channel mapping
    camera_tokens = filter_camera_sensors(sensor_list)
    
    if not camera_tokens:
        print("No camera sensors found with 'camera' modality.")
        return
    
    print(f"\nFound {len(camera_tokens)} camera sensor(s)")
    
    # Step 2: Filter calibration data for camera tokens
    print(f"\nFiltering calibration data from {len(calibration_list)} entries...")
    filtered_calibrations = filter_calibration_data(calibration_list, set(camera_tokens.keys()))
    
    # Step 3: Save filtered data for each camera channel
    os.makedirs(output_dir, exist_ok=True)
    
    for sensor_token, channel in camera_tokens.items():
        calibration_entries = filtered_calibrations.get(sensor_token, [])
        
        if calibration_entries:
            # Clean channel name for filename (remove special characters)
            safe_channel_name = "".join(c for c in channel if c.isalnum() or c in ('_', '-')).rstrip()
            output_file = os.path.join(output_dir, f"{safe_channel_name}.json")
            
            print(f"\nChannel: {channel}")
            print(f"  Sensor Token: {sensor_token}")
            print(f"  Calibration Entries: {len(calibration_entries)}")
            print(f"  Output File: {output_file}")
            
            # Save the calibration data
            if save_json_file(calibration_entries, output_file):
                print(f"  ✓ Successfully saved {len(calibration_entries)} calibration entries")
            else:
                print(f"  ✗ Failed to save calibration data")
        else:
            print(f"\nWarning: No calibration data found for sensor token {sensor_token} (channel: {channel})")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Filter camera calibration data from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 nuscenes_camera_parser.py sensors.json calibrations.json
  python3 nuscenes_camera_parser.py sensors.json calibrations.json --output-dir ./output
        """
    )
    
    parser.add_argument("sensor_file", help="Path to the sensor JSON file")
    parser.add_argument("calibration_file", help="Path to the calibration JSON file")
    parser.add_argument("--output-dir", "-o", default=".", 
                       help="Output directory for filtered files (default: current directory)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.sensor_file):
        print(f"Error: Sensor file '{args.sensor_file}' does not exist.")
        return
    
    if not os.path.exists(args.calibration_file):
        print(f"Error: Calibration file '{args.calibration_file}' does not exist.")
        return
    
    print("="*60)
    print("Camera Calibration Data Filter")
    print("="*60)
    print(f"Sensor file: {args.sensor_file}")
    print(f"Calibration file: {args.calibration_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Process the data
    process_camera_data(args.sensor_file, args.calibration_file, args.output_dir)
    
    print("\n" + "="*60)
    print("Processing completed!")
    print("="*60)

if __name__ == "__main__":
    main()
