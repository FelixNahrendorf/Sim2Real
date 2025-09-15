import json
import csv
import sys
import argparse
from pathlib import Path

def convert_json_to_csv(json_file_path, output_csv_path, verbose=True):
    """
    Convert JSON camera data to CSV format matching the camera parameters structure.
    
    Args:
        json_file_path (str): Path to input JSON file
        output_csv_path (str): Path to output CSV file
        verbose (bool): Whether to print status messages
    """
    
    # Define camera mapping for 6 cameras
    camera_names = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT", 
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]
    
    # Load JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        if verbose:
            print(f"Error: File {json_file_path} not found.")
        return False
    except json.JSONDecodeError:
        if verbose:
            print(f"Error: Invalid JSON in file {json_file_path}")
        return False
    
    # Validate data structure
    required_keys = ['coordinates', 'pitchs', 'yaws', 'fov']
    for key in required_keys:
        if key not in data:
            if verbose:
                print(f"Error: Missing key '{key}' in JSON data")
            return False
    
    # Check if all arrays have the same length
    lengths = [len(data[key]) for key in required_keys]
    if not all(length == lengths[0] for length in lengths):
        if verbose:
            print("Error: All arrays in JSON must have the same length")
        return False
    
    # Check if we have exactly 6 entries (number of cameras)
    if lengths[0] != 6:
        if verbose:
            print(f"Error: Expected 6 camera entries, got {lengths[0]}")
        return False
    
    # Prepare CSV data
    csv_data = []
    
    # CSV header
    header = ['Camera', 'Parameter', 'Mean', 'Std_Dev', 'Min', 'Max', 'Median']
    csv_data.append(header)
    
    # Process each camera
    for i, camera_name in enumerate(camera_names):
        # Extract coordinate values from JSON (original x, y, z)
        json_x, json_y, json_z = data['coordinates'][i]
        pitch = data['pitchs'][i]
        yaw = data['yaws'][i]
        fov = data['fov'][i]
        
        # Remap coordinates: json_x -> z, json_y -> x, json_z -> y
        x = json_x  # JSON y becomes x
        y = json_y  # JSON z becomes y
        z = json_z  # JSON x becomes z
        
        # Note: Since we only have single values, we'll use them as mean
        # and set other statistics to 0 or the same value
        parameters = [
            ('x', x),
            ('y', y), 
            ('z', z),
            ('pitch', pitch),
            ('yaw', yaw),
            ('fov', fov)
        ]
        
        # Add roll parameter with 0 (since it's not in the JSON)
        parameters.append(('roll', 0.0))
        
        # Add each parameter row
        for param_name, value in parameters:
            row = [
                camera_name,
                param_name,
                value,  # Mean
                0.0,    # Std_Dev (0 since we only have one value)
                value,  # Min (same as mean)
                value,  # Max (same as mean) 
                value   # Median (same as mean)
            ]
            csv_data.append(row)
    
    # Write CSV file
    try:
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        
        if verbose:
            print(f"Successfully converted {json_file_path} to {output_csv_path}")
            print(f"Processed {len(camera_names)} cameras")
            print(f"Generated {len(csv_data)-1} parameter rows")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"Error writing CSV file: {e}")
        return False

def main():
    """
    Main function to handle command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Convert JSON camera data to CSV format (6 cameras)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 json_to_csv_converter.py input.json output.csv
  python3 json_to_csv_converter.py input.json
  python3 json_to_csv_converter.py --input input.json --output output.csv
        """
    )
    
    # Positional arguments (for backward compatibility)
    parser.add_argument(
        'json_file',
        nargs='?',
        default='transforms_converted.json',
        help='Input JSON file path (default: transforms_converted.json)'
    )
    
    parser.add_argument(
        'csv_file',
        nargs='?',
        help='Output CSV file path (default: auto-generated from input name)'
    )
    
    # Named arguments
    parser.add_argument(
        '--input', '-i',
        dest='json_input',
        help='Input JSON file path (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--output', '-o',
        dest='csv_output',
        help='Output CSV file path (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress status messages'
    )
    
    args = parser.parse_args()
    
    # Determine input file
    json_file = args.json_input if args.json_input else args.json_file
    
    # Determine output file
    if args.csv_output:
        csv_file = args.csv_output
    elif args.csv_file:
        csv_file = args.csv_file
    else:
        # Generate output filename based on input
        csv_file = Path(json_file).stem + "_camera_parameters.csv"
    
    # Set verbosity
    verbose = not args.quiet
    
    if verbose:
        print(f"Converting {json_file} to {csv_file}")
        print("Configuration: Processing 6 cameras")
    
    success = convert_json_to_csv(json_file, csv_file, verbose)
    
    if success:
        if verbose:
            print("Conversion completed successfully!")
    else:
        if verbose:
            print("Conversion failed!")
        sys.exit(1)

# Configuration constants for easy modification
class Config:
    """Configuration class to easily modify default behavior."""
    
    # Default file names
    DEFAULT_JSON_INPUT = "transforms_converted.json"
    DEFAULT_CSV_OUTPUT_SUFFIX = "_camera_parameters.csv"

# Convenience function for programmatic use
def convert_cameras_to_csv(json_file, csv_file=None, verbose=True):
    """Convert JSON to CSV for 6 cameras."""
    if csv_file is None:
        csv_file = Path(json_file).stem + "_camera_parameters.csv"
    return convert_json_to_csv(json_file, csv_file, verbose=verbose)

if __name__ == "__main__":
    main()