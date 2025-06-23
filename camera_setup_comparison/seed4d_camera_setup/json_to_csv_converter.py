import json
import csv
import sys
from pathlib import Path

def convert_json_to_csv(json_file_path, output_csv_path):
    """
    Convert JSON camera data to CSV format matching the camera parameters structure.
    
    Args:
        json_file_path (str): Path to input JSON file
        output_csv_path (str): Path to output CSV file
    """
    
    # Define camera mapping in the specified order
    camera_names = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT", 
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        #"CAM_BACK_fov110",  # Special case for fov=110
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]
    
    # Load JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found.")
        return False
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {json_file_path}")
        return False
    
    # Validate data structure
    required_keys = ['coordinates', 'pitchs', 'yaws', 'fov']
    for key in required_keys:
        if key not in data:
            print(f"Error: Missing key '{key}' in JSON data")
            return False
    
    # Check if all arrays have the same length
    lengths = [len(data[key]) for key in required_keys]
    if not all(length == lengths[0] for length in lengths):
        print("Error: All arrays in JSON must have the same length")
        return False
    
    # Check if we have exactly 7 entries (number of cameras)
    if lengths[0] != 7: #was 7 for fov=110
        print(f"Error: Expected 7 camera entries, got {lengths[0]}")
        return False
    
    # Prepare CSV data
    csv_data = []
    
    # CSV header
    header = ['Camera', 'Parameter', 'Mean', 'Std_Dev', 'Min', 'Max', 'Median']
    csv_data.append(header)
    
    # Process each camera
    for i, camera_name in enumerate(camera_names):
        # Extract coordinate values (x, y, z)
        x, y, z = data['coordinates'][i]
        pitch = data['pitchs'][i]
        yaw = data['yaws'][i]
        fov = data['fov'][i]
        
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
        print(f"Successfully converted {json_file_path} to {output_csv_path}")
        print(f"Generated {len(csv_data)-1} parameter rows for {len(camera_names)} cameras")
        return True
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def main():
    """
    Main function to handle command line arguments or default file names.
    """
    if len(sys.argv) == 3:
        json_file = sys.argv[1]
        csv_file = sys.argv[2]
    elif len(sys.argv) == 2:
        json_file = sys.argv[1]
        # Generate output filename based on input
        csv_file = Path(json_file).stem + "_camera_parameters.csv"
    else:
        # Default filenames
        json_file = "transforms_converted.json"
        csv_file = "camera_parameters_output.csv"
    
    print(f"Converting {json_file} to {csv_file}")
    success = convert_json_to_csv(json_file, csv_file)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()