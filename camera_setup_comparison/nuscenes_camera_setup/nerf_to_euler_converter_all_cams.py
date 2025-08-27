import json
import math
import argparse
import os
import glob
import numpy as np
from decimal import Decimal, getcontext
from typing import Dict, List, Tuple, Any, Union

# Set high precision for decimal calculations
getcontext().prec = 50

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

def high_precision_atan2(y: Union[float, Decimal], x: Union[float, Decimal]) -> Decimal:
    """
    High-precision atan2 implementation using Decimal arithmetic.
    
    Args:
        y: Y coordinate
        x: X coordinate
    
    Returns:
        Angle in radians as Decimal
    """
    y_dec = Decimal(str(y))
    x_dec = Decimal(str(x))
    
    # Handle special cases
    if x_dec == 0:
        if y_dec > 0:
            return Decimal('1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # Ï€/2
        elif y_dec < 0:
            return Decimal('-1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # -Ï€/2
        else:
            return Decimal('0')
    
    # Calculate atan(y/x) using Taylor series expansion
    ratio = y_dec / x_dec
    
    # For |ratio| <= 1, use direct Taylor series
    if abs(ratio) <= 1:
        result = _atan_taylor_series(ratio)
    else:
        # For |ratio| > 1, use atan(x) = Ï€/2 - atan(1/x)
        pi_half = Decimal('1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')
        inv_ratio = Decimal('1') / ratio
        if ratio > 0:
            result = pi_half - _atan_taylor_series(inv_ratio)
        else:
            result = -pi_half - _atan_taylor_series(inv_ratio)
    
    # Adjust for quadrant
    if x_dec < 0:
        pi_val = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066471236220')
        if y_dec >= 0:
            result += pi_val
        else:
            result -= pi_val
    
    return result

def _atan_taylor_series(x: Decimal) -> Decimal:
    """
    Calculate arctan using Taylor series expansion with high precision.
    
    Series: atan(x) = x - x^3/3 + x^5/5 - x^7/7 + ...
    """
    if abs(x) > 1:
        raise ValueError("Input must be <= 1 for this series")
    
    result = Decimal('0')
    x_power = x
    x_squared = x * x
    
    # Calculate terms until convergence
    for n in range(1, 200, 2):  # Odd numbers only
        term = x_power / Decimal(str(n))
        if n % 4 == 3:  # Alternating signs
            result -= term
        else:
            result += term
        
        # Check for convergence
        if abs(term) < Decimal('1e-45'):
            break
            
        x_power *= x_squared
    
    return result

def high_precision_asin(x: Union[float, Decimal]) -> Decimal:
    """
    High-precision asin implementation using Decimal arithmetic.
    
    Args:
        x: Input value (-1 <= x <= 1)
    
    Returns:
        Angle in radians as Decimal
    """
    x_dec = Decimal(str(x))
    
    # Handle boundary cases
    if x_dec == 1:
        return Decimal('1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # Ï€/2
    elif x_dec == -1:
        return Decimal('-1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # -Ï€/2
    elif x_dec == 0:
        return Decimal('0')
    
    # For |x| close to 1, use asin(x) = atan(x/sqrt(1-x^2))
    if abs(x_dec) > Decimal('0.9'):
        sqrt_term = (Decimal('1') - x_dec * x_dec).sqrt()
        return high_precision_atan2(x_dec, sqrt_term)
    
    # For smaller values, use Taylor series
    # asin(x) = x + x^3/6 + 3*x^5/40 + 5*x^7/112 + ...
    result = x_dec
    x_squared = x_dec * x_dec
    x_power = x_dec * x_squared  # x^3
    
    # Coefficients for Taylor series
    coefficients = [
        Decimal('1') / Decimal('6'),  # 1/6
        Decimal('3') / Decimal('40'),  # 3/40
        Decimal('5') / Decimal('112'),  # 5/112
        Decimal('35') / Decimal('1152'),  # 35/1152
    ]
    
    for i, coeff in enumerate(coefficients):
        term = coeff * x_power
        result += term
        
        if abs(term) < Decimal('1e-45'):
            break
            
        x_power *= x_squared
        
        # Update coefficient for next term
        if i < len(coefficients) - 1:
            continue
        else:
            # Calculate next coefficient dynamically if needed
            n = 2 * (i + 2) + 1
            next_coeff = coeff * Decimal(str(2 * i + 3)) * Decimal(str(2 * i + 1)) / (Decimal(str(2 * i + 4)) * Decimal(str(n)))
            coefficients.append(next_coeff)
    
    return result

def apply_rotation_matrix_transformation(rotation_matrix) -> np.ndarray:
    """
    Apply coordinate transformation to rotation matrix.
    Coordinate transform: x_new = y_old, y_new = z_old, z_new = -x_old
    """
    # Transformation matrix for the coordinate change
    Trans = np.array([
        [0,  1,  0],   # x_new = y_old
        [0,  0,  1],   # y_new = z_old  
        [-1, 0,  0]    # z_new = -x_old
    ])

    # pitch=yaw+90, yaw=roll+90, roll=-pitch (transformation from carla to nerf)
    #yaw =pitch-90, roll=yaw-90, pitch=-roll

    return Trans @ rotation_matrix @ Trans.T

def apply_coordinate_transformation(position):
    """
    Apply coordinate transformation:
    x_new = y_old
    y_new = z_old
    z_new = -x_old
    """
    x_old, y_old, z_old = position
    return np.array([y_old, z_old, -x_old])

def extract_rotation_from_transform_matrix(transform_matrix: List[List[float]]) -> np.ndarray:
    """
    Extract 3x3 rotation matrix from 4x4 transformation matrix.
    
    Args:
        transform_matrix: 4x4 transformation matrix
    
    Returns:
        3x3 rotation matrix as numpy array
    """
    # Convert to numpy array and extract top-left 3x3 submatrix
    transform_np = np.array(transform_matrix)
    rotation_matrix = transform_np[:3, :3]
    return rotation_matrix

def extract_translation_from_transform_matrix(transform_matrix: List[List[float]]) -> List[float]:
    """
    Extract translation vector from 4x4 transformation matrix.
    
    Args:
        transform_matrix: 4x4 transformation matrix
    
    Returns:
        Translation vector [x, y, z]
    """
    # Extract the last column (first 3 elements) as translation
    transform_np = np.array(transform_matrix)
    translation = transform_np[:3, 3].tolist()
    return translation

def rotation_matrix_to_euler_high_precision(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw) with maximum precision.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
    
    Returns:
        Tuple of (roll, pitch, yaw) in radians with maximum precision
    """

    # Extract elements (0-indexed) 
    Rot = np.zeros((3, 3), dtype=np.float64)
    Rot[0, 0] = rotation_matrix[0, 0]  # First row, first column
    Rot[0, 1] = rotation_matrix[0, 1]  # First row, second column  
    Rot[0, 2] = rotation_matrix[0, 2]  # First row, third column
    
    Rot[1, 0] = rotation_matrix[1, 0]  # Second row, first column
    Rot[1, 1] = rotation_matrix[1, 1]  # Second row, second column
    Rot[1, 2] = rotation_matrix[1, 2]  # Second row, third column
    
    Rot[2, 0] = rotation_matrix[2, 0]  # Third row, first column
    Rot[2, 1] = rotation_matrix[2, 1]  # Third row, second column
    Rot[2, 2] = rotation_matrix[2, 2]  # Third row, third column
    
    # Apply rotation transformation
    R = apply_rotation_matrix_transformation(Rot)
    
    print(f"Rotation matrix after transformation:\n{R}")
    
    # Extract Euler angles from rotation matrix (ZYX convention)
    '''sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]) #works also
    
    # Gimbal lock threshold
    singular = sy < 1e-6
    
    if not singular:
        # Normal case - no gimbal lock
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock case - pitch is close to Â±90 degrees
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0  # Set yaw to zero in gimbal lock'''


#approach 2: from "Computing Euler angles from rotation matrix"

    # Ensure numerical precision
    R = np.array(R, dtype=np.float64)
    
    # Check if R31 != Â±1 (not at singularity)
    if abs(R[2, 0]) != 1.0 and abs(R[2, 0]) != -1.0:
        # Two possible solutions
        theta_1 = -np.arcsin(R[2, 0]) # asin(R31)
        theta_2 = np.pi - theta_1
        
        # Compute corresponding psi and phi values
        cos_theta_1 = np.cos(theta_1)
        cos_theta_2 = np.cos(theta_2)
        
        psi_1 = np.arctan2(R[2, 1] / cos_theta_1, R[2, 2] / cos_theta_1) # atan2(R32/cos theta_1, R33/cos theta_1)
        psi_2 = np.arctan2(R[2, 1] / cos_theta_2, R[2, 2] / cos_theta_2) # atan2(R32/cos theta_2, R33/cos theta_2)
        
        phi_1 = np.arctan2(R[1, 0] / cos_theta_1, R[0, 0] / cos_theta_1) # atan2(R21/cos theta_1, R11/cos theta_1)
        phi_2 = np.arctan2(R[1, 0] / cos_theta_2, R[0, 0] / cos_theta_2) # atan2(R21/cos theta_2, R11/cos theta_2)
        
        # Choose the first solution (could implement additional criteria to choose)
        phi = phi_1
        theta = theta_1
        psi = psi_1
        
    else:
        # Singularity case: R31 = Â±1
        phi = 0.0  # Set to 0 as suggested
        
        if R[2, 0] == -1.0:  # R31 = -1
            theta = np.pi / 2
            psi = phi + np.arctan2(R[0, 1], R[0, 2]) # atan2(R12, R13)
        else:  # R31 = 1
            theta = -np.pi / 2
            psi = -phi + np.arctan2(-R[0, 1], -R[0, 2]) # atan2(-R12, -R13)

    #yaw =pitch-90, roll=yaw-90, pitch=-roll ##from reversed carla_to_nerf_unnormalized
    roll = psi
    pitch = theta
    yaw = phi

    #print(f"Extracted Euler angles: roll={roll}, pitch={pitch}, yaw={yaw}")

    
    pitch = -(pitch + math.pi / 2) 

    yaw_correction = math.pi / 2
    yaw = -(yaw - yaw_correction) #+2* math.pi 
    
    return roll, pitch, yaw


def calculate_fov_from_focal_length(fl_x: float, image_width: int) -> float:
    """
    Calculate horizontal field of view from focal length with high precision.
    
    Args:
        fl_x: Focal length in pixels (x direction)
        image_width: Image width in pixels
    
    Returns:
        Horizontal field of view in degrees
    """
    # Use high precision calculations
    fx = Decimal(str(fl_x))
    width_dec = Decimal(str(image_width))
    
    # Calculate horizontal FOV with high precision
    # FOV = 2 * atan(width / (2 * fx))
    ratio = width_dec / (2 * fx)
    fov_rad_dec = 2 * high_precision_atan2(ratio, Decimal('1'))
    
    # Convert to degrees
    pi_val = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066471236220')
    fov_deg_dec = fov_rad_dec * Decimal('180') / pi_val
    
    return float(fov_deg_dec)


def adjust_to_target(value, target, tolerance=0.1):
    """
    Adjusts value to be within Â±10% of target (default).
    If not within range, add/subtract 0.5Ï€ until closest to target.
    """

    # Compute tolerance bounds
    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)

    # Check if there is a sign change, if so, adjust value to match target sign
    if value * target < 0:
        value = -value


    # Check if already within tolerance
    if lower <= value <= upper:
        return value

    # Otherwise adjust by Â±0.5Ï€ until closest
    best_val = value
    best_diff = abs(value - target)

    for k in range(-10, 11):  # shift up to Â±10 * (0.5Ï€) just in case
        candidate = value + k * (0.5 * math.pi)
        diff = abs(candidate - target)
        if diff < best_diff:
            best_val = candidate
            best_diff = diff

    return best_val




def process_nerf_data(nerf_data: Dict, file_counter: int) -> Dict[str, List[float]]:
    """
    Process NeRF-style JSON data and extract coordinates, roll, pitch, yaw, and fov with high precision.
    
    Args:
        nerf_data: NeRF-style JSON data with frames and camera intrinsics
        file_counter: Counter indicating which file is being processed (0-5)
    
    Returns:
        Dictionary with coordinates, rolls, pitchs, yaws, and fov lists
    """
    coordinates = []
    rolls = []
    pitchs = []
    yaws = []
    fovs = []


    target_rolls = [
                -1.5707964855517969,
                -1.570792290789168,
                -1.5708060248469724,
                -1.5707963015148056,
                -1.5707397483668046,
                -1.5707894944129062
                ]

    target_pitchs = [
                -0.002184605901413669,
                -0.00908179325149176,
                -0.003702805724416052,
                0.010903921170017539,
                0.0007322564726739024,
                -0.005943942229067065
                ]

    target_yaws = [
                4.720323669231367,
                3.717790651712274,
                5.676476373000298,
                1.5644901720670188,
                0.32445679467339517,
                2.762686748075455
                ]
    
    # Extract camera intrinsics
    fl_x = nerf_data.get("fl_x", 1142.5184053936916)
    image_width = nerf_data.get("w", 1600)
    
    # Calculate FOV once for all frames (assuming same camera)
    #fov_value = calculate_fov_from_focal_length(fl_x, image_width)
    fov_value = 90.0  # Default FOV value, can be adjusted if needed
    
    # For continuity checking
    previous_yaw = None
    previous_roll = None
    pi = math.pi
    
    frames = nerf_data.get("frames", [])
    
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        
        print('file_counter: ', file_counter)
        transform_matrix = frame.get("transform_matrix", [])
        if len(transform_matrix) != 4 or any(len(row) != 4 for row in transform_matrix):
            print(f"Warning: Invalid transform matrix format")
            continue
        
        try:
            # Extract translation vector
            translation = extract_translation_from_transform_matrix(transform_matrix)
            translation = apply_coordinate_transformation(translation)
            coordinates.append([
                float(translation[0]),
                float(translation[1]),
                float(translation[2])
            ])
            
            # Extract rotation matrix and convert to Euler angles
            rotation_matrix = extract_rotation_from_transform_matrix(transform_matrix)
            roll, pitch, yaw = rotation_matrix_to_euler_high_precision(rotation_matrix)
            
            # Apply continuity heuristics to avoid large jumps
            if previous_yaw is not None:
                # Check for yaw wraparound
                yaw_diff = yaw - previous_yaw
                if yaw_diff > pi:
                    yaw -= 2 * pi
                elif yaw_diff < -pi:
                    yaw += 2 * pi
            
            if previous_roll is not None:
                # Check for roll wraparound
                roll_diff = roll - previous_roll
                if roll_diff > pi:
                    roll -= 2 * pi
                elif roll_diff < -pi:
                    roll += 2 * pi

            print('target_index (file_counter):', file_counter)




            rolls.append(roll)
            pitchs.append(pitch)
            yaws.append(yaw)
            #rolls.append(adjust_to_target(roll, target_rolls[file_counter]))
            #pitchs.append(adjust_to_target(pitch, target_pitchs[file_counter]))
            #yaws.append(adjust_to_target(yaw, target_yaws[file_counter]))
            fovs.append(round(fov_value, 6))

            # Update previous values
            previous_yaw = yaw
            previous_roll = roll
            
        except Exception as e:
            print(f"Warning: Error processing frame: {e}")
            # Add default values
            coordinates.append([0.0, 0.0, 0.0])
            rolls.append(0.0)
            pitchs.append(0.0)
            yaws.append(0.0)
            fovs.append(90.0)

        
    
    return {
        "coordinates": coordinates,
        "rolls": rolls,
        "pitchs": pitchs,
        "yaws": yaws,
        "fov": fovs
    }

def process_single_nerf_file(file_path: str, output_dir: str, file_counter: int) -> bool:
    """
    Process a single NeRF JSON file.
    
    Args:
        file_path: Path to the NeRF JSON file
        output_dir: Output directory for processed files
        file_counter: Counter indicating which file is being processed (0-5)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing: {file_path} (counter: {file_counter})")
    
    # Load NeRF data
    nerf_data = load_json_file(file_path)
    if nerf_data is None:
        return False
    
    # Ensure we have the expected format
    if not isinstance(nerf_data, dict) or "frames" not in nerf_data:
        print(f"Error: Expected NeRF format with 'frames' key in {file_path}")
        return False
    
    frames = nerf_data.get("frames", [])
    print(f"  Found {len(frames)} frames")
    
    # Process the data with the file counter
    processed_data = process_nerf_data(nerf_data, file_counter)
    
    # Generate output filename: replace .json with _trans.json
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_trans.json")
    
    # Save processed data
    success = save_json_file(processed_data, output_file)
    
    if success:
        print(f"  ✓ Processed {len(processed_data['coordinates'])} entries")
        print(f"  ✓ Coordinates: {len(processed_data['coordinates'])}")
        print(f"  ✓ Roll values: {len(processed_data['rolls'])}")
        print(f"  ✓ Pitch values: {len(processed_data['pitchs'])}")
        print(f"  ✓ Yaw values: {len(processed_data['yaws'])}")
        print(f"  ✓ FOV values: {len(processed_data['fov'])}")
        
        # Print camera info
        fl_x = nerf_data.get("fl_x", "N/A")
        image_width = nerf_data.get("w", "N/A")
        image_height = nerf_data.get("h", "N/A")
        print(f"  ✓ Camera focal length (fx): {fl_x}")
        print(f"  ✓ Image dimensions: {image_width}x{image_height}")
        print(f"  ✓ Calculated FOV: {processed_data['fov'][0]:.6f}°")
    
    return success

def process_camera_files(input_dir: str, output_dir: str = ".") -> None:
    """
    Process camera JSON files from a directory in the specified order.
    
    Args:
        input_dir: Directory containing camera JSON files
        output_dir: Output directory for processed files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the processing order
    camera_order = [
        "CAM_FRONT_ego_transform_matrix.json",
        "CAM_FRONT_RIGHT_ego_transform_matrix.json", 
        "CAM_FRONT_LEFT_ego_transform_matrix.json",
        "CAM_BACK_ego_transform_matrix.json",
        "CAM_BACK_LEFT_ego_transform_matrix.json",
        "CAM_BACK_RIGHT_ego_transform_matrix.json"
    ]
    
    # Find existing files in the specified order
    files_to_process = []
    for camera_file in camera_order:
        file_path = os.path.join(input_dir, camera_file)
        if os.path.exists(file_path):
            files_to_process.append(file_path)
        else:
            print(f"Warning: {camera_file} not found in {input_dir}")
    
    if not files_to_process:
        print(f"No camera files found in directory: {input_dir}")
        return
    
    print(f"Found {len(files_to_process)} camera file(s) to process in order:")
    for i, file_path in enumerate(files_to_process):
        print(f"  {i}: {os.path.basename(file_path)}")
    
    # Process each file with its corresponding counter
    successful = 0
    for file_counter, file_path in enumerate(files_to_process):
        print(f"\n{'='*60}")
        print(f"Processing file {file_counter + 1}/{len(files_to_process)}")
        print(f"{'='*60}")
        
        if process_single_nerf_file(file_path, output_dir, file_counter):
            successful += 1
            print(f"✓ Successfully processed {os.path.basename(file_path)} with counter {file_counter}")
        else:
            print(f"✗ Failed to process {os.path.basename(file_path)}")
    
    print(f"\n" + "="*60)
    print(f"Processing Summary:")
    print(f"  Total files: {len(files_to_process)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {len(files_to_process) - successful}")
    print(f"  Output directory: {output_dir}")
    print("="*60)

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process camera JSON files from a directory and extract coordinates, roll, pitch, yaw, and FOV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 nerf_to_euler_converter.py /path/to/camera/files
  python3 nerf_to_euler_converter.py /path/to/camera/files --output-dir ./processed
        """
    )
    
    parser.add_argument("input_dir", 
                       help="Input directory containing camera JSON files (CAM_*.json)")
    parser.add_argument("--output-dir", "-o", default=".", 
                       help="Output directory for processed files (default: current directory)")
    parser.add_argument("--precision", type=int, default=50,
                       help="Decimal precision for calculations (default: 50)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
    
    # Set the precision for Decimal calculations
    getcontext().prec = args.precision
    
    print("="*60)
    print("Camera Files to Pose Converter")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Calculation precision: {args.precision} decimal places")
    print("="*60)
    
    # Process camera files
    process_camera_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()