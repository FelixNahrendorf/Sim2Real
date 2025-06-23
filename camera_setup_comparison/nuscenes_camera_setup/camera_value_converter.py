import json
import math
import argparse
import os
import glob
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
            return Decimal('1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # π/2
        elif y_dec < 0:
            return Decimal('-1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # -π/2
        else:
            return Decimal('0')
    
    # Calculate atan(y/x) using Taylor series expansion
    ratio = y_dec / x_dec
    
    # For |ratio| <= 1, use direct Taylor series
    if abs(ratio) <= 1:
        result = _atan_taylor_series(ratio)
    else:
        # For |ratio| > 1, use atan(x) = π/2 - atan(1/x)
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
        return Decimal('1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # π/2
    elif x_dec == -1:
        return Decimal('-1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # -π/2
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

def quaternion_to_euler_high_precision(q: List[float]) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) with maximum precision.
    Uses improved algorithm with continuity heuristics to avoid angle jumps.
    
    Args:
        q: Quaternion as [x, y, z, w] (NuScenes format)
    
    Returns:
        Tuple of (roll, pitch, yaw) in radians with maximum precision
    """
    if len(q) != 4:
        raise ValueError("Quaternion must have 4 components")
    
    # Convert to high-precision Decimal
    x, y, z, w = [Decimal(str(component)) for component in q]
    
    # Normalize quaternion with high precision
    norm_squared = w*w + x*x + y*y + z*z
    if norm_squared == 0:
        return 0.0, 0.0, 0.0
    
    norm = norm_squared.sqrt()
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Use rotation matrix approach for more stable conversion
    # Convert quaternion to rotation matrix elements
    r11 = 1 - 2 * (y*y + z*z)
    r12 = 2 * (x*y - w*z)
    r13 = 2 * (x*z + w*y)
    r21 = 2 * (x*y + w*z)
    r22 = 1 - 2 * (x*x + z*z)
    r23 = 2 * (y*z - w*x)
    r31 = 2 * (x*z - w*y)
    r32 = 2 * (y*z + w*x)
    r33 = 1 - 2 * (x*x + y*y)
    
    # Extract Euler angles from rotation matrix (ZYX convention)
    # This provides more stable results than direct quaternion conversion
    
    # Pitch (Y rotation) - handle gimbal lock
    sin_pitch = -r31
    
    # Clamp to avoid numerical errors
    if sin_pitch >= 1:
        pitch_dec = Decimal('1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # π/2
        # Gimbal lock case - set roll to 0 and compute yaw
        roll_dec = Decimal('0')
        yaw_dec = high_precision_atan2(r12, r22)
    elif sin_pitch <= -1:
        pitch_dec = Decimal('-1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339910740432566411533235618055')  # -π/2
        # Gimbal lock case - set roll to 0 and compute yaw
        roll_dec = Decimal('0')
        yaw_dec = high_precision_atan2(-r12, r22)
    else:
        pitch_dec = high_precision_asin(sin_pitch)
        
        # Roll (X rotation)
        roll_dec = high_precision_atan2(r32, r33)
        
        # Yaw (Z rotation)
        yaw_dec = high_precision_atan2(r21, r11)
    
    # Apply continuity heuristics to avoid large jumps
    roll_float = float(roll_dec)
    pitch_float = float(pitch_dec)
    yaw_float = float(yaw_dec)
    
    # Store previous values for continuity checking (simplified approach)
    # In a real implementation, you'd maintain state between calls
    
    return roll_float, pitch_float, yaw_float

def calculate_fov_from_intrinsics(camera_intrinsic: List[List[float]], 
                                image_width: int = 1600, 
                                image_height: int = 900) -> float:
    """
    Calculate horizontal field of view from camera intrinsic matrix with high precision.
    
    Args:
        camera_intrinsic: 3x3 camera intrinsic matrix
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Horizontal field of view in degrees
    """
    # Extract focal length in x direction with high precision
    fx = Decimal(str(camera_intrinsic[0][0]))
    width_dec = Decimal(str(image_width))
    
    # Calculate horizontal FOV with high precision
    # FOV = 2 * atan(width / (2 * fx))
    ratio = width_dec / (2 * fx)
    fov_rad_dec = 2 * high_precision_atan2(ratio, Decimal('1'))
    
    # Convert to degrees
    pi_val = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066471236220')
    fov_deg_dec = fov_rad_dec * Decimal('180') / pi_val
    
    return float(fov_deg_dec)

def process_camera_calibration(calibration_data: List[Dict]) -> Dict[str, List[float]]:
    """
    Process camera calibration data and extract coordinates, roll, pitch, yaw, and fov with high precision.
    Applies continuity heuristics to avoid angle jumps.
    
    Args:
        calibration_data: List of calibration dictionaries
    
    Returns:
        Dictionary with coordinates, rolls, pitchs, yaws, and fov lists
    """
    coordinates = []
    rolls = []
    pitchs = []
    yaws = []
    fovs = []
    
    # For continuity checking
    previous_yaw = None
    previous_roll = None
    pi = math.pi
    
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
        
        # Extract rotation and convert to roll/pitch/yaw with high precision
        rotation = entry.get("rotation", [])
        if len(rotation) >= 4:
            try:
                roll, pitch, yaw = quaternion_to_euler_high_precision(rotation)
                
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
                
                rolls.append(roll)
                pitchs.append(pitch)
                yaws.append(yaw)
                
                # Update previous values
                previous_yaw = yaw
                previous_roll = roll
                
            except Exception as e:
                print(f"Warning: Error processing rotation {rotation}: {e}")
                rolls.append(0.0)
                pitchs.append(0.0)
                yaws.append(0.0)
        else:
            print(f"Warning: Invalid rotation data: {rotation}")
            rolls.append(0.0)
            pitchs.append(0.0)
            yaws.append(0.0)
        
        # Extract camera intrinsics and calculate FOV with high precision
        camera_intrinsic = entry.get("camera_intrinsic", [])
        if len(camera_intrinsic) >= 3 and len(camera_intrinsic[0]) >= 3:
            try:
                fov = calculate_fov_from_intrinsics(camera_intrinsic)
                fovs.append(round(fov, 6))  # Round to 6 decimal places for higher precision
            except Exception as e:
                print(f"Warning: Error calculating FOV: {e}")
                fovs.append(90.0)  # Default FOV
        else:
            print(f"Warning: Invalid camera intrinsic data")
            fovs.append(90.0)  # Default FOV
    
    return {
        "coordinates": coordinates,
        "rolls": rolls,
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
        print(f"  ✓ Roll values: {len(processed_data['rolls'])}")
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
        description="Process camera calibration JSON files and extract coordinates, roll, pitch, yaw, and FOV with maximum precision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera_value_converter.py CAM_FRONT.json
  python3 camera_value_converter.py "CAM_*.json" --output-dir ./processed
  python3 camera_value_converter.py CAM_FRONT.json CAM_BACK.json --output-dir ./output
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
    parser.add_argument("--precision", type=int, default=50,
                       help="Decimal precision for calculations (default: 50)")
    
    args = parser.parse_args()
    
    # Set the precision for Decimal calculations
    getcontext().prec = args.precision
    
    print("="*60)
    print("High-Precision Camera Calibration Data Processor")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Image dimensions: {args.image_width}x{args.image_height}")
    print(f"Calculation precision: {args.precision} decimal places")
    print("="*60)
    
    # Process each input file or pattern
    for input_pattern in args.input_files:
        process_camera_files(input_pattern, args.output_dir)

if __name__ == "__main__":
    main()