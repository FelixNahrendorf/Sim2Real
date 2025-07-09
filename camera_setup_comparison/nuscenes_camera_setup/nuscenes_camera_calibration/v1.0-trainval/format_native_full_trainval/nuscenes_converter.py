import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# Frame transformation matrix (original → new)
R_no = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0]
])

def quaternion_to_euler_zyx(w, x, y, z):
    r = R.from_quat([x, y, z, w])
    euler_zyx = r.as_euler('zyx', degrees=False)
    roll, pitch, yaw = euler_zyx[2], euler_zyx[1], euler_zyx[0]
    return roll, pitch, yaw

def calculate_fov_from_intrinsics(camera_intrinsic, image_width=1600):
    fx = camera_intrinsic[0][0]
    fov_rad = 2 * math.atan(image_width / (2 * fx))
    return math.degrees(fov_rad)

def process_nuscenes_cameras():
    camera_files = [
        "CAM_FRONT.json",
        "CAM_FRONT_RIGHT.json", 
        "CAM_FRONT_LEFT.json",
        "CAM_BACK.json",
        "CAM_BACK_LEFT.json",
        "CAM_BACK_RIGHT.json"
    ]

    native = {"coordinates": [], "pitchs": [], "rolls": [], "yaws": [], "fov": []}
    transformed = {"coordinates": [], "pitchs": [], "rolls": [], "yaws": [], "fov": []}
    original_structure = []  # New: store first camera data in original JSON structure
    
    for camera_file in camera_files:
        try:
            print(f"Processing {camera_file}...")
            with open(camera_file, 'r') as f:
                camera_data = json.load(f)

            if not camera_data:
                print(f"  Warning: Empty file.")
                continue

            first = camera_data[0]
            translation = np.array(first["translation"])
            rotation = first["rotation"]
            cam_intrinsic = first["camera_intrinsic"]

            # Add first camera data to original structure list
            original_structure.append(first)

            w, x, y, z = rotation
            fov = round(calculate_fov_from_intrinsics(cam_intrinsic), 1)

            # Native values
            roll_nat, pitch_nat, yaw_nat = quaternion_to_euler_zyx(w, x, y, z)
            native["coordinates"].append(translation.tolist())
            native["rolls"].append(roll_nat)
            native["pitchs"].append(pitch_nat)
            native["yaws"].append(yaw_nat)
            native["fov"].append(fov)

            # Apply rule-based remapping:
            # New Roll = Pitch
            # New Pitch = Yaw
            # New Yaw = -Roll
            new_roll = pitch_nat
            new_pitch = yaw_nat
            new_yaw = -roll_nat  # no normalization

            trans_coord = (R_no @ translation).tolist()
            transformed["coordinates"].append(trans_coord)
            transformed["rolls"].append(new_roll)
            transformed["pitchs"].append(new_pitch)
            transformed["yaws"].append(new_yaw)
            transformed["fov"].append(fov)

        except FileNotFoundError:
            print(f"  Error: {camera_file} not found.")
        except Exception as e:
            print(f"  Error processing {camera_file}: {e}")

    # Write all output files
    with open("nuscene_native_first_cams_converted.json", 'w') as f:
        json.dump(native, f, indent=2)
    with open("nuscene_native_first_cams_transformed_to_match_carla.json", 'w') as f:
        json.dump(transformed, f, indent=2)
    with open("nuscene_first_cameras_original_structure.json", 'w') as f:
        json.dump(original_structure, f, indent=2)

    print("\n✅ Outputs written:")
    print("  - nuscene_native_first_cams_converted.json (original frame)")
    print("  - nuscene_native_first_cams_transformed_to_match_carla.json (transformed frame)")
    print("  - nuscene_first_cameras_original_structure.json (first camera data in original JSON structure)")
    
    return native, transformed, original_structure

if __name__ == "__main__":
    native_result, transformed_result, original_structure_result = process_nuscenes_cameras()

    print("\nSample (first 2 cameras):")
    for i in range(min(2, len(native_result["coordinates"]))):
        print(f"\nCamera {i}:")
        print("  Native:")
        print(f"    Coord: {native_result['coordinates'][i]}")
        print(f"    RPY  : {native_result['rolls'][i]:.6f}, {native_result['pitchs'][i]:.6f}, {native_result['yaws'][i]:.6f}")
        print("  Transformed:")
        print(f"    Coord: {transformed_result['coordinates'][i]}")
        print(f"    RPY  : {transformed_result['rolls'][i]:.6f}, {transformed_result['pitchs'][i]:.6f}, {transformed_result['yaws'][i]:.6f}")
    
    print(f"\n✅ Original structure file contains {len(original_structure_result)} camera entries in order: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT")