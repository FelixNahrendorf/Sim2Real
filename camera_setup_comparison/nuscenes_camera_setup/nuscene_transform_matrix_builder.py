#!/usr/bin/env python3

from nuscenes.nuscenes import NuScenes
import numpy as np
import json
import argparse
import os
 
def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix
    """
    w, x, y, z = q
   
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
   
    # Create rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R
 
def create_transform_matrix(translation, rotation):
    """
    Create 4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(rotation)
    T[:3, 3] = translation
    return T

def get_camera_transforms(nusc, sample):
    """
    Get transformation matrices for all six cameras
    """
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    camera_transforms = {}
    
    for cam_name in camera_names:
        if cam_name not in sample['data']:
            print(f"Warning: {cam_name} not found in sample data")
            continue
            
        # Get camera sample data
        cam_data = nusc.get('sample_data', sample['data'][cam_name])
        
        # Get ego pose (vehicle position in global coordinates)
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        ego_transform_matrix = create_transform_matrix(
            ego_pose['translation'],
            ego_pose['rotation']
        )
        
        # Get calibrated sensor (sensor position relative to ego vehicle)
        calibrated_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        sensor_transform_matrix = create_transform_matrix(
            calibrated_sensor['translation'],
            calibrated_sensor['rotation']
        )
        
        # Combine transformations: Global -> Ego -> Sensor
        # To get from global coordinates to camera coordinates, we need the inverse
        global_to_ego = np.linalg.inv(ego_transform_matrix)
        ego_to_sensor = np.linalg.inv(sensor_transform_matrix)
        
        # Combined transformation from global to camera
        global_to_camera = ego_to_sensor @ global_to_ego
        
        # Camera extrinsics (camera to global)
        camera_to_global = np.linalg.inv(global_to_camera)
        
        # Store all relevant matrices
        camera_transforms[cam_name] = {
            'ego_transform': ego_transform_matrix,
            'sensor_transform': sensor_transform_matrix,
            'global_to_camera': global_to_camera,
            'camera_to_global': camera_to_global,
            'calibrated_sensor': calibrated_sensor,
            'sample_data': cam_data
        }
    
    return camera_transforms
 
def get_camera_data_tokens(nusc, sample):
    """
    Helper function to get all camera data tokens
    """
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    camera_tokens = {}
    for cam_name in camera_names:
        if cam_name in sample['data']:
            camera_tokens[cam_name] = sample['data'][cam_name]
    
    return camera_tokens

def save_transforms_to_json(camera_transforms, output_path='transforms.json'):
    """
    Save camera transforms to JSON file in the specified format
    """
    # Create the output structure
    output_data = {
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": []
    }
    
    # Camera name mapping for file paths
    camera_mapping = {
        'CAM_FRONT': 0,
        'CAM_FRONT_RIGHT': 1,
        'CAM_FRONT_LEFT': 2, 
        'CAM_BACK': 3,
        'CAM_BACK_LEFT': 4,
        'CAM_BACK_RIGHT': 5
    }
    
    for cam_name, transforms in camera_transforms.items():
        if cam_name in camera_mapping:
            cam_idx = camera_mapping[cam_name]
            
            # Get camera intrinsics
            calibrated_sensor = transforms['calibrated_sensor']
            camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
            
            # Extract focal lengths and principal point
            fl_x = camera_intrinsic[0, 0]
            fl_y = camera_intrinsic[1, 1] 
            cx = camera_intrinsic[0, 2]
            cy = camera_intrinsic[1, 2]
            
            # Get image dimensions (you may need to adjust these based on your dataset)
            # These are typical NuScenes camera dimensions, but verify with your data
            w = 1600
            h = 900
            
            # Create frame entry
            frame = {
                "file_path": f"../sensors/{cam_idx}_rgb.png",
                "depth_file_path": f"../sensors/{cam_idx}_depth.png",
                "semantic_segmentation_file_path": f"../sensors/{cam_idx}_semantic_segmentation.png",
                "instance_segmentation_file_path": f"../sensors/{cam_idx}_instance_segmentation.png",
                "transform_matrix": transforms['camera_to_global'].tolist(),
                "fl_x": float(fl_x),
                "fl_y": float(fl_y), 
                "cx": float(cx),
                "cy": float(cy),
                "w": w,
                "h": h
            }
            
            output_data["frames"].append(frame)
    
    # Sort frames by camera index to maintain consistent ordering
    output_data["frames"].sort(key=lambda x: int(x["file_path"].split('/')[-1].split('_')[0]))
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    return output_data

def main(dataroot='/app/datasets/nuscenes_mini', version='v1.0-mini', scene_idx=0, output_file=None, verbose=False):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    scene = nusc.scene[scene_idx]
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    
    # Get all camera transforms
    camera_transforms = get_camera_transforms(nusc, sample)
    
    if verbose:
        # Print results for all cameras
        for cam_name, transforms in camera_transforms.items():
            print(f"\n{'='*50}")
            print(f"CAMERA: {cam_name}")
            print(f"{'='*50}")
            print(f"Ego Transform Matrix (Ego to Global):\n{transforms['ego_transform']}")
            print(f"\nSensor Transform Matrix (Sensor to Ego):\n{transforms['sensor_transform']}")
            print(f"\nGlobal to Camera Transform:\n{transforms['global_to_camera']}")
            print(f"\nCamera to Global Extrinsics:\n{transforms['camera_to_global']}")
            
            # Also print camera intrinsics if available
            if 'camera_intrinsic' in transforms['calibrated_sensor']:
                print(f"\nCamera Intrinsics:\n{np.array(transforms['calibrated_sensor']['camera_intrinsic'])}")
    
    # Save transforms to JSON file by default
    if output_file is None:
        output_file = 'transforms.json'
    
    json_data = save_transforms_to_json(camera_transforms, output_file)
    print(f"Camera transforms extracted and saved to: {output_file}")
    print(f"Processed {len(json_data['frames'])} cameras from scene {scene_idx}")
    
    # Return the transforms dictionary for further use
    return camera_transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract NuScenes camera transforms to JSON')
    parser.add_argument('--dataroot', type=str, default='/app/datasets/nuscenes_mini',
                       help='Path to NuScenes dataset (default: /app/datasets/nuscenes_mini)')
    parser.add_argument('--version', type=str, default='v1.0-mini', 
                       help='NuScenes version (default: v1.0-mini)')
    parser.add_argument('--scene', type=int, default=0,
                       help='Scene index to process (default: 0)')
    parser.add_argument('--output', '-o', type=str, default='transforms.json',
                       help='Output JSON filename (default: transforms.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed transformation matrices')
    
    args = parser.parse_args()
    
    # Check if dataroot exists
    if not os.path.exists(args.dataroot):
        print(f"Error: Dataset path '{args.dataroot}' does not exist!")
        print("Please specify the correct path using --dataroot")
        exit(1)
    
    try:
        camera_transforms = main(
            dataroot=args.dataroot,
            version=args.version, 
            scene_idx=args.scene,
            output_file=args.output,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error processing NuScenes data: {e}")
        exit(1)