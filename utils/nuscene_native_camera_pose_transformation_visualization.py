#!/usr/bin/env python3
"""
Merged Camera Pose Transformation and Visualization Script

This script combines camera pose transformation from nuScenes data and 3D visualization.
It processes nuScenes camera data, applies coordinate transformations, and generates
interactive visualizations without creating intermediate JSON files by default.

Features:
- Extracts camera poses from nuScenes dataset
- Applies coordinate transformation (x_new = -z_old, y_new = x_old, z_new = -y_old)
- Generates interactive 3D visualizations with coordinate tables
- Optional JSON file output with --save-json flag

Usage:
    python merged_camera_pose_script.py [options]

Options:
    --save-json         Save intermediate transform JSON files
    --output, -o        Output file prefix (default: sensor_poses)
    --no-interactive    Skip interactive visualization
    --no-static         Skip static PNG generation
    --no-ply            Skip PLY file generation
    --no-csv            Skip CSV coordinates file generation
    --output-dir        Directory for saving plots (default: output_plots)

Example:
    python merged_camera_pose_script.py --save-json --output my_poses
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyquaternion import Quaternion
import os
import json
from nuscenes.nuscenes import NuScenes
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import sys
from pathlib import Path
import csv
from mpl_toolkits.mplot3d import Axes3D

# ========================================
# CAMERA POSE TRANSFORMATION FUNCTIONS
# ========================================

def apply_coordinate_transformation(position):
    """
    Apply coordinate transformation:
    x_new = -z_old
    y_new = x_old
    z_new = -y_old
    """
    x_old, y_old, z_old = position
    return np.array([-z_old, x_old, -y_old])

def apply_rotation_transformation(quaternion):
    """
    Apply rotation transformation to match the coordinate system change:
    x_new = -z_old
    y_new = x_old
    z_new = -y_old
   
    This requires transforming the quaternion to the new coordinate frame
    """
    # Transformation matrix for coordinate change
    # [x_new]   [ 0  0 -1] [x_old]
    # [y_new] = [ 1  0  0] [y_old]  
    # [z_new]   [ 0 -1  0] [z_old]
   
    # Convert to rotation matrix
    rotation_matrix = quaternion.rotation_matrix
   
    # Transformation matrix
    T = np.array([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0, -1,  0]
    ])
   
    # Apply transformation: R_new = T * R_old
    transformed_rotation_matrix = T @ rotation_matrix
   
    # Convert back to quaternion
    transformed_quaternion = Quaternion(matrix=transformed_rotation_matrix)
   
    return transformed_quaternion

def quaternion_to_xyzw(quaternion):
    """
    Convert quaternion from pyquaternion format (w,x,y,z) to (x,y,z,w) format
    """
    return np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to euler angles (roll, pitch, yaw) in radians
    Using ZYX convention (yaw, pitch, roll)
    """
    # Extract quaternion components
    w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
   
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
   
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
   
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
   
    return np.array([roll, pitch, yaw])

def euler_to_degrees(euler_rad):
    """
    Convert euler angles from radians to degrees
    """
    return np.rad2deg(euler_rad)

def quaternion_to_transform_matrix(quaternion, translation):
    """
    Convert quaternion and translation to 4x4 transformation matrix
    """
    # Get rotation matrix from quaternion
    rotation_matrix = quaternion.rotation_matrix
    
    # Create 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    
    return transform_matrix

def get_all_camera_positions_with_images(nusc):
    """
    Get camera positions from any sample and include image data
    """
    # Get the first sample
    sample = nusc.sample[0]
    camera_data = {}
   
    # Look for camera data in the sample
    for data_key, sample_data_token in sample['data'].items():
        sample_data = nusc.get('sample_data', sample_data_token)
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        sensor = nusc.get('sensor', calibrated_sensor['sensor_token'])
       
        if sensor['modality'] == 'camera':
            # Get image file path
            image_path = os.path.join(nusc.dataroot, sample_data['filename'])
           
            # Get original position and apply transformation
            original_translation = np.array(calibrated_sensor['translation'])
            transformed_translation = apply_coordinate_transformation(original_translation)
           
            # Get quaternion and apply rotation transformation
            original_quaternion = Quaternion(calibrated_sensor['rotation'])
            x_axis_flip = Quaternion(axis=[1, 0, 0], angle=np.pi)
            original_quaternion = original_quaternion * x_axis_flip
           
            transformed_quaternion = apply_rotation_transformation(original_quaternion)
           
            # Convert to (x,y,z,w) format and euler angles for both original and transformed
            original_quaternion_xyzw = quaternion_to_xyzw(original_quaternion)
            transformed_quaternion_xyzw = quaternion_to_xyzw(transformed_quaternion)
           
            original_euler_rad = quaternion_to_euler(original_quaternion)
            transformed_euler_rad = quaternion_to_euler(transformed_quaternion)
           
            original_euler_deg = euler_to_degrees(original_euler_rad)
            transformed_euler_deg = euler_to_degrees(transformed_euler_rad)
           
            # Create transformation matrices
            original_transform_matrix = quaternion_to_transform_matrix(original_quaternion, original_translation)
            transformed_transform_matrix = quaternion_to_transform_matrix(transformed_quaternion, transformed_translation)
           
            camera_data[sensor['channel']] = {
                'translation': transformed_translation,
                'original_translation': original_translation,
                'rotation': transformed_quaternion,  # Transformed quaternion object
                'original_rotation': original_quaternion,  # Original quaternion object
                'rotation_xyzw': transformed_quaternion_xyzw,  # Transformed (x,y,z,w) format
                'original_rotation_xyzw': original_quaternion_xyzw,  # Original (x,y,z,w) format
                'euler_rad': transformed_euler_rad,  # Transformed euler angles in radians
                'euler_deg': transformed_euler_deg,  # Transformed euler angles in degrees
                'original_euler_rad': original_euler_rad,  # Original euler angles in radians
                'original_euler_deg': original_euler_deg,  # Original euler angles in degrees
                'transform_matrix': transformed_transform_matrix,  # Transformed 4x4 matrix
                'original_transform_matrix': original_transform_matrix,  # Original 4x4 matrix
                'sample_data_key': data_key,
                'image_path': image_path,
                'sample_data_token': sample_data_token,
                'camera_intrinsic': calibrated_sensor['camera_intrinsic']
            }
   
    return camera_data

def create_json_output(camera_data, output_path_original='transforms_original.json', output_path_transformed='transforms_transformed.json'):
    """
    Create JSON output files following the structure of transforms_ego.json
    """
    # Base structure for both JSONs
    base_structure = {
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": []
    }
    
    # Create original transforms JSON
    original_json = base_structure.copy()
    original_json["frames"] = []
    
    # Create transformed transforms JSON
    transformed_json = base_structure.copy()
    transformed_json["frames"] = []
    
    # Process each camera
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        # Get camera intrinsics
        camera_intrinsic = np.array(data['camera_intrinsic'])
        fl_x = camera_intrinsic[0, 0]
        fl_y = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        
        # Assume standard image dimensions (you may need to adjust these)
        w = 1600
        h = 900
        
        # Original frame
        original_frame = {
            "file_path": f"../sensors/{idx}_rgb.png",
            "depth_file_path": f"../sensors/{idx}_depth.png",
            "semantic_segmentation_file_path": f"../sensors/{idx}_semantic_segmentation.png",
            "instance_segmentation_file_path": f"../sensors/{idx}_instance_segmentation.png",
            "transform_matrix": data['original_transform_matrix'].tolist(),
            "fl_x": float(fl_x),
            "fl_y": float(fl_y),
            "cx": float(cx),
            "cy": float(cy),
            "w": w,
            "h": h,
            "camera_name": camera_name
        }
        
        # Transformed frame
        transformed_frame = {
            "file_path": f"../sensors/{idx}_rgb.png",
            "depth_file_path": f"../sensors/{idx}_depth.png",
            "semantic_segmentation_file_path": f"../sensors/{idx}_semantic_segmentation.png",
            "instance_segmentation_file_path": f"../sensors/{idx}_instance_segmentation.png",
            "transform_matrix": data['transform_matrix'].tolist(),
            "fl_x": float(fl_x),
            "fl_y": float(fl_y),
            "cx": float(cx),
            "cy": float(cy),
            "w": w,
            "h": h,
            "camera_name": camera_name
        }
        
        original_json["frames"].append(original_frame)
        transformed_json["frames"].append(transformed_frame)
    
    # Save JSON files
    with open(output_path_original, 'w') as f:
        json.dump(original_json, f, indent=4)
    
    with open(output_path_transformed, 'w') as f:
        json.dump(transformed_json, f, indent=4)
    
    print(f"Original transforms saved to: {output_path_original}")
    print(f"Transformed transforms saved to: {output_path_transformed}")
    
    return original_json, transformed_json

def plot_camera_images(camera_data, figsize=(20, 12), output_path='camera_images.png'):
    """
    Plot images from all cameras in a grid layout and save as PNG
    """
    n_cameras = len(camera_data)
   
    # Create subplots - adjust layout based on number of cameras
    if n_cameras <= 3:
        rows, cols = 1, n_cameras
    elif n_cameras <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
   
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
   
    # Handle case where there's only one subplot
    if n_cameras == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
   
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_cameras > 1 else axes
   
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        if idx >= len(axes_flat):
            break
           
        ax = axes_flat[idx]
       
        try:
            # Load and display image
            image = Image.open(data['image_path'])
            ax.imshow(image)
           
            # Create title with camera info (using transformed coordinates)
            translation = data['translation']
            title = f"{camera_name}\n"
            title += f"Position: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]"
           
            ax.set_title(title, fontsize=10, pad=10)
            ax.axis('off')
           
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image:\n{str(e)}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{camera_name} - Image Error", fontsize=10)
            ax.axis('off')
   
    # Hide unused subplots
    for idx in range(n_cameras, len(axes_flat)):
        axes_flat[idx].axis('off')
   
    plt.tight_layout()
    plt.suptitle('nuScenes Camera Views and Positions (Transformed Coordinates)', fontsize=16, y=0.98)
    
    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Camera images plot saved to: {output_path}")
    
    plt.show()

def plot_coordinate_comparison(camera_data, output_path='coordinate_comparison.png'):
    """
    Plot comparison between original and transformed coordinate systems and save as PNG
    """
    fig = plt.figure(figsize=(16, 8))
   
    # Original coordinates
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter([0], [0], [0], c='red', s=100, marker='s', label='Ego Vehicle')
   
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        orig_translation = data['original_translation']
        ax1.scatter(orig_translation[0], orig_translation[1], orig_translation[2],
                   s=60, label=camera_name, color=colors[idx % len(colors)])
        ax1.text(orig_translation[0], orig_translation[1], orig_translation[2] + 0.1,
                camera_name, fontsize=8)
   
    ax1.set_xlabel('X (original)')
    ax1.set_ylabel('Y (original)')
    ax1.set_zlabel('Z (original)')
    ax1.set_title('Original Coordinate System')
    ax1.legend()
   
    # Transformed coordinates
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter([0], [0], [0], c='red', s=100, marker='s', label='Ego Vehicle')
   
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        translation = data['translation']
        ax2.scatter(translation[0], translation[1], translation[2],
                   s=60, label=camera_name, color=colors[idx % len(colors)])
        ax2.text(translation[0], translation[1], translation[2] + 0.1,
                camera_name, fontsize=8)
   
    ax2.set_xlabel('X (transformed: -z_old)')
    ax2.set_ylabel('Y (transformed: x_old)')
    ax2.set_zlabel('Z (transformed: -y_old)')
    ax2.set_title('Transformed Coordinate System')
    ax2.legend()
   
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Coordinate comparison plot saved to: {output_path}")
    
    plt.show()

def print_camera_info(camera_data):
    """
    Print detailed camera information showing both original and transformed coordinates
    """
    print("Camera Information:")
    print("=" * 110)
   
    for camera_name, data in camera_data.items():
        print(f"\n{camera_name}:")
        print(f"  Original position:      {data['original_translation']}")
        print(f"  Transformed position:   {data['translation']}")
        print(f"  Original rotation (x,y,z,w):     {data['original_rotation_xyzw']}")
        print(f"  Transformed rotation (x,y,z,w):  {data['rotation_xyzw']}")
        print(f"  Original euler (deg):   Roll={data['original_euler_deg'][0]:6.1f}°, Pitch={data['original_euler_deg'][1]:6.1f}°, Yaw={data['original_euler_deg'][2]:6.1f}°")
        print(f"  Transformed euler (deg): Roll={data['euler_deg'][0]:6.1f}°, Pitch={data['euler_deg'][1]:6.1f}°, Yaw={data['euler_deg'][2]:6.1f}°")
        print(f"  Image path: {data['image_path']}")
        print(f"  Camera intrinsic shape: {np.array(data['camera_intrinsic']).shape}")
       
        # Show the transformation explicitly
        orig = data['original_translation']
        print(f"  Position transformation: [{orig[0]:.2f}, {orig[1]:.2f}, {orig[2]:.2f}] -> [{-orig[2]:.2f}, {orig[0]:.2f}, {-orig[1]:.2f}]")

# ========================================
# VISUALIZATION FUNCTIONS
# ========================================

def transform_matrix_to_pose(transform_matrix):
    """Extract position and rotation from 4x4 transform matrix"""
    transform = np.array(transform_matrix)
    position = transform[:3, 3]
    rotation = transform[:3, :3]
    return position, rotation

def rotation_matrix_to_euler(rotation_matrix):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees"""
    # Using ZYX convention (yaw-pitch-roll)
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # roll
        y = np.arctan2(-rotation_matrix[2, 0], sy)                    # pitch
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # yaw
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])  # roll
        y = np.arctan2(-rotation_matrix[2, 0], sy)                     # pitch
        z = 0                                                          # yaw
    
    # Convert to degrees
    return np.degrees([x, y, z])

def rotation_matrix_to_quaternion(rotation_matrix):
    """Convert rotation matrix to quaternion (x, y, z, w)"""
    R = rotation_matrix
    
    # Shepperd's method for numerical stability
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    # Normalize to ensure unit quaternion and return in x,y,z,w order
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    return np.array([x, y, z, w]) / norm

def plot_coordinate_system(ax, position, rotation, scale=0.1, alpha=0.7):
    """Plot coordinate system axes at given position and rotation"""
    # Standard colors: X=red, Y=green, Z=blue
    colors = ['red', 'green', 'blue']
    
    # Create unit vectors for each axis
    axes = np.eye(3) * scale
    
    # Transform axes by rotation matrix
    transformed_axes = rotation @ axes
    
    # Plot each axis
    for i, (axis, color) in enumerate(zip(transformed_axes.T, colors)):
        end_point = position + axis
        ax.plot([position[0], end_point[0]], 
                [position[1], end_point[1]], 
                [position[2], end_point[2]], 
                color=color, linewidth=2, alpha=alpha)

def write_ply_file(transform_data_list, output_path='sensor_poses.ply'):
    """Write sensor poses to PLY file format for 3D viewing"""
    vertices = []
    faces = []
    colors = []
    
    # Color palette for different files (RGB 0-255)
    color_palette = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [255, 192, 203], # Pink
        [128, 128, 128], # Gray
    ]
    
    vertex_count = 0
    
    for file_idx, (dataset_name, data) in enumerate(transform_data_list):
        file_color = color_palette[file_idx % len(color_palette)]
        
        # Process each frame/sensor
        for frame_idx, frame in enumerate(data.get('frames', [])):
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix is None:
                continue
                
            # Extract pose
            position, rotation = transform_matrix_to_pose(transform_matrix)
            
            # Add coordinate system vertices (origin + 3 axis endpoints)
            scale = 0.2
            axes_points = []
            
            # Origin
            axes_points.append(position)
            colors.append(file_color)
            
            # X, Y, Z axis endpoints
            for i in range(3):
                axis_color = [0, 0, 0]
                axis_color[i] = 255  # Red for X, Green for Y, Blue for Z
                
                axis_end = position + rotation[:, i] * scale
                axes_points.append(axis_end)
                colors.append(axis_color)
            
            vertices.extend(axes_points)
            
            # Add lines connecting origin to each axis endpoint
            for i in range(3):
                faces.append([vertex_count, vertex_count + 1 + i])
            
            vertex_count += 4
    
    # Write PLY file
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(faces)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        # Write vertices
        for vertex, color in zip(vertices, colors):
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
        
        # Write edges
        for face in faces:
            f.write(f"{face[0]} {face[1]}\n")
    
    print(f"PLY file saved to: {output_path}")

def create_interactive_plotly(transform_data_list, output_path='sensor_poses.html'):
    """Create interactive 3D visualization using Plotly"""
    fig = go.Figure()
    
    # Color palette
    color_palette = px.colors.qualitative.Set1
    
    all_positions = []
    coordinate_data = []
    
    for file_idx, (dataset_name, data) in enumerate(transform_data_list):
        file_color = color_palette[file_idx % len(color_palette)]
        
        # Collect all positions and rotations for this file
        positions = []
        rotations = []
        hover_texts = []
        
        for frame_idx, frame in enumerate(data.get('frames', [])):
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix is None:
                continue
                
            position, rotation = transform_matrix_to_pose(transform_matrix)
            euler_angles = rotation_matrix_to_euler(rotation)
            quaternion = rotation_matrix_to_quaternion(rotation)
            
            positions.append(position)
            rotations.append(rotation)
            all_positions.append(position)
            
            # Store coordinate data with rotation information (Euler and quaternion)
            coordinate_data.append({
                'file': dataset_name,
                'sensor': frame_idx,
                'x': position[0],
                'y': position[1],
                'z': position[2],
                'roll': euler_angles[0],
                'pitch': euler_angles[1],
                'yaw': euler_angles[2],
                'qx': quaternion[0],
                'qy': quaternion[1],
                'qz': quaternion[2],
                'qw': quaternion[3]
            })
            
            # Create hover text with coordinates, Euler angles, and quaternions
            hover_text = (f"Dataset: {dataset_name}<br>"
                         f"Sensor: {frame_idx}<br>"
                         f"X: {position[0]:.3f}<br>"
                         f"Y: {position[1]:.3f}<br>"
                         f"Z: {position[2]:.3f}<br>"
                         f"Roll: {euler_angles[0]:.1f}°<br>"
                         f"Pitch: {euler_angles[1]:.1f}°<br>"
                         f"Yaw: {euler_angles[2]:.1f}°<br>"
                         f"Quaternion (x,y,z,w): ({quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f})")
            hover_texts.append(hover_text)
        
        if not positions:
            continue
            
        positions = np.array(positions)
        
        # Create legend group name for this dataset
        legend_group = f"group_{file_idx}"
        
        # Add sensor positions as markers with coordinate information
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+text',
            text=[str(i) for i in range(len(positions))],
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo='text',
            marker=dict(
                size=8,
                color=file_color,
                symbol='circle'
            ),
            name=f'{dataset_name} ({len(positions)} sensors)',
            legendgroup=legend_group,
            showlegend=True
        ))
        
        # Collect all coordinate axes data for this dataset
        x_lines_x, x_lines_y, x_lines_z = [], [], []
        y_lines_x, y_lines_y, y_lines_z = [], [], []
        z_lines_x, z_lines_y, z_lines_z = [], [], []
        
        scale = 0.2
        for pos_idx, (position, rotation) in enumerate(zip(positions, rotations)):
            # X axis points
            x_end = position + rotation[:, 0] * scale
            x_lines_x.extend([position[0], x_end[0], None])
            x_lines_y.extend([position[1], x_end[1], None])
            x_lines_z.extend([position[2], x_end[2], None])
            
            # Y axis points
            y_end = position + rotation[:, 1] * scale
            y_lines_x.extend([position[0], y_end[0], None])
            y_lines_y.extend([position[1], y_end[1], None])
            y_lines_z.extend([position[2], y_end[2], None])
            
            # Z axis points
            z_end = position + rotation[:, 2] * scale
            z_lines_x.extend([position[0], z_end[0], None])
            z_lines_y.extend([position[1], z_end[1], None])
            z_lines_z.extend([position[2], z_end[2], None])
        
        # Add all X axes for this dataset as a single trace
        fig.add_trace(go.Scatter3d(
            x=x_lines_x,
            y=x_lines_y,
            z=x_lines_z,
            mode='lines',
            line=dict(color='red', width=4),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip',
            name=f'{dataset_name} X-axes'
        ))
        
        # Add all Y axes for this dataset as a single trace
        fig.add_trace(go.Scatter3d(
            x=y_lines_x,
            y=y_lines_y,
            z=y_lines_z,
            mode='lines',
            line=dict(color='green', width=4),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip',
            name=f'{dataset_name} Y-axes'
        ))
        
        # Add all Z axes for this dataset as a single trace
        fig.add_trace(go.Scatter3d(
            x=z_lines_x,
            y=z_lines_y,
            z=z_lines_z,
            mode='lines',
            line=dict(color='blue', width=4),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip',
            name=f'{dataset_name} Z-axes'
        ))
    
    # Create coordinate table HTML
    coord_table_html = create_coordinate_table_html(coordinate_data)
    
    # Update layout
    fig.update_layout(
        title='Interactive 3D Sensor Poses Visualization<br><sub>X=Red, Y=Green, Z=Blue | Use mouse to rotate, zoom, and pan | Hover over sensors for coordinates, Euler angles, and quaternions</sub>',
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=1000,
        showlegend=True
    )
    
    # Create custom HTML with coordinate table
    plot_html = fig.to_html(include_plotlyjs=True, div_id="plotly-div")
    
    # Combine plot and coordinate table
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Sensor Poses Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ display: flex; flex-direction: column; }}
            .plot-container {{ width: 100%; }}
            .coords-container {{ margin-top: 20px; }}
            .coords-table {{ width: 100%; border-collapse: collapse; }}
            .coords-table th, .coords-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .coords-table th {{ background-color: #f2f2f2; }}
            .coords-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .position-col {{ background-color: #e6f3ff; }}
            .rotation-col {{ background-color: #ffe6f0; }}
            .quaternion-col {{ background-color: #fff9e6; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="plot-container">
                {plot_html.split('<body>')[1].split('</body>')[0]}
            </div>
            <div class="coords-container">
                <h2>Sensor Coordinates, Euler Angles, and Quaternions</h2>
                <p><strong>Position columns (blue background):</strong> X, Y, Z coordinates in meters</p>
                <p><strong>Euler angles (pink background):</strong> Roll, Pitch, Yaw angles in degrees (ZYX convention)</p>
                <p><strong>Quaternion columns (yellow background):</strong> x, y, z, w components (unit quaternion, x,y,z,w order)</p>
                {coord_table_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save as HTML
    with open(output_path, 'w') as f:
        f.write(full_html)
    
    print(f"Interactive HTML with coordinates and rotations saved to: {output_path}")
    
    # Show interactive plot
    fig.show()

def create_coordinate_table_html(coordinate_data):
    """Create HTML table with coordinate, Euler angles, and quaternion information"""
    if not coordinate_data:
        return "<p>No coordinate data available.</p>"
    
    # Group by file
    files = {}
    for coord in coordinate_data:
        file_name = coord['file']
        if file_name not in files:
            files[file_name] = []
        files[file_name].append(coord)
    
    html = '<table class="coords-table">'
    html += '<thead><tr>'
    html += '<th>Dataset</th>'
    html += '<th>Sensor</th>'
    html += '<th class="position-col">X (m)</th>'
    html += '<th class="position-col">Y (m)</th>'
    html += '<th class="position-col">Z (m)</th>'
    html += '<th class="rotation-col">Roll (°)</th>'
    html += '<th class="rotation-col">Pitch (°)</th>'
    html += '<th class="rotation-col">Yaw (°)</th>'
    html += '<th class="quaternion-col">Q_x</th>'
    html += '<th class="quaternion-col">Q_y</th>'
    html += '<th class="quaternion-col">Q_z</th>'
    html += '<th class="quaternion-col">Q_w</th>'
    html += '</tr></thead>'
    html += '<tbody>'
    
    for file_name, coords in files.items():
        # Sort by sensor number
        coords.sort(key=lambda x: x['sensor'])
        
        for coord in coords:
            html += f'<tr>'
            html += f'<td>{coord["file"]}</td>'
            html += f'<td>{coord["sensor"]}</td>'
            html += f'<td class="position-col">{coord["x"]:.3f}</td>'
            html += f'<td class="position-col">{coord["y"]:.3f}</td>'
            html += f'<td class="position-col">{coord["z"]:.3f}</td>'
            html += f'<td class="rotation-col">{coord["roll"]:.1f}</td>'
            html += f'<td class="rotation-col">{coord["pitch"]:.1f}</td>'
            html += f'<td class="rotation-col">{coord["yaw"]:.1f}</td>'
            html += f'<td class="quaternion-col">{coord["qx"]:.4f}</td>'
            html += f'<td class="quaternion-col">{coord["qy"]:.4f}</td>'
            html += f'<td class="quaternion-col">{coord["qz"]:.4f}</td>'
            html += f'<td class="quaternion-col">{coord["qw"]:.4f}</td>'
            html += f'</tr>'
    
    html += '</tbody></table>'
    return html

def save_coordinates_csv(transform_data_list, output_path='sensor_coordinates.csv'):
    """Save sensor coordinates, Euler angles, and quaternions to CSV file"""
    coordinate_data = []
    
    for file_idx, (dataset_name, data) in enumerate(transform_data_list):
        for frame_idx, frame in enumerate(data.get('frames', [])):
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix is None:
                continue
                
            position, rotation = transform_matrix_to_pose(transform_matrix)
            euler_angles = rotation_matrix_to_euler(rotation)
            quaternion = rotation_matrix_to_quaternion(rotation)
            
            coordinate_data.append({
                'Dataset': dataset_name,
                'Sensor': frame_idx,
                'X': f"{position[0]:.6f}",
                'Y': f"{position[1]:.6f}",
                'Z': f"{position[2]:.6f}",
                'Roll_deg': f"{euler_angles[0]:.3f}",
                'Pitch_deg': f"{euler_angles[1]:.3f}",
                'Yaw_deg': f"{euler_angles[2]:.3f}",
                'Quaternion_x': f"{quaternion[0]:.6f}",
                'Quaternion_y': f"{quaternion[1]:.6f}",
                'Quaternion_z': f"{quaternion[2]:.6f}",
                'Quaternion_w': f"{quaternion[3]:.6f}",
                'Dataset_Name': dataset_name
            })
    
    # Write CSV
    if coordinate_data:
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['Dataset', 'Sensor', 'X', 'Y', 'Z', 'Roll_deg', 'Pitch_deg', 'Yaw_deg', 
                         'Quaternion_x', 'Quaternion_y', 'Quaternion_z', 'Quaternion_w', 'Dataset_Name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(coordinate_data)
        
        print(f"Coordinates, Euler angles, and quaternions CSV saved to: {output_path}")
    
    return coordinate_data

def plot_sensor_poses_matplotlib(transform_data_list, output_path='sensor_poses.png'):
    """Plot all sensor poses from multiple files using matplotlib"""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color palette for different files
    colors = plt.cm.Set1(np.linspace(0, 1, len(transform_data_list)))
    
    all_positions = []
    legend_elements = []
    
    for file_idx, (dataset_name, data) in enumerate(transform_data_list):
        print(f"Processing dataset {file_idx + 1}/{len(transform_data_list)}: {dataset_name}")
        
        file_color = colors[file_idx]
        
        # Process each frame/sensor
        for frame_idx, frame in enumerate(data.get('frames', [])):
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix is None:
                continue
                
            # Extract pose
            position, rotation = transform_matrix_to_pose(transform_matrix)
            all_positions.append(position)
            
            # Plot coordinate system
            plot_coordinate_system(ax, position, rotation, scale=0.2, alpha=0.8)
            
            # Plot sensor number with file color
            ax.text(position[0], position[1], position[2] + 0.1, 
                   str(frame_idx), 
                   color=file_color, 
                   fontsize=12, 
                   fontweight='bold',
                   ha='center')
        
        # Add legend entry for this file
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=file_color, markersize=10,
                                        label=f'{dataset_name} ({len(data.get("frames", []))} sensors)'))
    
    # Set up the plot
    if all_positions:
        all_positions = np.array(all_positions)
        
        # Set axis limits with some padding
        padding = 0.5
        ax.set_xlim(all_positions[:, 0].min() - padding, all_positions[:, 0].max() + padding)
        ax.set_ylim(all_positions[:, 1].min() - padding, all_positions[:, 1].max() + padding)
        ax.set_zlim(all_positions[:, 2].min() - padding, all_positions[:, 2].max() + padding)
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title('3D Sensor Poses Visualization\n(X=Red, Y=Green, Z=Blue)', fontsize=14, fontweight='bold')
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add coordinate system reference
    ax.text2D(0.02, 0.02, 'Coordinate System:\nX-axis: Red\nY-axis: Green\nZ-axis: Blue', 
              transform=ax.transAxes, fontsize=10, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Static PNG saved to: {output_path}")
    plt.show()

# ========================================
# MAIN EXECUTION FUNCTIONS
# ========================================

def process_nuscenes_data(nusc_path='/app/datasets/nuscenes_full/', version='v1.0-trainval'):
    """Process nuScenes data and return camera data"""
    try:
        # Initialize nuScenes
        nusc = NuScenes(version=version, dataroot=nusc_path, verbose=True)
        
        # Get camera data with images (now includes transformation)
        camera_data = get_all_camera_positions_with_images(nusc)
        
        return camera_data
    except Exception as e:
        print(f"Error processing nuScenes data: {e}")
        return None

def create_transform_data_from_camera_data(camera_data):
    """Convert camera data to transform JSON format for visualization"""
    # Base structure
    base_structure = {
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": []
    }
    
    # Create original transforms JSON
    original_json = base_structure.copy()
    original_json["frames"] = []
    
    # Create transformed transforms JSON
    transformed_json = base_structure.copy()
    transformed_json["frames"] = []
    
    # Process each camera
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        # Get camera intrinsics
        camera_intrinsic = np.array(data['camera_intrinsic'])
        fl_x = camera_intrinsic[0, 0]
        fl_y = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        
        # Assume standard image dimensions (you may need to adjust these)
        w = 1600
        h = 900
        
        # Original frame
        original_frame = {
            "file_path": f"../sensors/{idx}_rgb.png",
            "depth_file_path": f"../sensors/{idx}_depth.png",
            "semantic_segmentation_file_path": f"../sensors/{idx}_semantic_segmentation.png",
            "instance_segmentation_file_path": f"../sensors/{idx}_instance_segmentation.png",
            "transform_matrix": data['original_transform_matrix'].tolist(),
            "fl_x": float(fl_x),
            "fl_y": float(fl_y),
            "cx": float(cx),
            "cy": float(cy),
            "w": w,
            "h": h,
            "camera_name": camera_name
        }
        
        # Transformed frame
        transformed_frame = {
            "file_path": f"../sensors/{idx}_rgb.png",
            "depth_file_path": f"../sensors/{idx}_depth.png",
            "semantic_segmentation_file_path": f"../sensors/{idx}_semantic_segmentation.png",
            "instance_segmentation_file_path": f"../sensors/{idx}_instance_segmentation.png",
            "transform_matrix": data['transform_matrix'].tolist(),
            "fl_x": float(fl_x),
            "fl_y": float(fl_y),
            "cx": float(cx),
            "cy": float(cy),
            "w": w,
            "h": h,
            "camera_name": camera_name
        }
        
        original_json["frames"].append(original_frame)
        transformed_json["frames"].append(transformed_frame)
    
    return original_json, transformed_json

def save_all_outputs(camera_data, output_dir='output_plots'):
    """
    Save all plots and JSON files to the specified directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    camera_images_path = os.path.join(output_dir, 'camera_images.png')
    coordinate_comparison_path = os.path.join(output_dir, 'coordinate_comparison.png')
    original_json_path = os.path.join(output_dir, 'transforms_original.json')
    transformed_json_path = os.path.join(output_dir, 'transforms_transformed.json')
    
    print(f"Saving camera pose outputs to directory: {output_dir}")
    
    # Save camera images plot
    plot_camera_images(camera_data, output_path=camera_images_path)
    
    # Save coordinate comparison plot
    plot_coordinate_comparison(camera_data, output_path=coordinate_comparison_path)
    
    # Save JSON files
    create_json_output(camera_data, original_json_path, transformed_json_path)
    
    print(f"\nCamera pose outputs saved successfully in: {output_dir}")
    print(f"- Camera images: {camera_images_path}")
    print(f"- Coordinate comparison: {coordinate_comparison_path}")
    print(f"- Original transforms JSON: {original_json_path}")
    print(f"- Transformed transforms JSON: {transformed_json_path}")

def main():
    parser = argparse.ArgumentParser(description='Process nuScenes camera poses and create 3D visualizations')
    parser.add_argument('--save-json', action='store_true',
                       help='Save intermediate transform JSON files')
    parser.add_argument('--output', '-o', default='sensor_poses', 
                       help='Output file prefix (default: sensor_poses)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive visualization')
    parser.add_argument('--no-static', action='store_true',
                       help='Skip static PNG generation')
    parser.add_argument('--no-ply', action='store_true',
                       help='Skip PLY file generation')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV coordinates file generation')
    parser.add_argument('--output-dir', default='output_plots',
                       help='Directory for saving camera pose plots (default: output_plots)')
    parser.add_argument('--nuscenes-path', default='/app/datasets/nuscenes_full/',
                       help='Path to nuScenes dataset (default: /app/datasets/nuscenes_full/)')
    parser.add_argument('--nuscenes-version', default='v1.0-trainval',
                       help='nuScenes dataset version (default: v1.0-trainval)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MERGED CAMERA POSE TRANSFORMATION AND VISUALIZATION")
    print("="*80)
    
    # Step 1: Process nuScenes data
    print("\nStep 1: Processing nuScenes data...")
    camera_data = process_nuscenes_data(args.nuscenes_path, args.nuscenes_version)
    
    if camera_data is None:
        print("Error: Failed to process nuScenes data!")
        sys.exit(1)
    
    # Print camera information
    print_camera_info(camera_data)
    
    # Step 2: Save camera pose outputs (images, plots, and optionally JSON files)
    if args.save_json:
        print("\nStep 2: Saving camera pose outputs including JSON files...")
        save_all_outputs(camera_data, args.output_dir)
    else:
        print("\nStep 2: Saving camera pose outputs (skipping JSON files)...")
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Define output paths
        camera_images_path = os.path.join(args.output_dir, 'camera_images.png')
        coordinate_comparison_path = os.path.join(args.output_dir, 'coordinate_comparison.png')
        
        print(f"Saving camera pose outputs to directory: {args.output_dir}")
        
        # Save camera images plot
        plot_camera_images(camera_data, output_path=camera_images_path)
        
        # Save coordinate comparison plot
        plot_coordinate_comparison(camera_data, output_path=coordinate_comparison_path)
        
        print(f"\nCamera pose outputs saved successfully in: {args.output_dir}")
        print(f"- Camera images: {camera_images_path}")
        print(f"- Coordinate comparison: {coordinate_comparison_path}")
    
    # Step 3: Create transform data for visualization
    print("\nStep 3: Creating transform data for visualization...")
    original_json, transformed_json = create_transform_data_from_camera_data(camera_data)
    
    # Prepare data for visualization functions
    transform_data_list = [
        ("Original", original_json),
        ("Transformed", transformed_json)
    ]
    
    # Step 4: Generate visualizations
    print("\nStep 4: Generating 3D visualizations...")
    
    # Generate coordinate data first
    if not args.no_csv:
        print("Generating coordinates, Euler angles, and quaternions CSV...")
        coordinate_data = save_coordinates_csv(transform_data_list, f"{args.output}_coordinates.csv")
    
    # Generate visualizations
    if not args.no_interactive:
        print("Generating interactive 3D visualization with coordinate, Euler, and quaternion table...")
        create_interactive_plotly(transform_data_list, f"{args.output}.html")
    
    if not args.no_static:
        print("Generating static PNG...")
        plot_sensor_poses_matplotlib(transform_data_list, f"{args.output}.png")
    
    if not args.no_ply:
        print("Generating PLY 3D file...")
        write_ply_file(transform_data_list, f"{args.output}.ply")
    
    # Step 5: Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    
    print(f"Camera pose files saved in: {args.output_dir}")
    if args.save_json:
        print("- JSON transform files: transforms_original.json, transforms_transformed.json")
    print("- Camera images plot: camera_images.png")
    print("- Coordinate comparison plot: coordinate_comparison.png")
    
    print(f"\nVisualization files generated:")
    if not args.no_interactive:
        print(f"- Interactive HTML with coordinates, Euler angles, and quaternions: {args.output}.html")
    if not args.no_static:
        print(f"- Static PNG: {args.output}.png")
    if not args.no_ply:
        print(f"- 3D PLY file: {args.output}.ply")
    if not args.no_csv:
        print(f"- Coordinates, Euler angles, and quaternions CSV: {args.output}_coordinates.csv")
    
    # Print coordinate and rotation summary to console
    print(f"\n" + "="*120)
    print("COORDINATE, EULER ANGLES, AND QUATERNION SUMMARY")
    print("="*120)
    
    for dataset_name, data in [("Original", original_json), ("Transformed", transformed_json)]:
        print(f"\nDataset: {dataset_name}")
        print("-" * 100)
        print(f"{'Sensor':>6} {'X':>8} {'Y':>8} {'Z':>8} {'Roll':>8} {'Pitch':>8} {'Yaw':>8} {'Q_x':>8} {'Q_y':>8} {'Q_z':>8} {'Q_w':>8}")
        print("-" * 100)
        
        for frame_idx, frame in enumerate(data.get('frames', [])):
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix is None:
                continue
                
            position, rotation = transform_matrix_to_pose(transform_matrix)
            euler_angles = rotation_matrix_to_euler(rotation)
            quaternion = rotation_matrix_to_quaternion(rotation)
            
            print(f"{frame_idx:6d} {position[0]:8.3f} {position[1]:8.3f} {position[2]:8.3f} "
                  f"{euler_angles[0]:8.1f} {euler_angles[1]:8.1f} {euler_angles[2]:8.1f} "
                  f"{quaternion[0]:8.4f} {quaternion[1]:8.4f} {quaternion[2]:8.4f} {quaternion[3]:8.4f}")
    
    print("="*120)
    
    if not args.no_interactive:
        print(f"\nTo view the interactive 3D visualization with coordinate, Euler, and quaternion table, open {args.output}.html in your web browser.")
    if not args.no_ply:
        print(f"To view the PLY file, use software like MeshLab, Blender, or CloudCompare.")
    if not args.no_csv:
        print(f"The CSV file contains all sensor coordinates, Euler angles, and quaternions for data analysis.")

if __name__ == "__main__":
    main()