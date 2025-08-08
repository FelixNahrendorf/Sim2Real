#!/usr/bin/env python3
"""
3D Camera Extrinsics Visualizer

This script visualizes camera poses from text files containing Target and Context extrinsics
matrices in PyTorch tensor format. Each file's cameras are displayed with unique colors,
where Target cameras are shown as red dots and Context cameras as green dots.

Outputs:
- Interactive 3D plot (rotatable)
- Static PNG image
- PLY 3D file for external viewing
- HTML file with interactive 3D visualization (includes rotation data)
- CSV file with coordinates and rotation data

Usage:
    python visualize_camera_poses.py <path1.txt> <path2.txt> ... <pathN.txt>

Example:
    python visualize_camera_poses.py dataset1.txt dataset2.txt
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def extract_dataset_name(file_path):
    """Extract dataset name from file path or from content"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for the pattern "=== FINAL DATA FED TO MODEL ===" followed by dataset name
        match = re.search(r'=== FINAL DATA FED TO MODEL ===\s*\n\s*(\S+)', content)
        if match:
            return match.group(1)
    except Exception:
        pass
    
    # Fallback to filename if pattern not found
    return Path(file_path).stem

def parse_pytorch_tensor(tensor_text):
    """Parse PyTorch tensor from text format to numpy array"""
    # Remove 'tensor(' and the closing parenthesis, handling multiline format
    tensor_text = tensor_text.strip()
    if tensor_text.startswith('tensor('):
        tensor_text = tensor_text[7:]  # Remove 'tensor('
    
    # Find the last occurrence of ')' and remove everything after it
    last_paren = tensor_text.rfind(')')
    if last_paren != -1:
        tensor_text = tensor_text[:last_paren]
    
    # Remove all whitespace and normalize
    tensor_text = re.sub(r'\s+', '', tensor_text)
    
    # Extract all floating point numbers (including scientific notation)
    number_pattern = r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?'
    matches = re.findall(number_pattern, tensor_text)
    
    if not matches:
        print("No numbers found in tensor text")
        return None
    
    numbers = [float(n) for n in matches]
    total_numbers = len(numbers)
    
    print(f"  Found {total_numbers} numbers")
    
    # Common cases for transformation matrices
    if total_numbers % 16 == 0:  # 4x4 matrices
        num_matrices = total_numbers // 16
        print(f"  Detected {num_matrices} 4x4 matrices")
        return np.array(numbers).reshape(num_matrices, 4, 4)
    elif total_numbers % 9 == 0:  # 3x3 matrices  
        num_matrices = total_numbers // 9
        print(f"  Detected {num_matrices} 3x3 matrices")
        return np.array(numbers).reshape(num_matrices, 3, 3)
    else:
        print(f"  Cannot determine tensor structure. Total numbers: {total_numbers}")
        return None

def load_camera_data(file_path):
    """Load and parse camera data from text file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract dataset name
        dataset_name = extract_dataset_name(file_path)
        
        # Find Context extrinsics - improved pattern to handle multiline tensors
        context_pattern = r'Context extrinsics:\s*\n\s*(tensor\(\[\[\[.*?\]\]\]\))'
        context_match = re.search(context_pattern, content, re.DOTALL)
        context_extrinsics = None
        if context_match:
            context_tensor_text = context_match.group(1)
            print(f"Found Context extrinsics for {dataset_name}")
            context_extrinsics = parse_pytorch_tensor(context_tensor_text)
            if context_extrinsics is not None:
                print(f"  Parsed Context extrinsics shape: {context_extrinsics.shape}")
        
        # Find Target extrinsics - improved pattern to handle multiline tensors
        target_pattern = r'Target extrinsics:\s*\n\s*(tensor\(\[\[\[.*?\]\]\]\))'
        target_match = re.search(target_pattern, content, re.DOTALL)
        target_extrinsics = None
        if target_match:
            target_tensor_text = target_match.group(1)
            print(f"Found Target extrinsics for {dataset_name}")
            target_extrinsics = parse_pytorch_tensor(target_tensor_text)
            if target_extrinsics is not None:
                print(f"  Parsed Target extrinsics shape: {target_extrinsics.shape}")
        
        return {
            'dataset_name': dataset_name,
            'context_extrinsics': context_extrinsics,
            'target_extrinsics': target_extrinsics
        }
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def transform_matrix_to_pose(transform_matrix):
    """Extract position and rotation from 4x4 transform matrix"""
    if transform_matrix.shape != (4, 4):
        print(f"Warning: Expected 4x4 matrix, got {transform_matrix.shape}")
        return None, None
    
    position = transform_matrix[:3, 3]
    rotation = transform_matrix[:3, :3]
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

def write_ply_file(file_paths, output_path='camera_poses.ply'):
    """Write camera poses to PLY file format for 3D viewing"""
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
    
    for file_idx, file_path in enumerate(file_paths):
        data = load_camera_data(file_path)
        if data is None:
            continue
        
        # Process Context cameras (green)
        if data['context_extrinsics'] is not None:
            for cam_idx in range(data['context_extrinsics'].shape[0]):
                transform_matrix = data['context_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                
                # Add coordinate system vertices
                scale = 0.2
                axes_points = [position]
                colors.append([0, 255, 0])  # Green for context
                
                # Add axis endpoints
                for i in range(3):
                    axis_color = [0, 0, 0]
                    axis_color[i] = 255
                    axis_end = position + rotation[:, i] * scale
                    axes_points.append(axis_end)
                    colors.append(axis_color)
                
                vertices.extend(axes_points)
                
                # Add lines
                for i in range(3):
                    faces.append([vertex_count, vertex_count + 1 + i])
                vertex_count += 4
        
        # Process Target cameras (red)
        if data['target_extrinsics'] is not None:
            for cam_idx in range(data['target_extrinsics'].shape[0]):
                transform_matrix = data['target_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                
                # Add coordinate system vertices
                scale = 0.2
                axes_points = [position]
                colors.append([255, 0, 0])  # Red for target
                
                # Add axis endpoints
                for i in range(3):
                    axis_color = [0, 0, 0]
                    axis_color[i] = 255
                    axis_end = position + rotation[:, i] * scale
                    axes_points.append(axis_end)
                    colors.append(axis_color)
                
                vertices.extend(axes_points)
                
                # Add lines
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

def create_interactive_plotly(file_paths, output_path='camera_poses.html'):
    """Create interactive 3D visualization using Plotly"""
    fig = go.Figure()
    
    all_positions = []
    coordinate_data = []
    
    # Color scheme for different files
    context_colors = ['green', 'blue', 'purple', 'brown', 'pink', 'gray']
    target_colors = ['red', 'orange', 'yellow', 'magenta', 'cyan', 'lime']
    
    for file_idx, file_path in enumerate(file_paths):
        data = load_camera_data(file_path)
        if data is None:
            continue
            
        dataset_name = data['dataset_name']
        
        # Get colors for this file
        context_color = context_colors[file_idx % len(context_colors)]
        target_color = target_colors[file_idx % len(target_colors)]
        
        # Process Context cameras
        if data['context_extrinsics'] is not None:
            positions = []
            rotations = []
            hover_texts = []
            
            for cam_idx in range(data['context_extrinsics'].shape[0]):
                transform_matrix = data['context_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                    
                euler_angles = rotation_matrix_to_euler(rotation)
                quaternion = rotation_matrix_to_quaternion(rotation)
                
                positions.append(position)
                rotations.append(rotation)
                all_positions.append(position)
                
                # Store coordinate data
                coordinate_data.append({
                    'file': dataset_name,
                    'camera_type': 'Context',
                    'camera': cam_idx,
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
                
                hover_text = (f"Dataset: {dataset_name}<br>"
                             f"Type: Context Camera<br>"
                             f"Camera: {cam_idx}<br>"
                             f"X: {position[0]:.3f}<br>"
                             f"Y: {position[1]:.3f}<br>"
                             f"Z: {position[2]:.3f}<br>"
                             f"Roll: {euler_angles[0]:.1f}°<br>"
                             f"Pitch: {euler_angles[1]:.1f}°<br>"
                             f"Yaw: {euler_angles[2]:.1f}°<br>"
                             f"Quaternion (x,y,z,w): ({quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f})")
                hover_texts.append(hover_text)
            
            if positions:
                positions = np.array(positions)
                legend_group = f"group_{file_idx}"
                
                # Add context camera positions with file-specific color
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='markers+text',
                    text=[f'C{i}' for i in range(len(positions))],
                    textposition="top center",
                    hovertext=hover_texts,
                    hoverinfo='text',
                    marker=dict(
                        size=10,
                        color=context_color,
                        symbol='circle'
                    ),
                    name=f'{dataset_name} Context ({len(positions)})',
                    legendgroup=legend_group,
                    showlegend=True
                ))
                
                # Add coordinate axes for context cameras
                add_coordinate_axes(fig, positions, rotations, legend_group, context_color)
        
        # Process Target cameras
        if data['target_extrinsics'] is not None:
            positions = []
            rotations = []
            hover_texts = []
            
            for cam_idx in range(data['target_extrinsics'].shape[0]):
                transform_matrix = data['target_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                    
                euler_angles = rotation_matrix_to_euler(rotation)
                quaternion = rotation_matrix_to_quaternion(rotation)
                
                positions.append(position)
                rotations.append(rotation)
                all_positions.append(position)
                
                # Store coordinate data
                coordinate_data.append({
                    'file': dataset_name,
                    'camera_type': 'Target',
                    'camera': cam_idx,
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
                
                hover_text = (f"Dataset: {dataset_name}<br>"
                             f"Type: Target Camera<br>"
                             f"Camera: {cam_idx}<br>"
                             f"X: {position[0]:.3f}<br>"
                             f"Y: {position[1]:.3f}<br>"
                             f"Z: {position[2]:.3f}<br>"
                             f"Roll: {euler_angles[0]:.1f}°<br>"
                             f"Pitch: {euler_angles[1]:.1f}°<br>"
                             f"Yaw: {euler_angles[2]:.1f}°<br>"
                             f"Quaternion (x,y,z,w): ({quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f})")
                hover_texts.append(hover_text)
            
            if positions:
                positions = np.array(positions)
                legend_group = f"group_{file_idx}"
                
                # Add target camera positions with file-specific color
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='markers+text',
                    text=[f'T{i}' for i in range(len(positions))],
                    textposition="top center",
                    hovertext=hover_texts,
                    hoverinfo='text',
                    marker=dict(
                        size=10,
                        color=target_color,
                        symbol='circle'
                    ),
                    name=f'{dataset_name} Target ({len(positions)})',
                    legendgroup=legend_group,
                    showlegend=True
                ))
                
                # Add coordinate axes for target cameras
                add_coordinate_axes(fig, positions, rotations, legend_group, target_color)
    
    # Create coordinate table HTML
    coord_table_html = create_coordinate_table_html(coordinate_data)
    
    # Update layout
    fig.update_layout(
        title='Interactive 3D Camera Poses Visualization<br><sub>Target Cameras=Red, Context Cameras=Green | X=Red, Y=Green, Z=Blue | Use mouse to rotate, zoom, and pan</sub>',
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
        <title>3D Camera Poses Visualization</title>
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
            .context-row {{ background-color: #e6ffe6; }}
            .target-row {{ background-color: #ffe6e6; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="plot-container">
                {plot_html.split('<body>')[1].split('</body>')[0]}
            </div>
            <div class="coords-container">
                <h2>Camera Coordinates, Euler Angles, and Quaternions</h2>
                <p><strong>Position columns (blue background):</strong> X, Y, Z coordinates in meters</p>
                <p><strong>Euler angles (pink background):</strong> Roll, Pitch, Yaw angles in degrees (ZYX convention)</p>
                <p><strong>Quaternion columns (yellow background):</strong> x, y, z, w components (unit quaternion)</p>
                <p><strong>File 1 Context cameras (green background):</strong> First file context/reference cameras</p>
                <p><strong>File 1 Target cameras (red background):</strong> First file target cameras</p>
                <p><strong>File 2 Context cameras (blue background):</strong> Second file context/reference cameras</p>
                <p><strong>File 2 Target cameras (orange background):</strong> Second file target cameras</p>
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

def add_coordinate_axes(fig, positions, rotations, legend_group, base_color):
    """Add coordinate axes to the plotly figure with infinite Z-axes"""
    x_lines_x, x_lines_y, x_lines_z = [], [], []
    y_lines_x, y_lines_y, y_lines_z = [], [], []
    z_lines_x, z_lines_y, z_lines_z = [], [], []
    
    scale = 0.3
    z_scale = 20.0  # Make Z-axis much longer to simulate infinity
    
    for position, rotation in zip(positions, rotations):
        # X axis points (short)
        x_end = position + rotation[:, 0] * scale
        x_lines_x.extend([position[0], x_end[0], None])
        x_lines_y.extend([position[1], x_end[1], None])
        x_lines_z.extend([position[2], x_end[2], None])
        
        # Y axis points (short)
        y_end = position + rotation[:, 1] * scale
        y_lines_x.extend([position[0], y_end[0], None])
        y_lines_y.extend([position[1], y_end[1], None])
        y_lines_z.extend([position[2], y_end[2], None])
        
        # Z axis points (long - simulating infinity in positive direction)
        z_end = position + rotation[:, 2] * z_scale
        z_lines_x.extend([position[0], z_end[0], None])
        z_lines_y.extend([position[1], z_end[1], None])
        z_lines_z.extend([position[2], z_end[2], None])
    
    # Add X axes
    fig.add_trace(go.Scatter3d(
        x=x_lines_x, y=x_lines_y, z=z_lines_z,
        mode='lines', line=dict(color='red', width=4),
        legendgroup=legend_group, showlegend=False, hoverinfo='skip'
    ))
    
    # Add Y axes  
    fig.add_trace(go.Scatter3d(
        x=y_lines_x, y=y_lines_y, z=y_lines_z,
        mode='lines', line=dict(color='green', width=4),
        legendgroup=legend_group, showlegend=False, hoverinfo='skip'
    ))
    
    # Add Z axes (long blue lines)
    fig.add_trace(go.Scatter3d(
        x=z_lines_x, y=z_lines_y, z=z_lines_z,
        mode='lines', line=dict(color='blue', width=6),  # Make Z-axis thicker
        legendgroup=legend_group, showlegend=False, hoverinfo='skip'
    ))

def create_coordinate_table_html(coordinate_data):
    """Create HTML table with coordinate, Euler angles, and quaternion information"""
    if not coordinate_data:
        return "<p>No coordinate data available.</p>"
    
    # Group by file
    files = {}
    for coord in coordinate_data:
        file_name = coord['file']
        if file_name not in files:
            files[file_name] = {'Context': [], 'Target': []}
        files[file_name][coord['camera_type']].append(coord)
    
    html = '<table class="coords-table">'
    html += '<thead><tr>'
    html += '<th>Dataset</th>'
    html += '<th>Type</th>'
    html += '<th>Camera</th>'
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
    
    for file_name, camera_types in files.items():
        for camera_type in ['Context', 'Target']:
            coords = camera_types[camera_type]
            coords.sort(key=lambda x: x['camera'])
            
            for coord in coords:
                row_class = 'context-row' if camera_type == 'Context' else 'target-row'
                html += f'<tr class="{row_class}">'
                html += f'<td>{coord["file"]}</td>'
                html += f'<td>{coord["camera_type"]}</td>'
                html += f'<td>{coord["camera"]}</td>'
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

def save_coordinates_csv(file_paths, output_path='camera_coordinates.csv'):
    """Save camera coordinates, Euler angles, and quaternions to CSV file"""
    coordinate_data = []
    
    for file_path in file_paths:
        data = load_camera_data(file_path)
        if data is None:
            continue
            
        dataset_name = data['dataset_name']
        
        # Process Context cameras
        if data['context_extrinsics'] is not None:
            for cam_idx in range(data['context_extrinsics'].shape[0]):
                transform_matrix = data['context_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                    
                euler_angles = rotation_matrix_to_euler(rotation)
                quaternion = rotation_matrix_to_quaternion(rotation)
                
                coordinate_data.append({
                    'Dataset': dataset_name,
                    'Type': 'Context',
                    'Camera': cam_idx,
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
                    'File_Path': file_path
                })
        
        # Process Target cameras
        if data['target_extrinsics'] is not None:
            for cam_idx in range(data['target_extrinsics'].shape[0]):
                transform_matrix = data['target_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                    
                euler_angles = rotation_matrix_to_euler(rotation)
                quaternion = rotation_matrix_to_quaternion(rotation)
                
                coordinate_data.append({
                    'Dataset': dataset_name,
                    'Type': 'Target',
                    'Camera': cam_idx,
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
                    'File_Path': file_path
                })
    
    # Write CSV
    if coordinate_data:
        import csv
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['Dataset', 'Type', 'Camera', 'X', 'Y', 'Z', 'Roll_deg', 'Pitch_deg', 'Yaw_deg', 
                         'Quaternion_x', 'Quaternion_y', 'Quaternion_z', 'Quaternion_w', 'File_Path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(coordinate_data)
        
        print(f"Coordinates, Euler angles, and quaternions CSV saved to: {output_path}")
    
    return coordinate_data

def plot_camera_poses_matplotlib(file_paths, output_path='camera_poses.png'):
    """Plot all camera poses from multiple files using matplotlib"""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color scheme for different files
    context_colors = ['green', 'blue', 'purple', 'brown', 'pink', 'gray']
    target_colors = ['red', 'orange', 'yellow', 'magenta', 'cyan', 'lime']
    
    all_positions = []
    legend_elements = []
    
    for file_idx, file_path in enumerate(file_paths):
        print(f"Processing file {file_idx + 1}/{len(file_paths)}: {file_path}")
        
        # Load camera data
        data = load_camera_data(file_path)
        if data is None:
            continue
            
        dataset_name = data['dataset_name']
        
        # Get colors for this file
        context_color = context_colors[file_idx % len(context_colors)]
        target_color = target_colors[file_idx % len(target_colors)]
        
        context_count = 0
        target_count = 0
        
        # Process Context cameras
        if data['context_extrinsics'] is not None:
            for cam_idx in range(data['context_extrinsics'].shape[0]):
                transform_matrix = data['context_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                    
                all_positions.append(position)
                context_count += 1
                
                # Plot coordinate system
                plot_coordinate_system(ax, position, rotation, scale=0.3, alpha=0.8)
                
                # Plot camera number with context color
                ax.text(position[0], position[1], position[2] + 0.1, 
                       f'C{cam_idx}', 
                       color=context_color, 
                       fontsize=10, 
                       fontweight='bold',
                       ha='center')
        
        # Process Target cameras
        if data['target_extrinsics'] is not None:
            for cam_idx in range(data['target_extrinsics'].shape[0]):
                transform_matrix = data['target_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                    
                all_positions.append(position)
                target_count += 1
                
                # Plot coordinate system
                plot_coordinate_system(ax, position, rotation, scale=0.3, alpha=0.8)
                
                # Plot camera number with target color
                ax.text(position[0], position[1], position[2] + 0.1, 
                       f'T{cam_idx}', 
                       color=target_color, 
                       fontsize=10, 
                       fontweight='bold',
                       ha='center')
        
        # Add legend entries for this dataset
        if context_count > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=context_color, markersize=10,
                                            label=f'{dataset_name} Context ({context_count})'))
        if target_count > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=target_color, markersize=10,
                                            label=f'{dataset_name} Target ({target_count})'))
    
    # Set up the plot
    if all_positions:
        all_positions = np.array(all_positions)
        
        # Set axis limits with some padding
        padding = 1.0
        ax.set_xlim(all_positions[:, 0].min() - padding, all_positions[:, 0].max() + padding)
        ax.set_ylim(all_positions[:, 1].min() - padding, all_positions[:, 1].max() + padding)
        ax.set_zlim(all_positions[:, 2].min() - padding, all_positions[:, 2].max() + padding)
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title('3D Camera Poses Visualization\n(File 1: Context=Green, Target=Red | File 2: Context=Blue, Target=Orange | X=Red, Y=Green, Z=Blue)', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add coordinate system reference
    ax.text2D(0.02, 0.02, 'Coordinate System:\nX-axis: Red\nY-axis: Green\nZ-axis: Blue\n\nFile 1 (NuScenes):\nContext: Green (C0, C1, ...)\nTarget: Red (T0, T1, ...)\n\nFile 2 (Seed4D):\nContext: Blue (C0, C1, ...)\nTarget: Orange (T0, T1, ...)', 
              transform=ax.transAxes, fontsize=10, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Static PNG saved to: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D camera poses from text files with extrinsics matrices')
    parser.add_argument('files', nargs='+', help='Paths to text files containing Target and Context extrinsics')
    parser.add_argument('--output', '-o', default='camera_poses', 
                       help='Output file prefix (default: camera_poses)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive visualization')
    parser.add_argument('--no-static', action='store_true',
                       help='Skip static PNG generation')
    parser.add_argument('--no-ply', action='store_true',
                       help='Skip PLY file generation')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV coordinates file generation')
    
    args = parser.parse_args()
    
    # Validate input files
    valid_files = []
    for file_path in args.files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not valid_files:
        print("Error: No valid input files found!")
        sys.exit(1)
    
    print(f"Processing {len(valid_files)} file(s)...")
    
    # Generate coordinate data first
    if not args.no_csv:
        print("\nGenerating coordinates, Euler angles, and quaternions CSV...")
        coordinate_data = save_coordinates_csv(valid_files, f"{args.output}_coordinates.csv")
    
    # Generate visualizations
    if not args.no_interactive:
        print("\nGenerating interactive 3D visualization...")
        create_interactive_plotly(valid_files, f"{args.output}.html")
    
    if not args.no_static:
        print("\nGenerating static PNG...")
        plot_camera_poses_matplotlib(valid_files, f"{args.output}.png")
    
    if not args.no_ply:
        print("\nGenerating PLY 3D file...")
        write_ply_file(valid_files, f"{args.output}.ply")
    
    print("\nVisualization complete!")
    print(f"Files generated:")
    if not args.no_interactive:
        print(f"  - Interactive HTML: {args.output}.html")
    if not args.no_static:
        print(f"  - Static PNG: {args.output}.png")
    if not args.no_ply:
        print(f"  - 3D PLY file: {args.output}.ply")
    if not args.no_csv:
        print(f"  - Coordinates CSV: {args.output}_coordinates.csv")
    
    # Print summary to console
    print(f"\n" + "="*120)
    print("CAMERA COORDINATE AND ROTATION SUMMARY")
    print("="*120)
    
    for file_path in valid_files:
        data = load_camera_data(file_path)
        if data is None:
            continue
            
        dataset_name = data['dataset_name']
        print(f"\nDataset: {dataset_name}")
        print("-" * 100)
        print(f"{'Type':>8} {'Cam':>4} {'X':>8} {'Y':>8} {'Z':>8} {'Roll':>8} {'Pitch':>8} {'Yaw':>8} {'Q_x':>8} {'Q_y':>8} {'Q_z':>8} {'Q_w':>8}")
        print("-" * 100)
        
        # Print Context cameras
        if data['context_extrinsics'] is not None:
            for cam_idx in range(data['context_extrinsics'].shape[0]):
                transform_matrix = data['context_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                    
                euler_angles = rotation_matrix_to_euler(rotation)
                quaternion = rotation_matrix_to_quaternion(rotation)
                
                print(f"{'Context':>8} {cam_idx:4d} {position[0]:8.3f} {position[1]:8.3f} {position[2]:8.3f} "
                      f"{euler_angles[0]:8.1f} {euler_angles[1]:8.1f} {euler_angles[2]:8.1f} "
                      f"{quaternion[0]:8.4f} {quaternion[1]:8.4f} {quaternion[2]:8.4f} {quaternion[3]:8.4f}")
        
        # Print Target cameras
        if data['target_extrinsics'] is not None:
            for cam_idx in range(data['target_extrinsics'].shape[0]):
                transform_matrix = data['target_extrinsics'][cam_idx]
                position, rotation = transform_matrix_to_pose(transform_matrix)
                if position is None:
                    continue
                    
                euler_angles = rotation_matrix_to_euler(rotation)
                quaternion = rotation_matrix_to_quaternion(rotation)
                
                print(f"{'Target':>8} {cam_idx:4d} {position[0]:8.3f} {position[1]:8.3f} {position[2]:8.3f} "
                      f"{euler_angles[0]:8.1f} {euler_angles[1]:8.1f} {euler_angles[2]:8.1f} "
                      f"{quaternion[0]:8.4f} {quaternion[1]:8.4f} {quaternion[2]:8.4f} {quaternion[3]:8.4f}")
    
    print("="*120)
    print("\nVisualization Notes:")
    print("- File 1: Context cameras are GREEN, Target cameras are RED")
    print("- File 2: Context cameras are BLUE, Target cameras are ORANGE")
    print("- Additional files cycle through: purple/yellow, brown/magenta, pink/cyan, gray/lime")
    print("- Coordinate axes: X=Red, Y=Green, Z=Blue")
    print("- Hover over cameras in the interactive HTML for detailed information")

if __name__ == "__main__":
    main()