#!/usr/bin/env python3
"""
CSV Camera Parameters 3D Visualizer (Fixed Version)

This script visualizes camera poses from CSV files containing camera parameters statistics.
It uses the mean values to display camera positions and orientations in 3D space.
Each CSV file's cameras are displayed with a unique color, and coordinate systems show
consistent x (red), y (green), z (blue) axes.

Outputs:
- Interactive 3D plot (rotatable)
- Static PNG image
- PLY 3D file for external viewing
- HTML file with interactive 3D visualization

Usage:
    python csv_camera_visualizer.py <csv_path1> <csv_path2> ... <csv_pathN>

Example:
    python csv_camera_visualizer.py transforms_ego_adjusted.csv nuscene_trainval_camera_parameters_statistics.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from pathlib import Path
import sys
import traceback

# Try to import plotly, but make it optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    print("✓ Plotly available for interactive visualizations")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠ Plotly not available. Interactive visualizations will be skipped.")
    print("  Install with: pip install plotly")

def extract_dataset_name(file_path):
    """Extract dataset name from CSV file path"""
    return Path(file_path).stem

def load_csv_file(file_path):
    """Load and parse a CSV file with camera parameters"""
    try:
        print(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"  ✓ Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"  ✗ Error loading {file_path}: {e}")
        traceback.print_exc()
        return None

def parse_camera_parameters(df):
    """Parse camera parameters from CSV and extract mean values for position and rotation"""
    print("Parsing camera parameters...")
    cameras = {}
    
    # Get unique camera names
    unique_cameras = df['Camera'].unique()
    print(f"  Found cameras: {list(unique_cameras)}")
    
    for camera in unique_cameras:
        camera_data = df[df['Camera'] == camera]
        camera_params = {}
        
        # Extract mean values for each parameter
        for _, row in camera_data.iterrows():
            param = row['Parameter']
            mean_val = row['Mean']
            camera_params[param] = mean_val
        
        cameras[camera] = camera_params
        print(f"  {camera}: {list(camera_params.keys())}")
    
    return cameras

def extract_pose_data(camera_params):
    """Extract position and Euler angles directly from camera parameters"""
    print(f"    Extracting pose data from parameters: {list(camera_params.keys())}")
    
    # Extract position (try different common parameter names)
    x = camera_params.get('x', camera_params.get('X', camera_params.get('pos_x', 0.0)))
    y = camera_params.get('y', camera_params.get('Y', camera_params.get('pos_y', 0.0)))
    z = camera_params.get('z', camera_params.get('Z', camera_params.get('pos_z', 0.0)))
    
    position = np.array([x, y, z])
    print(f"    Position extracted: [{x:.3f}, {y:.3f}, {z:.3f}]")
    
    # Extract Euler angles (try different parameter naming conventions)
    roll = camera_params.get('roll', camera_params.get('Roll', camera_params.get('rot_x', 0.0)))
    pitch = camera_params.get('pitch', camera_params.get('Pitch', camera_params.get('rot_y', 0.0)))
    yaw = camera_params.get('yaw', camera_params.get('Yaw', camera_params.get('rot_z', 0.0)))
    
    # Ensure angles are in degrees for consistency
    euler_angles = np.array([roll, pitch, yaw])
    print(f"    Euler angles extracted: [roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}]")
    
    return position, euler_angles

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix (ZYX convention) - only for visualization axes"""
    # Convert to radians if needed
    if abs(roll) > 2*np.pi or abs(pitch) > 2*np.pi or abs(yaw) > 2*np.pi:
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
    else:
        roll_rad, pitch_rad, yaw_rad = roll, pitch, yaw
    
    # Roll (rotation around X-axis)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    # Pitch (rotation around Y-axis)
    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    # Yaw (rotation around Z-axis)
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix (ZYX order)
    rotation = R_z @ R_y @ R_x
    return rotation

def plot_coordinate_system(ax, position, euler_angles, scale=0.1, alpha=0.7):
    """Plot coordinate system axes at given position and Euler angles"""
    # Standard colors: X=red, Y=green, Z=blue
    colors = ['red', 'green', 'blue']
    
    # Convert Euler angles to rotation matrix for axis visualization
    rotation = euler_to_rotation_matrix(euler_angles[0], euler_angles[1], euler_angles[2])
    
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

def save_coordinates_csv(file_paths, output_path='camera_coordinates.csv'):
    """Save camera coordinates and Euler angles to CSV file"""
    print(f"Generating coordinates CSV: {output_path}")
    coordinate_data = []
    
    for file_idx, file_path in enumerate(file_paths):
        df = load_csv_file(file_path)
        if df is None:
            continue
            
        dataset_name = extract_dataset_name(file_path)
        cameras = parse_camera_parameters(df)
        
        for camera_name, camera_params in cameras.items():
            try:
                print(f"  Processing {dataset_name}/{camera_name}")
                position, euler_angles = extract_pose_data(camera_params)
                
                coordinate_data.append({
                    'Dataset': dataset_name,
                    'Camera': camera_name,
                    'X': f"{position[0]:.6f}",
                    'Y': f"{position[1]:.6f}",
                    'Z': f"{position[2]:.6f}",
                    'Roll_deg': f"{euler_angles[0]:.3f}",
                    'Pitch_deg': f"{euler_angles[1]:.3f}",
                    'Yaw_deg': f"{euler_angles[2]:.3f}",
                    'File_Path': file_path
                })
            except Exception as e:
                print(f"    ✗ Warning: Could not extract pose for camera {camera_name}: {e}")
                traceback.print_exc()
                continue
    
    # Write CSV
    if coordinate_data:
        import csv
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['Dataset', 'Camera', 'X', 'Y', 'Z', 'Roll_deg', 'Pitch_deg', 'Yaw_deg', 'File_Path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(coordinate_data)
        
        print(f"  ✓ Coordinates and Euler angles CSV saved: {output_path}")
        print(f"  ✓ Generated {len(coordinate_data)} coordinate entries")
    else:
        print("  ✗ No coordinate data generated!")
    
    return coordinate_data

def plot_camera_poses_matplotlib(file_paths, output_path='camera_poses.png'):
    """Plot all camera poses from multiple CSV files using matplotlib"""
    print(f"Generating static matplotlib visualization: {output_path}")
    
    try:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color palette for different files
        colors = plt.cm.Set1(np.linspace(0, 1, len(file_paths)))
        
        all_positions = []
        legend_elements = []
        total_cameras = 0
        
        for file_idx, file_path in enumerate(file_paths):
            print(f"  Processing file {file_idx + 1}/{len(file_paths)}: {file_path}")
            
            # Load CSV data
            df = load_csv_file(file_path)
            if df is None:
                continue
                
            # Extract dataset name
            dataset_name = extract_dataset_name(file_path)
            file_color = colors[file_idx]
            
            cameras = parse_camera_parameters(df)
            cameras_plotted = 0
            
            # Process each camera
            for camera_name, camera_params in cameras.items():
                try:
                    print(f"    Processing camera: {camera_name}")
                    position, euler_angles = extract_pose_data(camera_params)
                except Exception as e:
                    print(f"    ✗ Warning: Could not extract pose for camera {camera_name}: {e}")
                    continue
                    
                all_positions.append(position)
                
                # Plot coordinate system
                plot_coordinate_system(ax, position, euler_angles, scale=0.2, alpha=0.8)
                
                # Plot camera name with file color
                ax.text(position[0], position[1], position[2] + 0.1, 
                       camera_name, 
                       color=file_color, 
                       fontsize=10, 
                       fontweight='bold',
                       ha='center')
                
                cameras_plotted += 1
                total_cameras += 1
            
            # Add legend entry for this file
            if cameras_plotted > 0:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=file_color, markersize=10,
                                                label=f'{dataset_name} ({cameras_plotted} cameras)'))
                print(f"    ✓ Plotted {cameras_plotted} cameras")
        
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
            ax.set_title('3D Camera Poses Visualization\n(X=Red, Y=Green, Z=Blue)', fontsize=14, fontweight='bold')
            
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
            print(f"  ✓ Static PNG saved: {output_path}")
            print(f"  ✓ Plotted {total_cameras} total cameras")
            plt.show()
            
        else:
            print("  ✗ No positions to plot!")
            
    except Exception as e:
        print(f"  ✗ Error generating matplotlib plot: {e}")
        traceback.print_exc()

def create_interactive_plotly(file_paths, output_path='camera_poses.html'):
    """Create interactive 3D visualization using Plotly"""
    if not PLOTLY_AVAILABLE:
        print("  ⚠ Skipping interactive visualization (Plotly not available)")
        return
        
    print(f"Generating interactive plotly visualization: {output_path}")
    
    try:
        fig = go.Figure()
        
        # Color palette
        color_palette = px.colors.qualitative.Set1
        
        all_positions = []
        coordinate_data = []
        
        for file_idx, file_path in enumerate(file_paths):
            print(f"  Processing file {file_idx + 1}/{len(file_paths)}: {file_path}")
            df = load_csv_file(file_path)
            if df is None:
                continue
                
            dataset_name = extract_dataset_name(file_path)
            file_color = color_palette[file_idx % len(color_palette)]
            
            cameras = parse_camera_parameters(df)
            
            # Collect all positions and rotations for this file
            positions = []
            euler_angles_list = []
            hover_texts = []
            camera_names = []
            
            for camera_name, camera_params in cameras.items():
                try:
                    print(f"    Processing camera: {camera_name}")
                    position, euler_angles = extract_pose_data(camera_params)
                except Exception as e:
                    print(f"    ✗ Warning: Could not extract pose for camera {camera_name}: {e}")
                    continue
                    
                positions.append(position)
                euler_angles_list.append(euler_angles)
                camera_names.append(camera_name)
                all_positions.append(position)
                
                # Store coordinate data
                coordinate_data.append({
                    'file': dataset_name,
                    'camera': camera_name,
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],
                    'roll': euler_angles[0],
                    'pitch': euler_angles[1],
                    'yaw': euler_angles[2]
                })
                
                # Create hover text with coordinates and Euler angles
                hover_text = (f"Dataset: {dataset_name}<br>"
                             f"Camera: {camera_name}<br>"
                             f"X: {position[0]:.3f}<br>"
                             f"Y: {position[1]:.3f}<br>"
                             f"Z: {position[2]:.3f}<br>"
                             f"Roll: {euler_angles[0]:.1f}°<br>"
                             f"Pitch: {euler_angles[1]:.1f}°<br>"
                             f"Yaw: {euler_angles[2]:.1f}°")
                hover_texts.append(hover_text)
            
            if not positions:
                print(f"    ✗ No valid positions found for {dataset_name}")
                continue
                
            positions = np.array(positions)
            
            # Create legend group name for this dataset
            legend_group = f"group_{file_idx}"
            
            # Add camera positions as markers
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers+text',
                text=camera_names,
                textposition="top center",
                hovertext=hover_texts,
                hoverinfo='text',
                marker=dict(
                    size=8,
                    color=file_color,
                    symbol='circle'
                ),
                name=f'{dataset_name} ({len(positions)} cameras)',
                legendgroup=legend_group,
                showlegend=True
            ))
            
            # Collect all coordinate axes data for this dataset
            x_lines_x, x_lines_y, x_lines_z = [], [], []
            y_lines_x, y_lines_y, y_lines_z = [], [], []
            z_lines_x, z_lines_y, z_lines_z = [], [], []
            
            scale = 0.2
            for pos_idx, (position, euler_angles) in enumerate(zip(positions, euler_angles_list)):
                # Convert to rotation matrix for axis visualization
                rotation = euler_to_rotation_matrix(euler_angles[0], euler_angles[1], euler_angles[2])
                
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
                z=z_lines_z,
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
            
            print(f"    ✓ Added {len(positions)} cameras to interactive plot")
        
        # Create coordinate table HTML
        coord_table_html = create_coordinate_table_html(coordinate_data)
        
        # Update layout
        fig.update_layout(
            title='Interactive 3D Camera Poses Visualization<br><sub>X=Red, Y=Green, Z=Blue | Use mouse to rotate, zoom, and pan | Hover over cameras for coordinates and Euler angles</sub>',
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
            </style>
        </head>
        <body>
            <div class="container">
                <div class="plot-container">
                    {plot_html.split('<body>')[1].split('</body>')[0]}
                </div>
                <div class="coords-container">
                    <h2>Camera Coordinates and Euler Angles</h2>
                    <p><strong>Position columns (blue background):</strong> X, Y, Z coordinates in meters</p>
                    <p><strong>Euler angles (pink background):</strong> Roll, Pitch, Yaw angles in degrees</p>
                    {coord_table_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save as HTML
        with open(output_path, 'w') as f:
            f.write(full_html)
        
        print(f"  ✓ Interactive HTML with coordinates and rotations saved: {output_path}")
        
        # Show interactive plot
        fig.show()
        
    except Exception as e:
        print(f"  ✗ Error generating interactive plot: {e}")
        traceback.print_exc()

def create_coordinate_table_html(coordinate_data):
    """Create HTML table with coordinate and Euler angle information"""
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
    html += '<th>Camera</th>'
    html += '<th class="position-col">X (m)</th>'
    html += '<th class="position-col">Y (m)</th>'
    html += '<th class="position-col">Z (m)</th>'
    html += '<th class="rotation-col">Roll (°)</th>'
    html += '<th class="rotation-col">Pitch (°)</th>'
    html += '<th class="rotation-col">Yaw (°)</th>'
    html += '</tr></thead>'
    html += '<tbody>'
    
    for file_name, coords in files.items():
        # Sort by camera name
        coords.sort(key=lambda x: x['camera'])
        
        for coord in coords:
            html += f'<tr>'
            html += f'<td>{coord["file"]}</td>'
            html += f'<td>{coord["camera"]}</td>'
            html += f'<td class="position-col">{coord["x"]:.3f}</td>'
            html += f'<td class="position-col">{coord["y"]:.3f}</td>'
            html += f'<td class="position-col">{coord["z"]:.3f}</td>'
            html += f'<td class="rotation-col">{coord["roll"]:.1f}</td>'
            html += f'<td class="rotation-col">{coord["pitch"]:.1f}</td>'
            html += f'<td class="rotation-col">{coord["yaw"]:.1f}</td>'
            html += f'</tr>'
    
    html += '</tbody></table>'
    return html

def write_ply_file(file_paths, output_path='camera_poses.ply'):
    """Write camera poses to PLY file format for 3D viewing"""
    print(f"Generating PLY file: {output_path}")
    
    try:
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
        total_cameras = 0
        
        for file_idx, file_path in enumerate(file_paths):
            print(f"  Processing file {file_idx + 1}/{len(file_paths)}: {file_path}")
            df = load_csv_file(file_path)
            if df is None:
                continue
                
            cameras = parse_camera_parameters(df)
            file_color = color_palette[file_idx % len(color_palette)]
            
            # Process each camera
            for camera_name, camera_params in cameras.items():
                try:
                    print(f"    Processing camera: {camera_name}")
                    position, euler_angles = extract_pose_data(camera_params)
                    rotation = euler_to_rotation_matrix(euler_angles[0], euler_angles[1], euler_angles[2])
                except Exception as e:
                    print(f"    ✗ Warning: Could not extract pose for camera {camera_name}: {e}")
                    continue
                
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
                total_cameras += 1
        
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
        
        print(f"  ✓ PLY file saved: {output_path}")
        print(f"  ✓ Processed {total_cameras} cameras")
        
    except Exception as e:
        print(f"  ✗ Error generating PLY file: {e}")
        traceback.print_exc()

def print_csv_parameter_summary(file_paths):
    """Print available parameters in CSV files to help with debugging"""
    print("\n" + "="*120)
    print("CSV PARAMETER ANALYSIS")
    print("="*120)
    
    for file_path in file_paths:
        df = load_csv_file(file_path)
        if df is None:
            continue
            
        dataset_name = extract_dataset_name(file_path)
        print(f"\nDataset: {dataset_name}")
        print("-" * 80)
        
        # Show available parameters
        unique_params = df['Parameter'].unique()
        print(f"Available parameters: {', '.join(unique_params)}")
        
        # Show cameras
        unique_cameras = df['Camera'].unique()
        print(f"Cameras ({len(unique_cameras)}): {', '.join(unique_cameras)}")
        
        # Show a sample of the data
        print("\nSample data:")
        print(df.head(10).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D camera poses from CSV parameter files')
    parser.add_argument('files', nargs='+', help='Paths to CSV parameter files')
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
    parser.add_argument('--show-params', action='store_true',
                       help='Show available parameters in CSV files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CSV Camera Parameters 3D Visualizer (Fixed Version)")
    print("="*60)
    
    # Validate input files
    valid_files = []
    for file_path in args.files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
            print(f"✓ Found file: {file_path}")
        else:
            print(f"✗ Warning: File not found: {file_path}")
    
    if not valid_files:
        print("✗ Error: No valid input files found!")
        sys.exit(1)
    
    print(f"\n📊 Processing {len(valid_files)} CSV file(s)...")
    
    # Show parameter analysis if requested
    if args.show_params:
        print_csv_parameter_summary(valid_files)
        return
    
    # Debug: Always show parameter summary first if verbose
    if args.verbose:
        print_csv_parameter_summary(valid_files)
    
    # Generate coordinate data first
    if not args.no_csv:
        print(f"\n📄 Generating coordinates and Euler angles CSV...")
        try:
            coordinate_data = save_coordinates_csv(valid_files, f"{args.output}_coordinates.csv")
            if coordinate_data:
                print(f"✓ Successfully generated {len(coordinate_data)} coordinate entries")
            else:
                print("✗ No coordinate data generated!")
        except Exception as e:
            print(f"✗ Error generating CSV: {e}")
            traceback.print_exc()
    
    # Generate visualizations
    if not args.no_static:
        print(f"\n🎨 Generating static PNG...")
        try:
            plot_camera_poses_matplotlib(valid_files, f"{args.output}.png")
        except Exception as e:
            print(f"✗ Error generating static plot: {e}")
            traceback.print_exc()
    
    if not args.no_interactive:
        print(f"\n🌐 Generating interactive 3D visualization with coordinate and Euler angle table...")
        try:
            create_interactive_plotly(valid_files, f"{args.output}.html")
        except Exception as e:
            print(f"✗ Error generating interactive plot: {e}")
            traceback.print_exc()
    
    if not args.no_ply:
        print(f"\n🔧 Generating PLY 3D file...")
        try:
            write_ply_file(valid_files, f"{args.output}.ply")
        except Exception as e:
            print(f"✗ Error generating PLY file: {e}")
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎉 Visualization complete!")
    print("="*60)
    print(f"Files generated:")
    if not args.no_csv:
        print(f"  📄 Coordinates and Euler angles CSV: {args.output}_coordinates.csv")
    if not args.no_static:
        print(f"  🎨 Static PNG: {args.output}.png")
    if not args.no_interactive and PLOTLY_AVAILABLE:
        print(f"  🌐 Interactive HTML with coordinates and Euler angles: {args.output}.html")
    if not args.no_ply:
        print(f"  🔧 3D PLY file: {args.output}.ply")
    
    if not args.no_interactive and PLOTLY_AVAILABLE:
        print(f"\n💡 To view the interactive 3D visualization, open {args.output}.html in your web browser.")
    if not args.no_ply:
        print(f"💡 To view the PLY file, use software like MeshLab, Blender, or CloudCompare.")

if __name__ == "__main__":
    main()