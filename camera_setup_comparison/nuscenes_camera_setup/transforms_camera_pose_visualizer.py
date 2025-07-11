#!/usr/bin/env python3
"""
3D Sensor Pose Visualizer

This script visualizes sensor poses from multiple transform JSON files in 3D space.
Each file's sensors are displayed with a unique color, and coordinate systems show
consistent x (red), y (green), z (blue) axes.

Outputs:
- Interactive 3D plot (rotatable)
- Static PNG image
- PLY 3D file for external viewing
- HTML file with interactive 3D visualization

Usage:
    python visualize_poses.py <path1> <path2> ... <pathN>

Example:
    python visualize_poses.py /app/code/seed4d/data_nuscenes_adjusted/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle/nuscenes_invisible/transforms/transforms_ego.json
"""

import json
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
    """Extract dataset name from file path (string after 'seed4d')"""
    path_parts = Path(file_path).parts
    try:
        seed4d_idx = path_parts.index('seed4d')
        if seed4d_idx + 1 < len(path_parts):
            return path_parts[seed4d_idx + 1]
    except ValueError:
        pass
    # Fallback to filename if seed4d not found
    return Path(file_path).stem

def load_transform_file(file_path):
    """Load and parse a transforms JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def transform_matrix_to_pose(transform_matrix):
    """Extract position and rotation from 4x4 transform matrix"""
    transform = np.array(transform_matrix)
    position = transform[:3, 3]
    rotation = transform[:3, :3]
    return position, rotation

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

def write_ply_file(file_paths, output_path='sensor_poses.ply'):
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
    
    for file_idx, file_path in enumerate(file_paths):
        data = load_transform_file(file_path)
        if data is None:
            continue
            
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

def create_interactive_plotly(file_paths, output_path='sensor_poses.html'):
    """Create interactive 3D visualization using Plotly"""
    fig = go.Figure()
    
    # Color palette
    color_palette = px.colors.qualitative.Set1
    
    all_positions = []
    coordinate_data = []
    
    for file_idx, file_path in enumerate(file_paths):
        data = load_transform_file(file_path)
        if data is None:
            continue
            
        dataset_name = extract_dataset_name(file_path)
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
            positions.append(position)
            rotations.append(rotation)
            all_positions.append(position)
            
            # Store coordinate data
            coordinate_data.append({
                'file': dataset_name,
                'sensor': frame_idx,
                'x': position[0],
                'y': position[1],
                'z': position[2]
            })
            
            # Create hover text with coordinates
            hover_text = f"Dataset: {dataset_name}<br>Sensor: {frame_idx}<br>X: {position[0]:.3f}<br>Y: {position[1]:.3f}<br>Z: {position[2]:.3f}"
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
        title='Interactive 3D Sensor Poses Visualization<br><sub>X=Red, Y=Green, Z=Blue | Use mouse to rotate, zoom, and pan | Hover over sensors for coordinates</sub>',
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="plot-container">
                {plot_html.split('<body>')[1].split('</body>')[0]}
            </div>
            <div class="coords-container">
                <h2>Sensor Coordinates</h2>
                {coord_table_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save as HTML
    with open(output_path, 'w') as f:
        f.write(full_html)
    
    print(f"Interactive HTML with coordinates saved to: {output_path}")
    
    # Show interactive plot
    fig.show()

def create_coordinate_table_html(coordinate_data):
    """Create HTML table with coordinate information"""
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
    html += '<thead><tr><th>Dataset</th><th>Sensor</th><th>X (m)</th><th>Y (m)</th><th>Z (m)</th></tr></thead>'
    html += '<tbody>'
    
    for file_name, coords in files.items():
        # Sort by sensor number
        coords.sort(key=lambda x: x['sensor'])
        
        for coord in coords:
            html += f'<tr>'
            html += f'<td>{coord["file"]}</td>'
            html += f'<td>{coord["sensor"]}</td>'
            html += f'<td>{coord["x"]:.3f}</td>'
            html += f'<td>{coord["y"]:.3f}</td>'
            html += f'<td>{coord["z"]:.3f}</td>'
            html += f'</tr>'
    
    html += '</tbody></table>'
    return html

def save_coordinates_csv(file_paths, output_path='sensor_coordinates.csv'):
    """Save sensor coordinates to CSV file"""
    coordinate_data = []
    
    for file_idx, file_path in enumerate(file_paths):
        data = load_transform_file(file_path)
        if data is None:
            continue
            
        dataset_name = extract_dataset_name(file_path)
        
        for frame_idx, frame in enumerate(data.get('frames', [])):
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix is None:
                continue
                
            position, rotation = transform_matrix_to_pose(transform_matrix)
            
            coordinate_data.append({
                'Dataset': dataset_name,
                'Sensor': frame_idx,
                'X': f"{position[0]:.6f}",
                'Y': f"{position[1]:.6f}",
                'Z': f"{position[2]:.6f}",
                'File_Path': file_path
            })
    
    # Write CSV
    if coordinate_data:
        import csv
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['Dataset', 'Sensor', 'X', 'Y', 'Z', 'File_Path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(coordinate_data)
        
        print(f"Coordinates CSV saved to: {output_path}")
    
    return coordinate_data

def plot_sensor_poses_matplotlib(file_paths, output_path='sensor_poses.png'):
    """Plot all sensor poses from multiple files using matplotlib"""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color palette for different files
    colors = plt.cm.Set1(np.linspace(0, 1, len(file_paths)))
    
    all_positions = []
    legend_elements = []
    
    for file_idx, file_path in enumerate(file_paths):
        print(f"Processing file {file_idx + 1}/{len(file_paths)}: {file_path}")
        
        # Load transform data
        data = load_transform_file(file_path)
        if data is None:
            continue
            
        # Extract dataset name
        dataset_name = extract_dataset_name(file_path)
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

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D sensor poses from transform files')
    parser.add_argument('files', nargs='+', help='Paths to transform JSON files')
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
        print("\nGenerating coordinates CSV...")
        coordinate_data = save_coordinates_csv(valid_files, f"{args.output}_coordinates.csv")
    
    # Generate visualizations
    if not args.no_interactive:
        print("\nGenerating interactive 3D visualization with coordinate table...")
        create_interactive_plotly(valid_files, f"{args.output}.html")
    
    if not args.no_static:
        print("\nGenerating static PNG...")
        plot_sensor_poses_matplotlib(valid_files, f"{args.output}.png")
    
    if not args.no_ply:
        print("\nGenerating PLY 3D file...")
        write_ply_file(valid_files, f"{args.output}.ply")
    
    print("\nVisualization complete!")
    print(f"Files generated:")
    if not args.no_interactive:
        print(f"  - Interactive HTML with coordinates: {args.output}.html")
    if not args.no_static:
        print(f"  - Static PNG: {args.output}.png")
    if not args.no_ply:
        print(f"  - 3D PLY file: {args.output}.ply")
    if not args.no_csv:
        print(f"  - Coordinates CSV: {args.output}_coordinates.csv")
    
    if not args.no_interactive:
        print(f"\nTo view the interactive 3D visualization with coordinate table, open {args.output}.html in your web browser.")
    if not args.no_ply:
        print(f"To view the PLY file, use software like MeshLab, Blender, or CloudCompare.")
    if not args.no_csv:
        print(f"The CSV file contains all sensor coordinates for data analysis.")
        
    # Print coordinate summary to console
    print(f"\n" + "="*60)
    print("COORDINATE SUMMARY")
    print("="*60)
    
    for file_idx, file_path in enumerate(valid_files):
        data = load_transform_file(file_path)
        if data is None:
            continue
            
        dataset_name = extract_dataset_name(file_path)
        print(f"\nDataset: {dataset_name}")
        print("-" * 40)
        
        for frame_idx, frame in enumerate(data.get('frames', [])):
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix is None:
                continue
                
            position, _ = transform_matrix_to_pose(transform_matrix)
            print(f"Sensor {frame_idx:2d}: X={position[0]:8.3f}, Y={position[1]:8.3f}, Z={position[2]:8.3f}")
    
    print("="*60)

if __name__ == "__main__":
    main()