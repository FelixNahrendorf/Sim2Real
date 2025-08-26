#!/usr/bin/env python3
"""
Dual Dataset 3D Camera Visualizer
Compares camera positions and orientations between two datasets.
Enhanced with distinct axis styling and legend toggling.
"""

import numpy as np
import plotly.graph_objects as go

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix (ZYX convention)"""
    # Convert to radians if in degrees
    if abs(roll) > 2*np.pi or abs(pitch) > 2*np.pi or abs(yaw) > 2*np.pi:
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
    else:
        roll_rad, pitch_rad, yaw_rad = roll, pitch, yaw
    
    # Rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def create_dual_camera_visualization():
    """Create 3D visualization comparing two camera datasets"""
    
    # Dataset 1: nerf-to-euler #from csv
    nerf_data = {
        "coordinates": [
            [1.711546582487506, -0.00925438364241527, 1.5028675472910469],
            [1.5675328085130118, 0.49701484070991175, 1.5078454488957296],
            [1.5519070054468824, -0.4973630153835918, 1.5085220895456473],
            [0.04365895519582518, -0.007168419741461977, 1.5724962385613648],
            [1.042854011200353, -0.4830985561936272, 1.5758664294360236],
            [1.0390594047528943, 0.47302748339405754, 1.5566498046927528]
        ],
        "rolls": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
        ],
        "pitchs": [
        0.00604207372587012,
        0.012779727654633784,
        0.0027526839660454543,
        -0.004151409897312412,
        -0.0005133123938878268,
        0.006093952089494375
        ],
        "yaws": [
        1.562861457405587,
        2.5654127004916685,
        0.6067195899058673,
        -1.5644901568733787,
        -0.3245552430097445,
        -2.7626045865768387
        ],
        "fov": [64.5614792846793, 64.8772037435167, 64.6420610387791, 89.82514491401436, 64.99646878534186, 65.05495034029899]
    }
    
    # Dataset 2: nuscenes_adjusted
    nuscenes_data = {
        "coordinates": [
            [1.71154658, -0.00925438, 1.50286755],
            [1.56753281, 0.49701484, 1.50784545],
            [1.55190701, -0.49736302, 1.50852209],
            [0.05634104, -0.00716842, 1.57249624],
            [1.04285401, -0.48309856, 1.57586643],
            [1.0390594, 0.47302748, 1.5566498]
        ],
        "rolls": [0, 0, 0, 0, 0, 0],  # Assuming 0 roll for nuscenes
        "pitchs": [
            -0.002184605901413669, -0.00908179325149176, -0.003702805724416052,
            0.010903921170017539, 0.0007322564726739024, -0.005943942229067065
        ],
        "yaws": [
            4.720323669231367, 3.717790651712274, 5.676476373000298,
            1.5644901720670188, 0.32445679467339517, 2.762686748075455
        ],
        "fov": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    }
    
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Create figure
    fig = go.Figure()
    
    # Process both datasets with distinct styling
    datasets = [
        ("NeRF-to-Euler", nerf_data, "circle", ["#FF4444", "#CC0000"], 
         {"width": 4}, {"red": "#FF0000", "green": "#00AA00", "blue": "#0066FF"}),
        ("NuScenes Adjusted", nuscenes_data, "diamond", ["#4444FF", "#0000CC"], 
         {"width": 3, "dash": "dot"}, {"red": "#CC3333", "green": "#33AA33", "blue": "#3366CC"})
    ]
    
    scale = 0.3
    
    for dataset_name, data, symbol, colors, line_style, axis_colors in datasets:
        positions = np.array(data["coordinates"])
        rolls = data["rolls"]
        pitchs = data["pitchs"]
        yaws = data["yaws"]
        fovs = data["fov"]
        
        # Create hover text
        hover_texts = []
        for i, (pos, roll, pitch, yaw, fov, name) in enumerate(zip(positions, rolls, pitchs, yaws, fovs, camera_names)):
            # Convert angles to degrees for display
            roll_deg = np.degrees(roll) if abs(roll) <= 2*np.pi else roll
            pitch_deg = np.degrees(pitch) if abs(pitch) <= 2*np.pi else pitch
            yaw_deg = np.degrees(yaw) if abs(yaw) <= 2*np.pi else yaw
            
            hover_text = (f"Dataset: {dataset_name}<br>"
                         f"Camera: {name}<br>"
                         f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]<br>"
                         f"Roll: {roll_deg:.1f}°<br>"
                         f"Pitch: {pitch_deg:.1f}°<br>"
                         f"Yaw: {yaw_deg:.1f}°<br>"
                         f"FOV: {fov}°")
            hover_texts.append(hover_text)
        
        # Add camera positions with legendgroup for toggling
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
                size=12,
                color=colors[0],
                symbol=symbol,
                line=dict(width=2, color=colors[1])
            ),
            name=f'{dataset_name}',
            legendgroup=dataset_name,
            legendgrouptitle_text=dataset_name,
            showlegend=True
        ))
        
        # Collect axis lines for this dataset
        x_lines_x, x_lines_y, x_lines_z = [], [], []
        y_lines_x, y_lines_y, y_lines_z = [], [], []
        z_lines_x, z_lines_y, z_lines_z = [], [], []
        
        for i, (pos, roll, pitch, yaw) in enumerate(zip(positions, rolls, pitchs, yaws)):
            # Get rotation matrix
            rotation = euler_to_rotation_matrix(roll, pitch, yaw)
            
            # X axis
            x_end = pos + rotation[:, 0] * scale
            x_lines_x.extend([pos[0], x_end[0], None])
            x_lines_y.extend([pos[1], x_end[1], None])
            x_lines_z.extend([pos[2], x_end[2], None])
            
            # Y axis
            y_end = pos + rotation[:, 1] * scale
            y_lines_x.extend([pos[0], y_end[0], None])
            y_lines_y.extend([pos[1], y_end[1], None])
            y_lines_z.extend([pos[2], y_end[2], None])
            
            # Z axis
            z_end = pos + rotation[:, 2] * scale
            z_lines_x.extend([pos[0], z_end[0], None])
            z_lines_y.extend([pos[1], z_end[1], None])
            z_lines_z.extend([pos[2], z_end[2], None])
        
        # Add coordinate axes with dataset-specific styling and legendgroup
        # X axes
        fig.add_trace(go.Scatter3d(
            x=x_lines_x, y=x_lines_y, z=x_lines_z,
            mode='lines',
            line=dict(color=axis_colors["red"], **line_style),
            name=f'X-axes',
            legendgroup=dataset_name,
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Y axes
        fig.add_trace(go.Scatter3d(
            x=y_lines_x, y=y_lines_y, z=y_lines_z,
            mode='lines',
            line=dict(color=axis_colors["green"], **line_style),
            name=f'Y-axes',
            legendgroup=dataset_name,
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Z axes
        fig.add_trace(go.Scatter3d(
            x=z_lines_x, y=z_lines_y, z=z_lines_z,
            mode='lines',
            line=dict(color=axis_colors["blue"], **line_style),
            name=f'Z-axes',
            legendgroup=dataset_name,
            showlegend=True,
            hoverinfo='skip'
        ))
    

    
    # Calculate position differences for analysis
    nerf_positions = np.array(nerf_data["coordinates"])
    nuscenes_positions = np.array(nuscenes_data["coordinates"])
    position_diffs = np.linalg.norm(nerf_positions - nuscenes_positions, axis=1)
    avg_position_diff = np.mean(position_diffs)
    max_position_diff = np.max(position_diffs)
    
    # Update layout
    fig.update_layout(
        title=f'Dual Dataset 3D Camera Poses Comparison<br>'
              f'<sub>NeRF-to-Euler (circles, solid) vs NuScenes Adjusted (diamonds, dotted)<br>'
              f'Avg Position Difference: {avg_position_diff:.3f}m | Max: {max_position_diff:.3f}m<br>'
              f'<i>Click on dataset names in legend to toggle visibility</i></sub>',
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            zaxis=dict(showgrid=True, gridcolor='lightgray')
        ),
        width=1400,
        height=900,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            groupclick="toggleitem"  # Enable clicking on legend groups
        )
    )
    
    return fig, avg_position_diff, max_position_diff, position_diffs

def create_comparison_table(position_diffs):
    """Create HTML table comparing the datasets"""
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    table_html = '''
    <div class="comparison-table">
        <h3>Position Differences Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>Camera</th>
                    <th>Position Difference (m)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for i, (name, diff) in enumerate(zip(camera_names, position_diffs)):
        status = "Very Close" if diff < 0.001 else "Close" if diff < 0.01 else "Different"
        status_color = "#28a745" if diff < 0.001 else "#ffc107" if diff < 0.01 else "#dc3545"
        
        table_html += f'''
                <tr>
                    <td>{name}</td>
                    <td>{diff:.6f}</td>
                    <td style="color: {status_color}; font-weight: bold;">{status}</td>
                </tr>
        '''
    
    table_html += '''
            </tbody>
        </table>
    </div>
    '''
    
    return table_html

def generate_html_output(output_file='dual_camera_visualization.html'):
    """Generate HTML file with dual dataset 3D camera visualization"""
    
    # Create the visualization
    fig, avg_diff, max_diff, position_diffs = create_dual_camera_visualization()
    
    # Create comparison table
    comparison_table = create_comparison_table(position_diffs)
    
    # Generate HTML
    plot_html = fig.to_html(
        include_plotlyjs=True,
        div_id="camera-plot",
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
    )
    
    # Enhanced HTML with styling and comparison table
    enhanced_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dual Dataset 3D Camera Poses Comparison</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                text-align: center;
                margin-bottom: 20px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .plot-container {{
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 10px;
                margin-bottom: 20px;
            }}
            .comparison-table {{
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
            }}
            .comparison-table table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            .comparison-table th, .comparison-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .comparison-table th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            .comparison-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            h1 {{
                color: #333;
                margin: 0;
            }}
            h3 {{
                color: #333;
                margin-top: 0;
            }}
            .info {{
                color: #666;
                margin-top: 10px;
            }}
            .stats {{
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
            }}
            .legend-info {{
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Dual Dataset 3D Camera Poses Comparison</h1>
            <div class="info">
                Interactive comparison between NeRF-to-Euler and NuScenes Adjusted camera datasets<br>
                Use mouse to rotate, zoom, and pan | Hover over cameras for detailed information
            </div>
            <div class="legend-info">
                <strong>Interactive Legend:</strong> Click on dataset names in the legend to toggle visibility of cameras and their coordinate axes
            </div>
            <div class="stats">
                <strong>Quick Stats:</strong><br>
                Average Position Difference: {avg_diff:.6f} meters<br>
                Maximum Position Difference: {max_diff:.6f} meters<br>
                Number of Cameras: {len(position_diffs)}<br>
                <strong>Visual Differences:</strong><br>
                • NeRF-to-Euler: Circles with solid axes (brighter colors)<br>
                • NuScenes Adjusted: Diamonds with dotted axes (muted colors)
            </div>
        </div>
        <div class="plot-container">
            {plot_html.split('<body>')[1].split('</body>')[0]}
        </div>
        {comparison_table}
    </body>
    </html>
    """
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(enhanced_html)
    
    print(f"✓ Dual dataset HTML visualization saved to: {output_file}")
    print(f"📊 Average position difference: {avg_diff:.6f} meters")
    print(f"📊 Maximum position difference: {max_diff:.6f} meters")
    print("🎯 Features added:")
    print("   • Distinct axis colors for each dataset")
    print("   • Different line styles (solid vs dotted)")
    print("   • Clickable legend groups for toggling visibility")
    print("   • Enhanced visual differentiation")
    
    return enhanced_html

if __name__ == "__main__":
    # Generate the HTML output
    html_output = generate_html_output('dual_camera_visualization.html')
    print("Dual dataset 3D Camera visualization complete!")
    print("Open 'dual_camera_visualization.html' in your web browser to view the interactive comparison.")
    print("Click on 'NeRF-to-Euler Cameras' or 'NuScenes Adjusted Cameras' in the legend to toggle visibility!")