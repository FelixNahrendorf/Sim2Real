#!/usr/bin/env python3
"""
Simple 3D Camera Visualizer
Generates HTML visualization of camera positions and orientations.
"""

import numpy as np
import plotly.graph_objects as go

def euler_to_rotation_matrix(pitch, yaw, roll=0):
    """Convert Euler angles to rotation matrix (ZYX convention)"""
    # Convert to radians
    pitch_rad = np.radians(pitch) if abs(pitch) > 2*np.pi else pitch
    yaw_rad = np.radians(yaw) if abs(yaw) > 2*np.pi else yaw
    roll_rad = np.radians(roll) if abs(roll) > 2*np.pi else roll
    
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

def create_camera_visualization():
    """Create 3D visualization of camera poses"""
    
    # Camera data
    coordinates = [
        [0.81154658, -0.00925438, 1.50286755],
        [0.66753281, 0.49701484, 1.50784545],
        [0.65190701, -0.49736302, 1.50852209],
        [-0.85634104, -0.00716842, 1.57249624],
        [0.14285401, -0.48309856, 1.57586643],
        [0.1390594, 0.47302748, 1.5566498]
    ]
    
    pitchs = [
        -0.002184605901413669,
        -0.00908179325149176,
        -0.003702805724416052,
        0.010903921170017539,
        0.0007322564726739024,
        -0.005943942229067065
    ]
    
    yaws = [
        4.720323669231367,
        3.717790651712274,
        5.676476373000298,
        1.5644901720670188,
        0.32445679467339517,
        2.762686748075455
    ]
    
    fovs = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Convert to numpy arrays
    positions = np.array(coordinates)
    
    # Create figure
    fig = go.Figure()
    
    # Colors for different cameras
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    
    # Camera positions as markers
    hover_texts = []
    for i, (pos, pitch, yaw, fov, name) in enumerate(zip(positions, pitchs, yaws, fovs, camera_names)):
        hover_text = (f"Camera: {name}<br>"
                     f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]<br>"
                     f"Pitch: {np.degrees(pitch):.1f}°<br>"
                     f"Yaw: {np.degrees(yaw):.1f}°<br>"
                     f"FOV: {fov}°")
        hover_texts.append(hover_text)
    
    # Add camera positions
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
            size=10,
            color=colors[:len(positions)],
            symbol='circle'
        ),
        name='Cameras',
        showlegend=True
    ))
    
    # Add coordinate axes for each camera
    scale = 0.3
    
    # Collect all axis lines
    x_lines_x, x_lines_y, x_lines_z = [], [], []
    y_lines_x, y_lines_y, y_lines_z = [], [], []
    z_lines_x, z_lines_y, z_lines_z = [], [], []
    
    for i, (pos, pitch, yaw) in enumerate(zip(positions, pitchs, yaws)):
        # Get rotation matrix
        rotation = euler_to_rotation_matrix(pitch, yaw, 0)
        
        # X axis (red)
        x_end = pos + rotation[:, 0] * scale
        x_lines_x.extend([pos[0], x_end[0], None])
        x_lines_y.extend([pos[1], x_end[1], None])
        x_lines_z.extend([pos[2], x_end[2], None])
        
        # Y axis (green)
        y_end = pos + rotation[:, 1] * scale
        y_lines_x.extend([pos[0], y_end[0], None])
        y_lines_y.extend([pos[1], y_end[1], None])
        y_lines_z.extend([pos[2], y_end[2], None])
        
        # Z axis (blue)
        z_end = pos + rotation[:, 2] * scale
        z_lines_x.extend([pos[0], z_end[0], None])
        z_lines_y.extend([pos[1], z_end[1], None])
        z_lines_z.extend([pos[2], z_end[2], None])
    
    # Add X axes
    fig.add_trace(go.Scatter3d(
        x=x_lines_x, y=x_lines_y, z=z_lines_z,
        mode='lines',
        line=dict(color='red', width=6),
        name='X-axes',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add Y axes
    fig.add_trace(go.Scatter3d(
        x=y_lines_x, y=y_lines_y, z=y_lines_z,
        mode='lines',
        line=dict(color='green', width=6),
        name='Y-axes',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add Z axes
    fig.add_trace(go.Scatter3d(
        x=z_lines_x, y=z_lines_y, z=z_lines_z,
        mode='lines',
        line=dict(color='blue', width=6),
        name='Z-axes',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title='3D Camera Poses Visualization<br><sub>X=Red, Y=Green, Z=Blue | Hover over cameras for details</sub>',
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
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig

def generate_html_output(output_file='camera_visualization.html'):
    """Generate HTML file with 3D camera visualization"""
    
    # Create the visualization
    fig = create_camera_visualization()
    
    # Generate HTML
    html_content = fig.to_html(
        include_plotlyjs=True,
        div_id="camera-plot",
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
    )
    
    # Enhanced HTML with styling
    enhanced_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Camera Poses Visualization</title>
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
            }}
            h1 {{
                color: #333;
                margin: 0;
            }}
            .info {{
                color: #666;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>3D Camera Poses Visualization</h1>
            <div class="info">
                Interactive 3D visualization of camera positions and orientations<br>
                Use mouse to rotate, zoom, and pan | Hover over cameras for details
            </div>
        </div>
        <div class="plot-container">
            {html_content.split('<body>')[1].split('</body>')[0]}
        </div>
    </body>
    </html>
    """
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(enhanced_html)
    
    print(f"✓ HTML visualization saved to: {output_file}")
    return enhanced_html

if __name__ == "__main__":
    # Generate the HTML output
    html_output = generate_html_output('camera_visualization.html')
    print("3D Camera visualization complete!")
    print("Open 'camera_visualization.html' in your web browser to view the interactive 3D plot.")
