#!/usr/bin/env python3
"""
Triple Dataset 3D Camera Visualizer with FOV Circle Segments - FIXED UNIFORM COLORS & LEGEND CONTROL
Compares camera positions and orientations between three datasets.
Enhanced with FOV circle segments drawn in x-y plane facing minus y direction.
FIXED: Uniform colors AND proper legend toggle control for FOV segments.
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

def create_camera_triangle(position, rotation_matrix, length=1.0):
    """
    Create triangle marker with tip at camera origin and base towards minus y direction
    
    Args:
        position: Camera position [x, y, z] (will be the tip of the triangle)
        rotation_matrix: Camera rotation matrix
        length: Length of the triangle (1.0m)
    
    Returns:
        x, y, z coordinates for the triangle vertices
    """
    # Create triangle in local coordinate system with tip at origin
    # Base is towards minus y direction, base width is 1.5 times the length
    base_width = length * 1.5  # Base is 1.5 times longer than the length
    half_width = base_width * 0.5
    
    # Triangle vertices in local coordinates (tip at origin, base towards minus y)
    local_vertices = np.array([
        [0, 0, 0],                     # Tip at origin (camera position)
        [-half_width, -length, 0],     # Left base vertex (1.5x wider)
        [half_width, -length, 0],      # Right base vertex (1.5x wider)
        [0, 0, 0]                      # Close the triangle back to tip
    ])
    
    # Transform to world coordinates
    world_vertices = []
    for vertex in local_vertices:
        world_vertex = position + rotation_matrix @ vertex
        world_vertices.append(world_vertex)
    
    world_vertices = np.array(world_vertices)
    return world_vertices[:, 0], world_vertices[:, 1], world_vertices[:, 2]

def create_fov_segment(position, rotation_matrix, fov_degrees, radius=10.0, num_points=100):
    """
    Create FOV circle segment in x-y plane facing minus y direction
    Creates a full circle first, then extracts only the FOV segment with uniform color
    
    Args:
        position: Camera position [x, y, z]
        rotation_matrix: Camera rotation matrix
        fov_degrees: Field of view in degrees
        radius: Radius of the FOV visualization
        num_points: Number of points to create the full circle
    
    Returns:
        x, y, z coordinates for the FOV segment
    """
    # Convert FOV to radians
    fov_rad = np.radians(fov_degrees)
    half_fov = fov_rad / 2
    
    # Create FOV segment centered on minus y direction
    # In camera local coordinates, minus y direction is at angle = 0
    # We want the FOV segment to be symmetric around this direction
    
    # Calculate number of points for the FOV segment
    segment_points_count = max(10, int(num_points * fov_rad / (2*np.pi)))
    
    # Create angles for the FOV segment, centered on minus y (angle = 0)
    angles = np.linspace(-half_fov, half_fov, segment_points_count)
    
    # Create segment points in local camera coordinate system
    # Start with center point
    local_points = [[0, 0, 0]]  # Center point at camera origin
    
    # Add arc points, facing minus y direction
    for angle in angles:
        # In our coordinate system: minus y is at angle=0, x is left/right
        x = radius * np.sin(angle)
        y = -radius * np.cos(angle)  # Minus y direction
        z = 0
        local_points.append([x, y, z])
    
    # Convert to numpy array
    local_points = np.array(local_points)
    
    # Transform points to world coordinates using camera rotation and position
    world_points = []
    for point in local_points:
        # Apply rotation and translation
        world_point = position + rotation_matrix @ point
        world_points.append(world_point)
    
    world_points = np.array(world_points)
    
    return world_points[:, 0], world_points[:, 1], world_points[:, 2]

def create_triple_camera_visualization():
    """Create 3D visualization comparing three camera datasets with FOV segments"""
    
    # Dataset 1: NuScenes_native_to_euler
    nerf_data = {
        "coordinates": [
            [1.711546582487506, -0.00925438364241527, 1.5028675472910469],
            [1.5675328085130118, 0.49701484070991175, 1.5028675472910469],#1.5078454488957296], # modified for visualization
            [1.5519070054468824, -0.4973630153835918, 1.5028675472910469],#1.5085220895456473],
            [0.04365895519582518, -0.007168419741461977, 1.5028675472910469],#1.5724962385613648],
            [1.042854011200353, -0.4830985561936272, 1.5028675472910469],#1.5758664294360236],
            [1.0390594047528943, 0.47302748339405754, 1.5028675472910469]#1.5566498046927528]
        ],
        "rolls": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "pitchs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #[
        #    0.00604207372587012, 0.012779727654633784, 0.0027526839660454543, # modified for visualization
        #    -0.004151409897312412, -0.0005133123938878268, 0.006093952089494375
        #],
        "yaws": [
            1.562861457405587, 2.5654127004916685, 0.6067195899058673,
            -1.5644901568733787, -0.3245552430097445, -2.7626045865768387
        ],
        "fov": [64.5614792846793, 64.8772037435167, 64.6420610387791, 89.82514491401436, 64.99646878534186, 65.05495034029899]
    }
    
    # Dataset 2: NuScenes Old corrected
    nuscenes_data = {
        "coordinates": [
            [1.72200568, -0.00475453, 1.5028675472910469],#1.49491292],
            [1.58082566, 0.49907871, 1.5028675472910469],#1.51749368],
            [1.57525595, -0.50051938, 1.5028675472910469],#1.50696033],
            [0.05524611, -0.01078824, 1.5028675472910469],#1.56794287],
            [1.04852048, -0.48305813, 1.5028675472910469],#1.56210154],
            [1.05945173, 0.46720295, 1.5028675472910469]#1.55050858]
        ],
        "rolls": [0, 0, 0, 0, 0, 0],
        "pitchs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],# [
        #    -0.012229824837884262, -0.015832437375127517, -0.004017716027188145,
        #    0.011202230645773747, -0.002471878830597074, -0.001979743246308452
        #],
        "yaws": [ 
            1.560535171601681, 2.5756537342389154, 0.6049110566823916,
            -1.5615802076498657, -0.3242761715556668, -2.7485237221188865
        ],
        "fov": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    }
    
    # Dataset 3: NuScenes_old
    third_data = {
        "coordinates": [
            [1.72200568, -0.00475453, 1.49491292],
            [1.58082566, 0.49907871, 1.51749368],
            [1.57525595, -0.50051938, 1.50696033],
            [0.05524611, -0.01078824, 1.56794287],
            [1.04852048, -0.48305813, 1.56210154],
            [1.05945173, 0.46720295, 1.55050858]
        ],
        "rolls": [0, 0, 0, 0, 0, 0],
        "pitchs": [
            -0.012229824837884262, -0.015832437375127517, -0.004017716027188145,
            0.011202230645773747, -0.002471878830597074, -0.001979743246308452
        ],
        "yaws": [
            4.722650135577905, 3.707531572940671, 5.678274250497195,
            1.5615802076498657, 0.3242761715556668, 2.7485237221188865
        ],
        "fov": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    }
    
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Create figure
    fig = go.Figure()
    
    # Process all three datasets with distinct styling
    datasets = [
        ("NuScenes_native_to_euler", nerf_data, "#FF4444", "#CC0000", 
         {"width": 4}, {"red": "#FF0000", "green": "#00AA00", "blue": "#0066FF"}),
        ("NuScenes_old_modified", nuscenes_data, "#4444FF", "#0000CC", 
         {"width": 3, "dash": "dot"}, {"red": "#CC3333", "green": "#33AA33", "blue": "#3366CC"}),
        ("NuScenes_old", third_data, "#44FF44", "#00CC00", 
         {"width": 3, "dash": "dash"}, {"red": "#AA3333", "green": "#00CC00", "blue": "#3333AA"})
    ]
    
    scale = 0.3
    fov_radius = 10.0  # Radius for FOV visualization (10m)
    triangle_length = 1.0  # Length of triangle markers (1.0m)
    
    for dataset_name, data, primary_color, secondary_color, line_style, axis_colors in datasets:
        positions = np.array(data["coordinates"])
        rolls = data["rolls"]
        pitchs = data["pitchs"]
        yaws = data["yaws"]
        fovs = data["fov"]
        
        # Collect all triangle vertices for this dataset
        all_triangle_x, all_triangle_y, all_triangle_z = [], [], []
        triangle_meshes_x, triangle_meshes_y, triangle_meshes_z = [], [], []
        triangle_faces_i, triangle_faces_j, triangle_faces_k = [], [], []
        
        # Create hover text and triangle markers
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
            
            # Create triangle for this camera
            rotation = euler_to_rotation_matrix(roll, pitch, yaw)
            tri_x, tri_y, tri_z = create_camera_triangle(pos, rotation, triangle_length)
            
            # Add triangle outline
            if i > 0:
                all_triangle_x.extend([None])
                all_triangle_y.extend([None])
                all_triangle_z.extend([None])
            
            all_triangle_x.extend(tri_x.tolist())
            all_triangle_y.extend(tri_y.tolist())
            all_triangle_z.extend(tri_z.tolist())
            
            # Add triangle vertices for filled mesh (excluding the closing vertex)
            base_idx = len(triangle_meshes_x)
            triangle_meshes_x.extend(tri_x[:-1].tolist())  # Exclude closing vertex
            triangle_meshes_y.extend(tri_y[:-1].tolist())
            triangle_meshes_z.extend(tri_z[:-1].tolist())
            
            # Create triangle face (tip, left base, right base)
            triangle_faces_i.append(base_idx)      # Tip
            triangle_faces_j.append(base_idx + 1)  # Left base
            triangle_faces_k.append(base_idx + 2)  # Right base
        
        # Add triangle outlines as lines (black outlines)
        fig.add_trace(go.Scatter3d(
            x=all_triangle_x,
            y=all_triangle_y,
            z=all_triangle_z,
            mode='lines',
            line=dict(color='black', width=2),
            name=f'{dataset_name} Outlines',
            legendgroup=dataset_name,
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add filled triangles as Mesh3d
        fig.add_trace(go.Mesh3d(
            x=triangle_meshes_x,
            y=triangle_meshes_y,
            z=triangle_meshes_z,
            i=triangle_faces_i,
            j=triangle_faces_j,
            k=triangle_faces_k,
            color=primary_color,
            name=f'{dataset_name} Cameras',
            legendgroup=dataset_name,
            legendgrouptitle_text=dataset_name,
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add invisible markers at camera positions for hover info
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            hovertext=hover_texts,
            hoverinfo='text',
            marker=dict(
                size=8,
                color=primary_color,
                opacity=0  # Invisible markers just for hover
            ),
            name=f'{dataset_name} Info',
            legendgroup=dataset_name,
            showlegend=False
        ))
        
        # Collect axis lines for this dataset
        x_lines_x, x_lines_y, x_lines_z = [], [], []
        y_lines_x, y_lines_y, y_lines_z = [], [], []
        z_lines_x, z_lines_y, z_lines_z = [], [], []
        
        for i, (pos, roll, pitch, yaw, fov) in enumerate(zip(positions, rolls, pitchs, yaws, fovs)):
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
        
        # FIXED: Create FOV segments with uniform colors by building proper mesh geometry
        for i, (pos, roll, pitch, yaw, fov) in enumerate(zip(positions, rolls, pitchs, yaws, fovs)):
            rotation = euler_to_rotation_matrix(roll, pitch, yaw)
            fov_x, fov_y, fov_z = create_fov_segment(pos, rotation, fov, fov_radius)
            
            # Create proper mesh geometry for uniform coloring
            if len(fov_x) >= 3:
                # Create triangular faces from center to consecutive arc points
                faces_i, faces_j, faces_k = [], [], []
                
                # Create fan triangulation from center (index 0) to all arc points
                for j in range(1, len(fov_x) - 2):  # -2 to avoid out of bounds
                    faces_i.append(0)      # Center point
                    faces_j.append(j)      # Current arc point
                    faces_k.append(j + 1)  # Next arc point
                
                # Create unique legend group for this specific FOV camera
                fov_legend_group = f"{dataset_name}_FOV_{camera_names[i]}"
                
                # Add single Mesh3d trace with uniform color
                fig.add_trace(go.Mesh3d(
                    x=fov_x,
                    y=fov_y,
                    z=fov_z,
                    i=faces_i,
                    j=faces_j,
                    k=faces_k,
                    facecolor=[primary_color] * len(faces_i),  # Explicit uniform face colors
                    opacity=0.5,
                    name=f'FOV {camera_names[i]}',
                    legendgroup=fov_legend_group,  # Unique legend group
                    showlegend=True,
                    hoverinfo='skip',
                    # Ensure uniform coloring
                    lighting=dict(
                        ambient=1.0,    # Full ambient light for uniform appearance
                        diffuse=0.0,    # No diffuse lighting
                        fresnel=0.0,    # No fresnel effects
                        specular=0.0,   # No specular highlights
                        roughness=1.0   # Completely matte
                    ),
                    flatshading=True      # Flat shading for uniform color
                ))
    
    # Calculate position differences for analysis (comparing all datasets)
    nerf_positions = np.array(nerf_data["coordinates"])
    nuscenes_positions = np.array(nuscenes_data["coordinates"])
    third_positions = np.array(third_data["coordinates"])
    
    # Position differences between datasets
    nerf_nuscenes_diffs = np.linalg.norm(nerf_positions - nuscenes_positions, axis=1)
    nerf_third_diffs = np.linalg.norm(nerf_positions - third_positions, axis=1)
    nuscenes_third_diffs = np.linalg.norm(nuscenes_positions - third_positions, axis=1)
    
    # Average differences
    avg_nerf_nuscenes = np.mean(nerf_nuscenes_diffs)
    avg_nerf_third = np.mean(nerf_third_diffs)
    avg_nuscenes_third = np.mean(nuscenes_third_diffs)
    
    # Maximum differences
    max_nerf_nuscenes = np.max(nerf_nuscenes_diffs)
    max_nerf_third = np.max(nerf_third_diffs)
    max_nuscenes_third = np.max(nuscenes_third_diffs)
    
    # Update layout
    fig.update_layout(
        title=f'Triple Dataset 3D Camera Poses with FOV Visualization<br>'
              f'<sub>NuScenes_native_to_euler (red triangles) vs NuScenes_old_modified (blue triangles) vs NuScenes_old (green triangles)<br>'
              f'FOV segments: 10m radius, solid fill, x-y plane facing minus y<br>'
              f'Avg Diffs: Native↔Old_modified: {avg_nerf_nuscenes:.3f}m | Native↔Old: {avg_nerf_third:.3f}m | Old_modified↔Old: {avg_nuscenes_third:.3f}m<br>'
              f'<i>Use controls below to change view and visibility</i></sub>',
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(showgrid=True, gridcolor='lightgray', showline=True),
            yaxis=dict(showgrid=True, gridcolor='lightgray', showline=True),
            zaxis=dict(showgrid=True, gridcolor='lightgray', showline=True)
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
    
    return fig, avg_nerf_nuscenes, avg_nerf_third, avg_nuscenes_third, max_nerf_nuscenes, max_nerf_third, max_nuscenes_third, nerf_nuscenes_diffs, nerf_third_diffs, nuscenes_third_diffs

def create_comparison_table(nerf_nuscenes_diffs, nerf_third_diffs, nuscenes_third_diffs):
    """Create HTML table comparing the three datasets"""
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    table_html = '''
    <div class="comparison-table">
        <h3>Position Differences Analysis (meters)</h3>
        <table>
            <thead>
                <tr>
                    <th>Camera</th>
                    <th>Native ↔ Old_modified</th>
                    <th>Native ↔ Old</th>
                    <th>Old_modified ↔ Old</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for i, name in enumerate(camera_names):
        def get_status_color(diff):
            return "#28a745" if diff < 0.001 else "#ffc107" if diff < 0.01 else "#dc3545"
        
        table_html += f'''
                <tr>
                    <td>{name}</td>
                    <td style="color: {get_status_color(nerf_nuscenes_diffs[i])};">{nerf_nuscenes_diffs[i]:.6f}</td>
                    <td style="color: {get_status_color(nerf_third_diffs[i])};">{nerf_third_diffs[i]:.6f}</td>
                    <td style="color: {get_status_color(nuscenes_third_diffs[i])};">{nuscenes_third_diffs[i]:.6f}</td>
                </tr>
        '''
    
    table_html += '''
            </tbody>
        </table>
    </div>
    '''
    
    return table_html

def create_fov_comparison_table():
    """Create HTML table comparing FOV values between three datasets"""
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    nerf_fovs = [64.5614792846793, 64.8772037435167, 64.6420610387791, 89.82514491401436, 64.99646878534186, 65.05495034029899]
    nuscenes_fovs = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    third_fovs = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
    
    table_html = '''
    <div class="comparison-table">
        <h3>Field of View (FOV) Comparison</h3>
        <table>
            <thead>
                <tr>
                    <th>Camera</th>
                    <th>NuScenes_native_to_euler FOV (°)</th>
                    <th>NuScenes_old_modified FOV (°)</th>
                    <th>NuScenes_old FOV (°)</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for i, name in enumerate(camera_names):
        table_html += f'''
                <tr>
                    <td>{name}</td>
                    <td>{nerf_fovs[i]:.2f}</td>
                    <td>{nuscenes_fovs[i]:.2f}</td>
                    <td>{third_fovs[i]:.2f}</td>
                </tr>
        '''
    
    table_html += '''
            </tbody>
        </table>
    </div>
    '''
    
    return table_html

def generate_html_output(output_file='triple_camera_visualization_with_fov_fixed_colors.html'):
    """Generate HTML file with triple dataset 3D camera visualization including FOV segments with FIXED UNIFORM COLORS"""
    
    # Define FOV radius here so it's accessible in the HTML template
    fov_radius = 10.0
    triangle_length = 1.0
    
    # Create the visualization
    fig, avg_nerf_nuscenes, avg_nerf_third, avg_nuscenes_third, max_nerf_nuscenes, max_nerf_third, max_nuscenes_third, nerf_nuscenes_diffs, nerf_third_diffs, nuscenes_third_diffs = create_triple_camera_visualization()
    
    # Create comparison tables
    position_table = create_comparison_table(nerf_nuscenes_diffs, nerf_third_diffs, nuscenes_third_diffs)
    fov_table = create_fov_comparison_table()
    
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
    
    # Enhanced HTML with styling, comparison tables, and interactive controls
    enhanced_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Triple Dataset 3D Camera Poses with FOV Visualization (Fixed Colors & Legend)</title>
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
            .controls {{
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
                display: flex;
                gap: 30px;
                flex-wrap: wrap;
                justify-content: center;
            }}
            .control-group {{
                text-align: center;
            }}
            .control-group h4 {{
                margin-top: 0;
                margin-bottom: 10px;
                color: #333;
            }}
            .checkbox-group {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                align-items: flex-start;
            }}
            .checkbox-item {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            input[type="checkbox"] {{
                width: 18px;
                height: 18px;
                cursor: pointer;
            }}
            input[type="radio"] {{
                width: 18px;
                height: 18px;
                cursor: pointer;
            }}
            label {{
                cursor: pointer;
                font-size: 14px;
                color: #333;
            }}
            .plot-container {{
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 10px;
                margin-bottom: 20px;
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
            .fov-info {{
                background-color: #fff3cd;
                border: 1px solid #ffeeba;
                color: #856404;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 14px;
            }}
            .fix-info {{
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                color: #0c5460;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 14px;
            }}
            .table-container {{
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }}
            .table-container .comparison-table {{
                flex: 1;
                min-width: 400px;
            }}
        </style>
        <script>
            function updateView() {{
                const viewSelect = document.querySelector('input[name="view"]:checked');
                const showGrid = document.getElementById('showGrid').checked;
                const showAxes = document.getElementById('showAxes').checked;
                const showPlanes = document.getElementById('showPlanes').checked;
                
                let camera = {{}};
                let scene_update = {{}};
                
                // Set camera view based on selection
                switch(viewSelect.value) {{
                    case 'free':
                        camera = {{eye: {{x: 1.5, y: 1.5, z: 1.5}}}};
                        break;
                    case 'xy':
                        camera = {{eye: {{x: 0, y: 0, z: 3}}, up: {{x: 0, y: 1, z: 0}}}};
                        break;
                    case 'xz':
                        camera = {{eye: {{x: 0, y: 3, z: 0}}, up: {{x: 0, y: 0, z: 1}}}};
                        break;
                    case 'yz':
                        camera = {{eye: {{x: 3, y: 0, z: 0}}, up: {{x: 0, y: 0, z: 1}}}};
                        break;
                }}
                
                // Set grid, axes, and plane visibility
                scene_update = {{
                    camera: camera,
                    xaxis: {{
                        showgrid: showGrid,
                        gridcolor: 'lightgray',
                        showline: showAxes,
                        showticklabels: showAxes,
                        title: showAxes ? 'X (meters)' : '',
                        showbackground: showPlanes,
                        backgroundcolor: 'rgba(230, 230, 230, 0.3)'
                    }},
                    yaxis: {{
                        showgrid: showGrid,
                        gridcolor: 'lightgray',
                        showline: showAxes,
                        showticklabels: showAxes,
                        title: showAxes ? 'Y (meters)' : '',
                        showbackground: showPlanes,
                        backgroundcolor: 'rgba(230, 230, 230, 0.3)'
                    }},
                    zaxis: {{
                        showgrid: showGrid,
                        gridcolor: 'lightgray',
                        showline: showAxes,
                        showticklabels: showAxes,
                        title: showAxes ? 'Z (meters)' : '',
                        showbackground: showPlanes,
                        backgroundcolor: 'rgba(230, 230, 230, 0.3)'
                    }}
                }};
                
                Plotly.relayout('camera-plot', {{'scene': scene_update}});
            }}
            
            // Initialize controls when page loads
            window.addEventListener('load', function() {{
                document.querySelectorAll('input[name="view"]').forEach(radio => {{
                    radio.addEventListener('change', updateView);
                }});
                document.getElementById('showGrid').addEventListener('change', updateView);
                document.getElementById('showAxes').addEventListener('change', updateView);
                document.getElementById('showPlanes').addEventListener('change', updateView);
            }});
        </script>
    </head>
    <body>
        <div class="header">
            <h1>Triple Dataset 3D Camera Poses with FOV Visualization</h1>
            <div class="info">
                Interactive comparison between NuScenes_native_to_euler, NuScenes_old_modified, and NuScenes_old cameras<br>
                Use mouse to rotate, zoom, and pan | Hover over triangles for detailed camera information
            </div>
            <div class="fix-info">
                <strong>FIXED:</strong> FOV segments now have uniform colors AND proper legend toggle control!
            </div>
            <div class="legend-info">
                <strong>Interactive Legend:</strong> Click on any legend item to toggle visibility - including individual FOV cameras!
            </div>
            <div class="fov-info">
                <strong>FOV Visualization:</strong> Semi-transparent filled circle segments (50% opacity) show field of view in x-y plane, facing minus y direction from each camera's pose origin
            </div>
            <div class="stats">
                <strong>Quick Stats:</strong><br>
                Average Position Differences:<br>
                • Native ↔ Old_modified: {avg_nerf_nuscenes:.6f} meters<br>
                • Native ↔ Old: {avg_nerf_third:.6f} meters<br>
                • Old_modified ↔ Old: {avg_nuscenes_third:.6f} meters<br>
                <strong>Visual Differences:</strong><br>
                • NuScenes_native_to_euler: Red filled triangles with solid axes and uniform red FOV segments<br>
                • NuScenes_old_modified: Blue filled triangles with dotted axes and uniform blue FOV segments<br>
                • NuScenes_old: Green filled triangles with dashed axes and uniform green FOV segments<br>
                <strong>Camera Markers:</strong> Solid filled triangles with black outlines, tip at camera origin, base 1.5x wider towards minus y direction, length {triangle_length}m<br>
                <strong>FOV Segments:</strong> Semi-transparent (50% opacity), uniform colors, drawn at radius {fov_radius}m in x-y plane<br>
                <strong>Legend Control:</strong> Click any FOV item in legend to toggle that specific camera's FOV visibility
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <h4>View Plane</h4>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="radio" id="viewFree" name="view" value="free" checked>
                        <label for="viewFree">Free 3D View</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" id="viewXY" name="view" value="xy">
                        <label for="viewXY">X-Y Plane (Top View)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" id="viewXZ" name="view" value="xz">
                        <label for="viewXZ">X-Z Plane (Front View)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" id="viewYZ" name="view" value="yz">
                        <label for="viewYZ">Y-Z Plane (Side View)</label>
                    </div>
                </div>
            </div>
            
            <div class="control-group">
                <h4>Display Options</h4>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="checkbox" id="showGrid" checked>
                        <label for="showGrid">Show Grid</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="showAxes" checked>
                        <label for="showAxes">Show Global Axes & Labels</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="showPlanes" checked>
                        <label for="showPlanes">Show Coordinate Planes</label>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="plot-container">
            {plot_html.split('<body>')[1].split('</body>')[0]}
        </div>
        <div class="table-container">
            {position_table}
            {fov_table}
        </div>
    </body>
    </html>
    """
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(enhanced_html)
    
    print(f"COMPLETELY FIXED: Enhanced triple dataset HTML visualization saved to: {output_file}")
    print(f"FOV segments now have uniform colors AND proper legend toggle control!")
    print(f"Average position differences:")
    print(f"   • Native ↔ Old_modified: {avg_nerf_nuscenes:.6f} meters")
    print(f"   • Native ↔ Old: {avg_nerf_third:.6f} meters") 
    print(f"   • Old_modified ↔ Old: {avg_nuscenes_third:.6f} meters")
    print("Features:")
    print("   • Three datasets with filled triangle markers (1.0m length, 1.5x wider base, black outlines)")
    print("   • FIXED: Uniform color FOV segments (50% opacity, 10m radius)")
    print("   • FIXED: Individual FOV camera legend control - click to toggle each FOV!")
    print("   • Interactive view controls with coordinate plane toggle")
    print("   • Global axes/grid/plane visibility toggles")
    print("   • Enhanced comparison analysis")
    
    return enhanced_html

if __name__ == "__main__":
    # Generate the HTML output
    html_output = generate_html_output('triple_camera_visualization_with_fov_fixed_colors_and_legend.html')
    print("COMPLETELY FIXED: Enhanced triple dataset 3D Camera visualization complete!")
    print("Uniform FOV colors + Working legend toggle control!")
    print("Click on individual 'FOV CAM_FRONT', 'FOV CAM_BACK', etc. in legend to toggle each FOV segment!")
    print("Open 'triple_camera_visualization_with_fov_fixed_colors_and_legend.html' in your web browser to view the interactive comparison.")