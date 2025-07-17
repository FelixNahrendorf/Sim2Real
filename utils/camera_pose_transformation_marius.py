import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyquaternion import Quaternion
import os
from nuscenes.nuscenes import NuScenes
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize nuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/app/datasets/nuscenes_full/', verbose=True)
 
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
                'sample_data_key': data_key,
                'image_path': image_path,
                'sample_data_token': sample_data_token,
                'camera_intrinsic': calibrated_sensor['camera_intrinsic']
            }
   
    return camera_data
 
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
 
def plot_camera_positions_3d(camera_data, output_path='camera_positions_3d.html'):
    """
    Plot camera positions in 3D space relative to ego vehicle using Plotly (interactive HTML)
    """
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add ego vehicle at origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='red', symbol='square'),
        name='Ego Vehicle',
        text=['Ego Vehicle'],
        hoverinfo='text'
    ))
    
    # Add camera positions using transformed coordinates
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        translation = data['translation']
        fig.add_trace(go.Scatter3d(
            x=[translation[0]], 
            y=[translation[1]], 
            z=[translation[2]],
            mode='markers+text',
            marker=dict(size=8, color=colors[idx % len(colors)]),
            name=camera_name,
            text=[camera_name],
            textposition="top center",
            hovertemplate=f'<b>{camera_name}</b><br>' +
                         f'X: {translation[0]:.2f}<br>' +
                         f'Y: {translation[1]:.2f}<br>' +
                         f'Z: {translation[2]:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Camera Positions Relative to Ego Vehicle (Transformed Coordinates)',
        scene=dict(
            xaxis_title='X (transformed: -z_old)',
            yaxis_title='Y (transformed: x_old)',
            zaxis_title='Z (transformed: -y_old)',
            aspectmode='cube'
        ),
        width=1000,
        height=700
    )
    
    # Save as HTML
    fig.write_html(output_path)
    print(f"Interactive 3D plot saved to: {output_path}")
    
    # Show the plot
    fig.show()
 
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

def save_all_plots(camera_data, output_dir='output_plots'):
    """
    Save all plots to files in the specified directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    camera_images_path = os.path.join(output_dir, 'camera_images.png')
    coordinate_comparison_path = os.path.join(output_dir, 'coordinate_comparison.png')
    camera_positions_3d_path = os.path.join(output_dir, 'camera_positions_3d.html')
    
    print(f"Saving plots to directory: {output_dir}")
    
    # Save camera images plot
    plot_camera_images(camera_data, output_path=camera_images_path)
    
    # Save coordinate comparison plot
    plot_coordinate_comparison(camera_data, output_path=coordinate_comparison_path)
    
    # Save 3D camera positions plot
    plot_camera_positions_3d(camera_data, output_path=camera_positions_3d_path)
    
    print(f"\nAll plots saved successfully in: {output_dir}")
    print(f"- Camera images: {camera_images_path}")
    print(f"- Coordinate comparison: {coordinate_comparison_path}")
    print(f"- Interactive 3D plot: {camera_positions_3d_path}")
 
# Main execution
if __name__ == "__main__":
    # Assuming you have already initialized nusc
    # nusc = NuScenes(version='v1.0-mini', dataroot='/path/to/nuscenes', verbose=True)
   
    # Get camera data with images (now includes transformation)
    camera_data = get_all_camera_positions_with_images(nusc)
   
    # Print camera information
    print_camera_info(camera_data)
   
    # Save all plots to files
    save_all_plots(camera_data)
   
    # Print both original and transformed positions for comparison
    print("\nCamera positions and rotations comparison:")
    print("=" * 120)
    for camera, data in camera_data.items():
        orig_pos = data['original_translation']
        trans_pos = data['translation']
        orig_rot_xyzw = data['original_rotation_xyzw']
        trans_rot_xyzw = data['rotation_xyzw']
        orig_euler_deg = data['original_euler_deg']
        trans_euler_deg = data['euler_deg']
        print(f"{camera}:")
        print(f"  Original position:      [{orig_pos[0]:6.2f}, {orig_pos[1]:6.2f}, {orig_pos[2]:6.2f}]")
        print(f"  Transformed position:   [{trans_pos[0]:6.2f}, {trans_pos[1]:6.2f}, {trans_pos[2]:6.2f}]")
        print(f"  Original rotation (x,y,z,w):     [{orig_rot_xyzw[0]:6.3f}, {orig_rot_xyzw[1]:6.3f}, {orig_rot_xyzw[2]:6.3f}, {orig_rot_xyzw[3]:6.3f}]")
        print(f"  Transformed rotation (x,y,z,w):  [{trans_rot_xyzw[0]:6.3f}, {trans_rot_xyzw[1]:6.3f}, {trans_rot_xyzw[2]:6.3f}, {trans_rot_xyzw[3]:6.3f}]")
        print(f"  Original euler (deg):   Roll={orig_euler_deg[0]:6.1f}°, Pitch={orig_euler_deg[1]:6.1f}°, Yaw={orig_euler_deg[2]:6.1f}°")
        print(f"  Transformed euler (deg): Roll={trans_euler_deg[0]:6.1f}°, Pitch={trans_euler_deg[1]:6.1f}°, Yaw={trans_euler_deg[2]:6.1f}°")
        print()