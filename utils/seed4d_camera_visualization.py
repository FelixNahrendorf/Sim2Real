import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Camera positions extracted from SEED4D transforms_ego.json
# These are the translation components (last column, first 3 rows) of each transform matrix
camera_positions = [
    [-1.494913101196289, 1.7220057249069214, -0.0047545298002660275],      # Camera 0
    [-1.517493724822998, 1.580825686454773, 0.4990787208080292],          # Camera 1  
    [-1.506960391998291, 1.5752559900283813, -0.5005193948745728],        # Camera 2
    [-1.5679426193237305, 0.05524611100554466, -0.010788239538669586],    # Camera 3
    [-1.5679426193237305, 0.05524611100554466, -0.010788239538669586],    # Camera 4 (identical to 3)
    [-1.5621013641357422, 1.0485204458236694, -0.4830581247806549],       # Camera 5
    [-1.5505084991455078, 1.059451699256897, 0.46720296144485474]         # Camera 6
]

# Convert to numpy array for easier manipulation
coords = np.array(camera_positions)
x, y, z = coords[:, 1], coords[:, 2], coords[:, 0]

# Print camera positions
print("SEED4D Camera Positions from transforms_ego.json:")
print("="*60)
for i, pos in enumerate(camera_positions):
    print(f"Camera {i}: ({pos[0]:8.5f}, {pos[1]:8.5f}, {pos[2]:8.5f})")

print(f"\nNote: Cameras 3 and 4 have IDENTICAL positions!")
print(f"This explains the 6 NuScenes cameras vs 7 SEED4D sensors mismatch!")

# Create the 3D visualization
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# Define colors for each camera
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']

# Plot camera points
for i in range(len(camera_positions)):
    alpha = 0.5 if i == 4 else 0.8  # Make camera 4 more transparent since it's duplicate
    size = 100 if i == 4 else 150   # Make camera 4 smaller
    ax.scatter(x[i], y[i], z[i], c=colors[i], s=size, alpha=alpha, 
               label=f'Camera {i}' + (' (duplicate)' if i == 4 else ''))

# Add camera labels
for i, (xi, yi, zi) in enumerate(camera_positions):
    label_text = f'Cam {i}' + ('*' if i == 4 else '')  # Mark duplicate with *
    ax.text(xi + 0.02, yi + 0.02, zi + 0.02, label_text, 
            fontsize=10, fontweight='bold')

# Connect consecutive cameras with lines to show arrangement
for i in range(len(camera_positions) - 1):
    line_style = '--' if i == 3 else '-'  # Dashed line to duplicate camera
    ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 
            'gray', alpha=0.4, linewidth=1, linestyle=line_style)

# Draw lines from each camera to origin
for i, (xi, yi, zi) in enumerate(camera_positions):
    ax.plot([0, xi], [0, yi], [0, zi], 'lightblue', alpha=0.2, linewidth=1)

# Add origin point
ax.scatter([0], [0], [0], c='black', s=300, marker='*', 
           label='Origin', edgecolors='white', linewidth=2)

# Highlight the camera cluster around Y=1.5-1.7 (cameras 0,1,2,5,6)
cluster_cameras = [0, 1, 2, 5, 6]
cluster_x = [x[i] for i in cluster_cameras]
cluster_y = [y[i] for i in cluster_cameras]
cluster_z = [z[i] for i in cluster_cameras]

# Draw a circle around the main cluster
from matplotlib.patches import Circle
import matplotlib.patches as patches

# Set labels and title
ax.set_xlabel('X coordinate (meters)', fontsize=12)
ax.set_ylabel('Y coordinate (meters)', fontsize=12) 
ax.set_zlabel('Z coordinate (meters)', fontsize=12)
ax.set_title('SEED4D Camera Positions from transforms_ego.json\n' + 
             '7 Cameras with NuScenes-like Arrangement (Cameras 3&4 Identical)', 
             fontsize=14, fontweight='bold')

# Add legend
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Set equal aspect ratio for better visualization
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Add grid
ax.grid(True, alpha=0.3)

# Add annotations for camera groups
ax.text(-1.51, 1.6, 0.6, 'Front/Side Cameras\n(0,1,2,5,6)', fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
ax.text(-1.57, 0.0, -0.2, 'Back Camera\n(3,4 identical)', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

plt.tight_layout()

# Save as PNG file with high resolution
plt.savefig('seed4d_camera_positions.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'seed4d_camera_positions.png'")

plt.show()

# Print detailed statistics
print(f"\nCoordinate Statistics:")
print(f"X range: {x.min():.3f} to {x.max():.3f} (span: {x.max()-x.min():.3f})")
print(f"Y range: {y.min():.3f} to {y.max():.3f} (span: {y.max()-y.min():.3f})")
print(f"Z range: {z.min():.3f} to {z.max():.3f} (span: {z.max()-z.min():.3f})")
print(f"Center point: ({x.mean():.3f}, {y.mean():.3f}, {z.mean():.3f})")

# Calculate distances from origin
distances = np.sqrt(x**2 + y**2 + z**2)
print(f"\nDistance from origin:")
print(f"Min: {distances.min():.3f}, Max: {distances.max():.3f}, Mean: {distances.mean():.3f}")

# Analyze camera grouping
print(f"\nCamera Grouping Analysis:")
print(f"Front/Side cameras (0,1,2,5,6): Y ≈ 1.0-1.7 meters")
print(f"Back cameras (3,4): Y ≈ 0.055 meters")
print(f"This suggests a vehicle-mounted camera setup similar to NuScenes!")

# Check exact duplicate
print(f"\nDuplicate Camera Analysis:")
print(f"Camera 3 position: ({coords[3][0]:.6f}, {coords[3][1]:.6f}, {coords[3][2]:.6f})")
print(f"Camera 4 position: ({coords[4][0]:.6f}, {coords[4][1]:.6f}, {coords[4][2]:.6f})")
print(f"Exact match: {np.array_equal(coords[3], coords[4])}")

# Camera arrangement interpretation
print(f"\nCamera Arrangement Interpretation:")
print(f"This looks like a vehicle-mounted 6+1 camera system:")
print(f"- 5 unique cameras around the vehicle (0,1,2,5,6)")
print(f"- 1 back camera duplicated as cameras 3&4")
print(f"- Total: 6 unique positions, 7 camera definitions")
print(f"- This explains why NuScenes (6 cameras) maps to SEED4D (7 sensors)!")