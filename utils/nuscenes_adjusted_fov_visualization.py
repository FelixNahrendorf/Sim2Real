import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the coordinates
coordinates = [
    [0.81154658, -0.00925438, 1.50286755],
    [0.66753281, 0.49701484, 1.50784545],
    [0.65190701, -0.49736302, 1.50852209],
    [-0.85634104, -0.00716842, 1.57249624],
    [0.14285401, -0.48309856, 1.57586643],
    [0.1390594, 0.47302748, 1.5566498]
]

# Convert to numpy array for easier manipulation
coords = np.array(coordinates)
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

# Create the 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
scatter = ax.scatter(x, y, z, c=range(len(coordinates)), cmap='viridis', s=100, alpha=0.8)

# Add point labels
for i, (xi, yi, zi) in enumerate(coordinates):
    ax.text(xi, yi, zi, f'  Point {i}', fontsize=10)

# Connect points with lines to show relationships
# Draw lines from each point to the origin
for i, (xi, yi, zi) in enumerate(coordinates):
    ax.plot([0, xi], [0, yi], [0, zi], 'gray', alpha=0.3, linewidth=1)

# Draw lines connecting consecutive points
for i in range(len(coordinates) - 1):
    ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 'blue', alpha=0.5, linewidth=1)

# Add origin point
ax.scatter([0], [0], [0], c='red', s=200, marker='o', label='Origin')

# Set labels and title
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')
ax.set_title('3D Visualization of Camera Coordinates')

# Add colorbar
plt.colorbar(scatter, ax=ax, label='Point Index')

# Add legend
ax.legend()

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

plt.tight_layout()

# Save as PNG file with high resolution
plt.savefig('3d_coordinates_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved as '3d_coordinates_visualization.png'")

# Print coordinate statistics
print("Coordinate Statistics:")
print(f"X range: {x.min():.3f} to {x.max():.3f}")
print(f"Y range: {y.min():.3f} to {y.max():.3f}")
print(f"Z range: {z.min():.3f} to {z.max():.3f}")
print(f"Center point: ({x.mean():.3f}, {y.mean():.3f}, {z.mean():.3f})")

# Calculate distances from origin
distances = np.sqrt(x**2 + y**2 + z**2)
print(f"Distance from origin - Min: {distances.min():.3f}, Max: {distances.max():.3f}, Mean: {distances.mean():.3f}")

# Print individual coordinates with labels
print("\nIndividual Coordinates:")
for i, coord in enumerate(coordinates):
    print(f"Point {i}: ({coord[0]:8.5f}, {coord[1]:8.5f}, {coord[2]:8.5f})")