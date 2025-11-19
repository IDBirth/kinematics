""" DH transformation matrix testing on a single link """
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dh_transform(a, alpha, d, theta):
    """Standard (Classic) DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

# DH Parameters (change the values depends on the link configuration)
a = 0          # Link length (translation along x-axis)
alpha = np.pi/2  # Link twist (rotation about x-axis)= 90 degrees
d = 3          # Link offset (translation along z-axis)
theta = 0        # Joint angle (rotation about z-axis) 

# Input point coordinates (where the link originated)
x = 0
y = 0
z = 0

print("*** STANDARD DH TRANSFORMATION TEST ON SINGLE LINK ***")
print()
print("DH Parameters:")
print(f"  a     = {a} (link length)")
print(f"  alpha = {alpha:.6f} rad = {np.degrees(alpha):.2f}° (link twist)")
print(f"  d     = {d} (link offset)")
print(f"  theta = {theta:.6f} rad = {np.degrees(theta):.2f}° (joint angle)")
print()
print("Input Point:")
print(f"  x = {x}")
print(f"  y = {y}")
print(f"  z = {z}")
print(f"  Point = ({x}, {y}, {z})")
print()

# Compute transformation matrix
T = dh_transform(a, alpha, d, theta)

print("Transformation Matrix T:")
print(T)
print()

# Transform the point directly using matrix multiplication
# Convert point to homogeneous coordinates [x, y, z, 1]
point_homogeneous = np.array([x, y, z, 1.0])

# Apply transformation: T @ point
transformed_homogeneous = T @ point_homogeneous

# Extract 3D coordinates of where the link ends up (drop the last element which should be 1)
x_new, y_new, z_new = transformed_homogeneous[0], transformed_homogeneous[1], transformed_homogeneous[2]

print("Result:")
print(f"  Original point:    ({x}, {y}, {z})")
print(f"  Transformed point: ({x_new:.6f}, {y_new:.6f}, {z_new:.6f})")
print()

# ============================================================================
# 3D VISUALIZATION - Simple link drawing
# ============================================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the link from original point to transformed point
ax.plot([x, x_new], [y, y_new], [z, z_new], 'b-', linewidth=4, label='Link')

# Plot original point (joint start)
ax.scatter([x], [y], [z], color='red', s=150, marker='o', 
           label=f'Start ({x}, {y}, {z})', edgecolors='black', linewidths=2)

# Plot transformed point (joint end)
ax.scatter([x_new], [y_new], [z_new], color='green', s=150, marker='o', 
           label=f'End ({x_new:.2f}, {y_new:.2f}, {z_new:.2f})', 
           edgecolors='black', linewidths=2)

# Labels and title
ax.set_xlabel('X', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_zlabel('Z', fontsize=12, fontweight='bold')
ax.set_title(f'Single Link DH Transformation\na={a}, α={np.degrees(alpha):.1f}°, d={d}, θ={np.degrees(theta):.1f}°', 
             fontsize=14, fontweight='bold')

# Set equal aspect ratio
all_coords = np.array([[x, y, z], [x_new, y_new, z_new]])
max_range = np.array([np.ptp(all_coords[:, 0]),
                      np.ptp(all_coords[:, 1]),
                      np.ptp(all_coords[:, 2])]).max() / 2.0

# Add some padding
max_range = max(max_range, 1.0) * 1.2

mid_x = all_coords[:, 0].mean()
mid_y = all_coords[:, 1].mean()
mid_z = all_coords[:, 2].mean()

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_box_aspect([1, 1, 1])

# Grid and legend
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()
