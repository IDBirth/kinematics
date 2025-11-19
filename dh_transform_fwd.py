""" DH transformation matrix testing on a single link """

import numpy as np

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
a = 2.0          # Link length (translation along x-axis)
alpha = np.pi/4  # Link twist (rotation about x-axis) = 45 degrees
d = 1.5          # Link offset (translation along z-axis)
theta = 0  # Joint angle (rotation about z-axis) = 0 degrees

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
