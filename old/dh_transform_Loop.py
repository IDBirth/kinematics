""" DH transformation for 2-DOF RR robot with incremental 360° sweep and end-effector trail. """
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

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


# Robot parameters
L1 = 3.0  # Link 1 length (vertical)
L2 = 5.0  # Link 2 length (horizontal)

# Joint limits (degrees)
THETA1_MIN_DEG, THETA1_MAX_DEG = -180.0, 180.0
THETA2_MIN_DEG, THETA2_MAX_DEG = -180.0, 180.0


def forward_kinematics(theta1, theta2):
    """Compute P1 (joint 2) and P2 (end-effector) for given joint angles."""
    # Base position (origin)
    P0 = np.array([0, 0, 0, 1])

    # Link 1 transformation: Base to Joint 2
    T01 = dh_transform(a=0, alpha=np.pi/2, d=L1, theta=theta1)

    # Position of Joint 2
    P1_homogeneous = T01 @ P0
    P1 = P1_homogeneous[:3]

    # Link 2 transformation: Joint 2 to End-effector
    T12 = dh_transform(a=L2, alpha=0, d=0, theta=theta2)

    # Total transformation: Base to End-effector
    T02 = T01 @ T12

    # Position of End-effector
    P2_homogeneous = T02 @ P0
    P2 = P2_homogeneous[:3]

    return P1, P2, T01, T12, T02


def plot_robot_with_trail(ax, theta1, theta2, trail_points):
    """Update the 3D plot for the given joint angles and draw the trail."""
    joint2, end_effector, T01, T12, T02 = forward_kinematics(theta1, theta2)

    # Append current end-effector position to trail
    trail_points.append(end_effector.copy())

    # Optional: print some info in the terminal (comment out if too spammy)
    print(f"Theta1 = {np.degrees(theta1):7.2f} deg, "
          f"Theta2 = {np.degrees(theta2):7.2f} deg, "
          f"EE = ({end_effector[0]:.3f}, {end_effector[1]:.3f}, {end_effector[2]:.3f})")

    # Clear previous drawing
    ax.cla()

    base = np.array([0.0, 0.0, 0.0])
    P1 = joint2
    P2 = end_effector

    # Draw Link 1 (Base to Joint 2)
    ax.plot([base[0], P1[0]],
            [base[1], P1[1]],
            [base[2], P1[2]],
            'b-', linewidth=6, label=f'Link 1 (L={L1})')

    # Draw Link 2 (Joint 2 to End-effector)
    ax.plot([P1[0], P2[0]],
            [P1[1], P2[1]],
            [P1[2], P2[2]],
            'g-', linewidth=6, label=f'Link 2 (L={L2})')

    # Plot joints and EE
    ax.scatter([base[0]], [base[1]], [base[2]],
               color='red', s=200, marker='o',
               label='Base (Joint 1)', edgecolors='black', linewidths=2, zorder=5)

    ax.scatter([P1[0]], [P1[1]], [P1[2]],
               color='orange', s=200, marker='o',
               label='Joint 2', edgecolors='black', linewidths=2, zorder=5)

    ax.scatter([P2[0]], [P2[1]], [P2[2]],
               color='purple', s=200, marker='s',
               label='End-effector', edgecolors='black', linewidths=2, zorder=5)

    # Draw end-effector trail
    trail_arr = np.array(trail_points)
    ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2],
            'r--', linewidth=2, label='End-effector trail')

    # Draw coordinate frame at base
    frame_scale = 1.5
    ax.quiver(0, 0, 0, frame_scale, 0, 0,
              color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.6)
    ax.quiver(0, 0, 0, 0, frame_scale, 0,
              color='green', arrow_length_ratio=0.3, linewidth=2, alpha=0.6)
    ax.quiver(0, 0, 0, 0, 0, frame_scale,
              color='blue', arrow_length_ratio=0.3, linewidth=2, alpha=0.6)

    # Labels and title
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title(
        f'2-DOF RR Robot – Incremental Sweep\n'
        f'θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°',
        fontsize=14, fontweight='bold'
    )

    # Equal aspect ratio including trail
    all_points = np.vstack([base, P1, P2] + trail_points)
    max_range = np.array([np.ptp(all_points[:, 0]),
                          np.ptp(all_points[:, 1]),
                          np.ptp(all_points[:, 2])]).max() / 2.0
    max_range = max(max_range, 2.0) * 1.2

    mid_x = np.mean(all_points[:, 0])
    mid_y = np.mean(all_points[:, 1])
    mid_z = np.mean(all_points[:, 2])

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Info text
    info_text = (
        f'Robot Configuration:\n'
        f'L1 = {L1}\nL2 = {L2}\n'
        f'θ1 range: {THETA1_MIN_DEG:.0f}° to {THETA1_MAX_DEG:.0f}°\n'
        f'θ2 current: {np.degrees(theta2):.1f}°'
    )
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def main():
    # Use interactive mode for animation
    plt.ion()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Choose fixed θ2 (you can change this)
    theta2_deg = 181
    theta2 = np.radians(theta2_deg)

    # Set up sweep for θ1 across full 360° range
    n_steps = 181  # resolution of sweep (more = smoother, slower)
    theta1_values_deg = np.linspace(THETA1_MIN_DEG, THETA1_MAX_DEG, n_steps)

    trail_points = []

    print("Incremental 360° sweep of θ1")
    print(f"θ1 from {THETA1_MIN_DEG:.0f}° to {THETA1_MAX_DEG:.0f}°, θ2 fixed at {theta2_deg:.1f}°\n")

    for theta1_deg in theta1_values_deg:
        theta1 = np.radians(theta1_deg)

        plot_robot_with_trail(ax, theta1, theta2, trail_points)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.02)  # control animation speed (seconds)

    print("\n*** Sweep complete. Close the plot window to exit. ***")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
