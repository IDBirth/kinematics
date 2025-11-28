""" DH transformation for 2-DOF RR robot with incremental sweep of both joints
    and an end-effector trail showing the workspace.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


def dh_transform(a, alpha, d, theta):
    """Standard (Classic) DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])


# Robot parameters
L1 = 3.0  # Link 1 length (vertical)
L2 = 5.0  # Link 2 length (horizontal)

# Joint limits (degrees)
THETA1_MIN_DEG, THETA1_MAX_DEG = -360, 180.0
THETA2_MIN_DEG, THETA2_MAX_DEG = -360, 180.0


def forward_kinematics(theta1, theta2):
    """Compute P1 (joint 2) and P2 (end-effector) for given joint angles."""
    # Base position (origin) in homogeneous coordinates
    P0 = np.array([0, 0, 0, 1])

    # Link 1 transformation: Base to Joint 2
    T01 = dh_transform(a=0, alpha=np.pi / 2, d=L1, theta=theta1)

    # Position of Joint 2
    P1_h = T01 @ P0
    P1 = P1_h[:3]

    # Link 2 transformation: Joint 2 to End-effector
    T12 = dh_transform(a=L2, alpha=0, d=0, theta=theta2)

    # Total transformation: Base to End-effector
    T02 = T01 @ T12

    # Position of End-effector
    P2_h = T02 @ P0
    P2 = P2_h[:3]

    return P1, P2, T01, T12, T02


def plot_robot_with_trail(ax, theta1, theta2, trail_points):
    """Update the 3D plot for the given joint angles and draw the trail."""
    joint2, end_effector, T01, T12, T02 = forward_kinematics(theta1, theta2)

    # Add current EE position to trail
    trail_points.append(end_effector.copy())

    # Optional debug print (comment out if too spammy)
    print(
        f"Theta1 = {np.degrees(theta1):7.2f} deg, "
        f"Theta2 = {np.degrees(theta2):7.2f} deg, "
        f"EE = ({end_effector[0]:.3f}, {end_effector[1]:.3f}, {end_effector[2]:.3f})"
    )

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

    # Draw end-effector trail (all past EE positions)
    trail_arr = np.array(trail_points)
    ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2],
            'r--', linewidth=2, label='End-effector trail')

    # Draw coordinate axes at base
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
        f'2-DOF RR Robot – Joint Space Sweep\n'
        f'θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°',
        fontsize=14, fontweight='bold'
    )

    # Equal aspect ratio including all trail points
    all_points = np.vstack([base, P1, P2] + trail_points)
    max_range = np.array([
        np.ptp(all_points[:, 0]),
        np.ptp(all_points[:, 1]),
        np.ptp(all_points[:, 2])
    ]).max() / 2.0
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

    # Info text box
    info_text = (
        f'Robot Configuration:\n'
        f'L1 = {L1}\nL2 = {L2}\n'
        f'θ1 range: {THETA1_MIN_DEG:.0f}° to {THETA1_MAX_DEG:.0f}°\n'
        f'θ2 range: {THETA2_MIN_DEG:.0f}° to {THETA2_MAX_DEG:.0f}°'
    )
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def main():
    # Use interactive mode for animation
    plt.ion()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ------------------------------------------------------------------
    # Automatically choose steps from desired angle resolution (deg)
    # ------------------------------------------------------------------
    DESIRED_DTHETA1_DEG = 10.0   # desired step for θ1
    DESIRED_DTHETA2_DEG = 15.0   # desired step for θ2

    span1_deg = THETA1_MAX_DEG - THETA1_MIN_DEG   # e.g. 360
    span2_deg = THETA2_MAX_DEG - THETA2_MIN_DEG   # e.g. 360

    # Number of samples = floor(span/step) + 1 so we include both ends
    n_steps1 = int(np.floor(span1_deg / DESIRED_DTHETA1_DEG)) + 1
    n_steps2 = int(np.floor(span2_deg / DESIRED_DTHETA2_DEG)) + 1

    # Recompute actual step sizes so we exactly hit the limits
    if n_steps1 > 1:
        delta_theta1_deg = span1_deg / (n_steps1 - 1)
    else:
        delta_theta1_deg = 0.0

    if n_steps2 > 1:
        delta_theta2_deg = span2_deg / (n_steps2 - 1)
    else:
        delta_theta2_deg = 0.0

    # Build angle arrays explicitly using the computed steps
    theta1_vals_deg = [THETA1_MIN_DEG + i * delta_theta1_deg for i in range(n_steps1)]
    theta2_vals_deg = [THETA2_MIN_DEG + j * delta_theta2_deg for j in range(n_steps2)]

    trail_points = []

    print("Incremental sweep of BOTH θ1 and θ2 over their full ranges (auto step calc).")
    print(f"Requested Δθ1 ≈ {DESIRED_DTHETA1_DEG:.2f}°, actual Δθ1 = {delta_theta1_deg:.2f}°, "
          f"n_steps1 = {n_steps1}")
    print(f"Requested Δθ2 ≈ {DESIRED_DTHETA2_DEG:.2f}°, actual Δθ2 = {delta_theta2_deg:.2f}°, "
          f"n_steps2 = {n_steps2}\n")

    for theta2_deg in theta2_vals_deg:
        for theta1_deg in theta1_vals_deg:
            theta1 = np.radians(theta1_deg)
            theta2 = np.radians(theta2_deg)

            plot_robot_with_trail(ax, theta1, theta2, trail_points)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)  # animation speed (seconds per frame)

    print("\n*** Sweep complete. Close the plot window to exit. ***")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()