#!/usr/bin/env python3
"""
PDE4431 – 2-DOF manipulators: 2D planar and 3D vertical with visualization

- 2D planar 2-link RR arm (FK + IK)
- 3D vertical 2-link arm (FK + IK + 3D visualization)

3D arm equations (from notes):
    x = a2*cos(θ1)*cos(θ2)
    y = a2*sin(θ1)*cos(θ2)
    z = a2*sin(θ2) + d1
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


# ===============================
# 2D PLANAR 2-LINK (RR) FUNCTIONS
# ===============================

def forward_kinematics_2d(theta1_deg: float,
                          theta2_deg: float,
                          l1: float,
                          l2: float) -> tuple[float, float]:
    """Forward kinematics of a 2-link planar R-R arm."""
    t1 = np.deg2rad(theta1_deg)
    t2 = np.deg2rad(theta2_deg)

    x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
    y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
    return float(x), float(y)


def inverse_kinematics_2d(x: float,
                          y: float,
                          l1: float,
                          l2: float) -> list[tuple[float, float]]:
    """Inverse kinematics of a 2-link planar R-R arm (2 solutions: elbow up/down)."""
    r2 = x**2 + y**2
    r = np.sqrt(r2)

    if r > l1 + l2 or r < abs(l1 - l2):
        raise ValueError("Target (x, y) is outside the reachable workspace of the 2D arm.")

    # cos(theta2) by cosine law
    c2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)

    s2_pos = np.sqrt(1.0 - c2**2)   # elbow-down
    s2_neg = -s2_pos                # elbow-up

    k1 = l1 + l2 * c2

    def solve_theta1(s2: float) -> float:
        k2 = l2 * s2
        return np.arctan2(y, x) - np.arctan2(k2, k1)

    # Elbow-down
    theta2_down = np.arctan2(s2_pos, c2)
    theta1_down = solve_theta1(s2_pos)

    # Elbow-up
    theta2_up = np.arctan2(s2_neg, c2)
    theta1_up = solve_theta1(s2_neg)

    return [
        (float(np.rad2deg(theta1_down)), float(np.rad2deg(theta2_down))),
        (float(np.rad2deg(theta1_up)),   float(np.rad2deg(theta2_up))),
    ]


# ===============================
# 3D VERTICAL 2-LINK FUNCTIONS
# ===============================

def forward_kinematics_3d(theta1_deg: float,
                          theta2_deg: float,
                          a2: float,
                          d1: float) -> tuple[float, float, float]:
    """
    Forward kinematics for the 3D 2-link arm:

        x = a2*cos(θ1)*cos(θ2)
        y = a2*sin(θ1)*cos(θ2)
        z = a2*sin(θ2) + d1
    """
    t1 = np.deg2rad(theta1_deg)
    t2 = np.deg2rad(theta2_deg)

    x = a2 * np.cos(t1) * np.cos(t2)
    y = a2 * np.sin(t1) * np.cos(t2)
    z = a2 * np.sin(t2) + d1
    return float(x), float(y), float(z)


def inverse_kinematics_3d(x: float,
                          y: float,
                          z: float,
                          a2: float,
                          d1: float) -> tuple[float, float]:
    """
    Inverse kinematics for the 3D 2-link arm.

    Equations:
        x = a2*cos(θ1)*cos(θ2)
        y = a2*sin(θ1)*cos(θ2)
        z = a2*sin(θ2) + d1

    We solve:
        θ1 = atan2(y, x)
        sin(θ2) = (z - d1) / a2
        cos(θ2) = sqrt(1 - sin^2(θ2)), sign chosen to match horizontal reach.
    """
    rho = np.sqrt(x**2 + y**2)
    dz = z - d1

    theta1 = np.arctan2(y, x)

    s2 = dz / a2
    s2 = np.clip(s2, -1.0, 1.0)

    c2_mag = np.sqrt(1.0 - s2**2)
    # Choose cos sign so that a2*cos(θ2) ≈ ρ
    if rho >= 0:
        c2 = c2_mag
    else:
        c2 = -c2_mag

    theta2 = np.arctan2(s2, c2)

    return float(np.rad2deg(theta1)), float(np.rad2deg(theta2))


# ===============================
# 3D VISUALIZATION
# ===============================

def set_axes_equal(ax):
    """
    Make 3D plot axes have equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    x_middle = 0.5 * (x_limits[0] + x_limits[1])
    y_middle = 0.5 * (y_limits[0] + y_limits[1])
    z_middle = 0.5 * (z_limits[0] + z_limits[1])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_3d_arm(theta1_deg: float,
                theta2_deg: float,
                a2: float,
                d1: float,
                target: tuple[float, float, float] | None = None):
    """
    Plot the 3D 2-link arm configuration.

    Joints:
        Base at (0, 0, 0)
        Joint 1 at (0, 0, d1)
        End-effector at (x, y, z) from FK
    """
    # Joint positions
    base = np.array([0.0, 0.0, 0.0])
    joint1 = np.array([0.0, 0.0, d1])
    x, y, z = forward_kinematics_3d(theta1_deg, theta2_deg, a2, d1)
    ee = np.array([x, y, z])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"3D 2-Link Arm (θ1={theta1_deg:.1f}°, θ2={theta2_deg:.1f}°)")

    # Plot links: base -> joint1 -> ee
    xs = [base[0], joint1[0], ee[0]]
    ys = [base[1], joint1[1], ee[1]]
    zs = [base[2], joint1[2], ee[2]]

    ax.plot(xs, ys, zs, marker="o")

    # Plot target point (if any)
    if target is not None:
        ax.scatter([target[0]], [target[1]], [target[2]], marker="^", s=50)

    # Axes labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set limits roughly based on arm length
    max_range = a2 + abs(d1) + 1.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max_range)

    set_axes_equal(ax)
    plt.show()


# ===============================
# CLI
# ===============================

def main():
    print("=== 2-DOF Manipulators – FK, IK & 3D Visualization ===")
    print("Robot types:")
    print("  [1] 2D planar 2-link (RR)")
    print("  [2] 3D vertical 2-link (a2, d1)")

    robot_choice = input("Choose robot type [1/2]: ").strip()

    if robot_choice not in ("1", "2"):
        print("Unknown robot type, please choose 1 or 2.")
        return

    mode = input("Choose mode ([f]orward / [i]nverse): ").strip().lower()

    # ---------- 2D PLANAR ----------
    if robot_choice == "1":
        print("\n--- 2D planar 2-link (RR) ---")
        l1 = float(input("Enter link 1 length l1: "))
        l2 = float(input("Enter link 2 length l2: "))

        if mode.startswith("f"):
            print("\n--- Forward kinematics (2D) ---")
            theta1 = float(input("Enter θ1 (degrees): "))
            theta2 = float(input("Enter θ2 (degrees): "))

            x, y = forward_kinematics_2d(theta1, theta2, l1, l2)
            print(f"\nEnd-effector position:")
            print(f"  x = {x:.4f}")
            print(f"  y = {y:.4f}")

        elif mode.startswith("i"):
            print("\n--- Inverse kinematics (2D) ---")
            x = float(input("Enter target x: "))
            y = float(input("Enter target y: "))

            try:
                sols = inverse_kinematics_2d(x, y, l1, l2)
            except ValueError as e:
                print(f"\n[Error] {e}")
                return

            print("\nPossible joint solutions (θ1, θ2 in degrees):")
            print("  Elbow-down: θ1 = {:.2f}°, θ2 = {:.2f}°".format(*sols[0]))
            print("  Elbow-up  : θ1 = {:.2f}°, θ2 = {:.2f}°".format(*sols[1]))

            for name, (th1, th2) in zip(["Elbow-down", "Elbow-up"], sols):
                xx, yy = forward_kinematics_2d(th1, th2, l1, l2)
                print(f"\nCheck FK for {name}: FK(x, y) = ({xx:.4f}, {yy:.4f})")

        else:
            print("Unknown mode; please choose 'f' or 'i'.")

    # ---------- 3D VERTICAL ----------
    else:
        print("\n--- 3D vertical 2-link (a2, d1) ---")
        a2 = float(input("Enter link length a2: "))
        d1 = float(input("Enter base offset d1: "))

        if mode.startswith("f"):
            print("\n--- Forward kinematics (3D) ---")
            theta1 = float(input("Enter θ1 (deg, rotation about z): "))
            theta2 = float(input("Enter θ2 (deg, vertical angle): "))

            x, y, z = forward_kinematics_3d(theta1, theta2, a2, d1)
            print(f"\nEnd-effector position:")
            print(f"  x = {x:.4f}")
            print(f"  y = {y:.4f}")
            print(f"  z = {z:.4f}")

            do_plot = input("Visualize this configuration in 3D? [y/N]: ").strip().lower()
            if do_plot == "y":
                plot_3d_arm(theta1, theta2, a2, d1, target=(x, y, z))

        elif mode.startswith("i"):
            print("\n--- Inverse kinematics (3D) ---")
            x = float(input("Enter target x: "))
            y = float(input("Enter target y: "))
            z = float(input("Enter target z: "))

            try:
                theta1_deg, theta2_deg = inverse_kinematics_3d(x, y, z, a2, d1)
            except ValueError as e:
                print(f"\n[Error] {e}")
                return

            print("\nJoint solution (θ1, θ2 in degrees):")
            print(f"  θ1 = {theta1_deg:.2f}°")
            print(f"  θ2 = {theta2_deg:.2f}°")

            xx, yy, zz = forward_kinematics_3d(theta1_deg, theta2_deg, a2, d1)
            print(f"\nCheck FK: FK(x, y, z) = ({xx:.4f}, {yy:.4f}, {zz:.4f})")

            do_plot = input("Visualize this configuration (with target) in 3D? [y/N]: ").strip().lower()
            if do_plot == "y":
                plot_3d_arm(theta1_deg, theta2_deg, a2, d1, target=(x, y, z))

        else:
            print("Unknown mode; please choose 'f' or 'i'.")


if __name__ == "__main__":
    main()
