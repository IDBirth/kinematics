#!/usr/bin/env python3
"""
PDE4431 – 2-DOF manipulators: 2D planar and 3D vertical

Implements Forward & Inverse Kinematics for:
1) 2D planar 2-link RR arm:
   - Link lengths: l1, l2
   - Joint angles: θ1, θ2 (in degrees)

   FK:
       x = l1*cos(θ1) + l2*cos(θ1 + θ2)
       y = l1*sin(θ1) + l2*sin(θ1 + θ2)

2) 3D vertical 2-link arm from Judhi's lecture:
   - DH parameters:
       Link 1: a1 = 0, α1 = 90°, d1 (offset), θ1 (rotation about z)
       Link 2: a2 (length), α2 = 0°, d2 = 0, θ2 (vertical angle)
   - End-effector equations (translation part of 0T2):
       x = a2*cos(θ1)*cos(θ2)
       y = a2*sin(θ1)*cos(θ2)
       z = a2*sin(θ2) + d1
"""

import numpy as np


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

    # Workspace check
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
# (Judhi's example with a2, d1)
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
        cos(θ2) = sqrt(1 - sin^2(θ2)), sign chosen so that cos(θ2) has same sign as ρ/a2
    """
    # Horizontal distance
    rho = np.sqrt(x**2 + y**2)
    dz = z - d1

    # First angle θ1 is just the azimuth in the x-y plane
    theta1 = np.arctan2(y, x)

    # From z equation: sin(θ2)
    s2 = dz / a2
    # Numerical safety
    s2 = np.clip(s2, -1.0, 1.0)

    # Compute cos(θ2) from the unit circle identity
    c2_mag = np.sqrt(1.0 - s2**2)

    # Decide sign of cos(θ2) based on ρ (horizontal reach).
    # If ρ is positive, we want cos(θ2) ≥ 0; if ρ is near zero, sign doesn't matter much.
    if rho >= 0:
        c2 = c2_mag
    else:
        c2 = -c2_mag

    theta2 = np.arctan2(s2, c2)

    return float(np.rad2deg(theta1)), float(np.rad2deg(theta2))


# ===============================
# CLI
# ===============================

def main():
    print("=== 2-DOF Manipulators – Forward & Inverse Kinematics ===")
    print("Robot types:")
    print("  [1] 2D planar 2-link (RR)")
    print("  [2] 3D vertical 2-link (DH example with a2, d1)")

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

            # Check FK
            for name, (th1, th2) in zip(["Elbow-down", "Elbow-up"], sols):
                xx, yy = forward_kinematics_2d(th1, th2, l1, l2)
                print(f"\nCheck FK for {name}: FK(x, y) = ({xx:.4f}, {yy:.4f})")

        else:
            print("Unknown mode; please choose 'f' or 'i'.")

    # ---------- 3D VERTICAL ----------
    else:
        print("\n--- 3D vertical 2-link (a2, d1) ---")
        # Use generic a2, d1 so you can set to the example values (5, 3)
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

            # Verify with FK
            xx, yy, zz = forward_kinematics_3d(theta1_deg, theta2_deg, a2, d1)
            print(f"\nCheck FK: FK(x, y, z) = ({xx:.4f}, {yy:.4f}, {zz:.4f})")

        else:
            print("Unknown mode; please choose 'f' or 'i'.")


if __name__ == "__main__":
    main()
