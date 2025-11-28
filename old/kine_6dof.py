"""
Computes the forward/inverse kinematics for a 6-DOF RRRRRR robot (position-only IK).
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------- FORWARD KINEMATICS ----------

def forward_kinematics_6dof(theta, l, alpha):
    """
    Forward kinematics for a 6-DOF RRRRRR robot.
    Geometry:
      - Joint 1: base rotation about z (theta[0])
      - Links 2..6: lie in a vertical plane defined by theta[0]
        with cumulative elevation angles (theta2..theta6 + twists).

    Args:
        theta : array-like, shape (6,)
            Joint angles [theta1..theta6] in radians.
        l     : array-like, shape (6,)
            Link lengths [l1..l6].
        alpha : array-like, shape (6,)
            Twist offsets [alpha1..alpha6] (alpha1 is unused in this simple model).

    Returns:
        xs, ys, zs : np.ndarray
            Arrays of joint positions, length 7 (base + 6 joints).
        (x_end, y_end, z_end) : tuple
            End-effector position (same as xs[-1], ys[-1], zs[-1]).
    """
    theta = np.asarray(theta, dtype=float)
    l     = np.asarray(l, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    assert theta.shape == (6,)
    assert l.shape     == (6,)
    assert alpha.shape == (6,)

    # Base and link 1
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = 0.0, 0.0, l[0]

    xs = [x0, x1]
    ys = [y0, y1]
    zs = [z0, z1]

    theta1 = theta[0]

    # Elevation starts from joint 2
    elev = theta[1]
    x_prev, y_prev, z_prev = x1, y1, z1

    # Links 2..6 (indices 1..5 in arrays)
    for i in range(1, 6):
        # angle in the vertical plane for this link
        angle = elev + alpha[i]

        dx = l[i] * np.cos(theta1) * np.cos(angle)
        dy = l[i] * np.sin(theta1) * np.cos(angle)
        dz = l[i] * np.sin(angle)

        x = x_prev + dx
        y = y_prev + dy
        z = z_prev + dz

        xs.append(x)
        ys.append(y)
        zs.append(z)

        x_prev, y_prev, z_prev = x, y, z

        # accumulate elevation for the next link
        if i < 5:
            elev += theta[i + 1]

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    return xs, ys, zs, (xs[-1], ys[-1], zs[-1])


def end_effector_position(theta, l, alpha):
    """Helper: just returns (x, y, z) of the end-effector."""
    _, _, _, (x, y, z) = forward_kinematics_6dof(theta, l, alpha)
    return np.array([x, y, z])


# ---------- INVERSE KINEMATICS (NUMERIC, POSITION ONLY) ----------

def inverse_kinematics_6dof(x_target, y_target, z_target,
                            l, alpha,
                            max_iters=2000,
                            tol=1e-6,
                            step_size=0.1):
    """
    Numeric IK (Jacobian transpose) for 6-DOF RRRRRR arm.
    We only match the end-effector position (x, y, z), not orientation.

    Args:
        x_target, y_target, z_target : float
            Desired end-effector position.
        l, alpha : arrays, shape (6,)
            Link lengths and twist offsets.
        max_iters : int
            Maximum number of iterations.
        tol : float
            Position error tolerance.
        step_size : float
            Gradient step size.

    Returns:
        theta : np.ndarray, shape (6,)
            Joint angles that approximately reach the target.

    Raises:
        ValueError: if the target is clearly outside the reachable workspace.
    """
    l     = np.asarray(l, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    target = np.array([x_target, y_target, z_target], dtype=float)

    # Simple reachability check based on total arm length (from joint 1)
    r_xy = np.sqrt(x_target**2 + y_target**2)
    z_plane = z_target - l[0]  # remove first link height
    dist = np.sqrt(r_xy**2 + z_plane**2)

    max_reach = np.sum(l[1:])  # links 2..6
    if dist > max_reach + 1e-6:
        raise ValueError("Target is outside the robot's reachable workspace (approx check).")

    # Initial guess: point base towards (x, y), small bends
    theta = np.zeros(6, dtype=float)
    theta[0] = np.arctan2(y_target, x_target)  # base rotation

    def numerical_jacobian(theta_curr, eps=1e-5):
        """
        Numeric 3x6 Jacobian of end-effector position w.r.t thetas.
        """
        p0 = end_effector_position(theta_curr, l, alpha)
        J = np.zeros((3, 6), dtype=float)
        for i in range(6):
            t_pert = theta_curr.copy()
            t_pert[i] += eps
            p1 = end_effector_position(t_pert, l, alpha)
            J[:, i] = (p1 - p0) / eps
        return J

    for _ in range(max_iters):
        p = end_effector_position(theta, l, alpha)
        error = target - p
        err_norm = np.linalg.norm(error)
        if err_norm < tol:
            break

        J = numerical_jacobian(theta)
        # Jacobian transpose update (this was the line with the typo)
        delta_theta = step_size * (J.T @ error)

        theta += delta_theta
        # Wrap angles to [-pi, pi] to keep them reasonable
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

    return theta


# ---------- VALIDATION ----------

def validate_solution_6dof(x_target, y_target, z_target,
                           theta, l, alpha, tol=1e-6):
    """
    Validates that the forward kinematics of the 6-DOF solution
    matches the target position within a tolerance.
    """
    x_end, y_end, z_end = end_effector_position(theta, l, alpha)
    error = np.sqrt((x_end - x_target)**2 +
                    (y_end - y_target)**2 +
                    (z_end - z_target)**2)
    if error < tol:
        print(f"Solution is valid! Error = {error:.6e}")
    else:
        print(f"Solution is NOT exact. Error = {error:.6e}")


# ---------- PLOTTING ----------

def plot_robot_6dof(theta, l, alpha,
                    x_target=None, y_target=None, z_target=None,
                    title="6-DOF RRRRRR Robot Configuration"):
    """
    Plots the 6-DOF robot configuration in 3D.

    Args:
        theta : array-like, shape (6,)
        l, alpha : arrays, shape (6,)
        x_target, y_target, z_target : optional floats
            A target point to mark in the plot.
    """
    xs, ys, zs, _ = forward_kinematics_6dof(theta, l, alpha)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot links
    ax.plot(xs, ys, zs, 'o-', label="Links")

    # Plot target (if given)
    if x_target is not None and y_target is not None and z_target is not None:
        ax.scatter([x_target], [y_target], [z_target],
                   color="orange", s=50, label="Target")

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Equal axis scaling
    x_range = [np.min(xs), np.max(xs)]
    y_range = [np.min(ys), np.max(ys)]
    z_range = [np.min(zs), np.max(zs)]
    if x_target is not None:
        x_range = [min(x_range[0], x_target), max(x_range[1], x_target)]
        y_range = [min(y_range[0], y_target), max(y_range[1], y_target)]
        z_range = [min(z_range[0], z_target), max(z_range[1], z_target)]

    max_range = max(np.ptp(x_range), np.ptp(y_range), np.ptp(z_range)) / 2.0
    x_mid = np.mean(x_range)
    y_mid = np.mean(y_range)
    z_mid = np.mean(z_range)

    ax.set_xlim([x_mid - max_range, x_mid + max_range])
    ax.set_ylim([y_mid - max_range, y_mid + max_range])
    ax.set_zlim([z_mid - max_range, z_mid + max_range])

    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    ax.set_title(title)
    plt.show()


# ---------- EXAMPLE USAGE ----------

if __name__ == "__main__":
    # Link lengths
    l = np.array([1.0,  1.0, 0.5, 0.3, 0.3, 0.2])

    # Joint twists (you can tune these; alpha1 unused in this model)
    alpha = np.radians([
        -90,   # alpha1 (not used in current geometry)
        0,     # alpha2
        0,     # alpha3
        0,     # alpha4
        0,     # alpha5
        0      # alpha6
    ])

    # Target position
    x_target, y_target, z_target = 1.0, -0.5, 1.5
    print("Target:", x_target, y_target, z_target)

    try:
        # Solve IK numerically
        theta_sol = inverse_kinematics_6dof(
            x_target, y_target, z_target,
            l, alpha,
            max_iters=2000,
            tol=1e-6,
            step_size=0.1
        )

        print("Joint solution (rad):", theta_sol)
        print("Joint solution (deg):", np.degrees(theta_sol))

        # Check FK
        x_act, y_act, z_act = end_effector_position(theta_sol, l, alpha)
        print("Actual end-effector:", x_act, y_act, z_act)

        # Validate numerically
        validate_solution_6dof(x_target, y_target, z_target,
                               theta_sol, l, alpha)

        # Plot configuration
        plot_robot_6dof(theta_sol, l, alpha,
                        x_target, y_target, z_target)

    except ValueError as e:
        print("IK Error:", e)
