"""
PDE4431 Coursework 2 – Industrial Manipulator Kinematics Modelling
Author: YOUR NAME
Date: 2025

4-DOF RRPR Manipulator (Revolute–Revolute–Prismatic–Revolute)

- Custom Denavit–Hartenberg model (no external robotics toolbox)
- Analytical inverse kinematics for position control
- 3D stick-figure visualisation
- Floor with 3 objects (cubes)
- 3-level shelf
- Full block-stacking sequence:
    pick 3 cubes from floor and place onto 3 shelves
- Cube is carried along with the end-effector when "picked"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# -----------------------------------------------------------
# 1. Basic DH transform
# -----------------------------------------------------------

def dh_transform(a, alpha, d, theta):
    """
    Standard DH homogeneous transform.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,   -st*ca,  st*sa, a*ct],
        [st,    ct*ca, -ct*sa, a*st],
        [0.0,      sa,     ca,    d],
        [0.0,    0.0,   0.0,   1.0]
    ])

# -----------------------------------------------------------
# 2. RRPR Manipulator (custom DH)
# -----------------------------------------------------------

class RRPRManipulatorCustom:
    """
    4-DOF manipulator with joints:
      q1: Revolute  (base)
      q2: Revolute  (shoulder)
      q3: Prismatic (vertical lift)
      q4: Revolute  (wrist – orientation only, we keep it 0 in IK)

    DH Parameters (standard convention):

        i |  a_i   alpha_i   d_i       theta_i
        -----------------------------------------
        1 |  L1      0       0         q1
        2 |  L2      0       0         q2
        3 |  0       0       q3        0        (vertical lift)
        4 |  L4      0       0         q4

    Note:
    - All alpha_i = 0 → all joint axes are parallel (z-axis),
      so the arm moves in a planar XY workspace with vertical lift.
    """

    def __init__(self, L1=0.45, L2=0.35, L4=0.25):
        self.L1 = L1
        self.L2 = L2
        self.L4 = L4

        self.a =     [L1, L2, 0.0, L4]
        self.alpha = [0.0, 0.0, 0.0, 0.0]

        # Home configuration (nicely posed)
        self.q_home = np.array([
            np.deg2rad(25.0),   # q1
            np.deg2rad(40.0),   # q2
            0.20,               # q3 (vertical lift)
            0.0                 # q4
        ])

    # ------------- Forward Kinematics ---------------------

    def forward_kinematics(self, q):
        """
        Compute FK for q = [q1, q2, q3, q4].
        Returns 4x4 homogeneous transform of end-effector.
        """
        q1, q2, q3, q4 = q
        d =     [0.0, 0.0, q3, 0.0]
        theta = [q1,  q2,  0.0, q4]

        T = np.eye(4)
        for i in range(4):
            T = T @ dh_transform(self.a[i], self.alpha[i], d[i], theta[i])
        return T

    def joint_positions(self, q):
        """
        Return joint positions for plotting.
        Joints 0..4 (base + 4 joints).
        """
        q1, q2, q3, q4 = q
        d =     [0.0, 0.0, q3, 0.0]
        theta = [q1,  q2,  0.0, q4]

        Ts = [np.eye(4)]
        T = np.eye(4)
        for i in range(4):
            T = T @ dh_transform(self.a[i], self.alpha[i], d[i], theta[i])
            Ts.append(T)

        xs = [Ti[0,3] for Ti in Ts]
        ys = [Ti[1,3] for Ti in Ts]
        zs = [Ti[2,3] for Ti in Ts]
        return xs, ys, zs

    # ------------- Analytical IK (position only) ----------

    def ik_position(self, x, y, z, elbow_up=True):
        """
        Solve IK for a desired end-effector position p = (x, y, z).

        Approach:
        - Planar position (x, y) solved by 2R arm with link lengths:
              L1 and L_eq = L2 + L4
          (we treat the last revolute joint's offset as part of link 2).
        - Vertical position (z) is handled by the prismatic joint q3 = z.
        - We don't constrain orientation, so we set q4 = 0.

        Returns:
            q = [q1, q2, q3, q4]  or  None if unreachable.
        """
        L1 = self.L1
        L_eq = self.L2 + self.L4

        # radial distance in XY-plane
        r = np.hypot(x, y)

        # Check planar reach
        c2 = (r**2 - L1**2 - L_eq**2) / (2 * L1 * L_eq)
        if c2 < -1.0 or c2 > 1.0:
            print("IK: target (x,y) outside planar workspace")
            return None

        s2 = np.sqrt(max(0.0, 1.0 - c2**2))
        if not elbow_up:
            s2 = -s2

        q2 = np.arctan2(s2, c2)

        k1 = L1 + L_eq * c2
        k2 = L_eq * s2
        q1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        # Prismatic joint for vertical position
        q3 = z

        # Wrist orientation not used
        q4 = 0.0

        return np.array([q1, q2, q3, q4])


# -----------------------------------------------------------
# 3. Environment drawing (floor, cubes, shelf)
# -----------------------------------------------------------

def set_equal_3d(ax):
    """Make axes equal for proper 3D proportions."""
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xm, ym, zm = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    r = max(
        (xlim[1] - xlim[0]) / 2,
        (ylim[1] - ylim[0]) / 2,
        (zlim[1] - zlim[0]) / 2
    )
    ax.set_xlim3d([xm - r, xm + r])
    ax.set_ylim3d([ym - r, ym + r])
    ax.set_zlim3d([zm - r, zm + r])


def draw_floor(ax, size=1.5):
    s = size
    ax.plot([-s,s,s,-s,-s], [-s,-s,s,s,-s], [0,0,0,0,0],
            color='gray', linewidth=1)
    for g in np.linspace(-s, s, 7):
        ax.plot([-s,s], [g,g], [0,0], alpha=0.3)
        ax.plot([g,g], [-s,s], [0,0], alpha=0.3)


def draw_cube(ax, center, size=0.06, color='tab:orange'):
    cx, cy, cz = center
    s = size / 2.0
    corners = np.array([
        [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s],
        [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s],
        [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for e in edges:
        p1, p2 = corners[e[0]], corners[e[1]]
        ax.plot([p1[0],p2[0]],
                [p1[1],p2[1]],
                [p1[2],p2[2]],
                color=color, linewidth=1.3)


def draw_shelf(ax, origin, levels):
    x0, y0, z0 = origin
    width, depth = 0.5, 0.3
    x1, y1 = x0 + width, y0 - depth

    # Vertical posts
    for px, py in [(x0,y0),(x1,y0),(x0,y1),(x1,y1)]:
        ax.plot([px,px], [py,py],
                [z0, z0 + levels[-1] + 0.15],
                color='saddlebrown', linewidth=2)

    # Shelf boards
    for z in levels:
        ax.plot([x0,x1,x1,x0,x0],
                [y0,y0,y1,y1,y0],
                [z,z,z,z,z],
                color='peru', linewidth=2)


def draw_environment(ax, cubes, shelf_origin, shelf_levels):
    draw_floor(ax)
    for c in cubes:
        draw_cube(ax, c)
    draw_shelf(ax, shelf_origin, shelf_levels)


# -----------------------------------------------------------
# 4. Robot drawing & animation
# -----------------------------------------------------------

def plot_robot(ax, robot, q, env):
    ax.cla()
    draw_environment(ax, env['cubes'], env['shelf_origin'], env['shelf_levels'])

    xs, ys, zs = robot.joint_positions(q)
    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple']

    for i in range(4):
        ax.plot([xs[i],xs[i+1]],
                [ys[i],ys[i+1]],
                [zs[i],zs[i+1]],
                '-o', linewidth=3, markersize=7, color=colors[i])

    ax.scatter(xs[0], ys[0], zs[0], s=50, c='black')   # base
    ax.scatter(xs[-1], ys[-1], zs[-1], s=80, c='gold') # end-effector

    ax.view_init(elev=25, azim=135)
    set_equal_3d(ax)


def animate_motion(ax, robot, q_start, q_end, env, gui=None, steps=70):
    """
    Interpolate from q_start to q_end, updating the plot.
    If gui.carrying_object is not None, the selected cube
    is moved with the end-effector (carried).
    """
    for s in np.linspace(0.0, 1.0, steps):
        q = (1 - s) * q_start + s * q_end

        # If carrying a cube, attach it to the end-effector
        if gui is not None and gui.carrying_object is not None:
            T = robot.forward_kinematics(q)
            px, py, pz = T[0,3], T[1,3], T[2,3]
            gui.object_positions[gui.carrying_object][:] = [px, py, pz]

        plot_robot(ax, robot, q, env)
        plt.pause(0.02)

    return q_end


# -----------------------------------------------------------
# 5. GUI + Full Block-Stacking Sequence
# -----------------------------------------------------------

class ManipulatorGUI:
    def __init__(self):

        self.robot = RRPRManipulatorCustom()

        # Three floor cubes
        self.floor_objects = [
            np.array([0.55,  0.15, 0.035]),
            np.array([0.65,  0.00, 0.035]),
            np.array([0.55, -0.15, 0.035]),
        ]
        self.object_positions = [c.copy() for c in self.floor_objects]

        # Shelf definition
        self.shelf_origin = (0.9, 0.0, 0.0)
        self.shelf_levels = (0.30, 0.60, 0.90)

        self.env = {
            'cubes': self.object_positions,
            'shelf_origin': self.shelf_origin,
            'shelf_levels': self.shelf_levels
        }

        # Shelf target positions (EE positions)
        self.shelf_targets = [
            np.array([1.0, -0.15, 0.35]),
            np.array([1.0, -0.15, 0.65]),
            np.array([1.0, -0.15, 0.95]),
        ]

        # Named targets for manual buttons
        self.targets = {
            "Floor":  self.floor_objects[1] + np.array([0.0, 0.0, 0.07]),
            "Shelf1": self.shelf_targets[0],
            "Shelf2": self.shelf_targets[1],
            "Shelf3": self.shelf_targets[2],
        }

        # Precompute IK for button targets
        self.target_configs = {}
        for name, pos in self.targets.items():
            sol = self.robot.ik_position(*pos)
            if sol is None:
                print(f"[WARNING] IK failed for {name} at {pos}")
            self.target_configs[name] = sol

        self.q_current = self.robot.q_home.copy()

        # Index (0,1,2) of cube currently being carried
        self.carrying_object = None

        # Figure and axes
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)

        plot_robot(self.ax, self.robot, self.q_current, self.env)

        # Buttons
        axes = {
            "Floor":   [0.05, 0.05, 0.13, 0.07],
            "Shelf1":  [0.22, 0.05, 0.13, 0.07],
            "Shelf2":  [0.39, 0.05, 0.13, 0.07],
            "Shelf3":  [0.56, 0.05, 0.13, 0.07],
            "Seq":     [0.73, 0.05, 0.22, 0.07],
        }

        self.btn_floor = Button(plt.axes(axes["Floor"]),  "Floor")
        self.btn_s1    = Button(plt.axes(axes["Shelf1"]), "Shelf 1")
        self.btn_s2    = Button(plt.axes(axes["Shelf2"]), "Shelf 2")
        self.btn_s3    = Button(plt.axes(axes["Shelf3"]), "Shelf 3")
        self.btn_seq   = Button(plt.axes(axes["Seq"]),    "Play Sequence")

        self.btn_floor.on_clicked(lambda e: self.move_to("Floor"))
        self.btn_s1.on_clicked(lambda e: self.move_to("Shelf1"))
        self.btn_s2.on_clicked(lambda e: self.move_to("Shelf2"))
        self.btn_s3.on_clicked(lambda e: self.move_to("Shelf3"))
        self.btn_seq.on_clicked(self.full_sequence)

    # ---------------- Manual button motion ----------------

    def move_to(self, name):
        q_target = self.target_configs.get(name, None)
        if q_target is None:
            print(f"[ERROR] No IK solution for target '{name}'")
            return

        print(f"Moving to {name}: {self.targets[name]}")
        self.q_current = animate_motion(
            self.ax, self.robot,
            self.q_current, q_target,
            self.env, gui=self, steps=70
        )

    # ---------------- Full block-stacking seq --------------

    def full_sequence(self, event=None):
        print("\n--- Starting Full Block-Stacking Sequence ---")

        # 1. Go home
        self.q_current = animate_motion(
            self.ax, self.robot,
            self.q_current, self.robot.q_home,
            self.env, gui=self, steps=60
        )

        # 2. For each cube: pick from floor, place on corresponding shelf
        for i in range(3):
            print(f"\nPicking object {i+1}...")

            # Move to above cube i
            pick_pos = self.floor_objects[i] + np.array([0.0, 0.0, 0.07])
            q_pick = self.robot.ik_position(*pick_pos)
            if q_pick is None:
                print(f"[WARNING] Cannot reach cube {i+1}, skipping.")
                continue

            self.q_current = animate_motion(
                self.ax, self.robot,
                self.q_current, q_pick,
                self.env, gui=self, steps=60
            )

            # Attach cube to end-effector
            self.carrying_object = i

            # Move to corresponding shelf target
            place_pos = self.shelf_targets[i]
            q_place = self.robot.ik_position(*place_pos)
            if q_place is None:
                print(f"[WARNING] Cannot reach shelf {i+1} for place, skipping.")
                self.carrying_object = None
                continue

            print(f"Placing object {i+1} on shelf {i+1}...")
            self.q_current = animate_motion(
                self.ax, self.robot,
                self.q_current, q_place,
                self.env, gui=self, steps=60
            )

            # Detach cube at shelf
            self.carrying_object = None
            self.object_positions[i][:] = place_pos

        # 3. Return home
        print("\nReturning to home configuration...")
        self.q_current = animate_motion(
            self.ax, self.robot,
            self.q_current, self.robot.q_home,
            self.env, gui=self, steps=60
        )
        print("--- Sequence complete ---")

    # ------------------------------------------------------

    def run(self):
        plt.show()


# -----------------------------------------------------------
# 6. Main entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    gui = ManipulatorGUI()
    gui.run()
