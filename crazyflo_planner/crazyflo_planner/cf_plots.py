import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import cf_data

# Create figures and axes
f_states, a_states = plt.subplots(3, 2, sharex=True, figsize=(16, 10))

f_constr, a_constr = plt.subplots(2, 2, sharex=True, figsize=(16, 10))

f_3d = plt.figure(figsize=(12, 12))
a_3d = f_3d.add_subplot(projection="3d")

# Load OCP data
ocp_path = Path.home() / ".ros/crazyflo_planner" / "data" / "ocp_solution.npz"
ocp_data = np.load(ocp_path)

# Load rosbag data
ws_path = Path.home() /"winter-project/ws/bag"
bag_path = ws_path / "pose1557"
bag_data = cf_data.get_bag_data(bag_path)

# offset and total time for plots
t_offset = 11.0  # offset to align with ocp solution
t_total = 10.0
adjust_time_scale = True
plot_pl_only = False  # only plot payload trajectory without drones

# Save figures path
figures_path = Path.home() / ".ros/crazyflo_planner" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

colors = ['r', 'g', 'b']


def plot_states_cf(t, cf_p, cf_v=None, cf_a=None, linestyle='-'):
    """Plot drone states."""
    a_states[0, 0].set_ylabel(f"Altitude [m]")
    a_states[1, 0].set_ylabel(f"Speed drone [m/s]")
    a_states[2, 0].set_ylabel(f"Acceleration drone [m/s]")
    a_states[2, 0].set_xlabel("Time [s]")

    for i in range(3):
        a_states[0, 0].plot(t, cf_p[i, 2, :], label=f"Drone {i+1}", color=colors[i], linestyle=linestyle)

        if cf_v is not None:
            speeds = np.linalg.norm(cf_v[i, :, :], axis=0)
            a_states[1, 0].plot(t, speeds, label=f"Drone {i+1}", color=colors[i], linestyle=linestyle)
        if cf_a is not None:
            accels = np.linalg.norm(cf_a[i, :, :], axis=0)
            a_states[2, 0].plot(t[:-1], accels, label=f"Drone {i+1}", color=colors[i], linestyle=linestyle)

    for ax in a_states.flatten():
        ax.grid()
        ax.legend()

    f_states.tight_layout()


def plot_states_pl(t, pl_p, pl_v=None, pl_p_ref=None, linestyle='-'):
    """Plot payload states."""
    a_states[0, 0].plot(t, pl_p[2, :], label="payload", color='k', linestyle=linestyle)

    if pl_v is not None:
        speeds = np.linalg.norm(pl_v[:, :], axis=0)
    else:
        speeds = np.linalg.norm(np.gradient(pl_p, t, axis=1), axis=0)
    a_states[1, 0].plot(t, speeds, label="payload", color='k', linestyle=linestyle)

    if pl_v is not None:
        accels = np.linalg.norm(np.gradient(pl_v, t, axis=1), axis=0)
    else:
        accels = np.linalg.norm(np.gradient(np.gradient(pl_p, t, axis=1), t, axis=1), axis=0)
    a_states[2, 0].plot(t, accels, label="payload", color='k', linestyle=linestyle)

    # payload reference trajectory
    if pl_p_ref is not None:
        a_states[0, 0].plot(t, pl_p_ref[2, :], '-.', label="payload ref", color='gray')

    for ax in a_states.flatten():
        ax.grid()
        ax.legend()

    f_states.tight_layout()


def plot_3d_cf(cf_p, linestyle='-', label_suffix=''):
    """Plot 3D trajectory of drones."""
    for i in range(3):
        a_3d.plot(cf_p[i, 0], cf_p[i, 1], cf_p[i, 2], label=f"drone {i+1}{label_suffix}", color=colors[i], linestyle=linestyle)


def plot_3d_pl(pl_p, label='payload', color='k', linestyle='-'):
    """Plot 3D trajectory of payload."""
    a_3d.plot(pl_p[0],  pl_p[1],  pl_p[2],  label=label, color=color, linestyle=linestyle)


def plot_3d(cf_p, pl_p, pl_p_ref, cf_radius):
    """Plot 3D trajectory of drones and payload."""
    for i in range(3):
        a_3d.plot([cf_p[i, 0, -1], pl_p[0, -1]],
                [cf_p[i, 1, -1], pl_p[1, -1]],
                [cf_p[i, 2, -1], pl_p[2, -1]],
                'k--', linewidth=1)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = cf_radius * np.cos(u) * np.sin(v) + cf_p[i, 0, -1]
        y = cf_radius * np.sin(u) * np.sin(v) + cf_p[i, 1, -1]
        z = cf_radius * np.cos(v) + cf_p[i, 2, -1]
        a_3d.plot_surface(x, y, z, color=colors[i], alpha=0.3)

    plot_3d_cf(cf_p)

    # payload reference trajectory
    a_3d.scatter(pl_p_ref[0, 0], pl_p_ref[1, 0], pl_p_ref[2, 0],
               color='orange', s=10, label="start")
    a_3d.scatter(pl_p_ref[0, -1], pl_p_ref[1, -1], pl_p_ref[2, -1],
               color='purple', s=10, label="goal")

    for p, label, color, linestyle in [(pl_p, 'payload', 'k', '-'), (pl_p_ref, 'payload ref', 'gray', '-.')]:
        plot_3d_pl(p, label=label, color=color, linestyle=linestyle)


def set_3d_axis():
    x0, x1 = a_3d.get_xlim3d()
    y0, y1 = a_3d.get_ylim3d()
    z0, z1 = a_3d.get_zlim3d()
    xc, yc, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)

    s = max(sx, sy, sz)
    hx = hy = hz = s / 2

    a_3d.set_xlim3d(xc - hx, xc + hx)
    a_3d.set_ylim3d(yc - hy, yc + hy)
    a_3d.set_zlim3d(zc - hz, zc + hz)
    a_3d.set_box_aspect((1, 1, 1))

    a_3d.set_xlabel("X [m]")
    a_3d.set_ylabel("Y [m]")
    a_3d.set_zlabel("Z [m]")
    a_3d.legend()
    f_3d.tight_layout()


def plot_constraints(cf_p, pl_p, cf_cable_t, cable_l):
    """Plot constraints from OCP solution dictionary."""
    # cable tension
    a_constr[0, 0].set_ylabel(f"Cable tension [N]")
    a_constr[0, 0].set_xlabel("Time [s]")
    for i in range(3):
        a_constr[0, 0].plot(cf_cable_t[i, :], label=f"Drone {i+1}", color=colors[i])
    a_constr[0, 0].grid()
    a_constr[0, 0].legend()

    # cable angle
    a_constr[0, 1].set_ylabel(f"Cable angle [deg]")
    a_constr[0, 1].set_xlabel("Time [s]")
    z_axis = np.array([0.0, 0.0, -1.0])  # "down" as vertical
    for i in range(3):
        cable = cf_p[i, :, :] - pl_p[:, :]          # (3, N)
        cable_norm = np.linalg.norm(cable, axis=0)  # (N,)

        # cos(theta) = (cable · z_axis) / ||cable||
        cos_th = (cable.T @ z_axis) / (cable_norm + 1e-12)  # (N,)

        # numerical safety
        cos_th = np.clip(cos_th, -1.0, 1.0)

        angles = np.degrees(np.arccos(cos_th))
        a_constr[0, 1].plot(angles, label=f"Drone {i+1}", color=colors[i])
    a_constr[0, 1].grid()
    a_constr[0, 1].legend()

    # drone collision
    a_constr[1, 0].set_ylabel(f"Drone min. distance [m]")
    a_constr[1, 0].set_xlabel("Time [s]")
    for i in range(3):
        pos_cf_cf = np.linalg.norm(cf_p[(i+1) % 3, :, :] - cf_p[i, :, :], axis=0)
        a_constr[1, 0].plot(pos_cf_cf, label=f"Drone {i+1} to Drone {(i+2)%3 +1}", color=colors[i])
    a_constr[1, 0].grid()
    a_constr[1, 0].legend()

    # # cable length
    # a_constr[1, 1].set_ylabel(f"Cable length error [m]")
    # a_constr[1, 1].set_xlabel("Time [s]")
    # for i in range(3):
    #     cable = cf_p[i, :, :] - pl_p[:, :]          # (3, N)
    #     cable_norm = np.linalg.norm(cable, axis=0)  # (N,)
    #     cable_error = cable_norm - cable_l
    #     a_constr[1, 1].plot(cable_error, label=f"Drone {i+1}", color=colors[i])
    # a_constr[1, 1].grid()
    # a_constr[1, 1].legend()

    f_constr.tight_layout()


def plot_states_error(t, p_error, v_error=None, a_error=None):
    """Plot error between OCP solution and rosbag data."""
    a_states[0, 1].set_ylabel(f"Position error [m]")
    a_states[1, 1].set_ylabel(f"Velocity error [m/s]")
    a_states[2, 1].set_ylabel(f"Acceleration error [m/s]")
    a_states[2, 1].set_xlabel("Time [s]")

    for i in range(3):
        a_states[0, 1].plot(t, p_error[i, :], label=f"Drone {i+1}", color=colors[i])
        if v_error is not None:
            a_states[1, 1].plot(t, v_error[i, :], label=f"Drone {i+1}", color=colors[i])
        if a_error is not None:
            a_states[2, 1].plot(t[:-1], a_error[i, :], label=f"Drone {i+1}", color=colors[i])

    f_states.tight_layout()


def resample(data: dict, t_new: np.ndarray) -> dict:
    """Resample data to uniform time grid with step dt."""
    t_old = data["t"]
    resampled_data = {"t": t_new}

    for key in data.keys():
        if key == "t":
            continue

        signal_old = data[key]
        if key == "cf1_a" or key == "cf2_a" or key == "cf3_a":
            signal_new = np.vstack([
                np.interp(t_new[:-1], t_old[:-1], signal_old[0, :]),
                np.interp(t_new[:-1], t_old[:-1], signal_old[1, :]),
                np.interp(t_new[:-1], t_old[:-1], signal_old[2, :])
            ])
        else:
            signal_new = np.vstack([
                np.interp(t_new, t_old, signal_old[0, :]),
                np.interp(t_new, t_old, signal_old[1, :]),
                np.interp(t_new, t_old, signal_old[2, :])
            ])
        resampled_data[key] = signal_new

    return resampled_data


def plot_ocp(ocp_data: dict):
    """Plot data from OCP solution dictionary."""
    t = ocp_data["t"]
    pl_p = ocp_data["pl_p"]
    pl_v = ocp_data["pl_v"]
    cf_p = np.stack([ocp_data["cf1_p"], ocp_data["cf2_p"], ocp_data["cf3_p"]])
    cf_v = np.stack([ocp_data["cf1_v"], ocp_data["cf2_v"], ocp_data["cf3_v"]])
    cf_a = np.stack([ocp_data["cf1_a"], ocp_data["cf2_a"], ocp_data["cf3_a"]])
    cf_cable_t = np.stack([ocp_data["cf1_cable_t"], ocp_data["cf2_cable_t"], ocp_data["cf3_cable_t"]])
    pl_p_ref = ocp_data["pl_p_ref"]
    cable_l = ocp_data["cable_l"]
    cf_radius = ocp_data["cf_radius"]

    plot_states_cf(t, cf_p, cf_v, cf_a)
    plot_states_pl(t, pl_p, pl_v, pl_p_ref)
    plot_constraints(cf_p, pl_p, cf_cable_t, cable_l)
    if not plot_pl_only:
        plot_3d(cf_p, pl_p, pl_p_ref, cf_radius)
    else:
        plot_3d_pl(pl_p_ref, label='payload ref', color='gray', linestyle='-.')
        plot_3d_pl(pl_p, label='payload', color='k', linestyle='-')


def plot_bag(bag_data: dict, t_offset=0.0, t_total=10.0, cable_l=0.5):
    """Plot data from rosbag dictionary."""

    t = bag_data["t"]
    if adjust_time_scale:
        for ax in a_states.flatten():
            ax.set_xlim(-1, t_total + 1)

        t -= t_offset
        # truncate data from 0 to t_total
        mask = (t >= 0) & (t <= t_total)
        t = t[mask]
        bag_data["t"] = t
        for key in bag_data.keys():
            if key == "t":
                continue
            bag_data[key] = bag_data[key][:, mask]

    cf_p = np.stack([bag_data["cf1_p"], bag_data["cf2_p"], bag_data["cf3_p"]])
    # cf_v = np.stack([bag_data["cf1_v"], bag_data["cf2_v"], bag_data["cf3_v"]])
    # cf_a = np.stack([bag_data["cf1_a"], bag_data["cf2_a"], bag_data["cf3_a"]])

    plot_states_cf(t, cf_p, linestyle='--')
    pl_p = get_pl_pose(cf_p, cable_l)
    plot_states_pl(t, pl_p, linestyle='--')
    if not plot_pl_only:
        plot_3d_cf(cf_p, linestyle='--', label_suffix=' (bag)')
    else:
        plot_3d_pl(pl_p, label='payload (bag)', color='k', linestyle='--')
    set_3d_axis()

    # if adjust_time_scale:
    #     for ax in a_states.flatten():
    #         ax.set_xlim(-1, t_total + 1)


def plot_error(ocp_data: dict, bag_data: dict, t_offset=0.0, t_total=10.0):
    """Plot error between OCP solution and rosbag data."""

    bag_min = {"t": bag_data["t"], "cf1_p": bag_data["cf1_p"], "cf2_p": bag_data["cf2_p"], "cf3_p": bag_data["cf3_p"]}
    bag_min["t"] = bag_min["t"] - t_offset
    ocp_min = {"t": ocp_data["t"], "cf1_p": ocp_data["cf1_p"], "cf2_p": ocp_data["cf2_p"], "cf3_p": ocp_data["cf3_p"],
            "cf1_v": ocp_data["cf1_v"], "cf2_v": ocp_data["cf2_v"], "cf3_v": ocp_data["cf3_v"],
            "cf1_a": ocp_data["cf1_a"], "cf2_a": ocp_data["cf2_a"], "cf3_a": ocp_data["cf3_a"]}

    t_ocp = np.asarray(ocp_data["t"], dtype=float)
    t_bag = np.asarray(bag_data["t"], dtype=float)

    dt = max(np.median(np.diff(t_bag)), np.median(np.diff(t_ocp)))
    t_new = np.arange(0, min(t_bag[-1], t_ocp[-1]), dt)

    bag_r = resample(bag_min, t_new)
    ocp_r = resample(ocp_min, t_new)

    p_bag = np.stack([bag_r["cf1_p"], bag_r["cf2_p"], bag_r["cf3_p"]], axis=0)
    p_ocp = np.stack([ocp_r["cf1_p"], ocp_r["cf2_p"], ocp_r["cf3_p"]], axis=0)

    p_err = np.linalg.norm(p_ocp - p_bag, axis=1)

    v_bag = np.stack([np.gradient(p_bag[i], t_new, axis=1) for i in range(3)], axis=0)

    if all(k in ocp_r for k in ["cf1_v", "cf2_v", "cf3_v"]):
        v_ocp = np.stack([ocp_r["cf1_v"], ocp_r["cf2_v"], ocp_r["cf3_v"]], axis=0)
    else:
        v_ocp = np.stack([np.gradient(p_ocp[i], t_new, axis=1) for i in range(3)], axis=0)

    v_err = np.linalg.norm(v_ocp - v_bag, axis=1)

    a_bag = np.stack([np.gradient(v_bag[i], t_new, axis=1) for i in range(3)], axis=0)

    if all(k in ocp_r for k in ["cf1_a", "cf2_a", "cf3_a"]):
        a_ocp = np.stack([ocp_r["cf1_a"], ocp_r["cf2_a"], ocp_r["cf3_a"]], axis=0)
        # align lengths
        n = min(a_ocp.shape[-1], a_bag.shape[-1])
        a_err = np.linalg.norm(a_ocp[..., :n] - a_bag[..., :n], axis=1)

        # time vector for accel error
        if a_ocp.shape[-1] == len(t_new) - 1:
            t_a = t_new[:-1]
            t_a = t_a[:n]
        else:
            t_a = t_new[:n]
    else:
        a_ocp = np.stack([np.gradient(v_ocp[i], t_new, axis=1) for i in range(3)], axis=0)
        a_err = np.linalg.norm(a_ocp - a_bag, axis=1)
        t_a = t_new

    plot_states_error(t_new, p_err, v_err, a_err)


def get_pl_pose(cf_p: np.ndarray, cable_l: float):
    """Get payload position from rosbag data using cable geometry."""
    cf_p = np.asarray(cf_p, dtype=float)
    if cf_p.shape[0] != 3 or cf_p.shape[1] != 3:
        raise ValueError(f"Expected cf_p shape (3,3,T), got {cf_p.shape}")

    T = cf_p.shape[2]
    pl_p = np.zeros((3, T), dtype=float)

    for i in range(T):
        c0 = cf_p[0, :, i]
        c1 = cf_p[1, :, i]
        c2 = cf_p[2, :, i]
        pl_p[:, i] = get_pl_math(c0, c1, c2, cable_l)

    return pl_p


def get_pl_math(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, r: float, eps: float = 1e-12):
    """
    Return the lower (smaller z) intersection point of 3 spheres with equal radius r.
    If no real intersection (due to noise), it returns the best-fit point on the line
    and clamps the sqrt to 0 (tangent).
    """
    c0 = np.asarray(c0, dtype=float).reshape(3)
    c1 = np.asarray(c1, dtype=float).reshape(3)
    c2 = np.asarray(c2, dtype=float).reshape(3)

    # Two plane equations: (c1-c0)·x = (||c1||^2 - ||c0||^2)/2 and same for c2
    n1 = c1 - c0
    n2 = c2 - c0

    # Check degeneracy (centers nearly collinear / coincident)
    if np.linalg.norm(n1) < eps or np.linalg.norm(n2) < eps or np.linalg.norm(np.cross(n1, n2)) < eps:
        raise ValueError("Degenerate sphere configuration (centers nearly collinear or coincident).")

    d1 = 0.5 * (np.dot(c1, c1) - np.dot(c0, c0))
    d2 = 0.5 * (np.dot(c2, c2) - np.dot(c0, c0))

    v = np.cross(n1, n2)
    v_norm2 = np.dot(v, v)

    N = np.vstack([n1, n2])      # (2,3)
    d = np.array([d1, d2])       # (2,)
    x0, *_ = np.linalg.lstsq(N, d, rcond=None)  # point on both planes

    w = x0 - c0
    a = v_norm2
    b = 2.0 * np.dot(v, w)
    c = np.dot(w, w) - r * r

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0

    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)

    p1 = x0 + t1 * v
    p2 = x0 + t2 * v

    return p1 if p1[2] <= p2[2] else p2


if __name__ == "__main__":

    cable_l = ocp_data["cable_l"]
    plot_ocp(ocp_data)
    plot_bag(bag_data, t_offset=t_offset, t_total=t_total, cable_l=cable_l)
    plot_error(ocp_data, bag_data, t_offset=t_offset, t_total=t_total)

    f_states.savefig(figures_path / "cf_plot.png")
    f_constr.savefig(figures_path / "cf_constraints.png")
    f_3d.savefig(figures_path / "cf_3d.png")

    plt.show()
