import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pathlib import Path

COLORS = ['r', 'g', 'b']


def plot_ocp(ocp_data: dict, constraints=False, animate=False, folder=None):
    """Plot data from OCP solution dictionary."""
    if not constraints:
        f_constr = a_constr = None
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

    f_states, a_states = plot_states_cf(t, cf_p, cf_v, cf_a)
    plot_states_pl(t, pl_p, pl_v, pl_p_ref,
                   fig=f_states, axes=a_states)
    if constraints:
        f_constr, a_constr = plot_constraints(cf_p, pl_p, cf_cable_t, cable_l)
    f_3d, a_3d = plot_3d(cf_p, pl_p, pl_p_ref, cf_radius)

    set_3d_axis(fig=f_3d, axes=a_3d)

    if animate:
        animate_ocp(ocp_data)
    if folder is not None:
        save_plots(f_states, f_constr, f_3d, folder)

    return f_states, a_states, f_constr, a_constr, f_3d, a_3d


def plot_xyz(data, t_offset=0.0, t_total=None, fig=None, axes=None):
    """Plot x,y,z components of a trajectory."""
    if fig is None or axes is None:
        fig, axes = plt.subplots(3, 3, sharex=True, figsize=(20, 12))
    y_labels = ["x", "y", "z"]
    plot_labels = ["cf1", "cf2", "cf3"] if t_offset == 0.0 else ["cf1 (bag)", "cf2 (bag)", "cf3 (bag)"]
    t = data["t"]
    cf_p = np.stack([data["cf1_p"], data["cf2_p"], data["cf3_p"]])
    if cf_p.shape == (3, 3, t.shape[0]):
        pass  # already (cf, axis, time)
    elif cf_p.shape == (3, t.shape[0], 3):
        cf_p = cf_p.transpose(0, 2, 1)
    for i in range(3):
        for j in range(3):  # 3 drones
            axes[i, j].plot(t - t_offset, cf_p[j, i, :],
                         label=plot_labels[j], linewidth=1)
            axes[i, j].set_ylabel(f"Position {y_labels[i]} [m]")
            axes[i, j].grid(True)
            axes[i, j].legend()
            axes[2, j].set_xlabel("Time [s]")
            if t_total is not None:
                axes[2, j].set_xlim(0.0, t_total)
    fig.tight_layout()
    return fig, axes


def save_plots(f_states, f_constr, f_3d, folder):
    if f_states is not None:
        f_states.savefig(folder / "cf_plot.png")
    if f_constr is not None:
        f_constr.savefig(folder / "cf_constraints.png")
    if f_3d is not None:
        f_3d.savefig(folder / "cf_3d.png")
    print("Plots saved to:", folder)


def plot_states_cf(t, cf_p, cf_v=None, cf_a=None, linestyle='-',
                   fig=None, axes=None):
    """Plot drone states."""
    if fig is None or axes is None:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 14))
    axes[0].set_ylabel("Altitude [m]")
    axes[1].set_ylabel("Speed [m/s]")
    axes[2].set_ylabel("Acceleration [m/s]")
    # axes[3, 0].set_ylabel("Jerk [m/s]")
    # axes[3, 0].set_xlabel("Time [s]")
    axes[2].set_xlabel("Time [s]")

    for i in range(3):
        if len(cf_p[i, :, :]) == 0:
            continue

        axes[0].plot(
            t, cf_p[i, 2, :], label=f"Drone {i+1}",
            color=COLORS[i], linestyle=linestyle, linewidth=1)

        if cf_v is not None:
            speeds = np.linalg.norm(cf_v[i, :, :], axis=0)
            axes[1].plot(
                t, speeds, label=f"Drone {i+1}",
                color=COLORS[i], linestyle=linestyle, linewidth=1)
        if cf_a is not None:
            accels = np.linalg.norm(cf_a[i, :, :], axis=0)
            axes[2].plot(
                t[:-1], accels, label=f"Drone {i+1}",
                color=COLORS[i], linestyle=linestyle, linewidth=1)

            # jerks = np.linalg.norm(
            #     np.gradient(cf_a[i, :, :], t[:-1], axis=1), axis=0)
            # axes[3, 0].plot(
            #     t[:-1], jerks, label=f"Drone {i+1}",
            #     color=COLORS[i], linestyle=linestyle, linewidth=1)

    for ax in axes.flatten():
        ax.grid(True)
        if len(ax.get_lines()) > 0:
            ax.legend()

    fig.tight_layout()
    return fig, axes


def plot_states_pl(t, pl_p, pl_v=None, pl_p_ref=None, linestyle='-', fig=None, axes=None):
    """Plot payload states."""
    if fig is None or axes is None:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 14))
    axes[0].plot(
        t, pl_p[2, :], label="payload",
        color='k', linestyle=linestyle, linewidth=1)

    if pl_v is not None:
        speeds = np.linalg.norm(pl_v[:, :], axis=0)
    else:
        speeds = np.linalg.norm(np.gradient(pl_p, t, axis=1), axis=0)
    axes[1].plot(
        t, speeds, label="payload",
        color='k', linestyle=linestyle, linewidth=1)

    accels = np.gradient(speeds, t, axis=0)
    axes[2].plot(
        t, accels, label="payload",
        color='k', linestyle=linestyle, linewidth=1)

    # jerks = np.gradient(accels, t, axis=0)
    # axes[3].plot(
    #     t, jerks, label="payload",
    #     color='k', linestyle=linestyle, linewidth=1)

    # payload reference trajectory
    if pl_p_ref is not None:
        axes[0].plot(
            t, pl_p_ref[2, :],
            color='gray', linestyle='-.', label="payload ref")

    for ax in axes.flatten():
        ax.grid(True)
        if len(ax.get_lines()) > 0:
            ax.legend()

    fig.tight_layout()
    return fig, axes


def plot_constraints(cf_p, pl_p, cf_cable_t, cable_l, fig=None, axes=None):
    """Plot constraints from OCP solution dictionary."""
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(16, 10))

    N = min(cf_cable_t.shape[1], cf_p.shape[2], pl_p.shape[1])

    cf_cable_t = cf_cable_t[:, :N]
    cf_p = cf_p[:, :, :N]
    pl_p = pl_p[:, :N]

    # cable tension
    axes[0, 0].set_ylabel(f"Cable tension [N]")
    axes[0, 0].set_xlabel("Time [s]")
    for i in range(3):
        axes[0, 0].plot(cf_cable_t[i, :], label=f"Drone {i+1}", color=COLORS[i])
    axes[0, 0].axhline(0.05, color='gray',
                           linestyle='--', linewidth=1, label='min tension')
    axes[0, 0].axhline(0.15, color='gray',
                           linestyle='--', linewidth=1, label='max tension')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # cable tension z component
    axes[1, 0].set_ylabel(f"Cable tension z [N]")
    axes[1, 0].set_xlabel("Time [s]")
    for i in range(3):
        axes[1, 0].plot(
            cf_cable_t[i, :] * (cf_p[i, 2, :] - pl_p[2, :]) / (cable_l),
            label=f"Drone {i+1}", color=COLORS[i])
    axes[1, 0].axhline(0.15, color='gray',
                           linestyle='--', linewidth=1, label='max tension')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # cable angle
    axes[0, 1].set_ylabel(f"Cable angle [deg]")
    axes[0, 1].set_xlabel("Time [s]")
    z_axis = np.array([0.0, 0.0, 1.0])
    for i in range(3):
        cable = cf_p[i, :, :] - pl_p[:, :]
        cos_th = (cable.T @ z_axis) / (cable_l)
        angles = np.degrees(np.arccos(cos_th))
        axes[0, 1].plot(angles, label=f"Drone {i+1}", color=COLORS[i])
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # drone collision
    axes[1, 1].set_ylabel(f"Drone min distance [m]")
    axes[1, 1].set_xlabel("Time [s]")
    for i in range(3):
        pos_cf_cf = np.linalg.norm(cf_p[(i+1) % 3, :, :] - cf_p[i, :, :],
                                   axis=0)
        axes[1, 1].plot(
            pos_cf_cf,
            label=f"Drone {i+1} to Drone {(i+2)%3 +1}", color=COLORS[i])
    axes[1, 1].axhline(0.4, color='gray',
                           linestyle='--', linewidth=1, label='min distance')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    fig.suptitle("OCP Constraints")
    fig.tight_layout()

    return fig, axes


def plot_3d(cf_p, pl_p, pl_p_ref, cf_radius, fig=None, axes=None):
    """Plot 3D trajectory of drones and payload."""
    if fig is None or axes is None:
        fig = plt.figure(figsize=(12, 12))
        axes = fig.add_subplot(111, projection="3d")

    for i in range(3):
        axes.plot([cf_p[i, 0, -1], pl_p[0, -1]],
                  [cf_p[i, 1, -1], pl_p[1, -1]],
                  [cf_p[i, 2, -1], pl_p[2, -1]],
                  'k--', linewidth=1)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = cf_radius * np.cos(u) * np.sin(v) + cf_p[i, 0, -1]
        y = cf_radius * np.sin(u) * np.sin(v) + cf_p[i, 1, -1]
        z = cf_radius * np.cos(v) + cf_p[i, 2, -1]
        axes.plot_surface(x, y, z, color=COLORS[i], alpha=0.3)

    plot_3d_cf(cf_p, fig=fig, axes=axes)

    # payload reference trajectory
    axes.scatter(
        pl_p_ref[0, 0], pl_p_ref[1, 0], pl_p_ref[2, 0],
        color='orange', s=10, label="start")
    axes.scatter(
        pl_p_ref[0, -1], pl_p_ref[1, -1], pl_p_ref[2, -1],
        color='purple', s=10, label="goal")

    for p, label, color, linestyle in [(pl_p, 'payload', 'k', '-'),
                                       (pl_p_ref, 'payload ref', 'gray', '-.')]:
        plot_3d_pl(p, label=label, color=color, linestyle=linestyle,
                   fig=fig, axes=axes)
    return fig, axes


def plot_3d_cf(cf_p, linestyle='-', label_suffix='', fig=None, axes=None):
    """Plot 3D trajectory of drones."""
    if fig is None or axes is None:
        fig = plt.figure(figsize=(12, 12))
        axes = fig.add_subplot(111, projection="3d")
    for i in range(3):
        axes.plot(
            cf_p[i, 0], cf_p[i, 1], cf_p[i, 2],
            label=f"drone {i+1}{label_suffix}", color=COLORS[i], linestyle=linestyle)


def plot_3d_pl(pl_p, label='payload', color='k', linestyle='-', fig=None, axes=None):
    """Plot 3D trajectory of payload."""
    if fig is None or axes is None:
        fig = plt.figure(figsize=(12, 12))
        axes = fig.add_subplot(111, projection="3d")
    axes.plot(
        pl_p[0],  pl_p[1],  pl_p[2],
        label=label, color=color, linestyle=linestyle)


def set_3d_axis(fig, axes):
    x0, x1 = axes.get_xlim3d()
    y0, y1 = axes.get_ylim3d()
    z0, z1 = axes.get_zlim3d()
    xc, yc, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)

    s = max(sx, sy, sz)
    hx = hy = hz = s / 2

    axes.set_xlim3d(xc - hx, xc + hx)
    axes.set_ylim3d(yc - hy, yc + hy)
    axes.set_zlim3d(zc - hz, zc + hz)
    axes.set_box_aspect((1, 1, 1))

    axes.set_xlabel("X [m]")
    axes.set_ylabel("Y [m]")
    axes.set_zlabel("Z [m]")
    axes.legend()
    fig.tight_layout()


def animate_ocp(ocp_data: dict):
    t = ocp_data["t"]
    interval = 100

    pl_p_ref = ocp_data["pl_p_ref"]
    pl_p = ocp_data["pl_p"]
    cf_p = np.stack([ocp_data["cf1_p"],
                     ocp_data["cf2_p"],
                     ocp_data["cf3_p"]])

    cf_radius = ocp_data["cf_radius"]

    M = pl_p.shape[1]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # trajectory lines
    pl_ref_line, = ax.plot([], [], [], label="payload ref", color='gray', linestyle='-.', linewidth=1)
    pl_line, = ax.plot([], [], [], label="payload", color='k', linestyle='-', linewidth=1)
    cf_lines = [ax.plot([], [], [],label=f"cf{i+1}", color=COLORS[i], linestyle='-', linewidth=1)[0] for i in range(3)]

    # current markers
    pl_point, = ax.plot([], [], [], "ko", markersize=10)
    cf_points = [ax.plot([], [], [], "o", markersize=10, color=COLORS[i])[0] for i in range(3)]

    # cables
    cables = [ax.plot([], [], [], "k--")[0] for _ in range(3)]

    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # axis limits
    all_pts = np.concatenate([pl_p, cf_p.reshape(-1, M)], axis=0)
    lim = np.max(np.abs(all_pts)) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, lim)

    # labels
    title = ax.set_title("t = 0.00 s")
    cf_labels = [
        ax.text(0, 0, 0, f"cf_{i+1}", fontsize=10, color="k")
        for i in range(3)]

    def update(k):
        kk = k + 1

        title.set_text(f"t = {t[k]:.2f} s")

        # payload trail
        pl_line.set_data(pl_p[0, :kk], pl_p[1, :kk])
        pl_line.set_3d_properties(pl_p[2, :kk])

        # payload reference trail
        pl_ref_line.set_data(pl_p_ref[0, :kk], pl_p_ref[1, :kk])
        pl_ref_line.set_3d_properties(pl_p_ref[2, :kk])

        pl_point.set_data([pl_p[0, k]], [pl_p[1, k]])
        pl_point.set_3d_properties([pl_p[2, k]])

        for i in range(3):
            # drone trails
            cf_lines[i].set_data(cf_p[i, 0, :kk], cf_p[i, 1, :kk])
            cf_lines[i].set_3d_properties(cf_p[i, 2, :kk])

            # drone points
            x = cf_p[i, 0, k]
            y = cf_p[i, 1, k]
            z = cf_p[i, 2, k]

            cf_points[i].set_data([x], [y])
            cf_points[i].set_3d_properties([z])

            # label next to drone
            cf_labels[i].set_position((x, y))
            cf_labels[i].set_3d_properties(z, zdir='z')

            # cable
            cables[i].set_data([pl_p[0, k], x], [pl_p[1, k], y])
            cables[i].set_3d_properties([pl_p[2, k], z])

        return [
            pl_line, pl_point,
            *cf_lines, *cf_points,
            *cables, *cf_labels,
            title
        ]

    anim = FuncAnimation(
        fig,
        update,
        frames=M,
        interval=interval)

    plt.show()
    return anim


def plot_bag(bag_data: dict, t_offset=0.0, t_total=10.0, cable_l=0.5,
             fig=None, axes=None, fig_3d=None, axes_3d=None):
    """Plot data from rosbag dictionary."""
    if fig is None or axes is None:
        fig, axes = plt.subplots(4, 2, sharex=True, figsize=(16, 14))
        print("Warning: creating new figure and axes in plot_bag. This may cause multiple figures if called from plot_ocp.")
    if fig_3d is None or axes_3d is None:
        fig_3d = plt.figure(figsize=(12, 12))
        axes_3d = fig_3d.add_subplot(111, projection="3d")
        print("Warning: creating new 3d figure and axes in plot_bag. This may cause multiple figures if called from plot_ocp.")
    t = bag_data["t"]
    for ax in axes.flatten():
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

    plot_states_cf(
        t, cf_p, linestyle='--', fig=fig, axes=axes)
    pl_p = get_pl_pose(cf_p, cable_l)
    if pl_p.shape[1] > 1:
        plot_states_pl(t, pl_p, linestyle='--', fig=fig, axes=axes)
    plot_3d_cf(cf_p, linestyle='--', label_suffix=' (bag)', fig=fig_3d, axes=axes_3d)
    set_3d_axis(fig=fig_3d, axes=axes_3d)

    return fig, axes, fig_3d, axes_3d


def plot_error(ocp_data: dict, bag_data: dict, t_offset=0.0, t_total=10.0,
               fig=None, axes=None):
    """Plot error between OCP solution and rosbag data."""

    if fig is None or axes is None:
        fig, axes = plt.subplots(4, 2, sharex=True, figsize=(16, 14))
        print("Warning: creating new figure and axes in plot_error. This may cause multiple figures if called from plot_ocp.")

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

    plot_states_error(t_new, p_err, v_err, a_err, fig=fig, axes=axes)

    return fig, axes


def plot_states_error(t, p_error, v_error=None, a_error=None, fig=None, axes=None):
    """Plot error between OCP solution and rosbag data."""
    if fig is None or axes is None:
        fig, axes = plt.subplots(4, 2, sharex=True, figsize=(16, 14))
        print("Warning: creating new figure and axes in plot_states_error. This may cause multiple figures if called from plot_ocp.")
    axes[0, 1].set_ylabel(f"Position error [m]")
    axes[1, 1].set_ylabel(f"Velocity error [m/s]")
    axes[2, 1].set_ylabel(f"Acceleration error [m/s]")
    axes[2, 1].set_xlabel("Time [s]")

    for i in range(3):
        axes[0, 1].plot(t, p_error[i, :], label=f"Drone {i+1}", color=COLORS[i])
        if v_error is not None:
            axes[1, 1].plot(t, v_error[i, :], label=f"Drone {i+1}", color=COLORS[i])
        if a_error is not None:
            axes[2, 1].plot(t[:-1], a_error[i, :], label=f"Drone {i+1}", color=COLORS[i])

    fig.tight_layout()


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


def get_pl_math(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, r: float):
    """Get the lower intersection point of 3 spheres with radius r."""
    c0 = np.asarray(c0, dtype=float).reshape(3)
    c1 = np.asarray(c1, dtype=float).reshape(3)
    c2 = np.asarray(c2, dtype=float).reshape(3)

    # Two plane equations: (c1-c0)·x = (||c1||^2 - ||c0||^2)/2 and same for c2
    n1 = c1 - c0
    n2 = c2 - c0

    # Check degeneracy (centers nearly collinear / coincident)
    eps = 1e-12
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
