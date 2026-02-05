import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


def plot_states(t, pl_p, pl_v, cf_p, cf_v, cf_a, cf_t, pl_p_ref, L, cf_collision):
    """Plot payload and drone states."""
    figures_path = Path.home() / ".ros/crazyflo_planner" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    colors = ['r', 'g', 'b']

    # cf info
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(16, 10))
    axes[0, 0].set_ylabel(f"Altitude [m]")
    axes[1, 0].set_ylabel(f"Speed drone [m/s]")
    axes[2, 0].set_ylabel(f"Acceleration drone [m/s]")
    axes[0, 1].set_ylabel(f"Cable tension [N]")
    axes[1, 1].set_ylabel(f"Cable angle [deg]")
    axes[2, 1].set_ylabel(f"[m]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    for i in range(3):
        axes[0, 0].plot(t, cf_p[i, 2, :], label=f"Drone {i+1}", color=colors[i])

        speeds = np.linalg.norm(cf_v[i, :, :], axis=0)
        axes[1, 0].plot(t, speeds, label=f"Drone {i+1}", color=colors[i])

        accels = np.linalg.norm(cf_a[i, :, :], axis=0)
        axes[2, 0].plot(t[:-1], accels, label=f"Drone {i+1}", color=colors[i])

        axes[0, 1].plot(t[:-1], cf_t[i, :], label=f"Drone {i+1}", color=colors[i])

        angles = np.arccos(-cf_p[i, 2, :] + pl_p[2, :])  # angle from vertical
        angles *= 180.0 / np.pi  # to degrees
        axes[1, 1].plot(t, angles, label=f"Drone {i+1}", color=colors[i])
        # pos_errors = np.linalg.norm(cf_p[i, :, :] - pl_p, axis=0)
        # axes[2, 1].plot(t, pos_errors, label=f"Drone {i+1} to Payload")

        pos_cf_cf = np.linalg.norm(cf_p[(i+1) % 3, :, :] - cf_p[i, :, :], axis=0)
        axes[2, 1].plot(t, pos_cf_cf, label=f"Drone {i+1} to Drone {(i+2)%3 +1}", color=colors[i])

    axes[0, 0].plot(t, pl_p[2, :], 'k--', label="Payload")
    axes[1, 0].plot(t, np.linalg.norm(pl_v, axis=0), 'k--', label="Payload")
    axes[2, 1].plot(t, np.linalg.norm(pl_p - pl_p_ref, axis=0), '--',
                    color='gray', label="Payload ref. error")

    for ax in axes.flatten():
        ax.grid()
        ax.legend()
    plt.tight_layout()

    fig.savefig(figures_path / "cf_plot.png")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    ax.plot(pl_p[0],  pl_p[1],  pl_p[2],  label="payload", linewidth=3)
    ax.plot(cf_p[0, 0], cf_p[0, 1], cf_p[0, 2], label="drone 1", color=colors[0])
    ax.plot(cf_p[1, 0], cf_p[1, 1], cf_p[1, 2], label="drone 2", color=colors[1])
    ax.plot(cf_p[2, 0], cf_p[2, 1], cf_p[2, 2], label="drone 3", color=colors[2])

    # payload reference trajectory
    ax.scatter(pl_p_ref[0, 0], pl_p_ref[1, 0], pl_p_ref[2, 0],
               color='green', s=100, label="start")
    ax.scatter(pl_p_ref[0, -1], pl_p_ref[1, -1], pl_p_ref[2, -1],
               color='red', s=100, label="goal")
    ax.plot(pl_p_ref[0, :], pl_p_ref[1, :], pl_p_ref[2, :], '--',
            color='gray', label="reference")

    # plot cables at final time and spheres at drone positions
    for i in range(3):
        ax.plot([cf_p[i, 0, -1], pl_p[0, -1]],
                [cf_p[i, 1, -1], pl_p[1, -1]],
                [cf_p[i, 2, -1], pl_p[2, -1]],
                'k--', linewidth=1)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = cf_collision / 2.0 * np.cos(u) * np.sin(v) + cf_p[i, 0, -1]
        y = cf_collision / 2.0 * np.sin(u) * np.sin(v) + cf_p[i, 1, -1]
        z = cf_collision / 2.0 * np.cos(v) + cf_p[i, 2, -1]
        ax.plot_surface(x, y, z, color=colors[i], alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    set_axis(ax)

    plt.tight_layout()

    fig.savefig(figures_path / "cf_3d.png")
    plt.show()


def set_axis(ax):
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    xc, yc, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)

    s = max(sx, sy, sz)
    hx = hy = hz = s / 2

    ax.set_xlim3d(xc - hx, xc + hx)
    ax.set_ylim3d(yc - hy, yc + hy)
    ax.set_zlim3d(zc - hz, zc + hz)
    ax.set_box_aspect((1, 1, 1))


def plot_data(ocp_data):
    """Plot data from OCP solution dictionary."""
    t = ocp_data["t"]
    pl_p = ocp_data["pl_p"]
    pl_v = ocp_data["pl_v"]
    cf1_p = ocp_data["cf1_p"]
    cf2_p = ocp_data["cf2_p"]
    cf3_p = ocp_data["cf3_p"]
    cf1_v = ocp_data["cf1_v"]
    cf2_v = ocp_data["cf2_v"]
    cf3_v = ocp_data["cf3_v"]
    cf1_a = ocp_data["cf1_a"]
    cf2_a = ocp_data["cf2_a"]
    cf3_a = ocp_data["cf3_a"]
    cf1_t = ocp_data["cf1_t"]
    cf2_t = ocp_data["cf2_t"]
    cf3_t = ocp_data["cf3_t"]
    cf1_d = ocp_data["cf1_d"]
    cf2_d = ocp_data["cf2_d"]
    cf3_d = ocp_data["cf3_d"]
    cf_p = np.array([cf1_p, cf2_p, cf3_p])
    cf_v = np.array([cf1_v, cf2_v, cf3_v])
    cf_a = np.array([cf1_a, cf2_a, cf3_a])
    cf_t = np.array([cf1_t, cf2_t, cf3_t])
    pl_p_ref = ocp_data["pl_p_ref"]
    L = ocp_data["L"]
    cf_collision = ocp_data["cf_collision"]

    pl_p_ref = pl_p_ref.T  # (3, N+1)

    plot_states(t, pl_p, pl_v, cf_p, cf_v, cf_a, cf_t, pl_p_ref, L, cf_collision)


if __name__ == "__main__":
    ocp_data = np.load("crazyflo_planner/data/ocp_solution.npz")
    plot_data(ocp_data)
