import matplotlib.pyplot as plt
import numpy as np


def plot_states(t, p, vp, r, v, u, lam):
    """Plot payload and drone states."""
    fig, axes = plt.subplots(3, 3, sharex=True, figsize=(16, 8))
    labels = ['x', 'y', 'z']
    for i in range(3):
        axes[i, 0].grid()
        axes[i, 0].set_ylabel(f"Position {labels[i]} [m]")
        axes[i, 0].plot(t, p[i, :], label=f"Position {labels[i]} [m]")
        axes[i, 0].legend()

        axes[i, 1].grid()
        axes[i, 1].set_ylabel(f"Velocity {labels[i]} [m/s]")
        axes[i, 1].plot(t, vp[i, :], label=f"Velocity {labels[i]} [m/s]")
        axes[i, 1].legend()

        axes[i, 2].grid()
        axes[i, 2].set_ylabel(f"Tension {labels[i]} [N]")
        axes[i, 2].plot(t[:-1], lam[i, :], label=f"Tension {labels[i]} [N]")
        axes[i, 2].legend()

    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 2].set_xlabel("Time [s]")
    plt.tight_layout()

    fig.savefig("crazyflo_planner/figures/payload_states.png")

    # tension and cable angles
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 10))
    for i in range(3):
        axes[0].grid()
        axes[0].set_ylabel(f"Tension drone {i+1} [N]")
        axes[0].plot(t[:-1], lam[i, :], label=f"Drone {i+1}")
        axes[0].legend()

        angles = np.arccos(-r[i, 2, :] - p[2, :])  # angle from vertical
        axes[1].grid()
        axes[1].set_ylabel(f"Cable angle drone {i+1} [rad]")
        axes[1].plot(t, angles, label=f"Drone {i+1}")
        axes[1].legend()

        speeds = np.linalg.norm(v[i, :, :], axis=0)
        axes[2].grid()
        axes[2].set_ylabel(f"Speed drone {i+1} [m/s]")
        axes[2].plot(t, speeds, label=f"Drone {i+1}")
        axes[2].legend()
    plt.tight_layout()

    fig.savefig("crazyflo_planner/figures/cable_tensions_angles_speeds.png")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    ax.plot(p[0],  p[1],  p[2],  label="payload", linewidth=3)
    ax.plot(r[0, 0], r[0, 1], r[0, 2], label="drone 1")
    ax.plot(r[1, 0], r[1, 1], r[1, 2], label="drone 2")
    ax.plot(r[2, 0], r[2, 1], r[2, 2], label="drone 3")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    plt.tight_layout()

    fig.savefig("crazyflo_planner/figures/3d_trajectory.png")
    plt.show()


def plot_data(ocp_data):
    """Plot data from OCP solution dictionary."""
    t = ocp_data["t"]
    p = ocp_data["p"]
    vp = ocp_data["vp"]
    r1 = ocp_data["r1"]
    r2 = ocp_data["r2"]
    r3 = ocp_data["r3"]
    v1 = ocp_data["v1"]
    v2 = ocp_data["v2"]
    v3 = ocp_data["v3"]
    u1 = ocp_data["u1"]
    u2 = ocp_data["u2"]
    u3 = ocp_data["u3"]
    lam1 = ocp_data["lam1"]
    lam2 = ocp_data["lam2"]
    lam3 = ocp_data["lam3"] 
    r = np.array([r1, r2, r3])
    v = np.array([v1, v2, v3])
    u = np.array([u1, u2, u3])
    lam = np.array([lam1, lam2, lam3])

    plot_states(t, p, vp, r, v, u, lam)


if __name__ == "__main__":
    ocp_data = np.load("crazyflo_planner/data/ocp_solution.npz")
    plot_data(ocp_data)
