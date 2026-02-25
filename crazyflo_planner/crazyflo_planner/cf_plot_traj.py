import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _poly(coeffs, t):
    """Evaluate polynomial a0..a7 at scalar t."""
    return np.polynomial.polynomial.polyval(t, np.asarray(coeffs, dtype=float))


def plot_trajectory_from_csv(csv_path: str):
    """
    Uses each row's `duration` directly as its time parameter (t).
    Assumes one polynomial segment per row.
    """
    df = pd.read_csv(csv_path)

    xs, ys, zs, yaws, ts = [], [], [], [], []

    for _, r in df.iterrows():
        t = float(r["duration"])  # use provided duration directly
        ts.append(t)

        x_coeffs = r[[f"x^{i}" for i in range(8)]].to_numpy(dtype=float)
        y_coeffs = r[[f"y^{i}" for i in range(8)]].to_numpy(dtype=float)
        z_coeffs = r[[f"z^{i}" for i in range(8)]].to_numpy(dtype=float)
        yaw_coeffs = r[[f"yaw^{i}" for i in range(8)]].to_numpy(dtype=float)

        xs.append(_poly(x_coeffs, t))
        ys.append(_poly(y_coeffs, t))
        zs.append(_poly(z_coeffs, t))
        yaws.append(_poly(yaw_coeffs, t))

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    yaws = np.array(yaws)
    ts = np.array(ts)

    # ---- plots ----
    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(xs, ys, zs, marker="o")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Trajectory")

    ax_yaw = fig.add_subplot(1, 2, 2)
    ax_yaw.plot(ts, yaws, marker="o")
    ax_yaw.set_xlabel("t (duration)")
    ax_yaw.set_title("Yaw")

    plt.tight_layout()
    plt.show()

    return {
        "t": ts,
        "x": xs,
        "y": ys,
        "z": zs,
        "yaw": yaws,
    }


# Example
file = "/home/florian-jule/.ros/crazyflo_planner/data/traj_cf2.csv"
plot_trajectory_from_csv(file)