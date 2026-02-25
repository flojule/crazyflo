
import csv
import numpy as np
from pathlib import Path
import cf_traj


def save_time_pos_csv(sol, path):
    t = sol["t"]
    N = len(t)

    header = (
        "t,"
        "posx,posy,posz,"
        "velx,vely,velz,"
        "accx,accy,accz,"
        "jerkx,jerky,jerkz,"
        "snapx,snapy,snapz"
    )

    for i in [1, 2, 3]:
        filename = path / f"time_pos_cf{i}.csv"
        pos = sol.get(f"cf{i}_p", np.zeros((N, 3)))
        vel = sol.get(f"cf{i}_v", np.zeros((N, 3)))
        acc = sol.get(f"cf{i}_a", np.zeros((N, 3)))
        jerk = sol.get(f"cf{i}_j", np.zeros((N, 3)))
        snap = sol.get(f"cf{i}_s", np.zeros((N, 3)))

        pos = _as_N3(pos, N, f"cf{i}_p")
        vel = _as_N3(vel, N, f"cf{i}_v")
        acc = _as_N3(acc, N, f"cf{i}_a")
        jerk = _as_N3(jerk, N, f"cf{i}_j")
        snap = _as_N3(snap, N, f"cf{i}_s")

        data = np.column_stack([t, pos, vel, acc, jerk, snap])

        np.savetxt(
            filename,
            data,
            delimiter=",",
            header=header,
            comments="",
        )

    print(f"Time-position CSV files for cf1, cf2, cf3 saved to {path}")


def read_time_pos_csv(filename):
    data = np.genfromtxt(filename, delimiter=",", names=True)

    pos = np.vstack([data["posx"], data["posy"], data["posz"]])
    vel = np.vstack([data["velx"], data["vely"], data["velz"]])
    acc = np.vstack([data["accx"], data["accy"], data["accz"]])
    jerk = np.vstack([data["jerkx"], data["jerky"], data["jerkz"]])
    snap = np.vstack([data["snapx"], data["snapy"], data["snapz"]])
    pl_traj = {
        "t": data["t"],
        "p": pos.T,
        "v": vel.T,
        "a": acc.T,
        "j": jerk.T,
        "s": snap.T,
    }
    print(f"Read time-position CSV file {filename} with {len(data)} points.")
    return pl_traj


def _as_N3(x, N, name="array"):
    """
    Ensure x is shape (N,3). Accepts:
      - (N,3)
      - (3,N) -> transposed
      - (N-1,3) or (3,N-1) -> pad last row to length N
      - (3,) or (1,3) -> broadcast constant vector
      - scalar -> broadcast to (N,3)
    """
    x = np.asarray(x)

    # Handle transpose cases first
    if x.shape == (3, N):
        x = x.T
    elif x.shape == (3, N - 1):
        x = x.T  # becomes (N-1,3)

    # Now x is either (N,3), (N-1,3), etc.
    if x.shape == (N, 3):
        return x

    if x.shape == (N - 1, 3):
        # pad by repeating the last sample
        return np.vstack([x, x[-1:]])

    # constant vector
    if x.shape == (3,):
        return np.tile(x.reshape(1, 3), (N, 1))
    if x.shape == (1, 3):
        return np.tile(x, (N, 1))

    # scalar constant
    if x.shape == (1,) or x.shape == ():
        return np.full((N, 3), float(x))

    raise ValueError(f"{name} must be (N,3)/(3,N) or (N-1,3)/(3,N-1) or broadcastable; got {x.shape}, N={N}")


def save_poly7_csv(sol, folder, v_max=2.0, a_max=5.0):
    """
    Save Poly7 trajectory to CSV with columns:

    Duration,
    x^0..x^7,
    y^0..y^7,
    z^0..z^7,
    yaw^0..yaw^7
    """
    t = np.asarray(sol["t"], dtype=float).reshape(-1, 1)  # (N,1)

    for i in [1, 2, 3]:
        csv_path = folder / f"traj_cf{i}.csv"

        p_raw = np.asarray(sol[f"cf{i}_p"], dtype=float)
        if p_raw.shape == (t.shape[0], 3):
            p = p_raw
        elif p_raw.shape == (3, t.shape[0]):
            p = p_raw.T

        cf_i_traj = {"t": sol["t"], "p": p}

        poly7 = cf_traj.timed_waypoints_to_poly7(
            cf_i_traj, v_max=v_max, a_max=a_max,
            enforce_limits=False
        )

        path = Path(csv_path)

        t = np.asarray(poly7.t_knots, dtype=float)
        coeffs = np.asarray(poly7.coeffs, dtype=float)

        num_seg = coeffs.shape[0]

        yaw_coeffs = np.zeros((num_seg, 8), dtype=float)

        header = (
            ["Duration"]
            + [f"x^{i}" for i in range(8)]
            + [f"y^{i}" for i in range(8)]
            + [f"z^{i}" for i in range(8)]
            + [f"yaw^{i}" for i in range(8)]
        )

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i in range(num_seg):
                duration = float(t[i + 1] - t[i])

                row = [duration]

                # x, y, z coefficients
                for axis in range(3):
                    row.extend(coeffs[i, axis, :].tolist())

                # yaw coefficients
                row.extend(yaw_coeffs[i, :].tolist())

                writer.writerow(row)
    print(f"Poly7 trajectory CSV files for cf1, cf2, cf3 saved to {folder}")

