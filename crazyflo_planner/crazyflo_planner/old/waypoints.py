from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, List
import numpy as np
import math


@dataclass(frozen=True)
class DiscreteTrajectory3D:
    dt: float
    t: np.ndarray      # (K+1,)
    p: np.ndarray      # (K+1, 3)

    def vel(self) -> np.ndarray:
        return np.diff(self.p, axis=0) / self.dt

    def acc(self) -> np.ndarray:
        p = self.p
        return (p[2:] - 2 * p[1:-1] + p[:-2]) / (self.dt**2)

    def jerk(self) -> np.ndarray:
        p = self.p
        return (p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]) / (self.dt**3)


def _D3_matrix(n: int) -> np.ndarray:
    m = n - 3
    D3 = np.zeros((m, n), dtype=float)
    for k in range(m):
        D3[k, k]     = -1.0
        D3[k, k + 1] =  3.0
        D3[k, k + 2] = -3.0
        D3[k, k + 3] =  1.0
    return D3


def _max_norms(p: np.ndarray, dt: float) -> Tuple[float, float, float]:
    v = np.diff(p, axis=0) / dt
    a = (p[2:] - 2*p[1:-1] + p[:-2]) / (dt**2)
    j = (p[3:] - 3*p[2:-1] + 3*p[1:-2] - p[:-3]) / (dt**3)

    vmax = float(np.max(np.linalg.norm(v, axis=1))) if v.size else 0.0
    amax = float(np.max(np.linalg.norm(a, axis=1))) if a.size else 0.0
    jmax = float(np.max(np.linalg.norm(j, axis=1))) if j.size else 0.0
    return vmax, amax, jmax


def _build_constraints(n: int, idx: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    rows: List[np.ndarray] = []
    specs: List[Tuple[str, int]] = []

    N = idx.shape[0]
    K = n - 1

    # waypoint position pins
    for i in range(N):
        r = np.zeros(n, dtype=float)
        r[int(idx[i])] = 1.0
        rows.append(r)
        specs.append(("wp", i))

    # start (forward) discrete derivatives == 0
    r = np.zeros(n); r[1]=1; r[0]=-1
    rows.append(r); specs.append(("zero", -1))
    r = np.zeros(n); r[2]=1; r[1]=-2; r[0]=1
    rows.append(r); specs.append(("zero", -1))
    r = np.zeros(n); r[3]=1; r[2]=-3; r[1]=3; r[0]=-1
    rows.append(r); specs.append(("zero", -1))

    # end (backward) discrete derivatives == 0
    r = np.zeros(n); r[K]=1; r[K-1]=-1
    rows.append(r); specs.append(("zero", -1))
    r = np.zeros(n); r[K]=1; r[K-1]=-2; r[K-2]=1
    rows.append(r); specs.append(("zero", -1))
    r = np.zeros(n); r[K]=1; r[K-1]=-3; r[K-2]=3; r[K-3]=-1
    rows.append(r); specs.append(("zero", -1))

    return np.vstack(rows), specs


def _solve_min_jerk_constrained(wp: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    N, D = wp.shape
    n = K + 1
    if n < 8:
        raise ValueError("Need at least 8 samples for boundary v/a/j constraints.")

    A = _D3_matrix(n)
    H = A.T @ A

    C, specs = _build_constraints(n, idx)
    m = C.shape[0]

    d = np.zeros((m, D), dtype=float)
    for r, (kind, i) in enumerate(specs):
        if kind == "wp":
            d[r, :] = wp[i]
        else:
            d[r, :] = 0.0

    # small regularization for numerical stability
    eps = 1e-10
    H = H + eps * np.eye(n)

    KKT = np.zeros((n + m, n + m), dtype=float)
    KKT[:n, :n] = H
    KKT[:n, n:] = C.T
    KKT[n:, :n] = C

    rhs = np.zeros((n + m, D), dtype=float)
    rhs[n:, :] = d

    sol = np.linalg.solve(KKT, rhs)
    return sol[:n, :]


def _steps_from_waypoints(wp: np.ndarray, dt: float, v_max: float, min_steps_per_seg: int) -> np.ndarray:
    N = wp.shape[0]
    steps = np.zeros(N - 1, dtype=int)
    for i in range(N - 1):
        dist = float(np.linalg.norm(wp[i + 1] - wp[i]))
        Ti = dist / max(v_max, 1e-9)
        steps[i] = max(min_steps_per_seg, int(math.ceil(Ti / dt)))
    return steps


def _idx_from_steps(steps_per_seg: np.ndarray) -> np.ndarray:
    # idx length N where idx[0]=0, idx[i+1]=sum steps[0:i+1]
    idx = np.zeros(steps_per_seg.shape[0] + 1, dtype=int)
    idx[1:] = np.cumsum(steps_per_seg, dtype=np.int64)
    return idx


def smooth_traj(
    pl_waypoints: Sequence[Sequence[float]] | np.ndarray,
    dt: float,
    v_max: float,
    a_max: float,
    j_max: float,
    max_iters: int = 10,
    safety: float = 1.05,
    min_steps_per_seg: int = 1,
    max_samples: int = 8000,      # dense KKT, keep modest
    max_scale_per_iter: float = 5.0,  # prevents explosive growth
) -> DiscreteTrajectory3D:
    """
    Constrained minimum-jerk smoothing on a dt grid (3D):
      - hits waypoints exactly
      - discrete v/a/j == 0 at start and end (exact constraints)
      - C^3-like smoothness by minimizing sum of squared third differences
      - enforces v/a/j limits by increasing steps_per_seg (time stretching)

    Uses dense KKT => O(n^3) solve; keep n <= max_samples.
    """
    wp = np.asarray(pl_waypoints, dtype=float)
    if wp.ndim != 2 or wp.shape[1] != 3 or wp.shape[0] < 2:
        raise ValueError("pl_waypoints must be shape (N>=2, 3).")
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if v_max <= 0 or a_max <= 0 or j_max <= 0:
        raise ValueError("v_max, a_max, j_max must be > 0.")

    steps = _steps_from_waypoints(wp, dt, v_max=v_max, min_steps_per_seg=min_steps_per_seg)

    for _ in range(max_iters):
        idx = _idx_from_steps(steps)
        K = int(idx[-1])
        K = max(K, 8)  # boundary constraints need samples
        if K + 1 > max_samples:
            raise MemoryError(
                f"n={K+1} exceeds max_samples={max_samples} for dense solver. "
                f"Increase dt or use a sparse/CG solver."
            )

        # Ensure idx[-1] matches K (if we forced K>=8, adjust last segment)
        if idx[-1] < K:
            steps[-1] += (K - int(idx[-1]))
            idx = _idx_from_steps(steps)
            K = int(idx[-1])

        p = _solve_min_jerk_constrained(wp, idx, K)

        vmax_s, amax_s, jmax_s = _max_norms(p, dt)
        if not (math.isfinite(vmax_s) and math.isfinite(amax_s) and math.isfinite(jmax_s)):
            raise FloatingPointError("Non-finite v/a/j detected; check waypoints (NaN/inf) and dt.")

        vr = vmax_s / v_max
        ar = amax_s / a_max
        jr = jmax_s / j_max
        worst = max(vr, ar, jr)

        if worst <= 1.0 + 1e-6:
            t = np.arange(K + 1, dtype=float) * dt
            return DiscreteTrajectory3D(dt=dt, t=t, p=p)

        # scale factor for time stretching
        s = max(vr, math.sqrt(ar), jr ** (1.0 / 3.0)) * safety
        if not math.isfinite(s) or s <= 1.0:
            s = 1.1
        s = min(s, max_scale_per_iter)

        # Scale steps per segment (integer, monotone, safe)
        new_steps = np.maximum(min_steps_per_seg, np.ceil(steps.astype(float) * s).astype(np.int64))

        if np.array_equal(new_steps, steps):
            # ensure progress
            new_steps = steps.copy()
            new_steps[-1] += 1
        steps = new_steps.astype(int)

    # best-effort return
    idx = _idx_from_steps(steps)
    K = int(idx[-1])
    p = _solve_min_jerk_constrained(wp, idx, K)
    t = np.arange(K + 1, dtype=float) * dt
    return DiscreteTrajectory3D(dt=dt, t=t, p=p)


if __name__ == "__main__":
    pl_height = 0.5
    traj = 'ellipse'
    loops = 3

    N = 50  # number of trajectory points per loop
    grid = np.linspace(0, 1, N + 1)  # points for traj

    if traj == 'ellipse':
        # payload ref trajectory: ellipse
        r_A = 0.6
        r_B = 0.3
        waypoints = np.stack([
            r_A * (1.0 - np.cos(2 * np.pi * grid)),
            r_B * np.sin(2 * np.pi * grid),
            pl_height * np.ones(grid.shape),
        ], axis=1)
    else:
         waypoints = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.2],
                [2.0, 0.0, 0.4],
                [3.0, 0.2, 0.0],
            ], dtype=float)

    if loops > 1:
        # extend trajectory for infinite flight
        p0 = waypoints.copy()  # one loop, length N+1
        waypoints = np.concatenate([p0[:-1] for _ in range(loops - 1)] + [p0], axis=0)
        N = N * loops
        grid = np.linspace(0, 1, N + 1)

    tr = smooth_traj(
        waypoints,
        dt=0.02,
        v_max=1.0,
        a_max=4.0,
        j_max=10.0,
    )

    # plotting for visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(tr.t, tr.p[:, 0], label="x")
    axes[0].plot(tr.t, tr.p[:, 1], label="y")
    axes[0].legend()
    axes[0].set_title("Position")
    v = tr.vel()
    axes[1].plot(tr.t[:-1], v[:, 0], label="vx")
    axes[1].plot(tr.t[:-1], v[:, 1], label="vy")
    axes[1].legend()
    axes[1].set_title("Velocity")
    a = tr.acc()
    axes[2].plot(tr.t[1:-1], a[:, 0], label="ax")
    axes[2].plot(tr.t[1:-1], a[:, 1], label="ay")
    axes[2].legend()
    axes[2].set_title("Acceleration")
    j = tr.jerk()
    axes[3].plot(tr.t[1:-2], j[:, 0], label="jx")
    axes[3].plot(tr.t[1:-2], j[:, 1], label="jy")
    axes[3].legend()
    axes[3].set_title("Jerk")

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.plot(tr.p[:, 0], tr.p[:, 1], label="traj")
    ax.scatter(waypoints[:, 0], waypoints[:, 1], color="red", label="waypoints")
    plt.tight_layout()
    plt.show()
