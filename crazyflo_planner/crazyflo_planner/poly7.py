import numpy as np
from typing import Optional, Tuple, List, Iterable, Dict
import csv
from dataclasses import dataclass

@dataclass
class Poly7Segment:
    duration: float
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    coeffs_z: np.ndarray
    coeffs_yaw: np.ndarray


def _poly7_from_endpoint_conditions(
    dt: float,
    p0: float = 0.0, v0: float = 0.0, a0: float = 0.0, j0: float = 0.0,
    p1: float = 0.0, v1: float = 0.0, a1: float = 0.0, j1: float = 0.0,
) -> np.ndarray:
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Bad dt: {dt}")

    v0s = v0 * dt
    v1s = v1 * dt
    a0s = a0 * (dt * dt)
    a1s = a1 * (dt * dt)
    j0s = j0 * (dt * dt * dt)
    j1s = j1 * (dt * dt * dt)

    def row_p(s):
        return np.array([s**n for n in range(8)], dtype=float)

    def row_v(s):
        r = np.zeros(8, dtype=float)
        for n in range(1, 8):
            r[n] = n * (s ** (n - 1))
        return r

    def row_a(s):
        r = np.zeros(8, dtype=float)
        for n in range(2, 8):
            r[n] = n * (n - 1) * (s ** (n - 2))
        return r

    def row_j(s):
        r = np.zeros(8, dtype=float)
        for n in range(3, 8):
            r[n] = n * (n - 1) * (n - 2) * (s ** (n - 3))
        return r

    # Same matrix for all segments (evaluated at s=0 and s=1)
    M = np.stack([
        row_p(0.0),
        row_v(0.0),
        row_a(0.0),
        row_j(0.0),
        row_p(1.0),
        row_v(1.0),
        row_a(1.0),
        row_j(1.0),
    ], axis=0)

    b = np.array([p0, v0s, a0s, j0s, p1, v1s, a1s, j1s], dtype=float)

    # Solve for coefficients in s-domain: p(s) = sum c[n]*s^n
    c = np.linalg.solve(M, b)

    # Convert to t-domain: p(t) = sum c[n]*(t/dt)^n = sum (c[n]/dt^n) * t^n
    a = np.array([c[n] / (dt**n) for n in range(8)], dtype=float)
    return a


def _poly_derivative_coeffs(a: np.ndarray, order: int) -> np.ndarray:
    """Return coefficients of the derivative polynomial (same basis t^n)"""
    c = np.array(a, dtype=float)
    for _ in range(order):
        dc = np.zeros_like(c)
        for n in range(1, c.size):
            dc[n - 1] = n * c[n]
        c = dc
    return c


def _poly_eval(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate polynomial with coeffs a0..a7 at vector t."""
    # Horner
    y = np.zeros_like(t, dtype=float)
    for coeff in a[::-1]:
        y = y * t + coeff
    return y


def _segment_max_norms(coeffs_xyz: np.ndarray, T: float, samples: int = 200) -> Tuple[float, float, float]:
    """
    coeffs_xyz shape (3,8): x,y,z poly7 coeffs.
    Return max ||v||, max ||a||, max ||j|| by sampling.
    """
    ts = np.linspace(0.0, T, samples + 1)

    vmax = amax = jmax = 0.0
    for axis in range(3):
        ax = coeffs_xyz[axis]
        vx = _poly_derivative_coeffs(ax, 1)
        ax2 = _poly_derivative_coeffs(ax, 2)
        jx = _poly_derivative_coeffs(ax, 3)

        # evaluate
        v = _poly_eval(vx, ts)
        a = _poly_eval(ax2, ts)
        j = _poly_eval(jx, ts)

        if axis == 0:
            V = np.zeros((ts.size, 3))
            A = np.zeros((ts.size, 3))
            J = np.zeros((ts.size, 3))
        V[:, axis] = v
        A[:, axis] = a
        J[:, axis] = j

    vmax = float(np.max(np.linalg.norm(V, axis=1)))
    amax = float(np.max(np.linalg.norm(A, axis=1)))
    jmax = float(np.max(np.linalg.norm(J, axis=1)))
    return vmax, amax, jmax


def _finite_derivatives(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate v/a/j at samples y(t) with nonuniform t using finite differences.
    Returns (v,a,j) arrays same length.
    """
    t = np.asarray(t, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    K = t.size

    v = np.zeros(K, float)
    a = np.zeros(K, float)
    j = np.zeros(K, float)

    # v: central difference (nonuniform)
    v[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    v[0] = (y[1] - y[0]) / (t[1] - t[0])
    v[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])

    # a: second derivative (nonuniform)
    for k in range(1, K - 1):
        dt1 = t[k] - t[k - 1]
        dt2 = t[k + 1] - t[k]
        a[k] = 2.0 * (((y[k + 1] - y[k]) / dt2) - ((y[k] - y[k - 1]) / dt1)) / (dt1 + dt2)
    a[0] = a[1]
    a[-1] = a[-2]

    # j: derivative of a (reuse scheme)
    j[1:-1] = (a[2:] - a[:-2]) / (t[2:] - t[:-2])
    j[0] = (a[1] - a[0]) / (t[1] - t[0])
    j[-1] = (a[-1] - a[-2]) / (t[-1] - t[-2])

    return v, a, j


def fit_poly7_piecewise(
    p: np.ndarray,
    v_max: float = 5.0,
    a_max: float = 10.0,
    j_max: float = 50.0,
    t_grid: Optional[np.ndarray] = None,
    max_iter: int = 50,
    samples_per_seg: int = 200,
    max_seg_T: float = 20.0,
    min_seg_T: float = 1e-3,
) -> List[Poly7Segment]:
    """
    Build piecewise poly7 trajectory through waypoints.

    Modes:
      A) Untimed waypoints: provide waypoints (N,3), t_grid=None
         -> segment times are chosen to satisfy v/a/j limits.
      B) Timed waypoints: provide waypoints (N,3) AND t_grid (N,)
         -> segment times fixed; derivatives estimated from timed samples.

    Boundary conditions:
      Start and end: v=a=j=0 for x/y/z and yaw.

    Returns:
      list of Poly7Segment with duration and coeffs for x/y/z/yaw
    """
    P = np.asarray(p, dtype=float).reshape(-1, 3)
    N = P.shape[0]

    yaw = np.zeros(N, dtype=float)

    # time handling
    if t_grid is None:
        d = np.linalg.norm(np.diff(P, axis=0), axis=1)
        base = np.maximum(d / max(v_max, 1e-9), 1e-3)
        base *= 1.2
        seg_T = base.copy()
        t_wp = np.concatenate([[0.0], np.cumsum(seg_T)])
    else:
        t_wp = np.asarray(t_grid, dtype=float).reshape(-1)
        seg_T = np.diff(t_wp)

    # derivatives at waypoints
    V = np.zeros((N, 3), float)
    A = np.zeros((N, 3), float)
    J = np.zeros((N, 3), float)
    Vy = np.zeros(N, float)
    Ay = np.zeros(N, float)
    Jy = np.zeros(N, float)

    for axis in range(3):
        v, a, j = _finite_derivatives(t_wp, P[:, axis])
        V[:, axis], A[:, axis], J[:, axis] = v, a, j
    Vy, Ay, Jy = _finite_derivatives(t_wp, yaw)

    # enforce boundary derivatives = 0 as required
    V[0] = 0.0
    A[0] = 0.0
    J[0] = 0.0

    V[-1] = 0.0
    A[-1] = 0.0
    J[-1] = 0.0

    Vy[0] = 0.0
    Ay[0] = 0.0
    Jy[0] = 0.0

    Vy[-1] = 0.0
    Ay[-1] = 0.0
    Jy[-1] = 0.0

    def build_segments(seg_T_local: np.ndarray) -> List[Poly7Segment]:
        segs: List[Poly7Segment] = []
        for i in range(N - 1):
            T = float(seg_T_local[i])

            # solve x/y/z poly7
            coeffs_xyz = np.zeros((3, 8), float)
            for axis in range(3):
                coeffs_xyz[axis] = _poly7_from_endpoint_conditions(
                    dt=T,
                    p0=float(P[i, axis]), v0=float(V[i, axis]), a0=float(A[i, axis]), j0=float(J[i, axis]),
                    p1=float(P[i + 1, axis]), v1=float(V[i + 1, axis]), a1=float(A[i + 1, axis]), j1=float(J[i + 1, axis]),
                )

            # yaw poly7
            coeffs_yaw = _poly7_from_endpoint_conditions(
                dt=T,
                p0=float(yaw[i]), v0=float(Vy[i]), a0=float(Ay[i]), j0=float(Jy[i]),
                p1=float(yaw[i + 1]), v1=float(Vy[i + 1]), a1=float(Ay[i + 1]), j1=float(Jy[i + 1]),
            )

            segs.append(
                Poly7Segment(
                    duration=T,
                    coeffs_x=coeffs_xyz[0].copy(),
                    coeffs_y=coeffs_xyz[1].copy(),
                    coeffs_z=coeffs_xyz[2].copy(),
                    coeffs_yaw=coeffs_yaw.copy(),
                )
            )
        return segs

    if t_grid is not None:
        return build_segments(seg_T)

    seg_T_work = seg_T.copy()
    for _ in range(max_iter):
        segs = build_segments(seg_T_work)

        violations = 0
        for i, seg in enumerate(segs):
            coeffs_xyz = np.stack([seg.coeffs_x, seg.coeffs_y, seg.coeffs_z], axis=0)
            vmax_s, amax_s, jmax_s = _segment_max_norms(coeffs_xyz, seg.duration, samples=samples_per_seg)

            ok = (vmax_s <= v_max + 1e-9) and (amax_s <= a_max + 1e-9) and (jmax_s <= j_max + 1e-9)
            if not ok:
                violations += 1

                ratios = []
                if vmax_s > v_max:
                    ratios.append(vmax_s / v_max)  # v ~ 1/T
                if amax_s > a_max:
                    ratios.append(np.sqrt(amax_s / a_max))  # a ~ 1/T^2
                if jmax_s > j_max:
                    ratios.append((jmax_s / j_max) ** (1.0/3.0))  # j ~ 1/T^3

                scale = max(ratios) if ratios else 1.0
                seg_T_work[i] *= float(max(1.05, 1.02 * scale))
                seg_T_work[i] = float(np.clip(seg_T_work[i], min_seg_T, max_seg_T))

        if violations == 0:
            return segs
        
        t_wp = np.concatenate([[0.0], np.cumsum(seg_T_work)])
        for axis in range(3):
            v, a, j = _finite_derivatives(t_wp, P[:, axis])
            V[:, axis], A[:, axis], J[:, axis] = v, a, j
        Vy, Ay, Jy = _finite_derivatives(t_wp, yaw)

        # re-enforce boundary derivatives = 0
        V[0] = 0.0; A[0] = 0.0; J[0] = 0.0
        V[-1] = 0.0; A[-1] = 0.0; J[-1] = 0.0
        Vy[0] = 0.0; Ay[0] = 0.0; Jy[0] = 0.0
        Vy[-1] = 0.0; Ay[-1] = 0.0; Jy[-1] = 0.0

    return build_segments(seg_T_work)


def sample_segments(
    segments: Iterable,
    dt: float,
    include_endpoint: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert a list of Poly7Segment into a global time grid and samples.

    Returns dict with:
      t:   (M,)
      p:   (M,3)
      yaw: (M,)
      v:   (M,3)
      a:   (M,3)
      j:   (M,3)
    """
    segments = list(segments)

    t_all = []
    p_all = []
    yaw_all = []
    v_all = []
    a_all = []
    j_all = []

    t0_global = 0.0
    for i, seg in enumerate(segments):
        T = float(seg.duration)

        if include_endpoint:
            last = (i == len(segments) - 1)

            n = int(np.ceil(T / dt))
            t_local = dt * np.arange(n + 1)
            t_local = np.clip(t_local, 0.0, T)

            if not last:
                t_local = t_local[t_local < T]

            if last:
                if t_local.size == 0 or not np.isclose(t_local[-1], T):
                    t_local = np.concatenate([t_local, [T]])
                else:
                    t_local[-1] = T
        else:
            n = int(np.floor(T / dt))
            t_local = dt * np.arange(n + 1)
            t_local = t_local[t_local < T]

        t_global = t0_global + t_local

        # position
        x = _poly_eval(seg.coeffs_x, t_local)
        y = _poly_eval(seg.coeffs_y, t_local)
        z = _poly_eval(seg.coeffs_z, t_local)
        p = np.stack([x, y, z], axis=1)

        # yaw
        yaw = _poly_eval(seg.coeffs_yaw, t_local)

        # velocity/acc/jerk
        vx = _poly_eval(_poly_derivative_coeffs(seg.coeffs_x, 1), t_local)
        vy = _poly_eval(_poly_derivative_coeffs(seg.coeffs_y, 1), t_local)
        vz = _poly_eval(_poly_derivative_coeffs(seg.coeffs_z, 1), t_local)
        v = np.stack([vx, vy, vz], axis=1)

        ax = _poly_eval(_poly_derivative_coeffs(seg.coeffs_x, 2), t_local)
        ay = _poly_eval(_poly_derivative_coeffs(seg.coeffs_y, 2), t_local)
        az = _poly_eval(_poly_derivative_coeffs(seg.coeffs_z, 2), t_local)
        a = np.stack([ax, ay, az], axis=1)

        jx = _poly_eval(_poly_derivative_coeffs(seg.coeffs_x, 3), t_local)
        jy = _poly_eval(_poly_derivative_coeffs(seg.coeffs_y, 3), t_local)
        jz = _poly_eval(_poly_derivative_coeffs(seg.coeffs_z, 3), t_local)
        j = np.stack([jx, jy, jz], axis=1)

        # append
        t_all.append(t_global)
        p_all.append(p)
        yaw_all.append(yaw)
        v_all.append(v)
        a_all.append(a)
        j_all.append(j)

        t0_global += T

    return {
        "t": np.concatenate(t_all, axis=0),
        "p": np.concatenate(p_all, axis=0),
        "yaw": np.concatenate(yaw_all, axis=0),
        "v": np.concatenate(v_all, axis=0),
        "a": np.concatenate(a_all, axis=0),
        "j": np.concatenate(j_all, axis=0),
    }


def get_waypoint_states(segments):
    """
    Return states at segment boundaries (start + each segment end).

    Outputs dict with:
      t     (N,)
      pos   (N,3)
      vel   (N,3)
      acc   (N,3)
      jerk  (N,3)
      snap  (N,3)
    """
    def eval_poly(c, t):
        return sum(c[n] * (t ** n) for n in range(c.size))

    def eval_d1(c, t):
        return sum(n * c[n] * (t ** (n - 1)) for n in range(1, c.size))

    def eval_d2(c, t):
        return sum(n * (n - 1) * c[n] * (t ** (n - 2)) for n in range(2, c.size))

    def eval_d3(c, t):
        return sum(n * (n - 1) * (n - 2) * c[n] * (t ** (n - 3)) for n in range(3, c.size))

    def eval_d4(c, t):
        return sum(n * (n - 1) * (n - 2) * (n - 3) * c[n] * (t ** (n - 4)) for n in range(4, c.size))

    segments = list(segments)

    pos, vel, acc, jerk, snap = [], [], [], [], []
    times = [0.0]

    first = segments[0]
    t0 = 0.0

    p0 = np.array([eval_poly(first.coeffs_x, t0),
                   eval_poly(first.coeffs_y, t0),
                   eval_poly(first.coeffs_z, t0)], dtype=float)
    v0 = np.array([eval_d1(first.coeffs_x, t0),
                   eval_d1(first.coeffs_y, t0),
                   eval_d1(first.coeffs_z, t0)], dtype=float)
    a0 = np.array([eval_d2(first.coeffs_x, t0),
                   eval_d2(first.coeffs_y, t0),
                   eval_d2(first.coeffs_z, t0)], dtype=float)
    j0 = np.array([eval_d3(first.coeffs_x, t0),
                   eval_d3(first.coeffs_y, t0),
                   eval_d3(first.coeffs_z, t0)], dtype=float)
    s0 = np.array([eval_d4(first.coeffs_x, t0),
                   eval_d4(first.coeffs_y, t0),
                   eval_d4(first.coeffs_z, t0)], dtype=float)

    pos.append(p0); vel.append(v0); acc.append(a0); jerk.append(j0); snap.append(s0)

    # cumulative time and segment ends
    t_cum = 0.0
    for seg in segments:
        T = float(seg.duration)
        t_cum += T
        times.append(t_cum)

        p = np.array([eval_poly(seg.coeffs_x, T),
                      eval_poly(seg.coeffs_y, T),
                      eval_poly(seg.coeffs_z, T)], dtype=float)
        v = np.array([eval_d1(seg.coeffs_x, T),
                      eval_d1(seg.coeffs_y, T),
                      eval_d1(seg.coeffs_z, T)], dtype=float)
        a = np.array([eval_d2(seg.coeffs_x, T),
                      eval_d2(seg.coeffs_y, T),
                      eval_d2(seg.coeffs_z, T)], dtype=float)
        j = np.array([eval_d3(seg.coeffs_x, T),
                      eval_d3(seg.coeffs_y, T),
                      eval_d3(seg.coeffs_z, T)], dtype=float)
        s = np.array([eval_d4(seg.coeffs_x, T),
                      eval_d4(seg.coeffs_y, T),
                      eval_d4(seg.coeffs_z, T)], dtype=float)

        pos.append(p); vel.append(v); acc.append(a); jerk.append(j); snap.append(s)

    return {
        "t": np.array(times, dtype=float),
        "p": np.vstack(pos),
        "v": np.vstack(vel),
        "a": np.vstack(acc),
        "j": np.vstack(jerk),
        "s": np.vstack(snap),
    }


def get_time_grid(segments):
    """Return cumulative time grid from Poly7Segment list."""
    times = [0.0]
    t = 0.0
    for seg in segments:
        t += float(seg.duration)
        times.append(t)
    return np.array(times)


def write_multi_csv(path: str, segments: list[Poly7Segment]) -> None:
    """
    Write trajectory to a csv file in uav_trajectory/crazyflie format:
      Duration x0..x7  y0..y7  z0..z7  yaw0..yaw7
    """
    with open(path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Duration"]
        header += [f"x^{n}" for n in range(8)]
        header += [f"y^{n}" for n in range(8)]
        header += [f"z^{n}" for n in range(8)]
        header += [f"yaw^{n}" for n in range(8)]
        writer.writerow(header)
        for seg in segments:
            row = [seg.duration]
            row += seg.coeffs_x.tolist()
            row += seg.coeffs_y.tolist()
            row += seg.coeffs_z.tolist()
            row += seg.coeffs_yaw.tolist()
            writer.writerow(row)
