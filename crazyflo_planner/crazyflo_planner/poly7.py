import numpy as np
from typing import Optional, Sequence, Tuple, List, Iterable, Dict
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

    # Work in normalized time s in [0,1], where t = dt*s.
    # Convert derivatives from t-domain to s-domain:
    # v_s = v_t * dt, a_s = a_t * dt^2, j_s = j_t * dt^3
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
    v_max: float = 100.0,
    a_max: float = 1000.0,
    j_max: float = 10000.0,
    t_grid: Optional[np.ndarray] = None,
    yaw_waypoints: Optional[np.ndarray] = None,
    max_iter: int = 50,
    samples_per_seg: int = 200,
    time_scale_step: float = 1.15,
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
    if N < 2:
        raise ValueError("Need at least 2 waypoints.")

    # yaw handling
    if yaw_waypoints is None:
        yaw = np.zeros(N, dtype=float)
    else:
        yaw = np.asarray(yaw_waypoints, dtype=float).reshape(-1)
        if yaw.size != N:
            raise ValueError("yaw_waypoints must have same length as pl_waypoints")

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

    # if t_grid is not None:
    for axis in range(3):
        v, a, j = _finite_derivatives(t_wp, P[:, axis])
        V[:, axis], A[:, axis], J[:, axis] = v, a, j
    Vy, Ay, Jy = _finite_derivatives(t_wp, yaw)

    # enforce boundary derivatives = 0 as required
    V[0] = 0.0; A[0] = 0.0; J[0] = 0.0
    V[-1] = 0.0; A[-1] = 0.0; J[-1] = 0.0
    Vy[0] = 0.0; Ay[0] = 0.0; Jy[0] = 0.0
    Vy[-1] = 0.0; Ay[-1] = 0.0; Jy[-1] = 0.0

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


def _poly7_eval(coeffs: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate poly7 with coeffs a0..a7 at times t (vector)."""
    coeffs = np.asarray(coeffs, dtype=float).reshape(8)
    t = np.asarray(t, dtype=float)
    # Horner
    y = np.zeros_like(t, dtype=float)
    for a in coeffs[::-1]:
        y = y * t + a
    return y


def _poly7_derivative_coeffs(coeffs: np.ndarray, order: int) -> np.ndarray:
    """Return coeffs for the derivative polynomial (still in power basis)."""
    c = np.asarray(coeffs, dtype=float).copy()
    for _ in range(order):
        dc = np.zeros_like(c)
        for n in range(1, c.size):
            dc[n - 1] = n * c[n]
        c = dc
    return c


def sample_segments(
    segments: Iterable,
    dt: float,
    *,
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
    if not segments:
        raise ValueError("segments is empty")

    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be > 0")

    t_all = []
    p_all = []
    yaw_all = []
    v_all = []
    a_all = []
    j_all = []

    t0_global = 0.0
    for i, seg in enumerate(segments):
        T = float(seg.duration)

        # local time grid for this segment
        if include_endpoint:
            # include end on last segment; otherwise avoid duplicates
            last = (i == len(segments) - 1)
            t_local = np.arange(0.0, T + (0.5 * dt), dt)
            if not last:
                # drop the endpoint to avoid duplicate time at next segment start
                if t_local.size and np.isclose(t_local[-1], T):
                    t_local = t_local[:-1]
        else:
            t_local = np.arange(0.0, T, dt)

        t_global = t0_global + t_local

        # position
        x = _poly7_eval(seg.coeffs_x, t_local)
        y = _poly7_eval(seg.coeffs_y, t_local)
        z = _poly7_eval(seg.coeffs_z, t_local)
        p = np.stack([x, y, z], axis=1)

        # yaw
        yaw = _poly7_eval(seg.coeffs_yaw, t_local)

        # velocity/acc/jerk
        vx = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_x, 1), t_local)
        vy = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_y, 1), t_local)
        vz = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_z, 1), t_local)
        v = np.stack([vx, vy, vz], axis=1)

        ax = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_x, 2), t_local)
        ay = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_y, 2), t_local)
        az = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_z, 2), t_local)
        a = np.stack([ax, ay, az], axis=1)

        jx = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_x, 3), t_local)
        jy = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_y, 3), t_local)
        jz = _poly7_eval(_poly7_derivative_coeffs(seg.coeffs_z, 3), t_local)
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


def get_waypoint_positions(segments):
    """Return positions at segment boundaries."""
    pts = []

    # first point = start of first segment
    first = segments[0]
    p0 = np.array([
        first.coeffs_x[0],
        first.coeffs_y[0],
        first.coeffs_z[0],
    ])
    pts.append(p0)

    for seg in segments:
        T = seg.duration
        # evaluate end position
        def eval_poly(c, t):
            return sum(c[n] * t**n for n in range(len(c)))

        p_end = np.array([
            eval_poly(seg.coeffs_x, T),
            eval_poly(seg.coeffs_y, T),
            eval_poly(seg.coeffs_z, T),
        ])
        pts.append(p_end)

    return np.vstack(pts)


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
      dt  x0..x7  y0..y7  z0..z7  yaw0..yaw7
    """
    with open(path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["duration"]
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
