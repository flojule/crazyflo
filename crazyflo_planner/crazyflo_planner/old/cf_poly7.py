import csv
from dataclasses import dataclass
import numpy as np


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
    """
    Find coefficients a[0..7] such that:
      p(0)=p0, p'(0)=v0, p''(0)=a0, p'''(0)=j0
      p(dt)=p1, p'(dt)=v1, p''(dt)=a1, p'''(dt)=j1
    where p(t)=sum_{n=0..7} a[n]*t^n
    """
    # Build linear system M a = b
    # Derivatives:
    # p'(t) = sum n a[n] t^(n-1)
    # p''(t) = sum n(n-1) a[n] t^(n-2)
    # p'''(t)= sum n(n-1)(n-2)a[n] t^(n-3)
    def row_p(t):
        return np.array([t**n for n in range(8)], dtype=float)

    def row_v(t):
        r = np.zeros(8, dtype=float)
        for n in range(1, 8):
            r[n] = n * (t ** (n - 1))
        return r

    def row_a(t):
        r = np.zeros(8, dtype=float)
        for n in range(2, 8):
            r[n] = n * (n - 1) * (t ** (n - 2))
        return r

    def row_j(t):
        r = np.zeros(8, dtype=float)
        for n in range(3, 8):
            r[n] = n * (n - 1) * (n - 2) * (t ** (n - 3))
        return r

    M = np.stack([
        row_p(0.0),
        row_v(0.0),
        row_a(0.0),
        row_j(0.0),
        row_p(dt),
        row_v(dt),
        row_a(dt),
        row_j(dt),
    ], axis=0)

    b = np.array([p0, v0, a0, j0, p1, v1, a1, j1], dtype=float)
    return np.linalg.solve(M, b)


def fit_poly7_piecewise(
    t: np.ndarray,
    pos: np.ndarray,
    yaw: None | np.ndarray = None,
) -> list[Poly7Segment]:
    """Fit piecewise poly7 segments through sampled positions."""

    t = np.asarray(t, dtype=float).reshape(-1)
    if pos.shape[0] == 3 and pos.ndim == 2:
        P = pos
    else:
        P = np.asarray(pos, dtype=float).reshape(-1, 3).T

    K = t.size
    if yaw is None:
        yaw = np.zeros(K, dtype=float)
    else:
        yaw = np.asarray(yaw, dtype=float).reshape(-1)
        if yaw.size != K:
            raise ValueError("yaw must have same length as t")

    def deriv1(y):
        dy = np.zeros_like(y)
        dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
        dy[0] = (y[1] - y[0]) / (t[1] - t[0])
        dy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
        return dy

    def deriv2(y):
        d2 = np.zeros_like(y)
        for k in range(1, K - 1):
            dt1 = t[k] - t[k - 1]
            dt2 = t[k + 1] - t[k]
            d2[k] = 2.0 * ((
                y[k + 1] - y[k]) / dt2 - (y[k] - y[k - 1]) / dt1) / (
                    dt1 + dt2)
        d2[0] = d2[1]
        d2[-1] = d2[-2]
        return d2

    V = np.stack([deriv1(P[0]), deriv1(P[1]), deriv1(P[2])], axis=0)
    A = np.stack([deriv2(P[0]), deriv2(P[1]), deriv2(P[2])], axis=0)
    Vy = deriv1(yaw)
    Ay = deriv2(yaw)

    segment_every = 10
    segments: list[Poly7Segment] = []
    idx = 0
    while idx < K - 1:
        j0 = idx
        j1 = min(idx + segment_every, K - 1)
        if j1 == j0:
            break
        dt_seg = float(t[j1] - t[j0])

        coeffs = []
        for axis in range(3):
            c = _poly7_from_endpoint_conditions(
                dt=dt_seg,
                p0=float(P[axis, j0]), v0=float(V[axis, j0]), a0=float(A[axis, j0]),
                p1=float(P[axis, j1]), v1=float(V[axis, j1]), a1=float(A[axis, j1])
            )
            coeffs.append(c)

        c_yaw = _poly7_from_endpoint_conditions(
            dt=dt_seg,
            p0=float(yaw[j0]), v0=float(Vy[j0]), a0=float(Ay[j0]),
            p1=float(yaw[j1]), v1=float(Vy[j1]), a1=float(Ay[j1]),
        )

        segments.append(
            Poly7Segment(
                duration=dt_seg,
                coeffs_x=coeffs[0],
                coeffs_y=coeffs[1],
                coeffs_z=coeffs[2],
                coeffs_yaw=c_yaw,
            )
        )

        idx = j1

    return segments


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
