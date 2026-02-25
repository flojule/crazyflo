from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------- Public API ----------------------------

@dataclass(frozen=True)
class Poly7:
    """
    Piecewise 7th-degree polynomial trajectory.

    Segment i is valid on t in [t_knots[i], t_knots[i+1]].
    coeffs[i, ax, k] is coefficient a_k for axis ax in segment i:
        p(tau) = sum_{k=0..7} a_k * tau^k,   tau = t - t_knots[i]
    """
    t_knots: np.ndarray          # (N,)
    coeffs: np.ndarray           # (N-1, 3, 8)


def waypoints_to_poly7(
    waypoints: np.ndarray,
    v_max: float,
    a_max: float,
    time_safety: float = 1.15,
    continuity_order: int = 3,   # 2 => C2 (p,v,a), 3 => C3 (p,v,a,j)
) -> Poly7:
    """
    Input:
      waypoints: (N,3) xyz positions
    Output:
      Poly7 with time allocation from v_max/a_max and minimum-snap solution.
    Constraints:
      - position at all waypoints
      - start/end: v=a=j=0 (j only if continuity_order>=3)
      - internal knots: continuity up to 'continuity_order'
      - internal derivatives are NOT forced to 0 (chosen by min-snap objective)
    """
    w = np.asarray(waypoints, dtype=float)
    if w.ndim != 2 or w.shape[1] != 3 or w.shape[0] < 2:
        raise ValueError("waypoints must be shape (N,3) with N>=2")
    if v_max <= 0 or a_max <= 0:
        raise ValueError("v_max and a_max must be > 0")
    if continuity_order not in (2, 3):
        raise ValueError("continuity_order must be 2 or 3")

    seg_len = np.linalg.norm(w[1:] - w[:-1], axis=1)
    T = np.array([_min_time_1d(d, v_max, a_max) for d in seg_len], dtype=float) * float(time_safety)
    T = np.maximum(T, 1e-3)

    t_knots = np.zeros(w.shape[0], dtype=float)
    t_knots[1:] = np.cumsum(T)

    coeffs = np.zeros((len(T), 3, 8), dtype=float)
    for ax in range(3):
        coeffs[:, ax, :] = _solve_minsnap_axis(
            p_wp=w[:, ax],
            T=T,
            continuity_order=continuity_order,
        )

    return Poly7(t_knots=t_knots, coeffs=coeffs)


def poly7_to_timed_waypoints(poly7: Poly7) -> dict[str, np.ndarray]:
    """
    Returns timed waypoints as (N,4): [t, x, y, z] at knot times.
    These are exactly the original waypoint positions (up to numeric precision).
    """
    t = np.asarray(poly7.t_knots, dtype=float)
    N = len(t)

    p = np.zeros((N, 3), dtype=float)
    p[0] = _eval_segment(poly7.coeffs[0], tau=0.0, deriv=0)

    for i in range(1, N - 1):
        tau = t[i] - t[i - 1]   # end of segment i-1
        p[i] = _eval_segment(poly7.coeffs[i - 1], tau=tau, deriv=0)

    tau_end = t[-1] - t[-2]
    p[-1] = _eval_segment(poly7.coeffs[-1], tau=tau_end, deriv=0)

    return {'t': t, 'p': p}


def timed_waypoints_to_poly7(
    timed_waypoints: dict[str, np.ndarray],
    v_max: float,
    a_max: float,
    continuity_order: int = 3,   # 2 => C2, 3 => C3
    enforce_limits: bool = True,
    sample_per_seg: int = 80,
    max_iter: int = 20,
) -> Poly7:
    """
    Input:
      timed_waypoints: {'t': (N,), 'p': (N,3)}
      v_max, a_max: limits on ||v|| and ||a||

    Behavior:
      - Uses the provided times as initial knot times.
      - Fits a minimum-snap poly7 with position constraints at all waypoints,
        start/end derivatives zero, and continuity up to 'continuity_order'.
      - If enforce_limits=True, increases (stretches) offending segment durations
        and re-solves until sampled limits are satisfied or max_iter is hit.

    Output:
      Poly7 (note: if stretching happens, output t_knots may be later than input).
    """
    if not isinstance(timed_waypoints, dict) or "t" not in timed_waypoints or "p" not in timed_waypoints:
        raise ValueError("timed_waypoints must be a dict with keys {'t','p'}")

    t_in = np.asarray(timed_waypoints["t"], dtype=float).reshape(-1)
    w = np.asarray(timed_waypoints["p"], dtype=float)

    if t_in.ndim != 1 or t_in.shape[0] < 2:
        raise ValueError("timed_waypoints['t'] must be shape (N,) with N>=2")
    if w.ndim != 2 or w.shape[1] != 3 or w.shape[0] != t_in.shape[0]:
        raise ValueError("timed_waypoints['p'] must be shape (N,3) matching timed_waypoints['t']")

    if v_max <= 0 or a_max <= 0:
        raise ValueError("v_max and a_max must be > 0")
    if continuity_order not in (2, 3):
        raise ValueError("continuity_order must be 2 or 3")

    if not np.all(np.isfinite(t_in)) or not np.all(np.isfinite(w)):
        raise ValueError("timed_waypoints contains non-finite values")

    # Ensure strictly increasing times
    if np.any(np.diff(t_in) <= 0):
        raise ValueError("timed_waypoints['t'] must be strictly increasing")

    T = np.diff(t_in).astype(float)
    T = np.maximum(T, 1e-6)

    def solve_with_T(T_curr: np.ndarray) -> Poly7:
        t_knots = np.zeros(len(T_curr) + 1, dtype=float)
        t_knots[1:] = np.cumsum(T_curr)
        coeffs = np.zeros((len(T_curr), 3, 8), dtype=float)
        for ax in range(3):
            coeffs[:, ax, :] = _solve_minsnap_axis(
                p_wp=w[:, ax],
                T=T_curr,
                continuity_order=continuity_order,
            )
        return Poly7(t_knots=t_knots, coeffs=coeffs)

    poly = solve_with_T(T)

    if not enforce_limits:
        # Shift knots to match original start time
        return Poly7(t_knots=poly.t_knots + t_in[0], coeffs=poly.coeffs)

    # Time-stretch loop (per-segment) with re-solve
    for _ in range(max_iter):
        violated = False
        for seg in range(len(T)):
            Ti = float(T[seg])
            taus = np.linspace(0.0, Ti, sample_per_seg)

            vmax = 0.0
            amax = 0.0
            cseg = poly.coeffs[seg]  # (3,8)
            for tau in taus:
                v = _eval_segment(cseg, tau, deriv=1)
                a = _eval_segment(cseg, tau, deriv=2)
                vmax = max(vmax, float(np.linalg.norm(v)))
                amax = max(amax, float(np.linalg.norm(a)))

            if vmax > v_max * 1.0001 or amax > a_max * 1.0001:
                violated = True
                s_v = vmax / v_max
                s_a = np.sqrt(amax / a_max)
                s = max(1.05, s_v, s_a)  # conservative increase
                T[seg] *= s
                break

        if not violated:
            break
        poly = solve_with_T(T)

    # Shift knots to match original start time
    return Poly7(t_knots=poly.t_knots + t_in[0], coeffs=poly.coeffs)


# ---------------------------- Internals ----------------------------

def _min_time_1d(d: float, v_max: float, a_max: float) -> float:
    d = float(abs(d))
    if d < 1e-12:
        return 0.0

    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc * t_acc

    if 2.0 * d_acc >= d:
        return 2.0 * np.sqrt(d / a_max)  # triangular
    else:
        d_cruise = d - 2.0 * d_acc
        return 2.0 * t_acc + d_cruise / v_max  # trapezoidal


def _eval_segment(cseg: np.ndarray, tau: float, deriv: int) -> np.ndarray:
    # cseg: (3,8)
    return np.array([_poly_eval_7(cseg[ax], tau, deriv) for ax in range(3)], dtype=float)


def _poly_eval_7(c: np.ndarray, t: float, deriv: int) -> float:
    t = float(t)
    c = np.asarray(c, dtype=float)
    if deriv == 0:
        return float(sum(c[k] * (t ** k) for k in range(8)))
    if deriv == 1:
        return float(sum(k * c[k] * (t ** (k - 1)) for k in range(1, 8)))
    if deriv == 2:
        return float(sum(k * (k - 1) * c[k] * (t ** (k - 2)) for k in range(2, 8)))
    if deriv == 3:
        return float(sum(k * (k - 1) * (k - 2) * c[k] * (t ** (k - 3)) for k in range(3, 8)))
    raise ValueError("deriv must be 0,1,2,3")


def _solve_minsnap_axis(p_wp: np.ndarray, T: np.ndarray, continuity_order: int) -> np.ndarray:
    """
    1D min-snap with 7th degree per segment:
      p(tau)=sum a_k tau^k, tau in [0,T_i]
    Objective:
      minimize sum_i ∫_0^{T_i} (p''''(tau))^2 d tau
    Constraints:
      - p at all waypoints (start at seg0 tau=0, and each seg end tau=T_i)
      - start/end: v=a=j=0 (j included if continuity_order>=3)
      - internal continuity up to continuity_order (includes position)
    """
    p_wp = np.asarray(p_wp, dtype=float)
    T = np.asarray(T, dtype=float)
    M = len(T)
    Nvar = 8 * M

    # Hessian (block diagonal)
    H = np.zeros((Nvar, Nvar), dtype=float)
    for i in range(M):
        Hi = _snap_hessian_7th(float(T[i]))
        H[i*8:(i+1)*8, i*8:(i+1)*8] = Hi

    rows = []
    rhs = []

    def add_eval(seg_i: int, tau: float, deriv: int, value: float):
        row = np.zeros(Nvar, dtype=float)
        row[seg_i*8:(seg_i+1)*8] = _basis_7th(tau, deriv)
        rows.append(row)
        rhs.append(float(value))

    # Waypoint positions
    add_eval(0, 0.0, 0, p_wp[0])          # start
    for i in range(M):
        add_eval(i, float(T[i]), 0, p_wp[i + 1])  # segment ends

    # Boundary conditions: v=a=0 always; jerk if continuity_order>=3
    add_eval(0, 0.0, 1, 0.0)
    add_eval(0, 0.0, 2, 0.0)
    if continuity_order >= 3:
        add_eval(0, 0.0, 3, 0.0)

    add_eval(M - 1, float(T[M - 1]), 1, 0.0)
    add_eval(M - 1, float(T[M - 1]), 2, 0.0)
    if continuity_order >= 3:
        add_eval(M - 1, float(T[M - 1]), 3, 0.0)

    # Internal continuity between seg i-1 end and seg i start
    derivs = list(range(0, continuity_order + 1))  # includes position continuity
    for i in range(1, M):
        Ti_prev = float(T[i - 1])
        for d in derivs:
            row = np.zeros(Nvar, dtype=float)
            row[(i - 1)*8:i*8] = _basis_7th(Ti_prev, d)
            row[i*8:(i + 1)*8] = -_basis_7th(0.0, d)
            rows.append(row)
            rhs.append(0.0)

    A = np.vstack(rows)
    b = np.array(rhs, dtype=float)

    # KKT solve for equality-constrained QP:
    # [H  A^T][x] = [0]
    # [A   0 ][λ]   [b]
    eps = 1e-12
    H_reg = H + eps * np.eye(Nvar)

    KKT = np.block([
        [H_reg, A.T],
        [A, np.zeros((A.shape[0], A.shape[0]), dtype=float)]
    ])
    rhs_kkt = np.concatenate([np.zeros(Nvar, dtype=float), b])

    sol = np.linalg.solve(KKT, rhs_kkt)
    x = sol[:Nvar]
    return x.reshape(M, 8)


def _basis_7th(t: float, deriv: int) -> np.ndarray:
    t = float(t)
    out = np.zeros(8, dtype=float)
    if deriv == 0:
        out[:] = [t**k for k in range(8)]
        return out
    if deriv == 1:
        for k in range(1, 8):
            out[k] = k * (t ** (k - 1))
        return out
    if deriv == 2:
        for k in range(2, 8):
            out[k] = k * (k - 1) * (t ** (k - 2))
        return out
    if deriv == 3:
        for k in range(3, 8):
            out[k] = k * (k - 1) * (k - 2) * (t ** (k - 3))
        return out
    if deriv == 4:
        for k in range(4, 8):
            out[k] = k * (k - 1) * (k - 2) * (k - 3) * (t ** (k - 4))
        return out
    raise ValueError("deriv must be 0..4")


def _snap_hessian_7th(T: float) -> np.ndarray:
    """
    H such that  ∫_0^T (p''''(t))^2 dt = c^T H c  for 7th degree coefficients c.
    """
    T = float(T)
    H = np.zeros((8, 8), dtype=float)
    f = np.zeros(8, dtype=float)
    for k in range(4, 8):
        f[k] = k * (k - 1) * (k - 2) * (k - 3)

    for i in range(4, 8):
        for j in range(4, 8):
            power = i + j - 8  # integrand t^power
            H[i, j] = f[i] * f[j] * (T ** (power + 1)) / (power + 1)

    return H
