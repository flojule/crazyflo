#!/usr/bin/env python3
"""
Feb6 working OCP solver adapted to the current interface.
"""

import numpy as np
import casadi as ca
from scipy.interpolate import CubicSpline
from pathlib import Path


def _build_reference_grid(
    waypoints: np.ndarray,
    obstacles: list,
    cf_v_max: float,
    cf_a_max: float,
    dt_min: float,
    dt_max: float,
    M_max: int = 100,
):
    N_wp = waypoints.shape[0]
    M_min = max(N_wp, 10)

    seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    path_length = float(np.sum(seg_lengths))
    arc_wp = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    T_v = np.pi * path_length / (2.0 * cf_v_max)
    T_a = np.pi * np.sqrt(path_length / (2.0 * cf_a_max))
    T_min = max(T_v, T_a, 1e-3)

    target_dt = dt_min if obstacles else dt_max
    M = int(np.clip(round(T_min / target_dt) + 1, M_min, M_max))
    M = max(M, N_wp)
    dt = float(np.clip(T_min / max(M - 1, 1), dt_min, dt_max))
    T_grid = dt * (M - 1)

    t_nodes = np.linspace(0.0, T_grid, M)
    arc_nodes = (path_length / 2.0) * (1.0 - np.cos(np.pi * t_nodes / T_grid))
    arc_nodes = np.clip(arc_nodes, 0.0, path_length)

    arc_frac_wp = arc_wp / max(path_length, 1e-9)
    arc_frac_nodes = arc_nodes / max(path_length, 1e-9)

    if obstacles:
        cs = CubicSpline(arc_frac_wp, waypoints, bc_type='clamped')
        pl_p_ref = cs(arc_frac_nodes).T
    else:
        pl_p_ref = np.vstack([
            np.interp(arc_frac_nodes, arc_frac_wp, waypoints[:, dim])
            for dim in range(3)
        ])

    return M, dt, pl_p_ref, path_length


def _box_barrier(p, c_obs, l_obs, d_safe=0.3):
    eps = 1e-2
    alpha_in = 5.0
    gaps = [(ca.fabs(p[ax] - float(c_obs[ax])) - float(l_obs[ax])) / d_safe for ax in range(3)]

    def inside_w(ax):
        return 1.0 / (1.0 + ca.exp(alpha_in * gaps[ax]))

    def patched_log(g):
        return ca.if_else(
            g > eps,
            -ca.log(g),
            -ca.log(eps) - (1.0 / eps) * (g - eps) + (0.5 / eps ** 2) * (g - eps) ** 2,
        )

    cost = ca.MX(0.0)
    for idx in range(3):
        j, k_ax = (idx + 1) % 3, (idx + 2) % 3
        gap_i = gaps[idx]
        w_in = inside_w(j) * inside_w(k_ax)
        t = ca.fmax(ca.fmin(gap_i, 1.0), 0.0)
        blend = 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t ** 2)
        cost = cost + w_in * blend * patched_log(gap_i)
    return cost


def solve_ocp(
    waypoints: np.ndarray,
    cable_l: float,
    pl_mass: float = 0.03,
    g: float = 9.81,
    cf_v_max: float = 2.0,
    cf_a_max: float = 5.0,
    cf_j_max: float = 10.0,
    tension_min: float = 0.05,
    tension_max: float = 0.5,
    cf_radius: float = 0.2,
    pl_radius: float = 0.05,
    w_pl_p: float = 1e2,
    w_cf_p: float = 0.0,
    w_cf_j: float = 1e-3,
    w_cf_s: float = 1e-3,
    w_tension: float = 10.0,
    w_obs: float = 1e3,
    obstacles: list | None = None,
    soft_cf_pos_dyn: bool = False,
    dt_min: float = 0.1,
    dt_max: float = 0.3,
) -> dict:
    """Solve the feb6-style OCP using the current function signature."""
    if obstacles is None:
        obstacles = []

    M, dt, pl_p_ref, path_length = _build_reference_grid(
        waypoints, obstacles, cf_v_max, cf_a_max, dt_min, dt_max)
    t_grid = np.linspace(0.0, dt * (M - 1), M)
    cf_p0 = get_drones_p0(cable_l, float(waypoints[0, 2]))
    zero_vec = ca.DM(np.zeros((3, 1)))

    print(f"\n[feb6] Path length = {path_length:.2f} m, M = {M}, dt = {dt:.3f} s\n")

    opti = ca.Opti()

    pl_p = opti.variable(3, M)
    pl_v = opti.variable(3, M)
    cf_v = [opti.variable(3, M) for _ in range(3)]
    cf_a = [opti.variable(3, M - 1) for _ in range(3)]
    cf_cable_dir = [opti.variable(3, M) for _ in range(3)]
    cf_cable_t = [opti.variable(1, M) for _ in range(3)]

    def cf_p_it(i: int, k: int):
        return pl_p[:, k] + cable_l * cf_cable_dir[i][:, k]

    opti.subject_to(pl_p[:, 0] == pl_p_ref[:, 0])
    opti.subject_to(pl_p[:, M - 1] == pl_p_ref[:, -1])
    opti.subject_to(pl_v[:, 0] == zero_vec)
    opti.subject_to(pl_v[:, M - 1] == zero_vec)

    for i in range(3):
        opti.subject_to(cf_p_it(i, 0) == ca.DM(cf_p0[i]))
        opti.subject_to(cf_v[i][:, 0] == zero_vec)
        opti.subject_to(cf_v[i][:, M - 1] == zero_vec)
        opti.subject_to(cf_a[i][:, 0] == zero_vec)
        opti.subject_to(cf_a[i][:, M - 2] == zero_vec)

    for i in range(3):
        for k in range(M):
            opti.subject_to(ca.sumsqr(cf_v[i][:, k]) <= cf_v_max ** 2)
        for k in range(M - 1):
            opti.subject_to(ca.sumsqr(cf_a[i][:, k]) <= cf_a_max ** 2)
        for k in range(M):
            opti.subject_to(ca.sumsqr(cf_cable_dir[i][:, k]) == 1.0)
            opti.subject_to(cf_cable_t[i][0, k] >= tension_min)
            opti.subject_to(cf_cable_t[i][0, k] <= tension_max)

    for k in range(M):
        d1 = cf_p_it(0, k) - cf_p_it(1, k)
        d2 = cf_p_it(1, k) - cf_p_it(2, k)
        d3 = cf_p_it(2, k) - cf_p_it(0, k)
        opti.subject_to(ca.sumsqr(d1) >= (cf_radius * 2) ** 2)
        opti.subject_to(ca.sumsqr(d2) >= (cf_radius * 2) ** 2)
        opti.subject_to(ca.sumsqr(d3) >= (cf_radius * 2) ** 2)

    def payload_f(x, sum_tension):
        pl_v_ = x[3:6]
        pl_a_ = (1.0 / pl_mass) * sum_tension - ca.DM([0.0, 0.0, g])
        return ca.vertcat(pl_v_, pl_a_)

    def rk4_step(xk, fk, dt_):
        k1 = fk(xk)
        k2 = fk(xk + 0.5 * dt_ * k1)
        k3 = fk(xk + 0.5 * dt_ * k2)
        k4 = fk(xk + dt_ * k3)
        return xk + (dt_ / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    for k in range(M - 1):
        tension_sum = sum(cf_cable_t[i][:, k] * cf_cable_dir[i][:, k] for i in range(3))
        xk = ca.vertcat(pl_p[:, k], pl_v[:, k])
        x_next = rk4_step(xk, lambda xx, ts=tension_sum: payload_f(xx, ts), dt)
        opti.subject_to(pl_p[:, k + 1] == x_next[0:3])
        opti.subject_to(pl_v[:, k + 1] == x_next[3:6])
        for i in range(3):
            opti.subject_to(cf_v[i][:, k + 1] == cf_v[i][:, k] + dt * cf_a[i][:, k])
            opti.subject_to(cf_p_it(i, k + 1) - cf_p_it(i, k) == dt * cf_v[i][:, k])

    J = 0
    J_k = [ca.MX(0) for _ in range(M)]
    J_obs = [ca.MX(0) for _ in range(M)]

    for k in range(M):
        J += w_pl_p * ca.sumsqr(pl_p[:, k] - ca.DM(pl_p_ref[:, k]))
        J_k[k] += w_pl_p * ca.sumsqr(pl_p[:, k] - ca.DM(pl_p_ref[:, k]))

    for i in range(3):
        for k in range(M - 1):
            J += w_tension * ca.sumsqr(cf_cable_t[i][:, k])
            J_k[k] += w_tension * ca.sumsqr(cf_cable_t[i][:, k])
            if soft_cf_pos_dyn:
                pos_res = cf_p_it(i, k + 1) - cf_p_it(i, k) - dt * cf_v[i][:, k]
                J += w_cf_p * ca.sumsqr(pos_res)
                J_k[k] += w_cf_p * ca.sumsqr(pos_res)
        for k in range(M - 2):
            cf_j = (cf_a[i][:, k + 1] - cf_a[i][:, k]) / dt
            J += w_cf_j * ca.sumsqr(cf_j)
            J_k[k] += w_cf_j * ca.sumsqr(cf_j)
        for k in range(M - 3):
            cf_s = (cf_a[i][:, k + 2] - 2 * cf_a[i][:, k + 1] + cf_a[i][:, k]) / (dt ** 2)
            J += w_cf_s * ca.sumsqr(cf_s)
            J_k[k] += w_cf_s * ca.sumsqr(cf_s)

    obs_d_safe = cf_radius
    for obs in obstacles:
        if "nogo" not in obs:
            continue
        c = np.array(obs["nogo"]["center"])
        half_cf = np.array(obs["nogo"]["size"]) / 2.0 + cf_radius
        half_pl = np.array(obs["nogo"]["size"]) / 2.0 + pl_radius
        for k in range(M):
            barrier_pl = w_obs * _box_barrier(pl_p[:, k], c, half_pl, d_safe=obs_d_safe)
            J += barrier_pl
            J_k[k] += barrier_pl
            J_obs[k] += barrier_pl
            for i in range(3):
                barrier_cf = w_obs * _box_barrier(cf_p_it(i, k), c, half_cf, d_safe=obs_d_safe)
                J += barrier_cf
                J_k[k] += barrier_cf
                J_obs[k] += barrier_cf

    opti.minimize(J)

    opti.set_initial(pl_p, pl_p_ref)
    opti.set_initial(pl_v, 0.0)
    for i in range(3):
        base_dir = (cf_p0[i] - pl_p_ref[:, 0]) / cable_l
        opti.set_initial(cf_cable_dir[i], np.tile(base_dir.reshape(3, 1), (1, M)))
        opti.set_initial(cf_cable_t[i], 0.1)
        opti.set_initial(cf_v[i], 0.0)
        opti.set_initial(cf_a[i], 0.0)

    p_opts = {"expand": True}
    s_opts = {
        "print_level": 0,
        "linear_solver": "mumps",
    }
    opti.solver("ipopt", p_opts, s_opts)
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"\n[WARNING] IPOPT did not converge: {e}")
        print("Extracting best iterate found so far...\n")
        sol = opti.debug

    pl_p_sol = sol.value(pl_p)
    pl_v_sol = sol.value(pl_v)
    cf_v_sol = [sol.value(cf_v[i]) for i in range(3)]
    cf_a_sol = []
    cf_j_sol = []
    cf_s_sol = []
    for i in range(3):
        a_i = sol.value(cf_a[i])
        a_i = np.hstack([a_i, a_i[:, -1:]])
        cf_a_sol.append(a_i)
        if a_i.shape[1] > 1:
            j_i = np.diff(a_i, axis=1) / dt
        else:
            j_i = np.zeros((3, 1))
        cf_j_sol.append(j_i)
        if j_i.shape[1] > 1:
            s_i = np.diff(j_i, axis=1) / dt
            s_i = np.hstack([s_i, s_i[:, -1:]])
        else:
            s_i = np.zeros((3, 1))
        cf_s_sol.append(s_i)
    cf_cable_dir_sol = [sol.value(cf_cable_dir[i]) for i in range(3)]
    cf_cable_t_sol = [sol.value(cf_cable_t[i]) for i in range(3)]
    cf_p_sol = [pl_p_sol + cable_l * cf_cable_dir_sol[i] for i in range(3)]
    cost_sol = np.array([float(sol.value(J_k[k])) for k in range(M)])
    cost_obs_sol = np.array([float(sol.value(J_obs[k])) for k in range(M)])

    return {
        "dt": dt,
        "t": t_grid,
        "pl_p": pl_p_sol,
        "pl_v": pl_v_sol,
        "cf1_p": cf_p_sol[0], "cf2_p": cf_p_sol[1], "cf3_p": cf_p_sol[2],
        "cf1_v": cf_v_sol[0], "cf2_v": cf_v_sol[1], "cf3_v": cf_v_sol[2],
        "cf1_a": cf_a_sol[0], "cf2_a": cf_a_sol[1], "cf3_a": cf_a_sol[2],
        "cf1_j": cf_j_sol[0], "cf2_j": cf_j_sol[1], "cf3_j": cf_j_sol[2],
        "cf1_s": cf_s_sol[0], "cf2_s": cf_s_sol[1], "cf3_s": cf_s_sol[2],
        "cf1_cable_t": cf_cable_t_sol[0],
        "cf2_cable_t": cf_cable_t_sol[1],
        "cf3_cable_t": cf_cable_t_sol[2],
        "cf1_cable_dir": cf_cable_dir_sol[0],
        "cf2_cable_dir": cf_cable_dir_sol[1],
        "cf3_cable_dir": cf_cable_dir_sol[2],
        "pl_p_ref": pl_p_ref,
        "cable_l": cable_l,
        "cf_radius": cf_radius,
        "obstacles": obstacles,
        "cost": cost_sol,
        "cost_obs": cost_obs_sol,
        "waypoints": waypoints,
    }


def get_drones_p0(cable_l, pl_height):
    alpha = np.deg2rad(30.0)
    cf_height = pl_height + cable_l * np.cos(alpha)
    gamma = 2.0 * np.pi / 3.0
    cf_R = cable_l * np.cos(alpha)
    return [
        np.array([cf_R * np.cos(0), cf_R * np.sin(0), cf_height]),
        np.array([cf_R * np.cos(gamma), cf_R * np.sin(gamma), cf_height]),
        np.array([cf_R * np.cos(2 * gamma), cf_R * np.sin(2 * gamma), cf_height]),
    ]


def save_ocp(sol, filename="ocp.npz", path="."):
    path = Path(path)
    np.savez(path / filename, **{k: v for k, v in sol.items()})
    print(f"OCP solution saved to {path / filename}")


def print_ocp_stats(sol):
    print()
    print("-" * 30)
    print("Solution stats:")
    dt = sol["dt"]
    t_grid = sol["t"]
    print(f"dt: {dt:.3f}")
    print(f"total time: {t_grid[-1]:.2f} s\n")

    for cf in ["cf1", "cf2", "cf3"]:
        print(f"Start/end for {cf}:")
        for k in [0, -1]:
            x = np.round(sol[f"{cf}_p"][:, k], 2)
            v = np.linalg.norm(sol[f"{cf}_v"][:, k])
            a = np.linalg.norm(sol[f"{cf}_a"][:, k])
            print(f" x = {x}, v = {v:.2f}, a = {a:.2f}")
        v_max = np.max(np.linalg.norm(sol[f"{cf}_v"], axis=0))
        a_max = np.max(np.linalg.norm(sol[f"{cf}_a"], axis=0))
        print(f" v_max: {v_max:.2f} m/s,  a_max: {a_max:.2f} m/s^2")
        print(f" dimensions: {sol[f'{cf}_p'].shape}")

    print("-" * 50)
    print()