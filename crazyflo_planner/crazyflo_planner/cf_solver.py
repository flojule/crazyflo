#!/usr/bin/env python3
"""
3-drone payload OCP solver using CasADi.
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
    """Interpolate payload waypoints to a fixed-time grid.

    Time horizon is sized from a raised-cosine (sinusoidal) speed profile
    that starts and ends at rest and respects cf_v_max and cf_a_max:
      v(t) = v_peak * sin(pi*t/T),  a(t) = pi*v_peak/T * cos(pi*t/T)
      peak speed : pi*L/(2T) <= cf_v_max  ->  T >= pi*L / (2*v_max)
      peak accel : pi^2*L/(2T^2) <= cf_a_max ->  T >= pi*sqrt(L/(2*a_max))
    Arc-length reparametrisation maps the cosine profile onto the waypoint
    path so the reference slows near the endpoints instead of arriving at
    full speed.
    """
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

    # Arc-length at each time node via raised-cosine profile
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

    pl_v_ref = np.gradient(pl_p_ref, dt, axis=1)
    pl_v_ref[:, 0] = 0.0
    pl_v_ref[:, -1] = 0.0

    # Waypoint node indices: invert cosine profile to find time at each waypoint
    t_wp = (T_grid / np.pi) * np.arccos(np.clip(1.0 - 2.0 * arc_frac_wp, -1.0, 1.0))
    wp_nodes = np.clip(np.round(t_wp / dt).astype(int), 0, M - 1)

    return M, dt, wp_nodes, pl_p_ref, pl_v_ref, path_length


def solve_ocp(
    waypoints: np.ndarray,  # payload reference trajectory
    cable_l: float,  # cable length
    pl_mass: float = 0.03,  # payload mass in kg
    g: float = 9.81,  # gravity acceleration
    cf_v_max: float = 2.0,  # norm of velocity limits
    cf_a_max: float = 5.0,  # norm of acceleration limits
    cf_j_max: float = 50.0,  # norm of jerk limits
    tension_min: float = 0.05,  # min tension in N
    tension_max: float = 0.15,  # max tension in N
    cf_radius: float = 0.1,  # crazyflie radius for drone-drone collision in m
    pl_radius: float = 0.05,  # payload radius for obstacle avoidance in m
    w_T: float = 1e0,  # unused in fixed-time mode, kept for interface compatibility
    w_pl_p: float = 1e6,  # position weight
    w_pl_v: float = 1e0,  # velocity weight
    w_pl_a: float = 1e0,  # acceleration weight
    w_pl_j: float = 1e0,  # jerk weight
    w_pl_s: float = 1e0,  # snap weight
    w_cf_p: float = 1e3,  # derived drone position consistency weight
    w_cf_v: float = 1e0,  # velocity weight
    w_cf_a: float = 1e0,  # acceleration weight
    w_cf_j: float = 1e3,  # jerk weight
    w_cf_s: float = 1e1,  # snap weight
    w_tension: float = 1e0,  # tension weight
    w_dtension: float = 1e-1,  # tension change weight
    w_pl_pT: float = 1e0,  # terminal position weight
    w_pl_vT: float = 1e0,  # terminal velocity weight
    w_pl_aT: float = 1e0,  # terminal acceleration weight
    w_cf_pT: float = 1e0,  # terminal position weight (formation)
    w_obs: float = 1e3,  # obstacle avoidance weight
    obstacles: list | None = None,  # list of obstacles {'center', 'size'}
    soft_cf_pos_dyn: bool = False,  # relax derived drone position dynamics into a penalty term
    dt_min: float = 0.1,  # minimum time step for variable time grid
    dt_max: float = 0.3,  # maximum time step for variable time grid
    dt_guess: float = 0.1,  # initial guess for time step
) -> dict:
    """Solve the 3-drone payload OCP."""
    if obstacles is None:
        obstacles = []

    N_wp = waypoints.shape[0]
    M, dt, wp_nodes, pl_p_ref, pl_v_ref, path_length = _build_reference_grid(
        waypoints, obstacles, cf_v_max, cf_a_max, dt_min, dt_max)
    grid_nodes = np.linspace(0, 1, M)

    print(f"\nPath length = {path_length:.2f} m, N = {N_wp}, M = {M}, dt = {dt:.3f} s\n")

    cf_p0 = get_drones_p0(cable_l, float(waypoints[0, 2]))

    opti = ca.Opti()

    # ######### Decision variables #########
    pl_p = opti.variable(3, M)  # payload positions
    pl_v = opti.variable(3, M)  # payload velocities

    cf_v = [opti.variable(3, M) for _ in range(3)]  # drone velocities
    cf_a = [opti.variable(3, M) for _ in range(3)]  # drone accelerations
    cf_j = [opti.variable(3, M - 1) for _ in range(3)]  # drone jerks
    cf_cable_dir = [opti.variable(3, M) for _ in range(3)]  # unit cable directions
    cf_cable_t = [opti.variable(1, M) for _ in range(3)]  # cable tensions
    zero_vec_dm = ca.DM(np.zeros((3, 1)))
    zero_vec_mx = ca.MX(np.zeros((3, 1)))
    cf_pos_dyn_res = [[zero_vec_mx for _ in range(M - 1)] for _ in range(3)]

    def cf_p_it(i: int, k: int):  # drone i position at time step k
        return pl_p[:, k] + cable_l * cf_cable_dir[i][:, k]

    # ######### Constraints #########

    # Payload boundary conditions
    opti.subject_to(pl_p[:, 0] == pl_p_ref[:, 0])
    opti.subject_to(pl_p[:, M - 1] == pl_p_ref[:, M - 1])
    opti.subject_to(pl_v[:, 0] == zero_vec_dm)
    opti.subject_to(pl_v[:, M - 1] == zero_vec_dm)

    # Drone formation at start and end: fix cable directions to the nominal
    # equilateral-triangle formation so the drones begin and finish in formation.
    cf_p0_end = get_drones_p0(cable_l, float(waypoints[-1, 2]))
    for i in range(3):
        dir_start = np.array(cf_p0[i] - waypoints[0]) / cable_l
        dir_start /= np.linalg.norm(dir_start)
        dir_end = np.array(cf_p0_end[i] - waypoints[-1]) / cable_l
        dir_end /= np.linalg.norm(dir_end)
        opti.subject_to(cf_cable_dir[i][:, 0] == ca.DM(dir_start))
        opti.subject_to(cf_cable_dir[i][:, M - 1] == ca.DM(dir_end))

    # Drones boundary conditions
    for i in range(3):
        opti.subject_to(cf_v[i][:, 0] == zero_vec_dm)
        opti.subject_to(cf_v[i][:, M - 1] == zero_vec_dm)
        opti.subject_to(cf_a[i][:, 0] == zero_vec_dm)
        opti.subject_to(cf_a[i][:, M - 1] == zero_vec_dm)
        opti.subject_to(cf_j[i][:, 0] == zero_vec_dm)
        opti.subject_to(cf_j[i][:, M - 2] == zero_vec_dm)

    # Drones speed/acceleration/jerk limits
    for i in range(3):
        for k in range(M):
            opti.subject_to(ca.sumsqr(cf_v[i][:, k]) <= cf_v_max ** 2)
            opti.subject_to(ca.sumsqr(cf_a[i][:, k]) <= cf_a_max ** 2)
        for k in range(M - 1):
            opti.subject_to(ca.sumsqr(cf_j[i][:, k]) <= cf_j_max ** 2)

    # Drones collision avoidance
    for k in range(M):
        d1 = cf_p_it(0, k) - cf_p_it(1, k)
        d2 = cf_p_it(1, k) - cf_p_it(2, k)
        d3 = cf_p_it(2, k) - cf_p_it(0, k)
        opti.subject_to(ca.sumsqr(d1) >= (cf_radius*2) ** 2)
        opti.subject_to(ca.sumsqr(d2) >= (cf_radius*2) ** 2)
        opti.subject_to(ca.sumsqr(d3) >= (cf_radius*2) ** 2)

    # Cable unit-norm: hard equality
    for i in range(3):
        for k in range(M):
            opti.subject_to(ca.sumsqr(cf_cable_dir[i][:, k]) == 1.0)

    # Drones cable tension limits
    for i in range(3):
        opti.subject_to(cf_cable_t[i] >= tension_min)
        opti.subject_to(cf_cable_t[i] <= tension_max)

    # Drones above payload
    for i in range(3):
        for k in range(M):
            dz = cf_cable_dir[i][2, k]
            opti.subject_to(dz >= 1e-3)

    # Dynamics
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

    # Payload RK4 + drone constant-jerk state update
    for k in range(M - 1):
        tension_sum = sum(cf_cable_t[i][:, k] * cf_cable_dir[i][:, k] for i in range(3))
        xk = ca.vertcat(pl_p[:, k], pl_v[:, k])
        x_next = rk4_step(xk, lambda xx, ts=tension_sum: payload_f(xx, ts), dt)
        opti.subject_to(pl_p[:, k + 1] == x_next[0:3])
        opti.subject_to(pl_v[:, k + 1] == x_next[3:6])
        for i in range(3):
            a_next = cf_a[i][:, k] + dt * cf_j[i][:, k]
            opti.subject_to(cf_a[i][:, k + 1] == a_next)
            opti.subject_to(
                cf_v[i][:, k + 1] == cf_v[i][:, k] + 0.5 * dt * (cf_a[i][:, k] + cf_a[i][:, k + 1])
            )
            cf_pos_dyn_res[i][k] = (
                cf_p_it(i, k + 1)
                - cf_p_it(i, k)
                - 0.5 * dt * (cf_v[i][:, k] + cf_v[i][:, k + 1])
                + (dt ** 2 / 12.0) * (cf_a[i][:, k + 1] - cf_a[i][:, k])
            )
            if not soft_cf_pos_dyn:
                opti.subject_to(cf_pos_dyn_res[i][k] == 0)

    # ######### Cost function #########
    J = 0
    J_k = [ca.MX(0) for _ in range(M)]  # for debugging and plotting cost terms over time
    J_obs = [ca.MX(0) for _ in range(M)]

    # fixed-time mode: no time decision variable, so no time cost term

    # waypoints soft constraints
    for j in range(1, N_wp - 1):
        k = int(wp_nodes[j])
        J += w_pl_p * ca.sumsqr(pl_p[:, k] - ca.DM(waypoints[j]))
        J += w_pl_v * ca.sumsqr(pl_v[:, k])

        J_k[k] += w_pl_p * ca.sumsqr(pl_p[:, k] - ca.DM(waypoints[j]))
        J_k[k] += w_pl_v * ca.sumsqr(pl_v[:, k])

    # # payload smoothness (jerk + snap)
    # for k in range(M - 2):
    #     pl_a_k = (pl_v[:, k + 1] - pl_v[:, k]) / dt
    #     pl_a_k1 = (pl_v[:, k + 2] - pl_v[:, k + 1]) / dt
    #     j_k = (pl_a_k1 - pl_a_k) / dt
    #     J += w_pl_j * ca.sumsqr(j_k) * dt
    #     J_k[k] += w_pl_j * ca.sumsqr(j_k) * dt
    #     if k < M - 3:
    #         pl_a_k2 = (pl_v[:, k + 3] - pl_v[:, k + 2]) / dt
    #         j_k1 = (pl_a_k2 - pl_a_k1) / dt
    #         J += w_pl_s * ca.sumsqr(j_k1 - j_k) * dt
    #         J_k[k] += w_pl_s * ca.sumsqr(j_k1 - j_k) * dt

    # payload terminal cost
    # J += w_pl_pT * ca.sumsqr(pl_p[:, M - 1] - ca.DM(waypoints[-1]))
    J += w_pl_vT * ca.sumsqr(pl_v[:, M - 1])
    J += w_pl_aT * ca.sumsqr((pl_v[:, M - 1] - pl_v[:, M - 2]) / dt)

    # J_k[M - 1] += w_pl_pT * ca.sumsqr(pl_p[:, M - 1] - ca.DM(waypoints[-1]))
    J_k[M - 1] += w_pl_vT * ca.sumsqr(pl_v[:, M - 1])
    J_k[M - 1] += w_pl_aT * ca.sumsqr((pl_v[:, M - 1] - pl_v[:, M - 2]) / dt)

    # cf terminal cost
    cf_p0_T = get_drones_p0(cable_l, float(waypoints[-1, 2]))
    base_dirs_T = np.array([(cf_p0_T[i] - waypoints[-1]) / cable_l for i in range(3)])
    for i in range(3):
        J += w_cf_pT * ca.sumsqr(cf_cable_dir[i][:, M - 1] - ca.DM(base_dirs_T[i]))
        J_k[M - 1] += w_cf_pT * ca.sumsqr(cf_cable_dir[i][:, M - 1] - ca.DM(base_dirs_T[i]))

    # cf smoothness and optional soft consistency for derived positions
    for i in range(3):
        # for k in range(M):
        #     J += w_cf_v * ca.sumsqr(cf_v[i][:, k]) * dt
        #     J_k[k] += w_cf_v * ca.sumsqr(cf_v[i][:, k]) * dt
        # for k in range(M):
        #     J += w_cf_a * ca.sumsqr(cf_a[i][:, k]) * dt
        #     J_k[k] += w_cf_a * ca.sumsqr(cf_a[i][:, k]) * dt
        for k in range(M - 1):
            # J += w_cf_j * ca.sumsqr(cf_j[i][:, k]) * dt
            # J_k[k] += w_cf_j * ca.sumsqr(cf_j[i][:, k]) * dt
            if soft_cf_pos_dyn:
                J += w_cf_p * ca.sumsqr(cf_pos_dyn_res[i][k]) / dt
                J_k[k] += w_cf_p * ca.sumsqr(cf_pos_dyn_res[i][k]) / dt
        # for k in range(M - 2):
        #     snap_k = (cf_j[i][:, k + 1] - cf_j[i][:, k]) / dt
        #     J += w_cf_s * ca.sumsqr(snap_k) * dt
        #     J_k[k] += w_cf_s * ca.sumsqr(snap_k) * dt

    # cable tension
    # for k in range(M):
    #     t1, t2, t3 = cf_cable_t[0][:, k], cf_cable_t[1][:, k], cf_cable_t[2][:, k]
    #     J += w_tension * (ca.sumsqr(t1 - t2) + ca.sumsqr(t2 - t3) + ca.sumsqr(t3 - t1)) * dt
    #     J_k[k] += w_tension * (ca.sumsqr(t1 - t2) + ca.sumsqr(t2 - t3) + ca.sumsqr(t3 - t1)) * dt
    # for i in range(3):
    #     for k in range(M - 1):
    #         J += w_dtension * ca.sumsqr(cf_cable_t[i][:, k + 1] - cf_cable_t[i][:, k]) / dt
    #         J_k[k] += w_dtension * ca.sumsqr(cf_cable_t[i][:, k + 1] - cf_cable_t[i][:, k]) / dt

    def box_barrier(p, c_obs, l_obs, d_safe=0.3):
        """
        Conjunction log barrier for an axis-aligned box obstacle.

        Axis i is penalised only when the agent is inside the *other* two axes
        (smooth sigmoid indicator).  This gives a nonzero gradient even when
        the agent sits dead-centre on a thin slab (where the L-inf gradient
        would be zero), while still giving zero cost inside any passage gap.

        The log barrier is extended quadratically inside the obstacle so the
        solver always has finite, nonzero gradient to push the agent out.
        """
        eps = 1e-2      # log-barrier smoothing threshold
        alpha_in = 5.0  # sigmoid sharpness for "inside other axes" indicator

        # Normalised signed distances: >0 outside that axis range, <0 inside
        gaps = [(ca.fabs(p[ax] - float(c_obs[ax])) - float(l_obs[ax])) / d_safe
                for ax in range(3)]

        def inside_w(ax):
            """Smooth weight ≈1 when inside axis ax, ≈0 when outside."""
            return 1.0 / (1.0 + ca.exp(alpha_in * gaps[ax]))

        def patched_log(g):
            """Log barrier with quadratic extension inside (g < eps) for gradient."""
            return ca.if_else(
                g > eps,
                -ca.log(g),
                -ca.log(eps) - (1.0 / eps) * (g - eps)
                + (0.5 / eps ** 2) * (g - eps) ** 2,
            )

        cost = ca.MX(0.0)
        for i in range(3):
            j, k_ax = (i + 1) % 3, (i + 2) % 3
            gap_i = gaps[i]
            # Weight: activate axis-i barrier only when inside the other two axes
            w_in = inside_w(j) * inside_w(k_ax)
            # Quintic taper to zero at the safe horizon (gap_i = 1)
            t = ca.fmax(ca.fmin(gap_i, 1.0), 0.0)
            blend = 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t ** 2)
            cost = cost + w_in * blend * patched_log(gap_i)
        return cost

    obs_d_safe = cf_radius * 1.0  # body radius from wall surface ≥ 1 steps
    for obs in obstacles:
        if "nogo" in obs:
            c = np.array(obs["nogo"]["center"])
            half_cf = np.array(obs["nogo"]["size"]) / 2.0 + cf_radius
            half_pl = np.array(obs["nogo"]["size"]) / 2.0 + pl_radius
            for k in range(M):
                # payload
                J += w_obs * box_barrier(
                    pl_p[:, k], c, half_pl, d_safe=obs_d_safe)
                J_k[k] += w_obs * box_barrier(
                    pl_p[:, k], c, half_pl, d_safe=obs_d_safe)
                J_obs[k] += w_obs * box_barrier(
                    pl_p[:, k], c, half_pl, d_safe=obs_d_safe)
                # drones
                for i in range(3):
                    J += w_obs * box_barrier(
                        cf_p_it(i, k), c, half_cf, d_safe=obs_d_safe)
                    J_k[k] += w_obs * box_barrier(
                        cf_p_it(i, k), c, half_cf, d_safe=obs_d_safe)
                    J_obs[k] += w_obs * box_barrier(
                        cf_p_it(i, k), c, half_cf, d_safe=obs_d_safe)

    opti.minimize(J)

    # ######### Initial guess #########
    pl_v_guess = np.array(pl_v_ref, copy=True)
    pl_v_guess[:, 0] = 0.0
    pl_v_guess[:, -1] = 0.0

    opti.set_initial(pl_p, pl_p_ref)
    opti.set_initial(pl_v, pl_v_guess)

    # Build a time-varying cable direction guess by interpolating the formation
    # orientation from the start position to the end position.  This gives the
    # solver a hint about how the formation needs to rotate/translate along the
    # trajectory rather than holding a fixed initial orientation.
    for i in range(3):
        dir_start = (cf_p0[i] - pl_p_ref[:, 0]) / cable_l
        dir_end = (cf_p0_end[i] - pl_p_ref[:, -1]) / cable_l
        dir_start /= np.linalg.norm(dir_start)
        dir_end /= np.linalg.norm(dir_end)
        dirs_guess = np.outer(dir_start, 1 - grid_nodes) + np.outer(dir_end, grid_nodes)
        # re-normalise each column so the unit-norm constraint is satisfied
        dirs_guess /= np.linalg.norm(dirs_guess, axis=0, keepdims=True)
        cf_p_guess = pl_p_ref + cable_l * dirs_guess
        cf_v_guess = np.gradient(cf_p_guess, dt, axis=1)
        cf_a_guess = np.gradient(cf_v_guess, dt, axis=1)
        cf_j_guess = np.gradient(cf_a_guess, dt, axis=1)[:, :-1]
        cf_v_guess[:, 0] = 0.0
        cf_v_guess[:, -1] = 0.0
        cf_a_guess[:, 0] = 0.0
        cf_a_guess[:, -1] = 0.0
        if cf_j_guess.shape[1] > 0:
            cf_j_guess[:, 0] = 0.0
            cf_j_guess[:, -1] = 0.0
        opti.set_initial(cf_cable_dir[i], dirs_guess)
        opti.set_initial(cf_cable_t[i], 0.1)
        opti.set_initial(cf_v[i], cf_v_guess)
        opti.set_initial(cf_a[i], cf_a_guess)
        opti.set_initial(cf_j[i], cf_j_guess)

    # ######### Solver #########
    use_fast_ipopt = bool(obstacles) or M > 60
    s_opts = {
        # "max_iter": 2000,
        # "tol": 1e-4,
        # "acceptable_tol": 1e-2,
        # "acceptable_iter": 10,
        "linear_solver": "mumps",
        # "nlp_scaling_method": "gradient-based",
        # "mu_strategy": "adaptive",
        "print_level": 3,
    }
    if use_fast_ipopt:
        s_opts.update({
            "tol": 1e-3,
            "acceptable_tol": 5e-2,
            "hessian_approximation": "limited-memory",
            "print_level": 0,
        })
    p_opts = {"expand": False}
    opti.solver("ipopt", p_opts, s_opts)
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"\n[WARNING] IPOPT did not converge: {e}")
        print("Extracting best iterate found so far...\n")
        sol = opti.debug

    # Extract
    dt_sol = float(dt)
    t_grid = np.linspace(0.0, dt_sol * (M - 1), M)
    pl_p_sol = sol.value(pl_p)
    pl_v_sol = sol.value(pl_v)
    cf_v_sol = [sol.value(cf_v[i]) for i in range(3)]
    cf_a_sol = [sol.value(cf_a[i]) for i in range(3)]
    cf_j_sol = [sol.value(cf_j[i]) for i in range(3)]
    cf_cable_dir_sol = [sol.value(cf_cable_dir[i]) for i in range(3)]
    cf_cable_t_sol = [sol.value(cf_cable_t[i]) for i in range(3)]
    cf_p_sol = [pl_p_sol + cable_l * cf_cable_dir_sol[i] for i in range(3)]
    cf_s_sol = []
    for i in range(3):
        if M > 1:
            snap_i = np.diff(cf_j_sol[i], axis=1) / dt_sol
            if snap_i.shape[1] == 0:
                snap_i = np.zeros((3, 1))
            else:
                snap_i = np.hstack([snap_i, snap_i[:, -1:]])
        else:
            snap_i = np.zeros((3, 1))
        cf_s_sol.append(snap_i)
    cost_sol = np.array([float(sol.value(J_k[k])) for k in range(M)])
    cost_obs_sol = np.array([float(sol.value(J_obs[k])) for k in range(M)])

    # print for debugging
    print(f"\nOCP solution: dt: {dt_sol:.4f} s,  T_total: {t_grid[-1]:.2f} s\n")

    return {
        "dt": dt_sol,
        "t": t_grid,
        "pl_p": pl_p_sol, "pl_v": pl_v_sol,
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
        "waypoints": waypoints
    }


def get_drones_p0(cable_l, pl_height):
    # drones initial positions
    alpha = np.deg2rad(30.0)  # initial rope angle
    cf_height = pl_height + cable_l * np.cos(alpha)
    gamma = 2.0 * np.pi / 3.0  # initial drone separation angle
    cf_R = cable_l * np.cos(alpha)  # radius of drone formation
    cf_p0 = [
        np.array([cf_R * np.cos(0), cf_R * np.sin(0), cf_height]),
        np.array([cf_R * np.cos(gamma), cf_R * np.sin(gamma), cf_height]),
        np.array([cf_R * np.cos(2 * gamma), cf_R * np.sin(2 * gamma), cf_height]),
    ]
    return cf_p0


def save_ocp(sol, filename="ocp.npz", path="."):
    """Save OCP solution trajectories to npz file."""
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

    cf_name = ["cf1", "cf2", "cf3"]
    for cf in cf_name:

        print(f"Start/end for {cf}:")
        for k in [0, -1]:
            x = np.round(sol[f"{cf}_p"][:, k], 2)
            v = np.linalg.norm(sol[f"{cf}_v"][:, k])
            a = np.linalg.norm(sol[f"{cf}_a"][:, k])
            print(f' x = {x}, v = {v:.2f}, a = {a:.2f}')

        # print max velocity and acceleration
        v_max = np.max(np.linalg.norm(sol[f"{cf}_v"], axis=0))
        a_max = np.max(np.linalg.norm(sol[f"{cf}_a"], axis=0))
        print(f" v_max: {v_max:.2f} m/s,  a_max: {a_max:.2f} m/s^2")

        p = sol[f"{cf}_p"]
        print(f' dimensions: {p.shape}')

    print("-" * 50)
    print()
