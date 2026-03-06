#!/usr/bin/env python3
"""
3-drone payload OCP solver using CasADi.
"""

import numpy as np
import casadi as ca


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
    cf_radius: float = 0.2,  # crazyflie radius for drone-drone collision in m
    pl_radius: float = 0.05,  # payload radius for obstacle avoidance in m
    w_T: float = 1.0,  # time weight
    w_pl_p: float = 100.0,  # position weight
    w_pl_v: float = 0.01,  # velocity weight
    w_pl_a: float = 0.001,  # acceleration weight
    w_pl_j: float = 0.01,  # jerk weight
    w_pl_s: float = 0.01,  # snap weight
    w_cf_p: float = 0.001,  # position weight
    w_cf_v: float = 0.001,  # velocity weight
    w_cf_a: float = 0.001,  # acceleration weight
    w_cf_j: float = 0.1,  # jerk weight
    w_cf_s: float = 0.01,  # snap weight
    w_tension: float = 0.001,  # tension weight
    w_dtension: float = 0.0001,  # tension change weight
    w_pl_pT: float = 100.0,  # terminal position weight
    w_pl_vT: float = 0.1,  # terminal velocity weight
    w_pl_aT: float = 0.1,  # terminal acceleration weight
    w_cf_pT: float = 1.0,  # terminal position weight (formation)
    w_obs: float = 1e9,  # obstacle avoidance weight
    obstacles: list = [],  # list of obstacles {'center', 'size'}
    dt_min: float = 0.05,  # minimum time step for variable time grid
    dt_max: float = 0.5,  # maximum time step for variable time grid
    dt_guess: float = 0.1,  # initial guess for time step
) -> dict:
    """Solve the 3-drone payload OCP."""

    N_wp = waypoints.shape[0]

    M_max = 100
    M_min = max(N_wp, 10)
    path_length = float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))
    segment_length = cf_v_max * dt_max * 0.5
    M = int(np.clip(round(path_length / segment_length), M_min, M_max))
    print(f"\nOCP: path_length={path_length:.2f} m, solving with {M} segments\n")
    M = max(M, N_wp)
    grid_wp = np.linspace(0, 1, N_wp)
    grid_nodes = np.linspace(0, 1, M)

    wp_nodes = np.round(grid_wp * (M - 1)).astype(int)

    if obstacles:
        from scipy.interpolate import CubicSpline
        t_wp = np.linspace(0, 1, N_wp)
        cs = CubicSpline(t_wp, waypoints, bc_type='clamped')
        pl_p_ref = cs(grid_nodes).T
        pl_v_ref = cs(grid_nodes, 1).T
    else:
        pl_p_ref = np.vstack([
            np.interp(grid_nodes, grid_wp, waypoints[:, dim]) for dim in range(3)])
        pl_v_ref = np.gradient(pl_p_ref, dt_guess, axis=1)

    cf_p0 = get_drones_p0(cable_l, float(waypoints[0, 2]))

    opti = ca.Opti()

    # ######### Decision variables #########
    dt = opti.variable()
    pl_p = opti.variable(3, M)  # payload positions
    pl_v = opti.variable(3, M)  # payload velocities

    cf_v = [opti.variable(3, M) for _ in range(3)]  # drone velocities
    cf_a = [opti.variable(3, M - 1) for _ in range(3)]  # drone accelerations
    cf_cable_dir = [opti.variable(3, M) for _ in range(3)]  # unit cable directions
    cf_cable_t = [opti.variable(1, M) for _ in range(3)]  # cable tensions

    def cf_p_it(i: int, k: int):  # drone i position at time step k
        return pl_p[:, k] + cable_l * cf_cable_dir[i][:, k]

    # ######### Constraints #########

    # time step constraints
    opti.subject_to(dt >= dt_min)
    opti.subject_to(dt <= dt_max)

    # Payload boundary conditions
    opti.subject_to(pl_p[:, 0] == pl_p_ref[:, 0])
    opti.subject_to(pl_p[:, M - 1] == pl_p_ref[:, M - 1])
    opti.subject_to(pl_v[:, 0] == ca.DM.zeros(3, 1))
    opti.subject_to(pl_v[:, M - 1] == ca.DM.zeros(3, 1))

    # enforce static equilibrium at k=0 and k=M-1
    def sum_tension_vec(k: int):
        s = ca.MX.zeros(3, 1)
        for i in range(3):
            s += cf_cable_t[i][0, k] * cf_cable_dir[i][:, k]
        return s

    for k in [0, M-1]:
        opti.subject_to(pl_v[:, k] == ca.DM.zeros(3, 1))
        opti.subject_to(sum_tension_vec(k) == ca.DM([0.0, 0.0, pl_mass * g]))

    # Drones boundary conditions
    for i in range(3):
        opti.subject_to(cf_v[i][:, 0] == ca.DM.zeros(3, 1))
        opti.subject_to(cf_v[i][:, M - 1] == ca.DM.zeros(3, 1))
        opti.subject_to(cf_a[i][:, 0] == ca.DM.zeros(3, 1))
        opti.subject_to(cf_a[i][:, M - 2] == ca.DM.zeros(3, 1))

    # Drones speed/acceleration/jerk limits
    for i in range(3):
        for k in range(M):
            opti.subject_to(ca.sumsqr(cf_v[i][:, k]) <= cf_v_max ** 2)
        for k in range(M - 1):
            opti.subject_to(ca.sumsqr(cf_a[i][:, k]) <= cf_a_max ** 2)

    # Drones collision avoidance
    for k in range(M):
        d1 = cf_p_it(0, k) - cf_p_it(1, k)
        d2 = cf_p_it(1, k) - cf_p_it(2, k)
        d3 = cf_p_it(2, k) - cf_p_it(0, k)
        opti.subject_to(ca.sumsqr(d1) >= (cf_radius*2) ** 2)
        opti.subject_to(ca.sumsqr(d2) >= (cf_radius*2) ** 2)
        opti.subject_to(ca.sumsqr(d3) >= (cf_radius*2) ** 2)

    # Drones cable tension limits and unit directions
    for i in range(3):
        for k in range(M):
            d = cf_cable_dir[i][:, k]
            opti.subject_to(ca.sumsqr(d) == 1.0)
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

    # Payload RK4 + drone velocity/position update
    for k in range(M - 1):
        tension_sum = sum(cf_cable_t[i][:, k] * cf_cable_dir[i][:, k] for i in range(3))
        xk = ca.vertcat(pl_p[:, k], pl_v[:, k])
        x_next = rk4_step(xk, lambda xx, ts=tension_sum: payload_f(xx, ts), dt)
        opti.subject_to(pl_p[:, k + 1] == x_next[0:3])
        opti.subject_to(pl_v[:, k + 1] == x_next[3:6])
        for i in range(3):
            opti.subject_to(cf_v[i][:, k + 1] == cf_v[i][:, k] + dt * cf_a[i][:, k])
            opti.subject_to((cf_p_it(i, k + 1) - cf_p_it(i, k)) == dt * cf_v[i][:, k])

    # Obstacle avoidance
    # def box_sdf(p, c, h):
    #     """
    #     Signed distance from point p to box (center c, half-sizes h).
    #     Positive outside, zero on surface, negative inside.
    #     """
    #     q = ca.fabs(p - c) - h  # per-axis signed penetration

    #     outside = ca.vertcat(ca.fmax(q[0], 0),
    #                         ca.fmax(q[1], 0),
    #                         ca.fmax(q[2], 0))

    #     inside = ca.fmin(ca.fmax(ca.fmax(q[0], q[1]), q[2]), 0)

    #     return ca.norm_2(outside) + inside

    # for obs in obstacles:
    #     c_obs = np.array(obs["center"])
    #     h_cf = np.array(obs["size"]) / 2.0 + cf_radius
    #     h_pl = np.array(obs["size"]) / 2.0 + pl_radius

    #     for k in range(M):
    #         # payload
    #         opti.subject_to(box_sdf(pl_p[:, k], c_obs, h_pl) >= 0)

    #         # drones
    #         for i in range(3):
    #             opti.subject_to(box_sdf(cf_p_it(i, k), c_obs, h_cf) >= 0)

    # ######### Cost function #########
    J = 0

    # time cost
    J += w_T * (M - 1) * dt  # total time

    # waypoints soft constraints
    for j in range(1, N_wp - 1):
        k = int(wp_nodes[j])
        J += w_pl_p * ca.sumsqr(pl_p[:, k] - ca.DM(waypoints[j]))

    # payload smoothness (jerk + snap)
    for k in range(M - 2):
        pl_a_k = (pl_v[:, k + 1] - pl_v[:, k]) / dt
        pl_a_k1 = (pl_v[:, k + 2] - pl_v[:, k + 1]) / dt
        j_k = (pl_a_k1 - pl_a_k) / dt
        J += w_pl_j * ca.sumsqr(j_k) * dt
        if k < M - 3:
            pl_a_k2 = (pl_v[:, k + 3] - pl_v[:, k + 2]) / dt
            j_k1 = (pl_a_k2 - pl_a_k1) / dt
            J += w_pl_s * ca.sumsqr(j_k1 - j_k) * dt

    # payload terminal cost
    J += w_pl_pT * ca.sumsqr(pl_p[:, M - 1] - ca.DM(waypoints[-1]))
    J += w_pl_vT * ca.sumsqr(pl_v[:, M - 1])
    J += w_pl_aT * ca.sumsqr((pl_v[:, M - 1] - pl_v[:, M - 2]) / dt)

    # cf terminal cost
    cf_p0_T = get_drones_p0(cable_l, float(waypoints[-1, 2]))
    base_dirs_T = np.array([(cf_p0_T[i] - waypoints[-1]) / cable_l for i in range(3)])
    for i in range(3):
        J += w_cf_pT * ca.sumsqr(cf_cable_dir[i][:, M - 1] - ca.DM(base_dirs_T[i]))

    # cf smoothness
    for i in range(3):
        for k in range(M - 2):
            J += w_cf_j * ca.sumsqr(cf_a[i][:, k + 1] - cf_a[i][:, k]) / dt
        for k in range(M - 3):
            j_k = (cf_a[i][:, k + 1] - cf_a[i][:, k]) / dt
            j_k1 = (cf_a[i][:, k + 2] - cf_a[i][:, k + 1]) / dt
            J += w_cf_s * ca.sumsqr(j_k1 - j_k) / dt

    # cable tension
    for k in range(M):
        t1, t2, t3 = cf_cable_t[0][:, k], cf_cable_t[1][:, k], cf_cable_t[2][:, k]
        J += w_tension * (ca.sumsqr(t1 - t2) + ca.sumsqr(t2 - t3) + ca.sumsqr(t3 - t1)) * dt
    for i in range(3):
        for k in range(M - 1):
            J += w_dtension * ca.sumsqr(cf_cable_t[i][:, k + 1] - cf_cable_t[i][:, k]) / dt

    # obstacle avoidance
    def obs_penalty(p, c_obs, l_obs):
        # Normalised penetration per axis in [0, 1]
        pen_x = ca.fmax(l_obs[0] - ca.fabs(p[0] - c_obs[0]), 0) / l_obs[0]
        pen_y = ca.fmax(l_obs[1] - ca.fabs(p[1] - c_obs[1]), 0) / l_obs[1]
        pen_z = ca.fmax(l_obs[2] - ca.fabs(p[2] - c_obs[2]), 0) / l_obs[2]
        return (pen_x * pen_y * pen_z)

    for obs in obstacles:
        if "nogo" in obs:
            c = np.array(obs["nogo"]["center"])
            half_cf = np.array(obs["nogo"]["size"]) / 2.0 + cf_radius
            half_pl = np.array(obs["nogo"]["size"]) / 2.0 + pl_radius
            for k in range(M):
                # payload
                J += w_obs * obs_penalty(pl_p[:, k], c, half_pl)
                # drones
                for i in range(3):
                    J += w_obs * obs_penalty(cf_p_it(i, k), c, half_cf)

    opti.minimize(J)

    # ######### Initial guess #########
    opti.set_initial(dt, dt_guess)
    opti.set_initial(pl_p, pl_p_ref)
    opti.set_initial(pl_v, pl_v_ref)

    base_dirs = np.zeros((3, 3))
    for i in range(3):
        base_dirs[i] = (cf_p0[i] - pl_p_ref[:, 0]) / cable_l
        opti.set_initial(cf_cable_dir[i], np.tile(base_dirs[i].reshape(3, 1), (1, M)))
        opti.set_initial(cf_cable_t[i], 0.1)

    # ######### Solver #########
    s_opts = {
        "max_iter": 1000,
        "tol": 1e-4,
        "acceptable_tol": 1e-3,
        "acceptable_iter": 5,
        "linear_solver": "mumps",
        "nlp_scaling_method": "gradient-based",
        "mu_strategy": "adaptive",
        "print_level": 1,
        # "warm_start_init_point": "no",
    }
    p_opts = {"expand": True}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    # Extract
    dt_sol = float(sol.value(dt))
    t_grid = np.linspace(0.0, dt_sol * (M - 1), M)
    pl_p_sol = sol.value(pl_p)
    pl_v_sol = sol.value(pl_v)
    cf_v_sol = [sol.value(cf_v[i]) for i in range(3)]
    cf_a_sol = [sol.value(cf_a[i]) for i in range(3)]
    cf_cable_dir_sol = [sol.value(cf_cable_dir[i]) for i in range(3)]
    cf_cable_t_sol = [sol.value(cf_cable_t[i]) for i in range(3)]
    cf_p_sol = [pl_p_sol + cable_l * cf_cable_dir_sol[i] for i in range(3)]

    # print for debugging
    print(f"\nOCP solution: dt: {dt_sol:.4f} s,  T_total: {t_grid[-1]:.2f} s\n")

    return {
        "dt": dt_sol,
        "t": t_grid,
        "pl_p": pl_p_sol, "pl_v": pl_v_sol,
        "cf1_p": cf_p_sol[0], "cf2_p": cf_p_sol[1], "cf3_p": cf_p_sol[2],
        "cf1_v": cf_v_sol[0], "cf2_v": cf_v_sol[1], "cf3_v": cf_v_sol[2],
        "cf1_a": cf_a_sol[0], "cf2_a": cf_a_sol[1], "cf3_a": cf_a_sol[2],
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
        print(f" v_max: {v_max:.2f} m/s,  a_max: {a_max:.2f} m/s²")

        p = sol[f"{cf}_p"]
        print(f' dimensions: {p.shape}')

    print("-" * 50)
    print()
