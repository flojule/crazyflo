#!/usr/bin/env python3
"""
3-drone payload OCP solver using CasADi.
"""

import numpy as np
from pathlib import Path

import casadi as ca

ROOT_FOLDER = Path.home() / ".ros/crazyflo_planner"


def solve_ocp(
    pl_waypoints: np.ndarray,  # payload reference trajectory
    cf_p0: list[np.ndarray],  # initial drone positions
    cable_l: float,  # cable length
    pl_mass: float = 0.03,  # payload mass in kg
    g: float = 9.81,  # gravity acceleration
    cf_v_max: float = 2.0,  # norm of velocity limits
    cf_a_max: float = 5.0,  # norm of acceleration limits
    cf_j_max: float = 50.0,  # norm of jerk limits
    tension_min: float = 0.05,  # min tension in N
    tension_max: float = 0.15,  # max tension in N
    cf_radius: float = 0.2,  # crazyflie radius for drone-drone collision in m
    w_pl_p: float = 10.0,  # position weight
    w_pl_v: float = 0.01,  # velocity weight
    w_pl_a: float = 0.001,  # acceleration weight
    w_pl_j: float = 0.0001,  # jerk weight
    w_pl_s: float = 0.0001,  # snap weight
    w_cf_p: float = 0.001,  # position weight
    w_cf_v: float = 0.001,  # velocity weight
    w_cf_a: float = 0.001,  # acceleration weight
    w_cf_j: float = 0.01,  # jerk weight
    w_cf_s: float = 0.01,  # snap weight
    w_tension: float = 0.0001,  # tension weight
    w_dtension: float = 0.0001,  # tension change weight
    w_pl_pT: float = 0.01,  # terminal position weight
    w_pl_vT: float = 0.001,  # terminal velocity weight
    w_pl_aT: float = 0.001,  # terminal acceleration weight
    w_cf_pT: float = 0.001,  # terminal position weight
    ros: bool = False,
) -> dict:
    """Solve the 3-drone payload OCP."""
    if ros:
        from . import poly7
    else:
        import poly7

    pl_scale = 1.0  # scale max payload velocity/acceleration
    segments = poly7.fit_poly7_piecewise(
        pl_waypoints, cf_v_max*pl_scale, cf_a_max*pl_scale, cf_j_max*pl_scale)

    states = poly7.get_waypoint_states(segments)

    t_grid = states["t"]
    print(f"min seg dt: {np.min(np.diff(t_grid)):.4f}")
    print(f"max seg dt: {np.max(np.diff(t_grid)):.4f}")
    pl_p_ref = states["p"].T
    pl_v_ref = states["v"].T

    M = pl_p_ref.shape[1]  # number of reference points

    opti = ca.Opti()

    # ######### Decision variables #########
    pl_p = opti.variable(3, M)  # payload positions
    pl_v = opti.variable(3, M)  # payload velocities

    # cf_p = [opti.variable(3, M) for _ in range(3)]  # drone positions
    cf_v = [opti.variable(3, M) for _ in range(3)]  # drone velocities
    cf_a = [opti.variable(3, M - 1) for _ in range(3)]  # drone accelerations
    cf_cable_dir = [opti.variable(3, M) for _ in range(3)]  # unit cable directions
    cf_cable_t = [opti.variable(1, M) for _ in range(3)]  # cable tensions

    def cf_p_it(i: int, k: int):  # drone i position at time step k
        return pl_p[:, k] + cable_l * cf_cable_dir[i][:, k]

    # ######### Constraints #########
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
        # opti.subject_to(cf_p_it(i, 0) == cf_p0[i])
        # opti.subject_to(cf_p_it(i, 0)[2] == cf_p0[i][2])
        # opti.subject_to(cf_p_it(i, M - 1)[2] == cf_p0[i][2])
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
            eps = 1e-6
            d = cf_cable_dir[i][:, k]
            opti.subject_to(ca.sumsqr(d) >= 1.0 - eps)
            opti.subject_to(ca.sumsqr(d) <= 1.0 + eps)
        opti.subject_to(cf_cable_t[i] >= tension_min)
        opti.subject_to(cf_cable_t[i] <= tension_max)

    # Drones above payload
    for i in range(3):
        for k in range(M):
            dz = cf_cable_dir[i][2, k]
            opti.subject_to(dz >= 1e-3)

    # Dynamics
    def payload_f(x: ca.MX, sum_tension_vec: ca.MX) -> ca.MX:
        pl_v_ = x[3:6]
        pl_a_ = (1.0 / pl_mass) * sum_tension_vec - ca.DM([0.0, 0.0, g])
        return ca.vertcat(pl_v_, pl_a_)

    def rk4_step(xk: ca.MX, fk, dt_: float) -> ca.MX:
        k1 = fk(xk)
        k2 = fk(xk + 0.5 * dt_ * k1)
        k3 = fk(xk + 0.5 * dt_ * k2)
        k4 = fk(xk + dt_ * k3)
        return xk + (dt_ / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Payload RK4 + drone velocity/position update (variable step from t_grid)
    for k in range(M - 1):
        dt_k = float(t_grid[k + 1] - t_grid[k])  # <-- HERE

        sum_tension = ca.DM.zeros(3, 1)
        for i in range(3):
            sum_tension += cf_cable_t[i][:, k] * cf_cable_dir[i][:, k]

        xk = ca.vertcat(pl_p[:, k], pl_v[:, k])

        # payload RK4 step with dt_k
        x_next = rk4_step(
            xk, fk=lambda xx: payload_f(xx, sum_tension), dt_=dt_k)
        opti.subject_to(pl_p[:, k + 1] == x_next[0:3])
        opti.subject_to(pl_v[:, k + 1] == x_next[3:6])

        # drones position and velocity consistency with dt_k
        for i in range(3):
            opti.subject_to(cf_v[i][:, k + 1] == cf_v[i][:, k] + dt_k * cf_a[i][:, k])
            opti.subject_to((cf_p_it(i, k + 1) - cf_p_it(i, k)) == dt_k * cf_v[i][:, k])

    # ######### Cost function #########
    J = 0

    # payload position tracking, jerk and snap energy minimization
    for k in range(M):
        # J += w_pl_p * ca.sumsqr(pl_p[:, k] - ca.DM(pl_p_ref[:, k]))  # position tracking

        if k < M - 2:  # jerk
            dt_k = t_grid[k+1] - t_grid[k]

            J += w_pl_p * ca.sumsqr(pl_p[:, k] - ca.DM(pl_p_ref[:, k])) * dt_k  # position tracking
            # J += w_pl_v * ca.sumsqr(pl_v[:, k] - ca.DM(pl_v_ref[:, k])) * dt_k  # velocity tracking

            pl_a = ca.vertcat(pl_v[:, k+1] - pl_v[:, k]) / dt_k
            dt_k1 = t_grid[k+2] - t_grid[k+1]
            pl_a_k1 = ca.vertcat(pl_v[:, k+2] - pl_v[:, k+1]) / dt_k1
            dt_mid = 0.5*(dt_k + dt_k1)
            j_k = (pl_a_k1 - pl_a) / dt_mid
            J += w_pl_j * ca.sumsqr(j_k) * dt_mid  # jerk energy

            if k < M - 3:  # snap
                dt_k2 = t_grid[k+3] - t_grid[k+2]
                dt_mid2 = 0.5*(dt_k1 + dt_k2)
                pl_a_k2 = ca.vertcat(pl_v[:, k+3] - pl_v[:, k+2]) / dt_k2
                j_k1 = (pl_a_k2 - pl_a_k1) / dt_mid2
                s_k = (j_k1 - j_k) / dt_mid2
                J += w_pl_s * ca.sumsqr(s_k) * dt_mid2  # snap energy

    # payload terminal cost
    J += w_pl_pT * ca.sumsqr(pl_p[:, M - 1] - ca.DM(pl_p_ref[:, -1]))
    J += w_pl_vT * ca.sumsqr(pl_v[:, M - 1])
    pl_a_T = ca.vertcat(pl_v[:, M-1] - pl_v[:, M-2]) / (t_grid[M-1] - t_grid[M-2])
    J += w_pl_aT * ca.sumsqr(pl_a_T)  # terminal acceleration

    # cf terminal cost
    for i in range(3):  # max z position
        J += -w_cf_pT * cf_p_it(i, M - 1)[2]

    # cf smoothness
    for i in range(3):
        # for k in range(M - 1):
        #     J += w_cf_a * ca.sumsqr(cf_a[i][:, k])  # acceleration

        for k in range(M - 2):
            dt_k = t_grid[k+1] - t_grid[k]
            da = cf_a[i][:, k+1] - cf_a[i][:, k]
            J += w_cf_j * ca.sumsqr(da) / dt_k  # jerk energy

        for k in range(M - 3):
            dt_k = t_grid[k+1] - t_grid[k]
            dt_k1 = t_grid[k+2] - t_grid[k+1]
            j_k = (cf_a[i][:, k+1]-cf_a[i][:, k]) / dt_k
            j_k1 = (cf_a[i][:, k+2]-cf_a[i][:, k+1]) / dt_k1
            dt_mid = 0.5*(dt_k + dt_k1)
            J += w_cf_s * ca.sumsqr(j_k1 - j_k) / dt_mid  # snap energy

    # cable tension
    for i in range(3):
        for k in range(M - 2):  # cable tension
            dt_k = t_grid[k + 1] - t_grid[k]
            # J += w_tension * ca.sumsqr(cf_cable_t[i][:, k]) * dt_k  # tension energy

            d_tension = cf_cable_t[i][:, k + 1] - cf_cable_t[i][:, k]
            J += w_dtension * ca.sumsqr(d_tension) / dt_k  # tension change energy

    # minimize tension difference between drones
    for k in range(M):
        t1 = cf_cable_t[0][:, k]
        t2 = cf_cable_t[1][:, k]
        t3 = cf_cable_t[2][:, k]
        J += w_tension * (ca.sumsqr(t1 - t2) + ca.sumsqr(t2 - t3) + ca.sumsqr(t3 - t1))

    opti.minimize(J)

    # ######### Initial guess #########
    opti.set_initial(pl_p, pl_p_ref)
    opti.set_initial(pl_v, 0.0)

    base_dirs = np.zeros((3, 3))
    for i in range(3):
        base_dirs[i] = (cf_p0[i] - pl_p_ref[:, 0]) / cable_l
        opti.set_initial(cf_cable_dir[i], np.tile(base_dirs[i].reshape(3, 1), (1, M)))
        opti.set_initial(cf_cable_t[i], 0.1)

    # ######### Solver #########
    p_opts = {"expand": True}
    opti.solver("ipopt", p_opts)
    sol = opti.solve()

    # Extract
    pl_p_sol = sol.value(pl_p)
    pl_v_sol = sol.value(pl_v)
    cf_v_sol = [sol.value(cf_v[i]) for i in range(3)]
    cf_a_sol = [sol.value(cf_a[i]) for i in range(3)]
    cf_cable_dir_sol = [sol.value(cf_cable_dir[i]) for i in range(3)]
    cf_cable_t_sol = [sol.value(cf_cable_t[i]) for i in range(3)]
    cf_p_sol = [pl_p_sol + cable_l * cf_cable_dir_sol[i] for i in range(3)]

    # print for debugging
    print(f"min seg dt: {np.min(np.diff(t_grid)):.4f}")
    print(f"max seg dt: {np.max(np.diff(t_grid)):.4f}")
    print(f"total T: {t_grid[-1]:.2f}\n")

    return {
        "t": t_grid,
        "pl_p": pl_p_sol, "pl_v": pl_v_sol,
        "cf1_p": cf_p_sol[0], "cf2_p": cf_p_sol[1], "cf3_p": cf_p_sol[2],
        "cf1_v": cf_v_sol[0], "cf2_v": cf_v_sol[1], "cf3_v": cf_v_sol[2],
        "cf1_a": cf_a_sol[0], "cf2_a": cf_a_sol[1], "cf3_a": cf_a_sol[2],
        "cf1_cable_t": cf_cable_t_sol[0], "cf2_cable_t": cf_cable_t_sol[1], "cf3_cable_t": cf_cable_t_sol[2],
        "cf1_cable_dir": cf_cable_dir_sol[0], "cf2_cable_dir": cf_cable_dir_sol[1], "cf3_cable_dir": cf_cable_dir_sol[2],
        "pl_p_ref": pl_p_ref,
        "cable_l": cable_l,
        "cf_radius": cf_radius,
    }


def get_traj(traj='ellipse', loops=5, plot=True, save_csv=True, ros=False):
    """Solve an example OCP and export trajectories."""
    if ros:
        from . import cf_plots
        from . import poly7
        from . import cf_waypoints
    else:
        import cf_plots
        import poly7
        import cf_waypoints

    cf_height = 1.0  # solve at ~1m height
    cable_l = 0.5  # cable lengths
    alpha = np.deg2rad(30.0)  # initial rope angle
    gamma = 2.0 * np.pi / 3.0  # initial drone separation angle
    pl_height = cf_height - cable_l * np.cos(alpha)

    N = 20  # number of trajectory points for one loop
    grid = np.linspace(0, 1, N + 1)  # points for traj

    if traj == 'ellipse':
        pl_waypoints = cf_waypoints.generate_ellipse(r_A=0.6, r_B=0.3, height=pl_height, grid=grid)
    elif traj == 'figure8':
        pl_waypoints = cf_waypoints.generate_figure8(r_A=0.6, r_B=0.6, height=pl_height, grid=grid)
    elif traj == 'random':
        start = np.array([0, 0, pl_height])
        step_size = 0.5
        pl_waypoints = cf_waypoints.generate_random_walk(start=start, step_size=step_size, grid=grid)
    elif traj == 'line':  # straight line
        start = np.array([0, 0, pl_height])
        goal = np.array([10, 0, 2*pl_height])
        pl_waypoints = cf_waypoints.generate_line(start=start, end=goal, grid=grid)
    else:
        raise ValueError(f"Unknown traj type: {traj}")

    if loops > 1 and traj != 'line':
        p0 = pl_waypoints.copy()  # one loop, length N+1
        pl_waypoints = np.concatenate([p0[:-1] for _ in range(loops - 1)] + [p0], axis=0)

    # drone initial positions
    cf_R = cable_l * np.cos(alpha)  # radius of drone formation
    cf_p0 = [
        np.array([cf_R * np.cos(0), cf_R * np.sin(0), cf_height]),
        np.array([cf_R * np.cos(gamma), cf_R * np.sin(gamma), cf_height]),
        np.array([cf_R * np.cos(2 * gamma), cf_R * np.sin(2 * gamma), cf_height]),
    ]

    sol = solve_ocp(
        pl_waypoints=pl_waypoints,
        cf_p0=cf_p0,
        cable_l=cable_l,
        ros=ros,
    )

    if save_csv:
        print("Saving trajectories...")
        data_path = ROOT_FOLDER / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        cf_name = ["cf1", "cf2", "cf3"]
        for cf in cf_name:

            print(f"\nStarting trajectory for {cf}:")
            for k in range(1):
                print(f' x = {sol[f"{cf}_p"][:, k]}')
                print(f' v = {sol[f"{cf}_v"][:, k]}')
                print(f' a = {sol[f"{cf}_a"][:, k]}')

            # print max velocity and acceleration
            v_max = np.max(np.linalg.norm(sol[f"{cf}_v"], axis=0))
            a_max = np.max(np.linalg.norm(sol[f"{cf}_a"], axis=0))
            print(f" max velocity: {v_max:.2f} m/s")
            print(f" max acceleration: {a_max:.2f} m/s²")

            p = sol[f"{cf}_p"]
            print(f'dimensions: {p.shape}')
            segs = poly7.fit_poly7_piecewise(
                p=sol[f"{cf}_p"].T,
                t_grid=sol["t"])
            # poly7.show_min_max_coeffs(segs)

            out = data_path / f"traj_{cf}.csv"
            poly7.write_multi_csv(out, segs)
            print(f"Wrote {out} with {len(segs)} segments")

        np.savez(data_path / "ocp_solution.npz", **{k: v for k, v in sol.items()})

    if plot:
        cf_plots.plot_ocp(sol)
        cf_plots.animate_ocp(sol, time=True)


if __name__ == "__main__":
    get_traj()
    import matplotlib.pyplot as plt
    plt.show()


