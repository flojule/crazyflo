#!/usr/bin/env python3
"""
3-drone payload OCP solver using CasADi.
"""

import numpy as np
from pathlib import Path

import casadi as ca


def solve_ocp(
    pl_p_ref: np.ndarray,  # payload reference trajectory ( N+1, 3 )
    cf_p0: list[np.ndarray],  # initial drone positions
    t_grid: np.ndarray,  # time grid samples (N+1,)
    cable_l: float,  # cable length
    pl_mass: float = 0.03,  # payload mass in kg
    g: float = 9.81,  # gravity acceleration
    v_max: float = 2.0,  # norm of velocity limits
    a_max: float = 5.0,  # norm of acceleration limits
    j_max: float = 10.0,  # norm of jerk limits
    s_max: float = 1.0,  # norm of snap limits
    thrust_min: float = -5.0,  # vertical acceleration limits
    thrust_max: float = 5.0,  # vertical acceleration limits
    tension_min: float = 0.05,  # min tension in N
    tension_max: float = 0.5,  # max tension in N
    # cf_pl_max: float = 0.015,  # crazyflie max payload in kg
    cf_radius: float = 0.2,  # crazyflie radius for drone-drone collision in m
    w_p: float = 100.0,  # tracking weight
    w_a: float = 0.001,  # acceleration weight
    w_j: float = 0.001,  # jerk weight
    w_s: float = 0.001,  # snap weight
    w_tension: float = 10.0,  # tension weight
    # w_goal: float = 10.0,  # terminal position weight
) -> dict:
    """Solve the 3-drone payload OCP."""
    dt = t_grid[1] - t_grid[0]
    M = pl_p_ref.shape[0]  # number of reference points

    opti = ca.Opti()

    # ######### Decision variables #########
    pl_p = opti.variable(3, M)  # payload positions
    pl_v = opti.variable(3, M)  # payload velocities

    # cf_p = [opti.variable(3, M) for _ in range(3)]  # drone positions
    cf_v = [opti.variable(3, M) for _ in range(3)]  # drone velocities
    cf_a = [opti.variable(3, M - 1) for _ in range(3)]  # drone accelerations
    # cf_j = [opti.variable(3, M - 2) for _ in range(3)]  # drone jerks
    # cf_s = [opti.variable(3, M - 3) for _ in range(3)]  # drone snaps
    cf_cable_dir = [opti.variable(3, M) for _ in range(3)]  # unit cable directions
    cf_cable_t = [opti.variable(1, M - 1) for _ in range(3)]  # cable tensions

    def cf_p_it(i: int, k: int):  # drone i position at time step k
        return pl_p[:, k] + cable_l * cf_cable_dir[i][:, k]

    # ######### Constraints #########
    # Payload boundary conditions
    opti.subject_to(pl_p[:, 0] == pl_p_ref[0])
    opti.subject_to(pl_p[:, M - 1] == pl_p_ref[-1])
    opti.subject_to(pl_v[:, 0] == ca.DM.zeros(3, 1))
    opti.subject_to(pl_v[:, M - 1] == ca.DM.zeros(3, 1))

    # Drones boundary conditions
    for i in range(3):
        opti.subject_to(cf_p_it(i, 0) == cf_p0[i])
        opti.subject_to(cf_v[i][:, 0] == ca.DM.zeros(3, 1))
        opti.subject_to(cf_v[i][:, M - 1] == ca.DM.zeros(3, 1))
        opti.subject_to(cf_a[i][:, 0] == ca.DM.zeros(3, 1))
        opti.subject_to(cf_a[i][:, M - 2] == ca.DM.zeros(3, 1))
        # opti.subject_to(cf_j[i][:, 0] == ca.DM.zeros(3, 1))
        # opti.subject_to(cf_j[i][:, M - 3] == ca.DM.zeros(3, 1))

    # Drones speed/acceleration/jerk limits
    for i in range(3):
        for k in range(M):
            opti.subject_to(ca.sumsqr(cf_v[i][:, k]) <= v_max ** 2)
        for k in range(M - 1):
            opti.subject_to(ca.sumsqr(cf_a[i][:, k]) <= a_max ** 2)
            opti.subject_to(cf_a[i][2, k] >= thrust_min)
            opti.subject_to(cf_a[i][2, k] <= thrust_max)
        # for k in range(M - 2):
        #     opti.subject_to(ca.sumsqr(cf_j[i][:, k]) <= j_max ** 2)

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
            opti.subject_to(ca.sumsqr(cf_cable_dir[i][:, k]) == 1.0)
        opti.subject_to(cf_cable_t[i] >= tension_min)
        opti.subject_to(cf_cable_t[i] <= tension_max)

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

    # Payload RK4 + drone velocity RK4
    for k in range(M - 1):
        sum_tension = ca.DM.zeros(3, 1)
        for i in range(3):
            sum_tension += cf_cable_t[i][:, k] * cf_cable_dir[i][:, k]

        xk = ca.vertcat(pl_p[:, k], pl_v[:, k])

        # payload step
        x_next = rk4_step(
            xk,
            fk=lambda xx: payload_f(xx, sum_tension),
            dt_=dt
        )
        opti.subject_to(pl_p[:, k + 1] == x_next[0:3])
        opti.subject_to(pl_v[:, k + 1] == x_next[3:6])

        # drone velocity/position step
        for i in range(3):
            opti.subject_to(cf_v[i][:, k + 1] == cf_v[i][:, k] + dt * cf_a[i][:, k])
            opti.subject_to((cf_p_it(i, k + 1) - cf_p_it(i, k)) == dt * cf_v[i][:, k])

    # ######### Cost function #########
    J = 0
    for k in range(M):  # tracking error
        J += w_p * ca.sumsqr(pl_p[:, k] - ca.DM(pl_p_ref[k]))
    for i in range(3):  # cable tension
        for k in range(M - 1):
            J += w_tension * ca.sumsqr(cf_cable_t[i][:, k])
    for i in range(3):  # drone/payload smoothness
        # for k in range(M):  # payload acceleration
        #     pl_a = ca.vertcat(
        #         pl_v[:, k] - pl_v[:, k - 1] if k > 0 else ca.DM.zeros(3, 1)
        #     ) / dt
        #     J += w_a * ca.sumsqr(pl_a)
        # for k in range(M - 1):  # drone accelerations
        #     J += w_a * ca.sumsqr(cf_a[i][:, k])
        for k in range(M - 2):  # drone jerks
            cf_j = (ca.vertcat(
                cf_a[i][:, k + 1] - cf_a[i][:, k]
            )) / dt
            J += w_j * ca.sumsqr(cf_j)
        for k in range(M - 3):  # drone snaps
            cf_s = (ca.vertcat(
                cf_a[i][:, k + 2] - 2 * cf_a[i][:, k + 1] + cf_a[i][:, k]
            )) / (dt ** 2)
            J += w_s * ca.sumsqr(cf_s)
    # J += w_goal * ca.sumsqr(pl_p[:, M - 1] - ca.DM(pl_p_ref[-1]))
    opti.minimize(J)

    # ######### Initial guess #########
    opti.set_initial(pl_p, pl_p_ref.T)
    opti.set_initial(pl_v, 0.0)
    # for i in range(3):
    #     for k in range(M):
    #         opti.set_initial(cf_v[i][:, k], 0.0)
    #     for k in range(M - 1):
    #         opti.set_initial(cf_a[i][:, k], 0.0)

    base_dirs = np.zeros((3, 3))
    for i in range(3):
        base_dirs[i] = (cf_p0[i] - pl_p_ref[0]) / cable_l
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

    return {
        "t": t_grid,
        "pl_p": pl_p_sol, "pl_v": pl_v_sol,
        "cf1_p": cf_p_sol[0], "cf2_p": cf_p_sol[1], "cf3_p": cf_p_sol[2],
        "cf1_v": cf_v_sol[0], "cf2_v": cf_v_sol[1], "cf3_v": cf_v_sol[2],
        "cf1_a": cf_a_sol[0], "cf2_a": cf_a_sol[1], "cf3_a": cf_a_sol[2],
        "cf1_cable_t": cf_cable_t_sol[0], "cf2_cable_t": cf_cable_t_sol[1], "cf3_cable_t": cf_cable_t_sol[2],
        "cf1_cable_dir": cf_cable_dir_sol[0], "cf2_cable_dir": cf_cable_dir_sol[1], "cf3_cable_dir": cf_cable_dir_sol[2],
        "pl_p_ref": pl_p_ref.T,
        "cable_l": cable_l,
        "cf_radius": cf_radius,
    }


def get_traj(traj='circle', loops=5, plot=True, save_csv=True, ros=False):
    """Solve an example OCP and export trajectories."""
    cf_height = 1.0  # solve at ~1m height
    cable_l = 0.5  # cable lengths
    alpha = np.pi / 4.0  # rope angle
    gamma = 2.0 * np.pi / 3.0  # drone separation angle
    pl_height = cf_height - cable_l * np.cos(alpha)

    # time
    T = 2.0  # total time
    dt = 0.1  # 10 Hz
    N = int(T / dt)
    t_grid = np.linspace(0.0, T, N + 1)

    if traj == 'ellipse':
        # payload ref trajectory: ellipse
        r_A = 0.6
        r_B = 0.3
        pl_p_ref = np.stack([
            r_A * (1.0 - np.cos(2 * np.pi * t_grid / T)),
            r_B * np.sin(2 * np.pi * t_grid / T),
            pl_height * np.ones(N + 1),
        ], axis=1)
        pl_p_start = pl_p_ref[0]
        pl_p_goal = pl_p_ref[-1]
    else:  # straight line
        pl_p_start = np.array([0.0, 0.0, pl_height])
        pl_p_goal = np.array([-0.5, 0.0, pl_height])
        pl_p_ref = np.stack([(1 - (k / N)) * pl_p_start + (k / N) * pl_p_goal for k in range(N + 1)], axis=0)

    if loops > 1:
        # extend trajectory for infinite flight
        p0 = pl_p_ref.copy()  # one loop, length N+1
        pl_p_ref = np.concatenate([p0[:-1] for _ in range(loops - 1)] + [p0], axis=0)
        T = T * loops
        N = N * loops
        t_grid = np.linspace(0.0, T, N + 1)

    # drone initial positions
    cf_R = cable_l * np.cos(alpha)  # radius of drone formation
    cf_p0 = [
        np.array([cf_R * np.cos(0), cf_R * np.sin(0), cf_height]),
        np.array([cf_R * np.cos(gamma), cf_R * np.sin(gamma), cf_height]),
        np.array([cf_R * np.cos(2 * gamma), cf_R * np.sin(2 * gamma), cf_height]),
    ]

    sol = solve_ocp(
        pl_p_ref=pl_p_ref,
        t_grid=t_grid,
        cf_p0=cf_p0,
        cable_l=cable_l,
    )
    print("Solved. Payload at:", sol["pl_p"][:, -1])

    if ros:
        from . import cf_plots
        from .cf_poly7 import fit_poly7_piecewise, write_multi_csv
    else:
        import cf_plots
        from cf_poly7 import fit_poly7_piecewise, write_multi_csv

    if save_csv:
        print("Saving trajectories...")
        data_path = Path.home() / ".ros/crazyflo_planner" / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        t_grid = sol["t"]
        name = ["cf1", "cf2", "cf3"]
        for i, r_i in enumerate(("cf1_p", "cf2_p", "cf3_p")):
            pos = sol[r_i]  # (3, N+1)
            segs = fit_poly7_piecewise(t=t_grid, pos=pos, yaw=None, segment_every=10)
            out = data_path / f"traj_{name[i]}.csv"
            write_multi_csv(out, segs)
            print(f"Wrote {out} with {len(segs)} segments")

        np.savez(data_path / "ocp_solution.npz", **{k: v for k, v in sol.items()})

    if plot:
        cf_plots.plot_ocp(sol)


if __name__ == "__main__":
    get_traj(traj='ellipse', loops=5, plot=True, save_csv=True, ros=False)
    import matplotlib.pyplot as plt
    plt.show()
