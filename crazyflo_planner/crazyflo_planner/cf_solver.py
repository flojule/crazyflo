#!/usr/bin/env python3
"""
3-drone payload OCP (direct multiple shooting) with:
  (1) RK4 integration (payload + drone velocities)
  (2) optional payload reference trajectory input p_ref[k] or callable p_ref(t)
  (3) piecewise 7th-order polynomial fitting + Crazyflie/uav_trajectory-style export
"""

from __future__ import annotations
import numpy as np
import casadi as ca

import cf_plots
from cf_poly7 import fit_poly7_piecewise, write_multi_csv


def solve_ocp(
    p_start: np.ndarray,  # initial payload position
    p_goal: np.ndarray,  # goal payload position
    p_ref: None | np.ndarray = None,  # optional reference trajectory samples (N+1,3)
    r0: list[np.ndarray] = [np.array([0.5, 0.0, 1.0]),
                            np.array([-0.25, 0.433, 1.0]),
                            np.array([-0.25, -0.433, 1.0])],  # initial drone positions
    T: float = 4.0,  # total time
    N: int = 80,  # number of control intervals
    L: tuple[float, float, float] = (1.0, 1.0, 1.0),  # cable lengths
    mp: float = 0.05,  # payload mass in kg
    g: float = 9.81,  # gravity acceleration
    v_max: float = 1.0,  # norm of velocity limits
    a_max: float = 5.0,  # norm of acceleration limits
    uz_min: float = -5.0,  # vertical acceleration limits
    uz_max: float = 5.0,  # vertical acceleration limits
    lam_min: float = 0.1,  # min tension in N
    lam_max: float = 1.0,  # max tension in N
    w_track: float = 10.0,  # tracking weight
    w_u: float = 0.1,  # control effort weight
    w_lam: float = 0.01,  # tension weight
    w_terminal: float = 200.0,  # terminal position weight
) -> dict[str, np.ndarray]:
    """Solve the 3-drone payload OCP."""
    p_start = np.asarray(p_start, dtype=float).reshape(3)
    p_goal = np.asarray(p_goal, dtype=float).reshape(3)

    dt = float(T) / int(N)
    ez = np.array([0.0, 0.0, -1.0])

    # Payload trajectory to track
    t_grid = np.linspace(0.0, T, N + 1)
    if p_ref is None:
        p_d = np.stack([(1 - (k / N)) * p_start + (k / N) * p_goal for k in range(N + 1)], axis=0)
    else:
        p_d = np.asarray(p_ref, dtype=float).reshape(N + 1, 3)

    # Casadi optimizer
    opti = ca.Opti()

    # Decision variables
    p = opti.variable(3, N + 1)  # payload positions
    vp = opti.variable(3, N + 1)  # payload velocities

    v = [opti.variable(3, N + 1) for _ in range(3)]  # drone velocities
    u = [opti.variable(3, N) for _ in range(3)]  # drone accelerations
    s = [opti.variable(3, N + 1) for _ in range(3)]  # unit cable directions
    lam = [opti.variable(1, N) for _ in range(3)]  # cable tensions

    def r_i(i: int, k: int):  # drone i position at time step k
        return p[:, k] + L[i] * s[i][:, k]

    # Boundary conditions payload positions/velocities
    opti.subject_to(p[:, 0] == p_start)
    opti.subject_to(p[:, N] == p_goal)
    opti.subject_to(vp[:, 0] == ca.DM.zeros(3, 1))
    opti.subject_to(vp[:, N] == ca.DM.zeros(3, 1))

    # Unit cable directions
    for i in range(3):
        for k in range(N + 1):
            opti.subject_to(ca.sumsqr(s[i][:, k]) == 1.0)

    # Tension bounds
    for i in range(3):
        opti.subject_to(lam[i] >= lam_min)
        opti.subject_to(lam[i] <= lam_max)

    # Speed/accel bounds (squared)
    for i in range(3):
        for k in range(N + 1):
            opti.subject_to(ca.sumsqr(v[i][:, k]) <= v_max ** 2)
        for k in range(N):
            opti.subject_to(ca.sumsqr(u[i][:, k]) <= a_max ** 2)
            opti.subject_to(u[i][2, k] >= uz_min)
            opti.subject_to(u[i][2, k] <= uz_max)

    # Dynamics
    def payload_f(x: ca.MX, sum_tension_vec: ca.MX) -> ca.MX:
        # x = [p; vp]
        p_ = x[0:3]
        vp_ = x[3:6]
        ap_ = (1.0 / mp) * sum_tension_vec + ca.DM(ez) * g
        return ca.vertcat(vp_, ap_)

    def rk4_step(xk: ca.MX, fk, dt_: float) -> ca.MX:
        k1 = fk(xk)
        k2 = fk(xk + 0.5 * dt_ * k1)
        k3 = fk(xk + 0.5 * dt_ * k2)
        k4 = fk(xk + dt_ * k3)
        return xk + (dt_ / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Payload RK4 + drone velocity RK4 (trivial since dv=u constant)
    for k in range(N):
        sum_tension = ca.DM.zeros(3, 1)
        for i in range(3):
            sum_tension += lam[i][:, k] * s[i][:, k]

        xk = ca.vertcat(p[:, k], vp[:, k])

        # payload step with ZOH tension/s[k], lam[k]
        x_next = rk4_step(
            xk,
            fk=lambda xx: payload_f(xx, sum_tension),
            dt_=dt
        )
        opti.subject_to(p[:, k + 1] == x_next[0:3])
        opti.subject_to(vp[:, k + 1] == x_next[3:6])

        # drone velocity: v_{k+1} = v_k + dt*u_k (RK4 identical for constant u)
        for i in range(3):
            opti.subject_to(v[i][:, k + 1] == v[i][:, k] + dt * u[i][:, k])

            opti.subject_to((r_i(i, k + 1) - r_i(i, k)) == dt * v[i][:, k])

    # Cost
    J = 0
    for k in range(N + 1):
        J += w_track * ca.sumsqr(p[:, k] - ca.DM(p_d[k]))
    for i in range(3):
        for k in range(N):
            J += w_u * ca.sumsqr(u[i][:, k])
            J += w_lam * ca.sumsqr(lam[i][:, k])
    J += w_terminal * ca.sumsqr(p[:, N] - ca.DM(p_goal))
    opti.minimize(J)

    # Initial guess
    p_guess = np.linspace(p_start, p_goal, N + 1).T
    opti.set_initial(p, p_guess)
    opti.set_initial(vp, 0.0)

    base_dirs = np.zeros((3, 3))

    for i in range(3):
        d = r0[i] - p_start
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            raise ValueError(f"Drone {i} too close to payload.")
        base_dirs[i] = d / norm

    for i in range(3):
        opti.set_initial(s[i], np.tile(base_dirs[i].reshape(3, 1), (1, N + 1)))
        opti.set_initial(v[i], 0.0)
        opti.set_initial(u[i], 0.0)
        opti.set_initial(lam[i], 0.2)

    p_opts = {"expand": True}
    opti.solver("ipopt", p_opts)
    sol = opti.solve()

    # Extract
    p_sol = sol.value(p)
    vp_sol = sol.value(vp)
    v_sol = [sol.value(v[i]) for i in range(3)]
    u_sol = [sol.value(u[i]) for i in range(3)]
    s_sol = [sol.value(s[i]) for i in range(3)]
    lam_sol = [sol.value(lam[i]) for i in range(3)]
    r_sol = [p_sol + L[i] * s_sol[i] for i in range(3)]

    return {
        "t": t_grid,
        "p": p_sol, "vp": vp_sol,
        "r1": r_sol[0], "r2": r_sol[1], "r3": r_sol[2],
        "v1": v_sol[0], "v2": v_sol[1], "v3": v_sol[2],
        "u1": u_sol[0], "u2": u_sol[1], "u3": u_sol[2],
        "lam1": lam_sol[0], "lam2": lam_sol[1], "lam3": lam_sol[2],
        "s1": s_sol[0], "s2": s_sol[1], "s3": s_sol[2],
        "p_ref": p_d,
    }

def main():
    p_start = np.array([0.0, 0.0, 0.1])
    p_goal = np.array([2.0, 0.0, 0.1])

    r0 = [
        np.array([0.5, 0.0, 1.0]),
        np.array([-0.25, 0.433, 1.0]),
        np.array([-0.25, -0.433, 1.0]),
    ]

    sol = solve_ocp(
        p_start=p_start,
        p_goal=p_goal,
        r0=r0,
    )

    print("Solved. Payload at:", sol["p"][:, -1])

    t = sol["t"]
    folder = "crazyflo_planner/data"
    name = ["cf1", "cf2", "cf3"]
    for i, r_i in enumerate(("r1", "r2", "r3")):
        pos = sol[r_i]  # (3, N+1)
        segs = fit_poly7_piecewise(t=t, pos=pos, yaw=None, segment_every=10)
        out = f"{folder}/traj_{name[i]}.csv"
        write_multi_csv(out, segs)
        print(f"Wrote {out} with {len(segs)} segments")

    np.savez(f"{folder}/ocp_solution.npz", **{k: v for k, v in sol.items()})
    cf_plots.plot_data(sol)


if __name__ == "__main__":
    main()
