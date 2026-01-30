"""
Simple optimal control problem (OCP) solved with direct collocation + IPOPT.

System: double integrator
    x_dot = v
    v_dot = u

Goal: go from (x,v)=(0,0) to (1,0) in time T
Cost: integral u^2 dt
Constraints: |u| <= umax
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def solve_ocp(N: int = 50, T: float = 2.0, umax: float = 2.0):
    dt = T / N

    opti = ca.Opti()

    # Decision variables
    x = opti.variable(N + 1)  # position
    v = opti.variable(N + 1)  # velocity
    u = opti.variable(N)      # control (acceleration)

    # Boundary conditions
    opti.subject_to(x[0] == 0.0)
    opti.subject_to(v[0] == 0.0)
    opti.subject_to(x[-1] == 1.0)
    opti.subject_to(v[-1] == 0.0)

    # Control bounds
    opti.subject_to(opti.bounded(-umax, u, umax))

    # Dynamics constraints via trapezoidal collocation
    # x_{k+1} = x_k + dt/2 * (v_k + v_{k+1})
    # v_{k+1} = v_k + dt/2 * (u_k + u_k)
    for k in range(N):
        opti.subject_to(x[k + 1] == x[k] + 0.5 * dt * (v[k] + v[k + 1]))
        opti.subject_to(v[k + 1] == v[k] + dt * u[k])

    # Objective: integral u^2 dt
    J = ca.sumsqr(u) * dt
    opti.minimize(J)

    # Initial guess
    opti.set_initial(x, np.linspace(0, 1, N + 1))
    opti.set_initial(v, 0)
    opti.set_initial(u, 0)

    # IPOPT settings
    opti.solver(
        "ipopt",
        {"expand": True},
        {
            "print_level": 5,
            "max_iter": 2000,
            "tol": 1e-8,
        },
    )

    sol = opti.solve()

    t = np.linspace(0, T, N + 1)
    x_sol = sol.value(x)
    v_sol = sol.value(v)
    u_sol = sol.value(u)

    return t, x_sol, v_sol, u_sol

def plot_states(t, x, v, u):
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(16, 8))
    axes[0].grid()
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Position x [m]")
    axes[0].plot(t, x, label="Position x [m]")
    axes[0].legend()
    axes[1].grid()
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Velocity v [m/s]")
    axes[1].plot(t, v, label="Velocity v [m/s]", color="orange")
    axes[1].legend()
    axes[2].grid()
    axes[2].set_ylabel("Control u [m/s²]")
    axes[2].step(t[:-1], u, label="Control u [m/s²]", where="post", color="green")
    axes[2].legend()
    axes[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def main():
    t, x, v, u = solve_ocp(N=60, T=2.0, umax=2.0)

    print("Solved.")
    print(f"x(T)={x[-1]:.6f}, v(T)={v[-1]:.6f}")
    print(f"u range: [{u.min():.3f}, {u.max():.3f}]")
    plot_states(t, x, v, u)


if __name__ == "__main__":
    main()
