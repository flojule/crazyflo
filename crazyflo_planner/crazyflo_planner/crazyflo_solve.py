import cf_waypoints
import cf_solver
import cf_plots
import cf_csv
import cf_bag
import cf_traj

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT_FOLDER = Path.home() / "winter-project/ws/"

data_folder = ROOT_FOLDER / "data"
plot_folder = ROOT_FOLDER / "figures"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder.mkdir(parents=True, exist_ok=True)

PLOT_OCP = True


if __name__ == "__main__":
    pl_height = 0.5  # payload height in m
    cable_l = 0.5  # cable lengths
    pl_v_max, pl_a_max = 5.0, 10.0  # payload max velocity and acceleration
    cf_v_max, cf_a_max = 5.0, 10.0  # cf max velocity and acceleration

    # generate pl_waypoints and save to csv
    pl_waypoints = cf_waypoints.generate_waypoints(
        traj='ellipse', height=pl_height, N=6, folder=data_folder)

    # convert pl positions to polynomial trajectory with time grid
    pl_poly7 = cf_traj.waypoints_to_poly7(pl_waypoints, v_max=pl_v_max, a_max=pl_a_max)
    # convert pl_poly7 to timed waypoints for OCP
    pl_timed_waypoints = cf_traj.poly7_to_timed_waypoints(pl_poly7)

    print(f"traj total time {pl_timed_waypoints['t'][-1]:.2f} s, with {len(pl_timed_waypoints['t'])} points")

    # solve OCP for crazyflie trajectories
    sol = cf_solver.solve_ocp(
        pl_traj=pl_timed_waypoints,
        cable_l=cable_l,
        cf_v_max=cf_v_max,
        cf_a_max=cf_a_max,
    )

    # print OCP solution stats and save solution to csv
    cf_solver.print_ocp_stats(sol)
    cf_solver.save_ocp(sol, filename="ocp.npz", path=data_folder)
    cf_csv.save_time_pos_csv(sol, path=data_folder)

    # generate cf_waypoints and save to csv
    cf_csv.save_poly7_csv(sol, folder=data_folder, v_max=cf_v_max, a_max=cf_a_max)

    # plot OCP solution and animate
    if PLOT_OCP:
        f_states, a_states, f_constr, a_constr, f_3d, a_3d = cf_plots.plot_ocp(
            sol, animate=True, folder=plot_folder)
        plt.show()
