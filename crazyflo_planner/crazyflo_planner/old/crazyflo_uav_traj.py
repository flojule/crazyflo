import cf_waypoints
import cf_solver
import cf_plots
import cf_csv
import cf_bag
import cf_traj

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


ROOT_FOLDER = Path.home() / ".ros/crazyflo"
UAV_TRAJ_FOLDER = Path.home() / "winter-project/uav_trajectories"
BAG_FOLDER = Path.home() / "winter-project/ws/"

data_folder = ROOT_FOLDER / "data"
plot_folder = ROOT_FOLDER / "figures"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder.mkdir(parents=True, exist_ok=True)

PLOT_OCP = False
PLOT_BAG = False
LOCAL = True  # python vs C++ trajectory generation


if __name__ == "__main__":
    pl_height = 0.5  # payload height in m
    cable_l = 0.5  # cable lengths

    # generate pl_waypoints and save to csv
    pl_waypoints = cf_waypoints.generate_waypoints(
        traj='ellipse', height=pl_height, folder=data_folder)

    # convert pl positions to polynomial trajectory with time grid
    if LOCAL:
        pl_poly7 = cf_traj.waypoints_to_poly7(pl_waypoints, v_max=2.0, a_max=5.0)
        pl_timed_waypoints = cf_traj.poly7_to_timed_waypoints(pl_poly7)

    if not LOCAL:
        # ./build/genTrajectory -i crazyflo/pl_waypoints.csv --v_max 2.0 --a_max 5.0 -o crazyflo/pl_traj.csv
        # python3 scripts/gen_traj.py --traj crazyflo/pl_traj.csv --output crazyflo/pl_time_pos.csv --dt 0.1 --stretchtime 1.0
        # pl_traj_csv = data_folder / "pl_traj.csv"
        pl_time_pos_csv = UAV_TRAJ_FOLDER / "crazyflo/pl_time_pos.csv"
        pl_timed_waypoints = cf_csv.read_time_pos_csv(pl_time_pos_csv)

    print(f"traj total time {pl_timed_waypoints['t'][-1]:.2f} s, with {len(pl_timed_waypoints['t'])} points")

    # solve OCP for crazyflie trajectories
    sol = cf_solver.solve_ocp(
        pl_traj=pl_timed_waypoints,
        cable_l=cable_l,
    )

    # print OCP solution stats and save solution to csv
    cf_solver.print_ocp_stats(sol)
    cf_solver.save_ocp(sol, filename="ocp.npz", path=data_folder)
    cf_csv.save_time_pos_csv(sol, path=data_folder)

    # plot OCP solution and animate
    if PLOT_OCP:
        f_states, a_states, f_constr, a_constr, f_3d, a_3d = cf_plots.plot_ocp(sol)
        cf_plots.animate_ocp(sol)

    # # generate cf_waypoints and save to csv
    if LOCAL:
        cf_csv.save_poly7_csv(sol, folder=data_folder, v_max=2.0, a_max=5.0)

    # # C++
    # python3 ./scripts/generate_trajectory.py crazyflo/cf1_time_pos.csv crazyflo/cf1.csv --pieces 5
    # python3 ./scripts/generate_trajectory.py crazyflo/cf1_time_pos.csv crazyflo/cf2.csv --pieces 5
    # python3 ./scripts/generate_trajectory.py crazyflo/cf1_time_pos.csv crazyflo/cf3.csv --pieces 5

    # bag data
    if PLOT_BAG:
        ocp_path = ROOT_FOLDER / "data" / "ocp.npz"
        ocp_data = np.load(ocp_path)
        cable_l = ocp_data["cable_l"]
        t_total = ocp_data["t"][-1] - ocp_data["t"][0]
        t_offset = 0.0

        bag_name = "rosbag2_2026_02_24-17_33_56"  #  rosbag2_2026_02_24-17_30_27
        bag_path = BAG_FOLDER / bag_name
        bag_data = cf_bag.get_bag_data(bag_path)
        f_states, a_states, f_3d, a_3d = cf_plots.plot_bag(
            bag_data, t_offset, t_total, cable_l,
            fig=f_states, axes=a_states,
            fig_3d=f_3d, axes_3d=a_3d)
        f_states, a_states = cf_plots.plot_error(
            ocp_data, bag_data, t_offset, t_total,
            fig=f_states, axes=a_states)

    if PLOT_OCP:
        f_states.savefig(plot_folder / "cf_plot.png")
        f_constr.savefig(plot_folder / "cf_constraints.png")
        f_3d.savefig(plot_folder / "cf_3d.png")
        print("Plots saved to:", plot_folder)

        plt.show()
