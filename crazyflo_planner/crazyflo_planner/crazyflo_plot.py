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

plot_folder = ROOT_FOLDER / "figures"
plot_folder.mkdir(parents=True, exist_ok=True)

PLOT_BAG = False
if PLOT_BAG:
    data_folder = bag_path / "data"
else:
    data_folder = ROOT_FOLDER / "data"


if __name__ == "__main__":
    # load ocp
    ocp_path = data_folder / "ocp.npz"
    ocp_sol = np.load(ocp_path)
    cable_l = ocp_sol["cable_l"]
    t_total = ocp_sol["t"][-1] - ocp_sol["t"][0]

    # bag data

    # bag_folder = "rosbag2_ot1"
    # t_offset = 8.4

    bag_folder = "rosbag2_ot2"
    t_offset = 8.9

    # bag_folder = "rosbag2_ot3"
    # t_offset = 10.2

    # bag_folder = "rosbag2_ot4"
    # t_offset = 8.52

    # print OCP solution stats
    cf_solver.print_ocp_stats(ocp_sol)

    # plot OCP solution and animate
    f_states, a_states, f_constr, a_constr, f_3d, a_3d = cf_plots.plot_ocp(ocp_sol, constraints=False)
    # cf_plots.animate_ocp(ocp_sol)
    f_xyz, a_xyz = cf_plots.plot_xyz(ocp_sol)

    # bag data
    if PLOT_BAG:
        bag_path = ROOT_FOLDER / bag_folder
        bag_data = cf_bag.get_bag_data(bag_path)
        # f_states, a_states, f_3d, a_3d = cf_plots.plot_bag(
        #     bag_data, t_offset, t_total, cable_l,
        #     fig=f_states, axes=a_states,
        #     fig_3d=f_3d, axes_3d=a_3d)
        # f_states, a_states = cf_plots.plot_error(
        #     ocp_sol, bag_data, t_offset, t_total,
        #     fig=f_states, axes=a_states)
        f_xyz, a_xyz = cf_plots.plot_xyz(
            bag_data, t_offset, t_total,
            fig=f_xyz, axes=a_xyz)


    # cf_plots.save_plots(f_states, f_constr, f_3d, plot_folder)

    plt.show()
