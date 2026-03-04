import cf_solver
import cf_plots
import cf_bag

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


ROOT_FOLDER = Path.home() / "winter-project/ws/"
BAGS_FOLDER = ROOT_FOLDER / "bags"

plot_folder = ROOT_FOLDER / "figures"
plot_folder.mkdir(parents=True, exist_ok=True)

PLOT, PLOT_ANIMATE, PLOT_BAG = 1, 1, 0


if __name__ == "__main__":
    # bag data folder
    # bag_folder, t_offset = "rosbag2_ot1", 8.4
    # bag_folder, t_offset  = "rosbag2_ot2", 8.9
    # bag_folder, t_offset = "rosbag2_ot3", 10.2
    # bag_folder, t_offset = "rosbag2_ot4", 8.52
    bag_folder, t_offset = "rosbag2_0226_1", 0.0
    # bag_folder, t_offset = "rosbag2_0226_2", 8.3

    if PLOT_BAG:
        data_folder = BAGS_FOLDER / bag_folder / "data"
    else:
        data_folder = ROOT_FOLDER / "data"

    # load ocp
    ocp_path = data_folder / "ocp.npz"
    ocp_sol = np.load(ocp_path, allow_pickle=True)
    cable_l = ocp_sol["cable_l"]
    t_total = ocp_sol["t"][-1] - ocp_sol["t"][0]
    print(f"loaded OCP solution from {ocp_path}")

    # print OCP solution stats
    cf_solver.print_ocp_stats(ocp_sol)

    # plot OCP solution and animate
    if PLOT:
        f_states, a_states, f_constr, a_constr, f_3d, a_3d = cf_plots.plot_ocp(
            ocp_sol, constraints=False)
    if PLOT_ANIMATE:
        cf_plots.animate_ocp(ocp_sol)
    if PLOT_BAG:
        f_xyz, a_xyz = cf_plots.plot_xyz(ocp_sol)

    # bag data
    if PLOT_BAG:
        bag_path = BAGS_FOLDER / bag_folder
        bag_data = cf_bag.get_bag_data(bag_path)
        print(f"loaded bag data from {bag_path}")
        if PLOT:
            f_states, a_states, f_3d, a_3d = cf_plots.plot_bag(
                bag_data, t_offset, t_total, cable_l,
                fig=f_states, axes=a_states,
                fig_3d=f_3d, axes_3d=a_3d)
            # f_states, a_states = cf_plots.plot_error(
            #     ocp_sol, bag_data, t_offset, t_total,
            #     fig=f_states, axes=a_states)
        f_xyz, a_xyz = cf_plots.plot_xyz(
            bag_data, t_offset, t_total,
            fig=f_xyz, axes=a_xyz)


    # cf_plots.save_plots(f_states, f_constr, f_3d, plot_folder)

    plt.show()
