"""crazyflo_plot.py — post-flight analysis and visualisation script.

This script loads a saved OCP solution (ocp.npz) and optionally a rosbag2
recording, then produces plots for comparison and validation.

Flight flags
------------
Set the three flags at module level to choose which plots to generate:

    PLOT         – static trajectory / state plots and 3-D view.
    PLOT_ANIMATE – 3-D animation of the OCP solution.
    PLOT_BAG     – overlay rosbag2 measurements on OCP plots.
                   When enabled, ``data_folder`` is read from the bag
                   directory instead of the default data folder.

Bag alignment
-------------
Real flight recordings often start before the trajectory begins.
Set ``t_offset`` to the number of seconds that should be subtracted from
the bag timestamps so that the trajectory start aligns with t = 0 in the
OCP solution.

Usage
-----
    python crazyflo_plot.py
"""

import cf_solver
import cf_plots
import cf_bag

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_FOLDER = Path.home() / "winter-project/ws/"
BAGS_FOLDER = ROOT_FOLDER / "bags"   # parent directory for all rosbag2 recordings

plot_folder = ROOT_FOLDER / "figures"
plot_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot flags — set to 1 / True to enable each plot type
# ---------------------------------------------------------------------------
PLOT = 0          # static states + 3-D trajectory view
PLOT_ANIMATE = 0  # 3-D animation (slow for long trajectories)
PLOT_BAG = 0      # overlay real-flight bag data on OCP plots


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Bag selection
    # Comment/uncomment the desired bag folder and its time offset.
    # t_offset [s]: amount to subtract from bag timestamps so the
    # trajectory start aligns with t=0 of the OCP solution.
    # ------------------------------------------------------------------
    # bag_folder, t_offset = "rosbag2_ot1", 8.4
    # bag_folder, t_offset  = "rosbag2_ot2", 8.9
    # bag_folder, t_offset = "rosbag2_ot3", 10.2
    # bag_folder, t_offset = "rosbag2_ot4", 8.52
    bag_folder, t_offset = "rosbag2_0226_1", 0.0
    # bag_folder, t_offset = "rosbag2_0226_2", 8.3

    # When plotting bag data, load state CSVs that were saved into the
    # bag directory by the post-processing pipeline; otherwise use the
    # standard data folder from the planning step.
    if PLOT_BAG:
        data_folder = BAGS_FOLDER / bag_folder / "data"
    else:
        data_folder = ROOT_FOLDER / "data"

    # ------------------------------------------------------------------
    # Load OCP solution
    # ocp.npz contains the full solution dict produced by cf_solver.
    # ------------------------------------------------------------------
    ocp_path = data_folder / "ocp.npz"
    ocp_sol = np.load(ocp_path, allow_pickle=True)
    cable_l = ocp_sol["cable_l"]                       # cable length [m]
    t_total = ocp_sol["t"][-1] - ocp_sol["t"][0]      # total trajectory duration [s]
    print(f"loaded OCP solution from {ocp_path}")

    # Print a summary of the OCP result (cost breakdown, constraint violations)
    cf_solver.print_ocp_stats(ocp_sol)

    # ------------------------------------------------------------------
    # OCP plots
    # ------------------------------------------------------------------
    if PLOT:
        # Drone states (altitude, speed, acceleration) + 3-D trajectory
        f_states, a_states, f_constr, a_constr, f_3d, a_3d = cf_plots.plot_ocp(
            ocp_sol, constraints=False)
    if PLOT_ANIMATE:
        # 3-D animation; opens a separate matplotlib window
        cf_plots.animate_ocp(ocp_sol)
    if PLOT_BAG:
        # Per-axis position over time (useful for timeline alignment check)
        f_xyz, a_xyz = cf_plots.plot_xyz(ocp_sol)

    # ------------------------------------------------------------------
    # Bag data overlay
    # ------------------------------------------------------------------
    if PLOT_BAG:
        bag_path = BAGS_FOLDER / bag_folder
        bag_data = cf_bag.get_bag_data(bag_path)  # read /cf{1,2,3}/pose topics
        print(f"loaded bag data from {bag_path}")
        if PLOT:
            # Overlay measured drone positions on top of planned trajectories
            f_states, a_states, f_3d, a_3d = cf_plots.plot_bag(
                bag_data, t_offset, t_total, cable_l,
                fig=f_states, axes=a_states,
                fig_3d=f_3d, axes_3d=a_3d)
            # Uncomment to add a tracking-error plot:
            # f_states, a_states = cf_plots.plot_error(
            #     ocp_sol, bag_data, t_offset, t_total,
            #     fig=f_states, axes=a_states)
        # Per-axis position for bag data (overlaid on OCP axes)
        f_xyz, a_xyz = cf_plots.plot_xyz(
            bag_data, t_offset, t_total,
            fig=f_xyz, axes=a_xyz)

    # ------------------------------------------------------------------
    # Cost plot — always shown regardless of other flags
    # ------------------------------------------------------------------
    # cf_plots.save_plots(f_states, f_constr, f_3d, plot_folder)  # optional save
    cf_plots.plot_cost(ocp_sol)  # bar chart of weighted OCP cost terms

    plt.show()
