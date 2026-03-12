"""crazyflo_solve.py - top-level planning script.

Workflow
--------
1. Choose a reference trajectory type and obstacle layout.
2. Generate payload waypoints with cf_waypoints.
3. Optionally build a static obstacle environment with cf_obstacles.
4. Solve the 3-drone payload OCP with cf_solver (CasADi backend).
5. Save the solution under data/<mission>/ as:
   - ocp.npz            (full NumPy solution, for plotting / debugging)
   - time_pos_cf{1,2,3}.csv  (dense time / pos / vel / acc samples)
   - traj_cf{1,2,3}.csv     (7th-degree polynomial segments, Crazyflie format)
   where <mission> = <traj> or <traj>_<obstacles_type> (e.g. 'line_course').
6. Optionally display plots and a 3-D animation.

Usage
-----
    python crazyflo_solve.py

Change the variables in the ``if __name__ == "__main__":`` block to
configure the run without touching the module code.
"""

import cf_waypoints
import cf_solver
import cf_plots
import cf_csv
import cf_obstacles

from pathlib import Path
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_FOLDER = Path.home() / "winter-project/ws/"

_data_root = ROOT_FOLDER / "data"       # parent for all mission subfolders
_plot_root = ROOT_FOLDER / "figures"    # parent for all mission figure subfolders

# Set to True to display plots and animation after solving.
PLOT_OCP = True


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Trajectory type selection
    #    Pick one index from the list below.
    #    'line'     - straight segment from [0,0,h] to [length,0,h]
    #    'ellipse'  - horizontal ellipse
    #    'figure8'  - figure-8 (lemniscate)
    #    'random'   - random walk (non-repeatable)
    # ------------------------------------------------------------------
    trajs = ['line', 'ellipse', 'figure8', 'random']
    traj = trajs[1]  # <-- select trajectory type here

    # ------------------------------------------------------------------
    # 2. Obstacle layout selection
    #    Obstacles are only used when traj == 'line'.
    #    None                - obstacle-free environment
    #    'wall'              - single blocking wall
    #    'vertical_passage'  - two walls with a lateral gap
    #    'horizontal_passage'- two walls with a vertical gap
    #    'course'            - three-obstacle slalom course
    # ------------------------------------------------------------------
    obstacles_types = [None, 'wall', 'vertical_passage', 'horizontal_passage', 'course']
    obstacles_type = obstacles_types[0]  # <-- select obstacle type here
    gap = 0.6    # gap size for passage obstacles [m]
    if obstacles_type == 'course':
        length = 10.0  # total length of the trajectory / obstacle course [m]
    else:
        length = 2.0  # total length of the trajectory / obstacle course [m]

    # ------------------------------------------------------------------
    # 3. Physical parameters
    # ------------------------------------------------------------------
    pl_height = 0.5          # payload flight height [m]
    cable_l = 0.5            # cable length from each drone to payload [m]
    cf_v_max, cf_a_max = 2.0, 5.0  # drone velocity [m/s] and acceleration [m/s^2] limits

    # ------------------------------------------------------------------
    # 4. Build mission name and output folders
    #    mission = '<traj>' or '<traj>_<obstacles_type>'
    #    All files are written to data/<mission>/ and figures/<mission>/
    # ------------------------------------------------------------------
    use_obstacles = (obstacles_type is not None and traj == 'line')
    mission = f"{traj}_{obstacles_type}" if use_obstacles else traj

    data_folder = _data_root / mission
    plot_folder = _plot_root / mission
    data_folder.mkdir(parents=True, exist_ok=True)
    plot_folder.mkdir(parents=True, exist_ok=True)
    print(f"Mission: '{mission}'  ->  {data_folder}")

    # ------------------------------------------------------------------
    # 5. Generate payload reference waypoints
    #    Waypoints are saved to data_folder/pl_waypoints.csv.
    # ------------------------------------------------------------------
    waypoints = cf_waypoints.generate_waypoints(
        traj=traj, height=pl_height, length=length, N=10, folder=data_folder)

    # ------------------------------------------------------------------
    # 6. Build obstacle list
    # ------------------------------------------------------------------
    if not use_obstacles:
        obstacles = []
    else:
        obstacles = cf_obstacles.get_obstacles(obstacles_type, gap=gap, length=length)
        if obstacles_type != 'wall':
            # insert waypoints at each passage centre so the OCP has a
            # strong hint to route through the gap
            waypoints = cf_obstacles.update_waypoints(waypoints, obstacles)

    # ------------------------------------------------------------------
    # 7. Solve the OCP
    #    cf_solver.solve_ocp returns a dict with the full solution:
    #      't'           - time vector (M,)
    #      'pl_p', 'pl_v'- payload position / velocity  (M, 3)
    #      'cf{1,2,3}_p/v/a/j/s' - per-drone states
    #      'cf{1,2,3}_cable_t'   - cable tensions
    #      metadata: 'cable_l', 'cf_radius', 'obstacles', 'waypoints', ...
    # ------------------------------------------------------------------
    sol = cf_solver.solve_ocp(
        waypoints=waypoints,
        cable_l=cable_l,
        cf_v_max=cf_v_max,
        cf_a_max=cf_a_max,
        obstacles=obstacles,
    )

    # ------------------------------------------------------------------
    # 8. Persist the solution
    #    print_ocp_stats  - log cost, constraint violations, timing
    #    save_ocp         - write ocp.npz (full NumPy arrays)
    #    save_time_pos_csv- write dense per-drone state CSVs
    #    save_poly7_csv   - write firmware-compatible poly-7 CSVs
    # ------------------------------------------------------------------
    cf_solver.print_ocp_stats(sol)
    cf_solver.save_ocp(sol, path=data_folder)
    cf_csv.save_time_pos_csv(sol, path=data_folder)

    # Convert OCP states to 7th-degree polynomial segments and export
    # traj_cf{1,2,3}.csv for upload to the Crazyflie firmware.
    cf_csv.save_poly7_csv(sol, folder=data_folder, v_max=cf_v_max, a_max=cf_a_max)

    # ------------------------------------------------------------------
    # 9. Visualisation (optional)
    # ------------------------------------------------------------------
    if PLOT_OCP:
        cf_plots.plot_cost(sol)  # bar chart of OCP cost terms
        f_states, a_states, f_constr, a_constr, f_3d, a_3d = cf_plots.plot_ocp(
            sol, animate=True, folder=plot_folder)  # states + 3-D + animation
        plt.show()
