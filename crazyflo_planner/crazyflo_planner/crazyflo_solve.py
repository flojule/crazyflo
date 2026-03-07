import cf_waypoints
import cf_solver
import cf_plots
import cf_csv
import cf_obstacles

from pathlib import Path
import matplotlib.pyplot as plt


ROOT_FOLDER = Path.home() / "winter-project/ws/"

data_folder = ROOT_FOLDER / "data"
plot_folder = ROOT_FOLDER / "figures"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder.mkdir(parents=True, exist_ok=True)

PLOT_OCP = True


if __name__ == "__main__":
    trajs = ['line', 'ellipse', 'figure8', 'random']
    traj = trajs[0]
    obstacles_types = [None, 'wall', 'vertical_passage', 'horizontal_passage', 'course']
    obstacles_type = obstacles_types[-1]
    gap = 0.6  # gap size for obstacles in m
    length = 10.0  # length of obstacle course in m

    pl_height = 0.5  # payload height in m
    cable_l = 0.5  # cable lengths
    cf_v_max, cf_a_max = 2.0, 5.0  # cf max velocity and acceleration

    # generate pl_waypoints and save to csv
    waypoints = cf_waypoints.generate_waypoints(
        traj=traj, height=pl_height, length=length, N=10, folder=data_folder)

    # obstacles
    if obstacles_type is None or traj != 'line':
        obstacles = []  # only add obstacles for non-line trajs
    else:
        obstacles = cf_obstacles.get_obstacles(obstacles_type, gap=gap, length=length)
        if obstacles_type is not None and obstacles_type != 'wall':
            waypoints = cf_obstacles.update_waypoints(waypoints, obstacles)

    # solve OCP for crazyflie trajectories
    sol = cf_solver.solve_ocp(
        waypoints=waypoints,
        cable_l=cable_l,
        cf_v_max=cf_v_max,
        cf_a_max=cf_a_max,
        obstacles=obstacles,
    )

    # print OCP solution stats and save solution to csv
    cf_solver.print_ocp_stats(sol)
    cf_solver.save_ocp(sol, path=data_folder)
    cf_csv.save_time_pos_csv(sol, path=data_folder)

    # generate cf_waypoints and save to csv
    cf_csv.save_poly7_csv(sol, folder=data_folder, v_max=cf_v_max, a_max=cf_a_max)

    # plot OCP solution and animate
    if PLOT_OCP:
        cf_plots.plot_cost(sol)
        f_states, a_states, f_constr, a_constr, f_3d, a_3d = cf_plots.plot_ocp(
            sol, animate=True, folder=plot_folder)
        plt.show()
