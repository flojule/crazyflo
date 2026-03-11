# crazyflo_planner

A ROS 2 package for planning and executing cooperative payload transport with a swarm of three Crazyflie nano-drones connected to a shared payload via cables.

## Overview

The package solves an **Optimal Control Problem (OCP)** to generate collision-free, dynamically feasible trajectories for three Crazyflies carrying a hanging payload. It covers the full workflow:

1. Define a payload reference path (ellipse, figure-8, straight line, random walk, or obstacle course).
2. Solve the OCP with [CasADi](https://web.casadi.org/) to obtain per-drone trajectories that respect velocity, acceleration, jerk, cable tension, and obstacle constraints.
3. Convert the solution into 7th-degree polynomial segments compatible with the Crazyflie firmware.
4. Upload and execute the trajectories on real or simulated drones using [Crazyswarm2](https://imrclab.github.io/crazyswarm2/).
5. Compare planned vs. executed trajectories using rosbag2 data.

## Package Structure

```
crazyflo_planner/
├── crazyflo_planner/       # Python source modules
│   ├── cf_solver.py        # CasADi OCP formulation and solver
│   ├── cf_traj.py          # Minimum-snap 7th-degree polynomial trajectory
│   ├── cf_waypoints.py     # Reference waypoint generators
│   ├── cf_obstacles.py     # Static obstacle definitions
│   ├── cf_csv.py           # CSV I/O for OCP solutions
│   ├── cf_plots.py         # Matplotlib visualisation and animation
│   ├── cf_plot_traj.py     # Plot helper for poly-7 CSV files
│   ├── cf_bag.py           # rosbag2 reader (pose / vel / accel)
│   ├── cf_debug.py         # Quick inspection of saved OCP & CSV data
│   ├── crazyflo_solve.py   # Top-level script: solve OCP and save results
│   ├── crazyflo_plot.py    # Top-level script: load and plot results
│   ├── crazyflo_mission.py # Execute pre-computed trajectories on real drones
│   └── crazyflo_sim.py     # ROS 2 node: simulate payload position from TF
├── config/
│   ├── crazyflo.yaml       # OCP planner parameters
│   ├── sim.yaml            # Crazyswarm2 robot config (simulation)
│   ├── real.yaml           # Crazyswarm2 robot config (real drones)
│   ├── real1.yaml          # Alternate real-drone config
│   ├── config.rviz         # RViz visualisation preset
│   └── config1.rviz
├── launch/
│   ├── launch.py           # Base launch file (backend-agnostic)
│   ├── sim.launch.xml      # Launch simulation + payload node + RViz
│   ├── real.launch.xml     # Launch real-world flight
│   └── real1.launch.xml
├── data/                   # Pre-computed trajectory CSV files
│   ├── figure8.csv
│   ├── traj_cf{1,2,3}.csv
│   └── ...
├── urdf/
│   ├── crazyflie_body.xacro
│   └── crazyflie_description.urdf
└── setup.py
```

## Dependencies

| Dependency | Purpose |
|---|---|
| ROS 2 (Humble or later) | Middleware |
| [Crazyswarm2 / crazyflie_py](https://imrclab.github.io/crazyswarm2/) | Drone communication |
| [CasADi](https://web.casadi.org/) | OCP solver |
| NumPy / SciPy | Numerics |
| Matplotlib | Plots & animation |
| rosbag2_py | Bag reading |

Install Python dependencies:

```bash
pip install casadi numpy scipy matplotlib
```

Build the ROS 2 package:

```bash
cd ~/ws
colcon build --packages-select crazyflo_planner
source install/setup.bash
```

## Quick Start

### 1 — Solve the OCP (off-line planning)

Edit the parameters at the top of `crazyflo_solve.py` (trajectory type, obstacle layout, cable length, …), then run:

```bash
cd crazyflo_planner/crazyflo_planner
python crazyflo_solve.py
```

This will:
- Generate payload waypoints and save them to `data/pl_waypoints.csv`.
- Solve the OCP and save the solution to `data/ocp.npz`.
- Export per-drone polynomial CSV files (`data/traj_cf1.csv`, `traj_cf2.csv`, `traj_cf3.csv`).
- Optionally display 3-D plots and animations.

### 2 — Simulate

```bash
ros2 launch crazyflo_planner sim.launch.xml
```

This starts:
- The Crazyswarm2 server in `sim` (software-in-the-loop) backend.
- The `crazyflo_sim` ROS 2 node that estimates payload position from drone TF frames.
- RViz for real-time visualisation.

### 3 — Fly on real drones

```bash
ros2 launch crazyflo_planner real.launch.xml
```

Then, in a separate terminal:

```bash
cd crazyflo_planner/crazyflo_planner
python crazyflo_mission.py
```

The `crazyflo_mission.py` script:
1. Loads the pre-computed trajectory CSVs.
2. Waits for button press → takeoff.
3. Positions each drone at its trajectory start point.
4. Waits for button press → starts all trajectories simultaneously.
5. Waits for button press → lands.

### 4 — Plot results

```bash
cd crazyflo_planner/crazyflo_planner
python crazyflo_plot.py
```

Enable `PLOT_BAG = 1` and set the correct `bag_folder` to overlay rosbag2 measurements on top of the OCP solution.

## Configuration

### `config/crazyflo.yaml` — Planner parameters

| Parameter | Default | Description |
|---|---|---|
| `payload_start` | `[0, 0, 1]` | Payload start position (m) |
| `payload_goal` | `[2, 0, 1]` | Payload goal position (m) |
| `cf_{1,2,3}_start` | equilateral triangle | Initial drone positions (m) |
| `max_velocity` | `1.0` | Drone velocity limit (m/s) |
| `max_acceleration` | `1.0` | Drone acceleration limit (m/s²) |
| `max_jerk` | `2.0` | Drone jerk limit (m/s³) |
| `max_snap` | `5.0` | Drone snap limit (m/s⁴) |
| `cable_length` | `1.0` | Cable length from drone to payload (m) |

### `config/sim.yaml` / `config/real.yaml` — Robot configuration

These are standard Crazyswarm2 configuration files. `sim.yaml` uses the `udp://` transport for simulation; `real.yaml` uses `radio://` URIs with the physical drone addresses and enables custom firmware logging topics.

## Module Reference

### `cf_solver.py` — OCP formulation

```python
sol = cf_solver.solve_ocp(
    waypoints,          # (N,3) payload reference path
    cable_l=0.5,        # cable length [m]
    cf_v_max=2.0,       # drone speed limit [m/s]
    cf_a_max=5.0,       # drone accel limit [m/s²]
    obstacles=[],       # list of {'nogo': {'center', 'size'}} dicts
)
```

The solver uses a **direct collocation** approach on a variable-time grid:
- Decision variables: payload position/velocity, drone velocity/acceleration, cable directions, cable tensions.
- Drone position is derived as `payload_pos + cable_length * cable_direction`.
- Constraints: dynamics (Euler integration), boundary conditions (start/end at rest, static equilibrium), speed/acceleration limits, unit-length cable directions, tension bounds, inter-drone collision avoidance, and obstacle avoidance.
- Objective: weighted sum of time, tracking error, derivative magnitudes, and obstacle penalty.

Returns a `dict` containing time vector, all states, and metadata for post-processing.

### `cf_traj.py` — Minimum-snap polynomial trajectory

```python
from cf_traj import waypoints_to_poly7, Poly7

poly = waypoints_to_poly7(
    waypoints,          # (N,3) positions
    v_max=2.0,
    a_max=5.0,
    continuity_order=3, # C2 or C3 continuity at internal knots
)
```

`Poly7` stores:
- `t_knots`: (N,) knot times.
- `coeffs`: (N-1, 3, 8) polynomial coefficients per segment per axis.

Helper functions: `poly7_to_timed_waypoints`, `timed_waypoints_to_poly7`, `eval_poly7`.

### `cf_waypoints.py` — Reference path generators

| Function | Description |
|---|---|
| `generate_ellipse(r_A, r_B, height)` | Horizontal ellipse |
| `generate_figure8(r_A, r_B, height)` | Lemniscate (figure-8) |
| `generate_line(start, end)` | Straight segment |
| `generate_random_walk(start, step_size)` | Random walk |
| `generate_waypoints(traj, height, length, loops, N, folder)` | High-level dispatcher + CSV export |

### `cf_obstacles.py` — Obstacle definitions

| Function | Description |
|---|---|
| `get_obstacles(label, gap, length, height)` | Return obstacle list by name |
| `create_wall(...)` | Single blocking wall |
| `create_vertical_passage(...)` | Two walls with lateral gap |
| `create_horizontal_passage(...)` | Two walls with vertical gap |
| `create_course(...)` | Three-obstacle slalom course |
| `update_waypoints(waypoints, obstacles)` | Insert passage waypoints into path |

Obstacles are represented as axis-aligned boxes: `{'nogo': {'center': np.ndarray(3), 'size': np.ndarray(3)}}`.

### `cf_csv.py` — CSV I/O

| Function | Description |
|---|---|
| `save_time_pos_csv(sol, path)` | Save per-drone time/pos/vel/acc/jerk/snap CSV |
| `read_time_pos_csv(filename)` | Load CSV back into a state dict |
| `save_poly7_csv(sol, folder, v_max, a_max)` | Convert OCP solution to Crazyflie-compatible poly-7 CSV |

### `cf_plots.py` — Visualisation

| Function | Description |
|---|---|
| `plot_ocp(ocp_data, animate, folder)` | Full OCP result (states, 3-D, optional animation) |
| `plot_xyz(data, t_offset, t_total)` | Per-axis position over time |
| `plot_states_cf(t, cf_p, cf_v, cf_a)` | Altitude, speed, acceleration |
| `plot_cost(sol)` | OCP cost breakdown |
| `animate_ocp(ocp_data)` | 3-D animation with `FuncAnimation` |
| `save_plots(f_states, f_constr, f_3d, folder)` | Save figures to disk |

### `cf_bag.py` — rosbag2 reader

```python
bag_data = cf_bag.get_bag_data(bag_path)
```

Reads `/cf{1,2,3}/pose` topics from an MCAP bag and returns a dict with `t`, `cf{1,2,3}_p`, `cf{1,2,3}_v`, `cf{1,2,3}_a` arrays for comparison with the OCP solution.

### `payload_sim.py` — ROS 2 node

Estimates the payload position by computing the geometric centroid of the cable attachment points from the TF tree. Publishes a `visualization_msgs/Marker` on `payload_marker` and broadcasts a TF frame for the payload. Parameters:

| Parameter | Default | Description |
|---|---|---|
| `rate_hz` | `200.0` | Update rate (Hz) |
| `cable_length` | `0.5` | Cable length (m) |

## Data Files

Pre-computed trajectories for three drones are stored in `data/`:
- `traj_cf{1,2,3}.csv` — Default 7th-degree polynomial trajectories (Crazyflie format).
- `figure8.csv` — Figure-8 example trajectory.
- `ocp_solution.npz` — Cached NumPy OCP solution.

## License

MIT — see [LICENSE](LICENSE).
