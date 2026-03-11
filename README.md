# crazyflo

Cooperative payload transport with a swarm of three [Crazyflie](https://www.bitcraze.io/products/crazyflie-2-1/) drones connected to a shared payload via cables.

## What it does

Three drones fly in formation, each attached to the same payload by a cable. An **Optimal Control Problem (OCP)** is solved offline to generate dynamically feasible, collision-free trajectories for all drones simultaneously. The resulting trajectories are uploaded to the firmware and executed in a button-gated sequence (takeoff → align → fly → land).

## Repository layout

```
crazyflo/
└── crazyflo_planner/   # ROS 2 Python package (planning + execution)
```

See [crazyflo_planner/README.md](crazyflo_planner/README.md) for the full module reference, configuration details, and step-by-step usage.

## Quick start

```bash
# 1 — Source ROS 2 and the workspace
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source ~/winter-project/venvs/crazyswarm2/bin/activate

# 2 — Plan trajectories (offline, no drones needed)
cd crazyflo_planner/crazyflo_planner
python crazyflo_solve.py

# 3 — Launch simulation
ros2 launch crazyflo_planner sim.launch.xml

# 4 — Or launch on real drones + run the mission script
ros2 launch crazyflo_planner real.launch.xml
ros2 run crazyflo_planner mission
```

## Key dependencies

| Tool | Role |
|---|---|
| ROS 2 Jazzy | Middleware |
| [Crazyswarm2](https://imrclab.github.io/crazyswarm2/) | Drone communication & simulation |
| [CasADi](https://web.casadi.org/) | OCP solver |
| NumPy / SciPy / Matplotlib | Numerics & visualisation |

## License

MIT — see [crazyflo_planner/LICENSE](crazyflo_planner/LICENSE).
