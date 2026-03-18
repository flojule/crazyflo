# crazyflo

[Portfolio post](https://flojule.github.io/projects/nu-499/crazyflo/)

Cooperative payload transport with a swarm of three [Crazyflie](https://www.bitcraze.io/) drones connected to a shared payload via cables.

[Watch the demo video](https://github.com/user-attachments/assets/48989a06-b6cf-4c2b-961a-385476a8ae29)

## What it does

Three drones fly in formation, each attached to the same payload by a cable. An **Optimal Control Problem (OCP)** is solved offline to generate dynamically feasible, collision-free trajectories for all drones simultaneously. The resulting trajectories are uploaded to the firmware and executed.

![ellipse sim](crazyflo_planner/figures/ellipse/ellipse_sim.gif)


### Figure-8

| Animation | States (altitude / speed / acceleration) |
|:---:|:---:|
| ![figure8 animation](crazyflo_planner/figures/figure8/animation.gif) | ![figure8 states](crazyflo_planner/figures/figure8/cf_plot.png) |

### Ellipse

| Animation | States (altitude / speed / acceleration) |
|:---:|:---:|
| ![ellipse animation](crazyflo_planner/figures/ellipse/animation.gif) | ![ellipse states](crazyflo_planner/figures/ellipse/cf_plot.png) |

## Repository layout

```
crazyflo/
└── crazyflo_planner/   # ROS 2 Python package (planning + execution)
```

See [crazyflo_planner/README.md](crazyflo_planner/README.md) for the full module reference, configuration details, and step-by-step usage.

## Quick start

```bash
# 1 — Plan trajectories (offline, no drones needed)
python crazyflo_solve.py

# 2 — Launch simulation or real drones
ros2 launch crazyflo_planner sim.launch.xml
ros2 launch crazyflo_planner real.launch.xml

# 3 — Run the mission script (add <mission> arguments)
ros2 run crazyflo_planner mission
```


## Installation

### 1. Clone this repository

```bash
git clone https://github.com/flojule/crazyflo.git
```

### 2. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install ROS 2 (Jazzy)

Follow the [official ROS 2 installation guide](https://docs.ros.org/en/jazzy/Installation.html).

### 4. Install Crazyswarm2

Follow the [Crazyswarm2 installation instructions](https://imrclab.github.io/crazyswarm2/installation.html):

### 5. Install cfclient (for Crazyflie hardware configuration)

Follow the [cfclient installation instructions](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/installation/):

### 6. Build the ROS 2 package

```bash
cd ~/ws
colcon build --packages-select crazyflo_planner
source install/setup.bash
```

---

## Key dependencies

| Tool | Role |
|---|---|
| ROS 2 Jazzy | Middleware |
| [Crazyswarm2](https://imrclab.github.io/crazyswarm2/) | Drone communication & simulation |
| [cfclient](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/) | Crazyflie configuration |
| [CasADi](https://web.casadi.org/) | OCP solver |
