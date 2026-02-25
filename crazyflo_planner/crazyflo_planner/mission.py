#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory

from crazyflo_planner.cf_solver import get_traj

TRIALS = 1
TIMESCALE = 1.0

BUFFER_TIME = 2.0

TAKEOFF_TIME = 2.0
INITIAL_HEIGHT = 0.4

LANDING_TIME = 2.0

# ROOT_FOLDER = Path.home() / ".ros/crazyflo_planner"
ROOT_FOLDER = Path.home() / "winter-project/ws/src/crazyflo/crazyflo_planner"
# ROOT_FOLDER = Path.home() / "winter-project/ws/src/crazyflo/crazyflo_planner/data/multi_trajectory"

data_path = ROOT_FOLDER / "data"


def main():
    """Run the Crazyflie swarm with predefined trajectories."""
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # get_traj(ros=True)  # generate trajectories
    csv_cf1 = data_path / 'traj_cf1.csv'
    csv_cf2 = data_path / 'traj_cf2.csv'
    csv_cf3 = data_path / 'traj_cf3.csv'

    csv_paths = [csv_cf1, csv_cf2, csv_cf3]

    while len(csv_paths) < len(allcfs.crazyflies):
        csv_paths.append(csv_paths[-1])

    traj_cfs = []
    for i in range(len(allcfs.crazyflies)):
        traj_cfs.append(Trajectory())
        traj_cfs[i].loadcsv(csv_paths[i])

    allcfs.setParam('usd.logging', 1)  # enable logging

    print('uploading trajectories to crazyflies...')
    for i, cf in enumerate(allcfs.crazyflies):
        cf.uploadTrajectory(0, 0, traj_cfs[i])

    print('press button to takeoff...')
    swarm.input.waitUntilButtonPressed()
    allcfs.takeoff(targetHeight=INITIAL_HEIGHT, duration=TAKEOFF_TIME)
    timeHelper.sleep(BUFFER_TIME)

    for i, cf in enumerate(allcfs.crazyflies):
        start = traj_cfs[i].eval(0.0)
        x, y, _ = start.pos
        yaw0 = start.yaw
        cf.goTo((x, y, INITIAL_HEIGHT), yaw0, 2.0)
        # timeHelper.sleep(BUFFER_TIME)

    print('press button to go to start position...')
    swarm.input.waitUntilButtonPressed()
    for i, cf in enumerate(allcfs.crazyflies):
        start = traj_cfs[i].eval(0.0)
        p0 = start.pos
        yaw0 = start.yaw
        cf.goTo(p0, yaw0, 4.0)
        # timeHelper.sleep(BUFFER_TIME)

    print('press button to start trajectory...')
    swarm.input.waitUntilButtonPressed()
    allcfs.startTrajectory(0, timescale=TIMESCALE)
    # timeHelper.sleep(traj_cfs[0].duration * TIMESCALE + BUFFER_TIME)

    print('press button to land...')
    swarm.input.waitUntilButtonPressed()
    allcfs.land(targetHeight=0.03, duration=LANDING_TIME)
    timeHelper.sleep(BUFFER_TIME)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()
