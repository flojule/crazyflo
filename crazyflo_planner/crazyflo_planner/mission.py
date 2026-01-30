#!/usr/bin/env python

from pathlib import Path

from ament_index_python import get_package_share_directory
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
from cflib.crtp import scan_interfaces

import numpy as np

TRIALS = 1
TIMESCALE = 1.0

BUFFER_TIME = 2.0

TAKEOFF_TIME = 2.0
INITIAL_HEIGHT = 1.0

LANDING_TIME = 2.0

csv_cf1 = Path(get_package_share_directory(
    'crazyflo_planner')) / 'data' / f'traj_cf1.csv'
csv_cf2 = Path(get_package_share_directory(
    'crazyflo_planner')) / 'data' / f'traj_cf2.csv'
csv_cf3 = Path(get_package_share_directory(
    'crazyflo_planner')) / 'data' / f'traj_cf3.csv'

csv_path_default = [csv_cf1, csv_cf2, csv_cf3]
# csv_path_default = [csv_cf1]


def main(csv_paths=csv_path_default):
    """Run the Crazyflie swarm with predefined trajectories."""
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    
    # print(scan_interfaces())

    while len(csv_paths) < len(allcfs.crazyflies):
        csv_paths.append(csv_paths[-1])

    traj_cfs = []
    for i in range(len(allcfs.crazyflies)):
        traj_cfs.append(Trajectory())
        traj_cfs[i].loadcsv(csv_paths[i])

    allcfs.setParam('usd.logging', 1)  # enable logging

    for i in range(TRIALS):
        for i, cf in enumerate(allcfs.crazyflies):
            cf.uploadTrajectory(0, 0, traj_cfs[i])

        print('press button to takeoff...')
        swarm.input.waitUntilButtonPressed()
        allcfs.takeoff(targetHeight=INITIAL_HEIGHT, duration=TAKEOFF_TIME)
        timeHelper.sleep(BUFFER_TIME)

        print('press button to start trajectory...')
        swarm.input.waitUntilButtonPressed()
        allcfs.startTrajectory(0, timescale=TIMESCALE)
        timeHelper.sleep(traj_cfs[0].duration * TIMESCALE + BUFFER_TIME)

        print('press button to land...')
        swarm.input.waitUntilButtonPressed()
        allcfs.land(targetHeight=0.03, duration=LANDING_TIME)
        timeHelper.sleep(BUFFER_TIME)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()
