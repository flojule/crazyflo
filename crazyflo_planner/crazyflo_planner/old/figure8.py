#!/usr/bin/env python

from crazyflo_planner import mission
from pathlib import Path
from ament_index_python import get_package_share_directory


def main():
    csv_path = Path(get_package_share_directory(
        'crazyflo_planner')) / 'data' / 'figure8.csv'
    mission.main(csv_paths=[csv_path])


if __name__ == '__main__':
    main()
