import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from pathlib import Path

import cf_data
import cf_plots

ROOT_FOLDER = Path.home() / ".ros/crazyflo_planner"
PLOT_OCP = True
PLOT_BAG = False
OFFSET_BAG = 0.0  # offset in seconds to align with ocp solution



# Load OCP data
ocp_path = ROOT_FOLDER / "data" / "ocp_solution.npz"
ocp_data = np.load(ocp_path)

data_path = ROOT_FOLDER / "data"
cf_name = ["cf1", "cf2", "cf3"]

# # Load rosbag data
# bag_path = Path.home() / "winter-project/ws/bag"
# bag_path = bag_path / "rosbag2_2026_02_06-17_56_10"
# bag_data = cf_data.get_bag_data(bag_path)

# print some info about the data
print(f"\nLoaded OCP data from {ocp_path}")
# print first 5 entries of OCP data
print("OCP data keys:", ocp_data.files)
print("OCP data shapes:")
for key in ocp_data.files:
    print(f"  {key}: {ocp_data[key].shape}")

# open csv files and print first 5 lines
for i in range(3):
    csv_path = data_path / f"traj_{cf_name[i]}.csv"
    print(f"\nLoaded CSV data from {csv_path}")
    with open(csv_path, "r") as f:
        for _ in range(5):
            line = f.readline().strip()
            print(line)
