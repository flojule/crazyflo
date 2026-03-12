import numpy as np

from pathlib import Path


ROOT_FOLDER = Path.home() / ".ros/crazyflo_planner"
PLOT_OCP = True
PLOT_BAG = False
OFFSET_BAG = 0.0  # offset in seconds to align with ocp solution



if __name__ == "__main__":
    # Load OCP data
    ocp_path = ROOT_FOLDER / "data" / "ocp_solution.npz"
    ocp_data = np.load(ocp_path, allow_pickle=True)

    data_path = ROOT_FOLDER / "data"
    cf_name = ["cf1", "cf2", "cf3"]

    # print some info about the data
    print(f"\nLoaded OCP data from {ocp_path}")
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
