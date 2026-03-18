"""crazyflo_animate.py - save multi-view real-time animation for a mission.

Loads the OCP solution for MISSION and saves the animation to the mission
figure folder as animation.mp4.

"""

import cf_plots
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
MISSION = "ellipse"
ROOT_FOLDER = Path.home() / "winter-project/ws/"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _data_root = ROOT_FOLDER / "data"
    _plot_root = ROOT_FOLDER / "figures"

    data_folder = _data_root / MISSION
    plot_folder = _plot_root / MISSION
    plot_folder.mkdir(parents=True, exist_ok=True)

    ocp_path = data_folder / "ocp.npz"
    ocp_sol = np.load(ocp_path, allow_pickle=True)
    print(f"Loaded OCP solution from {ocp_path}")

    cam = (1, 2, 0.5)  # camera position for all views

    cf_plots.animate_ocp(ocp_sol, folder=plot_folder, cam=cam)
