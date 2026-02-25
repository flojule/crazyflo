import numpy as np


def generate_ellipse(r_A, r_B, height, grid):
    """Generate an elliptical trajectory in the horizontal plane."""
    r_A = 0.6
    r_B = 0.3
    waypoints = np.stack([r_A * (1.0 - np.cos(2 * np.pi * grid)),
                          r_B * np.sin(2 * np.pi * grid),
                          height * np.ones(grid.shape)
                          ], axis=1)
    return waypoints


def generate_figure8(r_A, r_B, height, grid):
    """Generate a figure-8 trajectory in the horizontal plane."""
    r_A = 0.6
    r_B = 0.3
    waypoints = np.stack([r_A * np.sin(2 * np.pi * grid),
                          r_B * np.sin(4 * np.pi * grid) / 2,
                          height * np.ones(grid.shape)
                          ], axis=1)
    return waypoints


def generate_line(start, end, grid=None):
    """Generate a straight line trajectory."""
    if grid is None:
        grid = np.linspace(0, 1, 2)
    waypoints = np.stack([np.linspace(start[0], end[0], grid.shape[0]),
                          np.linspace(start[1], end[1], grid.shape[0]),
                          np.linspace(start[2], end[2], grid.shape[0])
                          ], axis=1)
    return waypoints


def generate_random_walk(start, step_size, grid):
    """Generate a random walk trajectory."""
    waypoints = [start]
    for _ in range(1, grid.shape[0]):
        step = np.random.uniform(-step_size, step_size, size=3)
        waypoints.append(waypoints[-1] + step)
    return np.stack(waypoints, axis=0)


def generate_waypoints(traj='ellipse', height=0.5, loops=5, N=20, folder=''):
    grid = np.linspace(0, 1, N + 1)  # points for traj

    if traj == 'ellipse':
        waypoints = generate_ellipse(
            r_A=0.6, r_B=0.3, height=height, grid=grid)
    elif traj == 'figure8':
        waypoints = generate_figure8(
            r_A=0.6, r_B=0.6, height=height, grid=grid)
    elif traj == 'random':
        start = np.array([0, 0, height])
        step_size = 0.5
        waypoints = generate_random_walk(
            start=start, step_size=step_size, grid=grid)
    elif traj == 'line':  # straight line
        start = np.array([0, 0, height])
        goal = np.array([2, 0, height])
        waypoints = generate_line(
            start=start, end=goal, grid=grid)
    else:
        raise ValueError(f"Unknown traj type: {traj}")

    if loops > 1 and traj != 'line' and traj != 'random':
        p0 = waypoints.copy()  # one loop, length N+1
        waypoints = np.concatenate(
            [p0[:-1] for _ in range(loops - 1)] + [p0], axis=0)

    # save pl_waypoints to csv
    out = folder / "pl_waypoints.csv"
    np.savetxt(out, waypoints, delimiter=",")
    print(f"Wrote {out} with {len(waypoints)} waypoints.")

    return waypoints
