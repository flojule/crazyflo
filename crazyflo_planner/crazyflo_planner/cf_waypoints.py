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


def generate_line(start, end, grid):
    """Generate a straight line trajectory."""
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

