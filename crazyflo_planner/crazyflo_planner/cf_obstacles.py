import numpy as np

BIG = 10.0


def get_obstacles(label="vertical_passage", gap=0.5, length=2.0, height=0.5):
    """Define static obstacles in the environment."""
    obstacles = list()
    if label == "vertical_passage":
        obstacles = create_vertical_passage(obstacles, gap=gap, x=length/2, y=0.0, z=height)
    elif label == "horizontal_passage":
        obstacles = create_horizontal_passage(obstacles, gap=gap, x=length/2, y=0.0, z=height)
    elif label == "course":
        obstacles = create_course(obstacles, gap=gap, length=length)
    elif label == "wall":
        obstacles = create_wall(obstacles, x=length/2, y=0.0, z=height)
    return obstacles


def create_wall(obstacles=None, x=5.0, y=0.0, z=0.5):
    """Create a wall obstacle."""
    if obstacles is None:
        obstacles = []
    x_size = 0.1
    y_size = 0.5
    z_size = 0.5
    x_center = x
    y_center = y
    z_center = z
    obstacles.append({"nogo": {"center": np.array([x_center, y_center, z_center]),
                 "size": np.array([x_size, y_size, z_size])}})
    return obstacles


def create_vertical_passage(obstacles=None, gap=0.5, x=5.0, y=0.0, z=0.5):
    """Create two boxes with an open vertical passage (y)."""
    if obstacles is None:
        obstacles = []
    x_size = 0.1
    y_size = BIG/2
    z_size = BIG
    x_center = x
    y_center = gap/2 + y_size/2
    z_center = z
    obstacles.append({"nogo": {"center": np.array([x_center, y + y_center, z_center]),
                               "size": np.array([x_size, y_size, z_size])}})
    obstacles.append({"nogo": {"center": np.array([x_center, y - y_center, z_center]),
                               "size": np.array([x_size, y_size, z_size])}})
    obstacles.append({"passage": {"center": np.array([x_center, y, z_center]),
                                  "size": np.array([x_size, gap, z_size])}})
    return obstacles


def create_horizontal_passage(obstacles=None, gap=0.5, x=5.0, y=0.0, z=0.0):
    """Create two boxes with an open horizontal passage (z)."""
    if obstacles is None:
        obstacles = []
    x_size = 0.1
    y_size = BIG
    z_size = BIG/2
    x_center = x
    y_center = y
    z_center = gap/2 + z_size/2
    obstacles.append({"nogo": {"center": np.array([x_center, y_center, z + z_center]),
                               "size": np.array([x_size, y_size, z_size])}})
    obstacles.append({"nogo": {"center": np.array([x_center, y_center, z - z_center]),
                               "size": np.array([x_size, y_size, z_size])}})
    obstacles.append({"passage": {"center": np.array([x_center, y_center, z]),
                                  "size": np.array([x_size, y_size, gap])}})
    return obstacles


def create_course(obstacles=None, gap=0.5, length=2.0, height=0.5):
    """Create a course with multiple vertical and horizontal passages."""
    if obstacles is None:
        obstacles = []
    obstacles = create_vertical_passage(obstacles, gap=gap, x=length*1/4, y=1.0, z=height)
    obstacles = create_horizontal_passage(obstacles, gap=gap, x=length*2/4, y=0.0, z=height)
    obstacles = create_vertical_passage(obstacles, gap=gap, x=length*3/4, y=-1.0, z=height)
    return obstacles


def update_waypoints(waypoints, obstacles):
    """Add waypoints to navigate through the obstacle passages."""
    for obs in obstacles:
        if "passage" in obs:
            center = obs["passage"]["center"]
            waypoints = np.vstack([waypoints, center])
    waypoints = waypoints[np.argsort(waypoints[:, 0])]  # sort by x
    return waypoints
