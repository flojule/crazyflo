import numpy as np

BIG = 10.0


def get_obstacles(label="vertical_passage", gap=0.5, length=2.0):
    """Define static obstacles in the environment."""
    obstacles = list()
    if label == "vertical_passage":
        obstacles = create_vertical_passage(obstacles, gap=gap, x=length/2)
    elif label == "horizontal_passage":
        obstacles = create_horizontal_passage(obstacles, gap=gap, x=length/2)
    elif label == "course":
        obstacles = create_course(obstacles, gap=gap, length=length)
    return obstacles


def create_vertical_passage(list=[], gap=0.5, x=5.0, y=0.0):
    """Create two boxes with an open vertical passage (y)."""
    x_size = 0.1
    y_size = BIG/2
    z_size = BIG
    x_center = x
    y_center = gap/2 + y_size/2
    z_center = 0.5
    list.append({"center": np.array([x_center, y + y_center, z_center]),
                 "size": np.array([x_size, y_size, z_size])})
    list.append({"center": np.array([x_center, y - y_center, z_center]),
                 "size": np.array([x_size, y_size, z_size])})
    return list


def create_horizontal_passage(list=[], gap=0.5, x=5.0, z=0.0):
    """Create two boxes with an open horizontal passage (z)."""
    x_size = 0.1
    y_size = BIG
    z_size = BIG/2
    x_center = x
    y_center = 0.0
    z_center = gap/2 + z_size/2
    list.append({"center": np.array([x_center, y_center, z + z_center]),
                 "size": np.array([x_size, y_size, z_size])})
    list.append({"center": np.array([x_center, y_center, z - z_center]),
                 "size": np.array([x_size, y_size, z_size])})
    return list


def create_course(list=[], gap=0.5, length=2.0):
    """Create a course with multiple vertical and horizontal passages."""
    list = create_vertical_passage(list, gap=gap, x=length*1/3, y=1.0)
    list = create_horizontal_passage(list, gap=gap, x=length/2, z=0.5)
    list = create_vertical_passage(list, gap=gap, x=length*2/3, y=-1.0)
    return list
