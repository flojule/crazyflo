import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict

import numpy as np


def get_bag_data(bag_path):
    """Read a rosbag2 bag file and return a dictionary."""
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id="mcap"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    print("Topics in bag:")
    for topic in type_map:
        print(f"  {topic}: {type_map[topic]}")
    print()

    cf1_pose = "/cf1/pose"
    cf2_pose = "/cf2/pose"
    cf3_pose = "/cf3/pose"
    topics = [cf1_pose, cf2_pose, cf3_pose]
    filter_poses = rosbag2_py.StorageFilter(topics=topics)
    reader.set_filter(filter_poses)

    bag_data_topics = {}
    bag_data = {}

    while reader.has_next():
        topic, raw, timestamp = reader.read_next()

        msg_type = get_message(type_map[topic])
        msg = deserialize_message(raw, msg_type)
        msg_dict = message_to_ordereddict(msg)

        if topic not in bag_data_topics:
            bag_data_topics[topic] = []

        bag_data_topics[topic].append({
            "time": timestamp,
            "msg": msg_dict
        })
    # print(bag_data[cf1_pose][0]["msg"]["pose"]["position"]["x"])

    t1, t2, t3 = [], [], []
    cf1_p, cf2_p, cf3_p = [], [], []
    cf1_v, cf2_v, cf3_v = [], [], []
    cf1_a, cf2_a, cf3_a = [], [], []

    poses = ["/cf1/pose", "/cf2/pose", "/cf3/pose"]
    vels = ["/cf1/vel", "/cf2/vel", "/cf3/vel"]
    accels = ["/cf1/acceleration", "/cf2/acceleration", "/cf3/acceleration"]
    cf_p = [cf1_p, cf2_p, cf3_p]
    cf_v = [cf1_v, cf2_v, cf3_v]
    cf_a = [cf1_a, cf2_a, cf3_a]
    t = [t1, t2, t3]

    for j in range(3):
        pose = poses[j]
        if pose not in bag_data_topics or len(bag_data_topics[pose]) == 0:
            t[j] = np.array([])
            cf_p[j] = np.empty((3, 0))
            continue
        for i in range(len(bag_data_topics[pose])):
            # t_ = bag_data_topics[pose][i]["msg"]["header"]["stamp"]["sec"] + bag_data_topics[pose][i]["msg"]["header"]["stamp"]["nanosec"] * 1e-9
            t_ = bag_data_topics[pose][i]["time"] * 1e-9
            t[j].append(t_)
            cf_p[j].append(
                [bag_data_topics[pose][i]["msg"]["pose"]["position"]["x"],
                    bag_data_topics[pose][i]["msg"]["pose"]["position"]["y"],
                    bag_data_topics[pose][i]["msg"]["pose"]["position"]["z"]])
            # cf_v[j].append(
            #     [bag_data_topics[vels[j]][i]["msg"]["vector"]["x"],
            #         bag_data_topics[vels[j]][i]["msg"]["vector"]["y"],
            #         bag_data_topics[vels[j]][i]["msg"]["vector"]["z"]])
            # cf_a[j].append(
            #     [bag_data_topics[accels[j]][i]["msg"]["vector"]["x"],
            #         bag_data_topics[accels[j]][i]["msg"]["vector"]["y"],
            #         bag_data_topics[accels[j]][i]["msg"]["vector"]["z"]])

        t[j] = np.array(t[j]) - t[j][0]  # (N,)
        cf_p[j] = np.array(cf_p[j]).T  # (3, N)
        # cf_v[j] = np.array(cf_v[j]).T  # (3, N)
        # cf_a[j] = np.array(cf_a[j]).T  # (3, N)

    if len(t2) == 0 or len(t3) == 0:
        t_new = np.asarray(t1)
    else:
        # find common time range
        t_start = max(t1[0], t2[0], t3[0])
        t_end = min(t1[-1], t2[-1], t3[-1])
        # average dt
        dt = np.median(np.diff(t1))
        t_new = np.arange(0, t_end - t_start, dt)

    # print(f"Original time lengths: {len(t1)}, {len(t2)}, {len(t3)}")
    # print(f"New time length: {len(t_new)}")

    cf_p_new, cf_v_new, cf_a_new = [], [], []
    for j in range(3):
        if len(t[j]) == 0:
            cf_p_new.append(np.empty((3, len(t_new))))
            cf_v_new.append(np.empty((3, len(t_new))))
            cf_a_new.append(np.empty((3, len(t_new))))
        else:
            pose = np.vstack([
                np.interp(t_new, t[j], cf_p[j][0, :]),
                np.interp(t_new, t[j], cf_p[j][1, :]),
                np.interp(t_new, t[j], cf_p[j][2, :])
            ])
            cf_p_new.append(pose)
            # cf_v[j] = np.interp(t_new, t[j], cf_v[j])
            # cf_a[j] = np.interp(t_new, t[j], cf_a[j])

    bag_data["t"] = t_new
    bag_data["cf1_p"] = cf_p_new[0]
    bag_data["cf2_p"] = cf_p_new[1]
    bag_data["cf3_p"] = cf_p_new[2]

    # bag_data["cf1_v"] = cf1_v
    # bag_data["cf2_v"] = cf2_v
    # bag_data["cf3_v"] = cf3_v
    # bag_data["cf1_a"] = cf1_a
    # bag_data["cf2_a"] = cf2_a
    # bag_data["cf3_a"] = cf3_a

    return bag_data
