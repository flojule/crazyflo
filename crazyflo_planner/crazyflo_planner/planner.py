#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path


class PayloadPlanner(Node):

    def __init__(self):
        super().__init__("payload_planner_node")

        self.get_logger().info("Starting payload planner node...")

        # Parameters
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.cable_length = self.declare_parameter("cable_length", 1.0).value
        self.rate_hz = self.declare_parameter("rate_hz", 50.0).value

        # timer
        self.timer = self.create_timer(1/self.rate_hz, self.timer_callback)

        # Publishers
        self.setpoints_pub = self.create_publisher(
            PoseArray, "planner/cfs_setpoints", 10)
        self.paths_pub = self.create_publisher(
            Path, "planner/cfs_paths", 10)

    def timer_callback(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = PayloadPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
