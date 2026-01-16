import math
from dataclasses import dataclass
from typing import List

import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point

from crazyflie_interfaces.srv import Takeoff, Land, GoTo


@dataclass
class Waypoint:
    x: float
    y: float
    z: float


def dur(s: float) -> Duration:
    d = Duration()
    ns = int(s * 1e9)
    d.sec = ns // 1_000_000_000
    d.nanosec = ns % 1_000_000_000
    return d


class MultiMission(Node):
    def __init__(self):
        super().__init__("multi_mission")

        # ---- parameters ----
        self.declare_parameter("drones", ["cf_1", "cf_2", "cf_3", "cf_4"])
        self.declare_parameter("takeoff_height", 0.5)
        self.declare_parameter("takeoff_duration_s", 2.0)

        self.declare_parameter("segment_duration_s", 2.0)
        self.declare_parameter("yaw_deg", 0.0)
        self.declare_parameter("relative", False)

        self.declare_parameter("land_height", 0.0)
        self.declare_parameter("land_duration_s", 2.0)

        self.drones: List[str] = list(self.get_parameter("drones").value)

        # A simple square-ish path at z=0.5 (edit as you like)
        self.trajectory: List[Waypoint] = [
            Waypoint(0.0, 0.0, 0.5),
            Waypoint(0.5, 0.0, 0.5),
            Waypoint(0.5, 0.5, 0.5),
            Waypoint(0.0, 0.5, 0.5),
            Waypoint(0.0, 0.0, 0.5),
        ]
        self.traj_i = 0

        # ---- service clients per drone ----
        self.takeoff_clients = {}
        self.goto_clients = {}
        self.land_clients = {}

        for name in self.drones:
            self.takeoff_clients[name] = self.create_client(Takeoff, f"/{name}/takeoff")
            self.goto_clients[name] = self.create_client(GoTo, f"/{name}/go_to")
            self.land_clients[name] = self.create_client(Land, f"/{name}/land")

        # ---- state ----
        self.phase = "wait_services"
        self.pending_futures = []

        self.timer = self.create_timer(0.1, self.tick)  # 10 Hz

    def wait_all_services(self) -> bool:
        ok = True
        for name in self.drones:
            ok &= self.takeoff_clients[name].wait_for_service(timeout_sec=0.0)
            ok &= self.goto_clients[name].wait_for_service(timeout_sec=0.0)
            ok &= self.land_clients[name].wait_for_service(timeout_sec=0.0)
        if not ok:
            self.get_logger().warn("Waiting for /<cf>/{takeoff,go_to,land} services...")
        return ok

    @staticmethod
    def all_done(futures) -> bool:
        return all(f.done() for f in futures)

    def start_takeoff(self):
        h = float(self.get_parameter("takeoff_height").value)
        t = float(self.get_parameter("takeoff_duration_s").value)

        self.pending_futures = []
        for name in self.drones:
            req = Takeoff.Request()
            req.group_mask = 0
            req.height = float(h)
            req.duration = dur(t)
            self.pending_futures.append(self.takeoff_clients[name].call_async(req))

        self.get_logger().info(f"Takeoff all drones to {h:.2f} m over {t:.1f}s")

    def start_goto_segment(self):
        seg_t = float(self.get_parameter("segment_duration_s").value)
        yaw = float(self.get_parameter("yaw_deg").value)
        rel = bool(self.get_parameter("relative").value)

        wp = self.trajectory[self.traj_i]

        self.pending_futures = []
        for name in self.drones:
            req = GoTo.Request()
            req.group_mask = 0
            req.relative = rel
            req.goal = Point(x=float(wp.x), y=float(wp.y), z=float(wp.z))
            req.yaw = float(yaw)  # degrees in this interface
            req.duration = dur(seg_t)
            self.pending_futures.append(self.goto_clients[name].call_async(req))

        self.get_logger().info(
            f"Segment {self.traj_i+1}/{len(self.trajectory)}: go_to "
            f"({wp.x:.2f}, {wp.y:.2f}, {wp.z:.2f}) "
            f"{'[relative]' if rel else '[absolute]'} duration {seg_t:.1f}s"
        )

    def start_land(self):
        h = float(self.get_parameter("land_height").value)
        t = float(self.get_parameter("land_duration_s").value)

        self.pending_futures = []
        for name in self.drones:
            req = Land.Request()
            req.group_mask = 0
            req.height = float(h)
            req.duration = dur(t)
            self.pending_futures.append(self.land_clients[name].call_async(req))

        self.get_logger().info(f"Land all drones to {h:.2f} m over {t:.1f}s")

    def tick(self):
        if self.phase == "wait_services":
            if self.wait_all_services():
                self.phase = "takeoff"
                self.start_takeoff()

        elif self.phase == "takeoff":
            if self.all_done(self.pending_futures):
                self.phase = "traj"
                self.traj_i = 0
                self.start_goto_segment()

        elif self.phase == "traj":
            if self.all_done(self.pending_futures):
                self.traj_i += 1
                if self.traj_i >= len(self.trajectory):
                    self.phase = "land"
                    self.start_land()
                else:
                    self.start_goto_segment()

        elif self.phase == "land":
            if self.all_done(self.pending_futures):
                self.get_logger().info("Mission complete.")
                self.timer.cancel()
                rclpy.shutdown()


def main():
    rclpy.init()
    node = MultiMission()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
