# from crazyflie_py import Crazyswarm
from enum import Enum, auto
from rclpy.node import Node
import rclpy
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from tf2_ros import TransformBroadcaster, TransformListener, Buffer, TransformException
import numpy as np
from geometry_msgs.msg import Pose, TransformStamped, PoseStamped
from std_srvs.srv import Empty


class PayloadState(Enum):
    """Payload attachment state."""

    DETACHED = auto()
    ATTACHED = auto()


class PayloadSim(Node):
    """Simulate payload position based on drone positions and cable length."""

    def __init__(self):
        super().__init__('payload_sim_node')

        self.get_logger().info('Starting payload simulation node...')

        self.declare_parameter('rate_hz', 200.0)
        self.rate_hz = self.get_parameter(
            'rate_hz').get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)

        self.declare_parameter('cable_length', 0.5)
        self.cable_length = self.get_parameter(
            'cable_length').get_parameter_value().double_value

        self.payload_pose = None
        self.payload_state = PayloadState.ATTACHED

        markerQoS = QoSProfile(
            depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.payload_pub_ = self.create_publisher(
            Marker, 'payload_marker', markerQoS)

        self.tf_names = ["cf1", "cf2", "cf3"]
        self.tfs = [None] * len(self.tf_names)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('Payload publisher node started.')

        # service to attach/detach payload
        self.attach_srv = self.create_service(
            Empty,
            'attach_payload',
            self.attach_payload_callback,
        )
        self.detach_srv = self.create_service(
            Empty,
            'detach_payload',
            self.detach_payload_callback,
        )

        # initial payload position
        self.payload_pose = Pose()
        self.payload_pose.position.x = 0.0
        self.payload_pose.position.y = 0.0
        self.payload_pose.position.z = 0.0
        self.payload_pose.orientation.w = 1.0

        self.time = 0.0

    def timer_callback(self):
        """Timer callback to update payload position."""
        self.time += 1.0 / self.rate_hz
        if self.time > 1.0:
            if self.payload_state == PayloadState.ATTACHED:
                self.calc_payload_position()
                self.publish_payload_marker()

    def calc_payload_position(self):
        """Calculate payload position via trilateration."""
        self.get_tfs()
        valid = [tf for tf in self.tfs if tf is not None]
        if len(valid) < 3:
            self.get_logger().warn("Need at least 3 valid drone transforms.")
            return

        # Drone positions in world frame
        P = []
        for tf in valid[:3]:
            tr = tf.transform.translation
            P.append(np.array([tr.x, tr.y, tr.z], dtype=float))
        P1, P2, P3 = P
        L = float(self.cable_length)

        # Trilateration (equal radii)
        ex = (P2 - P1)
        d = np.linalg.norm(ex)
        if d < 1e-6:
            self.get_logger().warn("P1 and P2 too close for trilateration.")
            return
        ex = ex / d

        i = np.dot(ex, (P3 - P1))
        temp = (P3 - P1) - i * ex
        temp_norm = np.linalg.norm(temp)
        if temp_norm < 1e-6:
            self.get_logger().warn("Points nearly collinear; trilateration unstable.")
            return
        ey = temp / temp_norm
        ez = np.cross(ex, ey)
        j = np.dot(ey, (P3 - P1))

        # equal radii => x = d/2
        x = d / 2.0
        # y from third sphere constraint
        y = (i*i + j*j) / (2.0*j) - (i/j) * x

        z2 = L*L - x*x - y*y
        if z2 < 0.0:
            self.get_logger().warn("No real intersection (z^2 < 0).")
            return
        z = np.sqrt(z2)

        sol1 = P1 + x*ex + y*ey + z*ez
        sol2 = P1 + x*ex + y*ey - z*ez

        # choose lower z (payload hangs below)
        payload = sol1 if sol1[2] < sol2[2] else sol2

        payload_pose = Pose()
        if payload[2] < 0.0:  # ground contact, no position update
            payload_pose.position.x = self.payload_pose.position.x
            payload_pose.position.y = self.payload_pose.position.y
            payload_pose.position.z = 0.0
        else:
            payload_pose.position.x = float(payload[0])
            payload_pose.position.y = float(payload[1])
            payload_pose.position.z = float(payload[2])
        payload_pose.orientation.w = 1.0

        self.payload_pose = payload_pose


    def get_tfs(self):
        """Get transforms for all drones."""
        for (i, tf) in enumerate(self.tf_names):
            try:
                self.tfs[i] = self.tf_buffer.lookup_transform(
                    'world', tf, rclpy.time.Time())
            except TransformException:
                self.get_logger().warn(
                    f'Could not find transform for {tf}')
                continue

    def publish_payload_marker(self):
        """Publish payload marker and transform."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'payload'
        t.transform.translation.x = self.payload_pose.position.x
        t.transform.translation.y = self.payload_pose.position.y
        t.transform.translation.z = self.payload_pose.position.z
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "payload"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.b = 1.0
        marker.pose = self.payload_pose
        self.payload_pub_.publish(marker)

    def attach_payload_callback(self, request, response):
        """Attach payload service callback."""
        if self.payload_state == PayloadState.ATTACHED:
            self.get_logger().info("Payload already attached.")
        else:
            self.payload_state = PayloadState.ATTACHED
            self.get_logger().info("Payload attached.")
        return response

    def detach_payload_callback(self, request, response):
        """Detach payload service callback."""
        if self.payload_state == PayloadState.DETACHED:
            self.get_logger().info("Payload already detached.")
        else:
            self.payload_state = PayloadState.DETACHED
            self.get_logger().info("Payload detached.")
        return response
    
    def get_payload_pose(self):
        """Get current payload pose."""
        return self.payload_pose


def main(args=None):
    """Start node."""
    rclpy.init(args=args)
    node = PayloadSim()
    rclpy.spin(node)
    rclpy.shutdown()

