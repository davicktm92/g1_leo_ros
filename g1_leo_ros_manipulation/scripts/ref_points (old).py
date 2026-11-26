#!/usr/bin/env python3
"""
Ejecutor de las dos trayectorias cartesianas predefinidas del Unitree G1.

Comandos admitidos en /ref_points/cmd (std_msgs/msg/String):
    arrastre
    agarre

Cada comando publica UNA MultiDOFJointTrajectory con todos sus waypoints.
Se conserva la comprobación de que el robot esté en modo 2 y de que
hw_interface esté suscrito al topic de trayectorias.
"""

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Transform
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int8, String
from trajectory_msgs.msg import (
    MultiDOFJointTrajectory,
    MultiDOFJointTrajectoryPoint,
)

LEFT_LINK = "left_wrist_yaw_link"
RIGHT_LINK = "right_wrist_yaw_link"

# Cada waypoint: (tiempo_desde_inicio, pose_izquierda, pose_derecha)
# Cada pose: (x, y, z, qx, qy, qz, qw)
TRAJECTORIES = {
    "arrastre": [
        (
            1.0,
            (0.1, 0.2, 0.83, 0.0, 0.0, 0.0, 1.0),
            (0.1, -0.2, 0.83, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            2.0,
            (-0.054, 0.181, 0.842, -0.011, 0.134, 0.0, 0.991),
            (0.184, -0.199, 0.987, -0.016, 0.108, 0.012, 0.994),
        ),
        (
            3.0,
            (-0.054, 0.181, 0.842, -0.011, 0.134, 0.0, 0.991),
            (0.162, -0.199, 0.947, -0.602, 0.083, 0.066, 0.791),
        ),
        (
            4.0,
            (-0.054, 0.181, 0.842, -0.011, 0.134, 0.0, 0.991),
            (0.276, -0.065, 1.001, -0.602, 0.084, 0.066, 0.791),
        ),
    ],
    "agarre": [
        (
            1.0,
            (-0.054, 0.181, 0.842, 0.0, 0.0, 0.0, 1.0),
            (0.22, -0.1, 0.93, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            2.0,
            (0.1, 0.2, 0.83, 0.0, 0.0, 0.0, 1.0),
            (0.1, -0.2, 0.83, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            3.0,
            (0.246, 0.3, 0.99, 0.0, 0.0, 0.0, 1.0),
            (0.246, -0.3, 0.99, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            4.0,
            (0.246, 0.3, 0.99, 0.085, 0.332, -0.081, 0.785),
            (0.246, -0.3, 0.99, -0.085, 0.332, 0.081, 0.785),
        ),
        (
            5.0,
            (0.246, 0.3, 0.99, 0.085, 0.632, -0.081, 0.785),
            (0.246, -0.3, 0.99, -0.085, 0.632, 0.081, 0.785),
        ),
    ],
}


class RefTrajectories(Node):
    def __init__(self):
        super().__init__("ref_trajectories")

        self.declare_parameter("waypoints_topic", "arm_cartesian_waypoints")
        self.declare_parameter("mode_topic", "mode")
        self.declare_parameter("command_topic", "/ref_points/cmd")
        self.declare_parameter("frame_id", "pelvis")
        self.declare_parameter("arm_mode_value", 2)
        self.declare_parameter("mode_timeout", 1.0)
        self.declare_parameter("command_cooldown", 0.5)
        self.declare_parameter("max_abs_coordinate", 1.5)

        self.waypoints_topic = str(self.get_parameter("waypoints_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.arm_mode_value = int(self.get_parameter("arm_mode_value").value)
        self.mode_timeout = float(self.get_parameter("mode_timeout").value)
        self.command_cooldown = float(
            self.get_parameter("command_cooldown").value
        )
        self.max_abs_coordinate = float(
            self.get_parameter("max_abs_coordinate").value
        )

        self.mode = None
        self.mode_time = None
        self.last_send = None

        self.pub = self.create_publisher(
            MultiDOFJointTrajectory, self.waypoints_topic, 10
        )
        self.create_subscription(
            Int8,
            str(self.get_parameter("mode_topic").value),
            self.mode_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("command_topic").value),
            self.command_callback,
            10,
        )

        self.get_logger().info(
            f"Nodo listo. Trayectorias disponibles: arrastre, agarre. "
            f"Entrada: {self.get_parameter('command_topic').value}; "
            f"salida: {self.waypoints_topic}; frame: {self.frame_id}."
        )
        active_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.active_pub = self.create_publisher(String, "/ref_points/active", active_qos)
            

    def mode_callback(self, msg: Int8):
        self.mode = int(msg.data)
        self.mode_time = time.monotonic()

    def command_callback(self, msg: String):
        self.handle_command(msg.data)

    @staticmethod
    def duration_msg(seconds: float) -> Duration:
        sec = int(seconds)
        nanosec = int(round((seconds - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return Duration(sec=sec, nanosec=nanosec)

    @staticmethod
    def make_transform(pose):
        x, y, z, qx, qy, qz, qw = pose
        transform = Transform()
        transform.translation.x = x
        transform.translation.y = y
        transform.translation.z = z
        transform.rotation.x = qx
        transform.rotation.y = qy
        transform.rotation.z = qz
        transform.rotation.w = qw
        return transform

    def mode_is_valid(self):
        if self.mode is None or self.mode_time is None:
            return False, "todavía no se ha recibido el topic de modo"
        if time.monotonic() - self.mode_time > self.mode_timeout:
            return False, (
                f"el topic de modo lleva más de {self.mode_timeout:.1f} s "
                "sin actualizarse"
            )
        if self.mode != self.arm_mode_value:
            return False, (
                f"el modo actual es {self.mode}; se requiere "
                f"{self.arm_mode_value}"
            )
        return True, ""

    def build_trajectory(self, name):
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.joint_names = [LEFT_LINK, RIGHT_LINK]

        previous_time = 0.0
        for waypoint_time, left_pose, right_pose in TRAJECTORIES[name]:
            if waypoint_time <= previous_time:
                raise ValueError("Los tiempos de los waypoints deben aumentar.")
            previous_time = waypoint_time

            for arm, pose in (("left", left_pose), ("right", right_pose)):
                if len(pose) != 7 or not all(math.isfinite(v) for v in pose):
                    raise ValueError(f"Pose no válida para el brazo {arm}.")
                if any(abs(v) > self.max_abs_coordinate for v in pose[:3]):
                    raise ValueError(
                        f"Coordenada fuera de ±{self.max_abs_coordinate} m "
                        f"en {arm}: {pose[:3]}"
                    )
                quat_norm = math.sqrt(sum(v * v for v in pose[3:7]))
                if quat_norm < 1e-6:
                    raise ValueError(f"Cuaternión nulo en {arm}.")

            point = MultiDOFJointTrajectoryPoint()
            point.transforms = [
                self.make_transform(left_pose),
                self.make_transform(right_pose),
            ]
            point.time_from_start = self.duration_msg(waypoint_time)
            msg.points.append(point)

        return msg

    def handle_command(self, command):
        name = command.strip().lower()
        if name not in TRAJECTORIES:
            self.get_logger().error(
                f"Comando '{name}' no reconocido. Usa: arrastre o agarre."
            )
            return

        valid, reason = self.mode_is_valid()
        if not valid:
            self.get_logger().warning(
                f"Trayectoria '{name}' NO enviada: {reason}."
            )
            return

        if self.pub.get_subscription_count() == 0:
            self.get_logger().warning(
                f"Trayectoria '{name}' NO enviada: nadie está suscrito a "
                f"{self.waypoints_topic}. Comprueba el topic de hw_interface."
            )
            return

        now = time.monotonic()
        if (
            self.last_send is not None
            and now - self.last_send < self.command_cooldown
        ):
            self.get_logger().warning(
                "Comando ignorado: espera al menos "
                f"{self.command_cooldown:.1f} s entre envíos."
            )
            return

        try:
            msg = self.build_trajectory(name)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return

        self.pub.publish(msg)
        self.active_pub.publish(String(data=name))
        self.last_send = now
        final_time = TRAJECTORIES[name][-1][0]
        self.get_logger().info(
            f"Trayectoria '{name}' enviada: {len(msg.points)} puntos, "
            f"duración total {final_time:.1f} s."
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RefTrajectories()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
