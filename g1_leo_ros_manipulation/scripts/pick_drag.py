#!/usr/bin/env python3
"""Presets cartesianos del Unitree G1: Joy + ctrl_arms -> waypoints.

Suscripciones:
  /joy               sensor_msgs/msg/Joy
  /ctrl_arms         std_msgs/msg/Int8   (both_arms_teleop.py)
  /mode              std_msgs/msg/Int8   (2 = control de brazos)
Publicacion:
  /arm_cartesian_waypoints  trajectory_msgs/msg/MultiDOFJointTrajectory

Condiciones: modo 2, ctrl_arms=4, L1 presionado (Joy.buttons[4]).
Cruceta vertical (Joy.axes[7]): ARRIBA -> arrastre, ABAJO -> agarre.
Se exige soltar la cruceta entre activaciones. No publica /ref_points/cmd
ni /ref_points/active. Preserva las poses y tiempos originales de ref_points.py.
No calcula IK, no accede al hardware y no modifica las ganancias PD.
ATENCION: un waypoint publicado puede mover el G1 si hw_interface esta activo.
"""

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Transform
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Int8
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

LEFT_LINK = "left_wrist_yaw_link"
RIGHT_LINK = "right_wrist_yaw_link"

# (tiempo desde inicio [s], pose izquierda, pose derecha).
# Pose = (x, y, z [m], qx, qy, qz, qw).
# Valores exactamente iguales a los del ref_points.py adjunto y a las pruebas CLI.
# La normalizacion de cuaterniones corresponde a hw_interface, como en CLI.
TRAJECTORIES = {
    "arrastre": [
        (1.0, (0.1, 0.2, 0.83, 0.0, 0.0, 0.0, 1.0),
              (0.1, -0.2, 0.83, 0.0, 0.0, 0.0, 1.0)),
        (2.0, (-0.054, 0.181, 0.842, -0.011, 0.134, 0.0, 0.991),
              (0.184, -0.199, 0.987, -0.016, 0.108, 0.012, 0.994)),
        (3.0, (-0.054, 0.181, 0.842, -0.011, 0.134, 0.0, 0.991),
              (0.162, -0.199, 0.947, -0.602, 0.083, 0.066, 0.791)),
        (4.0, (-0.054, 0.181, 0.842, -0.011, 0.134, 0.0, 0.991),
              (0.276, -0.065, 1.001, -0.602, 0.084, 0.066, 0.791)),
    ],
    "agarre": [
        (1.0, (-0.054, 0.181, 0.842, 0.0, 0.0, 0.0, 1.0),
              (0.22, -0.1, 0.93, 0.0, 0.0, 0.0, 1.0)),
        (2.0, (0.1, 0.2, 0.83, 0.0, 0.0, 0.0, 1.0),
              (0.1, -0.2, 0.83, 0.0, 0.0, 0.0, 1.0)),
        (3.0, (0.246, 0.3, 0.99, 0.0, 0.0, 0.0, 1.0),
              (0.246, -0.3, 0.99, 0.0, 0.0, 0.0, 1.0)),
        (4.0, (0.246, 0.3, 0.99, 0.085, 0.332, -0.081, 0.785),
              (0.246, -0.3, 0.99, -0.085, 0.332, 0.081, 0.785)),
        (5.0, (0.246, 0.3, 0.99, 0.085, 0.632, -0.081, 0.785),
              (0.246, -0.3, 0.99, -0.085, 0.632, 0.081, 0.785)),
    ],
}


def build_trajectory(name: str) -> MultiDOFJointTrajectory:
    msg = MultiDOFJointTrajectory()
    # Mantener stamp=0, igual que el mensaje CLI que ya se ha validado.
    msg.header.frame_id = "pelvis"
    msg.joint_names = [LEFT_LINK, RIGHT_LINK]

    for elapsed_s, left_pose, right_pose in TRAJECTORIES[name]:
        point = MultiDOFJointTrajectoryPoint()
        point.transforms = []
        for pose in (left_pose, right_pose):
            x, y, z, qx, qy, qz, qw = pose
            tf = Transform()
            tf.translation.x, tf.translation.y, tf.translation.z = x, y, z
            tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = (
                qx, qy, qz, qw
            )
            point.transforms.append(tf)
        seconds = int(elapsed_s)
        point.time_from_start = Duration(
            sec=seconds,
            nanosec=int(round((elapsed_s - seconds) * 1_000_000_000)),
        )
        msg.points.append(point)
    return msg


class JoyPresetTrajectories(Node):
    def __init__(self):
        super().__init__("joy_preset_trajectories")
        self.declare_parameter("joy_topic", "joy")
        self.declare_parameter("ctrl_arms_topic", "ctrl_arms")
        self.declare_parameter("mode_topic", "mode")
        self.declare_parameter("waypoints_topic", "arm_cartesian_waypoints")
        self.declare_parameter("dpad_vertical_axis", 7)
        self.declare_parameter("deadman_button", 4)   # L1
        self.declare_parameter("mode_timeout", 1.0)
        self.declare_parameter("ctrl_timeout", 1.0)
        self.declare_parameter("quiet_stick_deadzone", 0.08)
        self.declare_parameter("min_holdoff_after_trajectory", 0.5)

        p = lambda key: self.get_parameter(key).value
        self.dpad_index = int(p("dpad_vertical_axis"))
        self.deadman_index = int(p("deadman_button"))
        self.mode_timeout = float(p("mode_timeout"))
        self.ctrl_timeout = float(p("ctrl_timeout"))
        self.deadzone = float(p("quiet_stick_deadzone"))
        self.holdoff = float(p("min_holdoff_after_trajectory"))
        if not (0.0 <= self.deadzone < 1.0):
            raise ValueError("quiet_stick_deadzone debe estar en [0, 1)")
        if min(self.mode_timeout, self.ctrl_timeout, self.holdoff) < 0:
            raise ValueError("Timeouts y holdoff deben ser >= 0")

        self.mode = None
        self.ctrl_arms = None
        self.mode_seen = None
        self.ctrl_seen = None
        self.dpad_released = False  # requiere una lectura en neutro antes de armar
        self.busy_until = 0.0

        self.pub = self.create_publisher(
            MultiDOFJointTrajectory, str(p("waypoints_topic")), 10
        )
        self.create_subscription(Joy, str(p("joy_topic")), self.on_joy, 10)
        self.create_subscription(Int8, str(p("ctrl_arms_topic")), self.on_ctrl, 10)
        self.create_subscription(Int8, str(p("mode_topic")), self.on_mode, 10)
        self.get_logger().info(
            "Listo: /joy + /ctrl_arms=4 + /mode=2 + L1; "
            "cruceta ARRIBA=arrastre, ABAJO=agarre; "
            "publica solo en /arm_cartesian_waypoints"
        )

    def on_mode(self, msg):
        self.mode = int(msg.data)
        self.mode_seen = time.monotonic()

    def on_ctrl(self, msg):
        self.ctrl_arms = int(msg.data)
        self.ctrl_seen = time.monotonic()

    @staticmethod
    def _axis(msg, idx):
        return float(msg.axes[idx]) if 0 <= idx < len(msg.axes) else 0.0

    @staticmethod
    def _button(msg, idx):
        return bool(msg.buttons[idx]) if 0 <= idx < len(msg.buttons) else False

    def _valid_mode_and_ctrl(self, now):
        return (
            self.mode == 2
            and self.ctrl_arms == 4
            and self.mode_seen is not None
            and self.ctrl_seen is not None
            and now - self.mode_seen <= self.mode_timeout
            and now - self.ctrl_seen <= self.ctrl_timeout
        )

    def _controls_neutral(self, msg):
        # Evita lanzar presets al tiempo que se teleopera con sticks o botones.
        # indices usados por both_arms_teleop para los sticks y las rotaciones.
        sticks = (0, 1, 3, 4)
        if any(abs(self._axis(msg, i)) > self.deadzone for i in sticks):
            return False
        if abs(self._axis(msg, 6)) > 0.5:  # cruceta horizontal = giro yaw
            return False
        if any(self._button(msg, i) for i in (0, 1, 2, 3)):
            return False  # botones X, O, triangulo, cuadrado
        return True

    def on_joy(self, msg: Joy):
        now = time.monotonic()
        v = self._axis(msg, self.dpad_index)
        direction = 1 if v > 0.5 else (-1 if v < -0.5 else 0)

        if direction == 0:
            self.dpad_released = True
            return
        if not self.dpad_released:
            return  # pulsacion sostenida: nunca se repite
        self.dpad_released = False  # consume flanco, aunque falle el permiso

        if not self._valid_mode_and_ctrl(now):
            self.get_logger().warning(
                "Ignorada cruceta: requiere /mode=2 y /ctrl_arms=4 recientes"
            )
            return
        if not self._button(msg, self.deadman_index):
            self.get_logger().warning("Ignorada cruceta: mantener L1 pulsado")
            return
        if not self._controls_neutral(msg):
            self.get_logger().warning(
                "Ignorada cruceta: sticks, cruceta horizontal o botones activos"
            )
            return
        if now < self.busy_until:
            self.get_logger().warning(
                f"Ignorada nueva orden: intervalo de proteccion "
                f"({self.busy_until - now:.1f} s restantes)"
            )
            return
        if self.pub.get_subscription_count() < 1:
            self.get_logger().warning(
                "No enviada: no hay suscriptores en /arm_cartesian_waypoints"
            )
            return

        name = "arrastre" if direction > 0 else "agarre"
        msg_out = build_trajectory(name)
        self.pub.publish(msg_out)
        # Bloquea comandos repetidos durante la duracion nominal de la trayectoria.
        self.busy_until = now + TRAJECTORIES[name][-1][0] + self.holdoff
        self.get_logger().info(
            f"PUBLICADA {name}: {len(msg_out.points)} waypoints, "
            f"{TRAJECTORIES[name][-1][0]:g}s en /arm_cartesian_waypoints"
        )


def main(args=None):
    rclpy.init(args=args)
    node = JoyPresetTrajectories()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
