#!/usr/bin/env python3

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Int8
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

try:
	from unitree_hg.msg import HandCmd, HandState, MotorCmd
except ImportError:            # sin unitree_hg el control de manos queda desactivado
	HandCmd = HandState = MotorCmd = None

# === Manos Dex3-1: valores de manos_agua.py (mano DERECHA) ===
HAND_MOTORS = 7
HAND_MAX_RIGHT = [1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75]
HAND_MIN_RIGHT = [-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0]

# Posturas de la mano IZQUIERDA (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1).
# La derecha se obtiene con hand_mirror_signs.
HAND_POSES_LEFT = {
	"open":   [0.0, 0.2, 0.4, -0.2, -0.6, -0.2, -0.6],    # L2: abierta
	"closed": [0.0, 0.8, 0.1, -1.0, -0.4, -1.0, -0.4],    # L2: cerrada
	"fist":   [0.0, 1.0, 1.5, -1.5, -1.6, -1.5, -1.6],    # R1: puño
}


class ArmsTeleop(Node):
	def __init__(self):
		super().__init__("both_arms_teleop")
		self.declare_parameter("arm_mode_value", 2)
		self.declare_parameter("deadman_button", 4)
		self.declare_parameter("axis_left_horizontal", 0)
		self.declare_parameter("axis_left_vertical", 1)
		self.declare_parameter("axis_right_vertical", 4)
		self.declare_parameter("axis_right_horizontal", 3)
		self.declare_parameter("axis_dpad_vertical", 7)
		self.declare_parameter("button_x", 0)
		self.declare_parameter("button_triangle", 2)
		self.declare_parameter("linear_speed", 0.12)
		self.declare_parameter("deadzone", 0.08)
		self.declare_parameter("control_rate", 50.0)
		self.declare_parameter("frame_id", "pelvis")
		self.declare_parameter("axis_dpad_horizontal", 6)
		self.declare_parameter("ctrl_arms_button", 7)     
		self.declare_parameter("button_square", 3)
		self.declare_parameter("button_circle", 1)
		self.declare_parameter("angular_speed", 0.6)
		self.declare_parameter("mirror_angular_signs", [-1.0, 1.0, -1.0])   # modo 4: signo de roll, pitch, yaw del brazo derecho
				# --- manos (modo 4) ---
		self.declare_parameter("enable_hands", True)
		self.declare_parameter("button_l2", 6)
		self.declare_parameter("button_r1", 5) 
		self.declare_parameter("hand_mirror_signs", [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])   # izquierda <-> derecha
		self.declare_parameter("hand_kp", 1.5)
		self.declare_parameter("hand_kd", 0.1)
		self.declare_parameter("hand_rate", 100.0)
		self.declare_parameter("hand_free_on_exit", True)
		self.declare_parameter("hand_max_speed", 0.5)     # rad/s: velocidad articular máxima en las transiciones
		self.declare_parameter("hand_min_time", 0.5)      # s: duración mínima de una transición
		self.declare_parameter("hand_cmd_topic_left", "/dex3/left/cmd")
		self.declare_parameter("hand_cmd_topic_right", "/dex3/right/cmd")
		self.declare_parameter("hand_state_topic_left", "/lf/dex3/left/state")
		self.declare_parameter("hand_state_topic_right", "/lf/dex3/right/state")
		self.declare_parameter("z_speed_scale", 0.5)      # factor de velocidad del eje z 


		self.arm_mode_value = self.get_parameter("arm_mode_value").value
		self.deadman_button = self.get_parameter("deadman_button").value
		self.axis_left_horizontal = self.get_parameter("axis_left_horizontal").value
		self.axis_left_vertical = self.get_parameter("axis_left_vertical").value
		self.axis_right_vertical = self.get_parameter("axis_right_vertical").value
		self.axis_right_horizontal = self.get_parameter("axis_right_horizontal").value
		self.axis_dpad_vertical = self.get_parameter("axis_dpad_vertical").value
		self.button_x = self.get_parameter("button_x").value
		self.button_triangle = self.get_parameter("button_triangle").value
		self.linear_speed = self.get_parameter("linear_speed").value
		self.deadzone = self.get_parameter("deadzone").value
		self.control_rate = self.get_parameter("control_rate").value
		self.frame_id = self.get_parameter("frame_id").value
		self.axis_dpad_horizontal = self.get_parameter("axis_dpad_horizontal").value
		self.ctrl_arms_button = self.get_parameter("ctrl_arms_button").value
		self.button_square = self.get_parameter("button_square").value
		self.button_circle = self.get_parameter("button_circle").value
		self.angular_speed = self.get_parameter("angular_speed").value
		self.z_speed_scale = self.get_parameter("z_speed_scale").value
		self.mirror_angular_signs = [float(v) for v in self.get_parameter("mirror_angular_signs").value]
		if len(self.mirror_angular_signs) != 3:
			raise ValueError("mirror_angular_signs debe tener 3 valores (roll, pitch, yaw)")

		self.mode = 0
		self.last_joy_time = 0.0
		self.left_input = [0.0, 0.0, 0.0]
		self.right_input = [0.0, 0.0, 0.0]
		self.ctrl_arms = 1            # 1 = ambos, 2 = solo izquierdo, 3 = solo derecho
		self.prev_ctrl_button = 0
		self.left_ang = [0.0, 0.0, 0.0]
		self.right_ang = [0.0, 0.0, 0.0]
		self.ctrl_arms_pub = self.create_publisher(Int8, "ctrl_arms", 10)
	
		self.create_subscription(Joy, "joy", self.joy_callback, 10)
		self.create_subscription(Int8, "mode", self.mode_callback, 10)
		self.create_timer(1.0 / self.control_rate, self.control_callback)
		self.velocity_pub = self.create_publisher(
    		MultiDOFJointTrajectory, "arm_cartesian_velocities", 10
		)

		self._init_hands()
		
		self.get_logger().info(
			"Publishing Cartesian arm velocities; hardware and IK are handled by hw_interface"
		)


		
	def _init_hands(self):
		self.hand_pubs = None
		self.hand_enabled = False        # no se publica nada hasta la primera pulsación de L2 o R1
		self.prev_l2 = 0
		self.prev_r1 = 0
		self.last_hand_tick = None
		self.hand_state_q = {"left": None, "right": None}
		self.hand_q = None               # consigna actual enviada: {"left": [7], "right": [7]}
		self.hand_start = None           # postura al empezar la transición actual
		self.hand_target = None          # postura destino de la transición actual
		self.hand_progress = 1.0         # 0..1
		self.hand_duration = 1.0         # s
		self.l2_closed = False           # último destino pedido con L2 (False = abierta, True = cerrada)
		self.r1_active = False           # True mientras las manos están en la postura de R1
		self.r1_saved = None             # postura anterior a R1, a la que se vuelve con la 2ª pulsación

		if not self.get_parameter("enable_hands").value:
			self.get_logger().info("Control de manos desactivado (enable_hands=false)")
			return
		if HandCmd is None:
			self.get_logger().warning("No se encontró unitree_hg: control de manos desactivado")
			return

		self.button_l2 = self.get_parameter("button_l2").value
		self.button_r1 = self.get_parameter("button_r1").value
		self.hand_kp = float(self.get_parameter("hand_kp").value)
		self.hand_kd = float(self.get_parameter("hand_kd").value)
		self.hand_max_speed = float(self.get_parameter("hand_max_speed").value)
		self.hand_min_time = float(self.get_parameter("hand_min_time").value)
		self.hand_free_on_exit = bool(self.get_parameter("hand_free_on_exit").value)
		hand_rate = float(self.get_parameter("hand_rate").value)
		signs = [float(v) for v in self.get_parameter("hand_mirror_signs").value]
		if len(signs) != HAND_MOTORS:
			raise ValueError("hand_mirror_signs debe tener 7 valores")
		if hand_rate <= 0.0 or self.hand_max_speed <= 0.0 or self.hand_min_time <= 0.0:
			raise ValueError("hand_rate, hand_max_speed y hand_min_time deben ser positivos")

		# Posturas: izquierda tal cual; derecha = izquierda con los signos de espejo.
		# Límites: los de la derecha (manos_agua.py); la izquierda, con los signos.
		self.hand_poses = {"left": {}, "right": {}}
		for name, q in HAND_POSES_LEFT.items():
			self.hand_poses["left"][name] = [float(v) for v in q]
			self.hand_poses["right"][name] = [sg * float(v) + 0.0 for sg, v in zip(signs, q)]
		self.hand_limits = {
			"right": (list(HAND_MIN_RIGHT), list(HAND_MAX_RIGHT)),
			"left": ([min(sg * a, sg * b) + 0.0 for sg, a, b in zip(signs, HAND_MIN_RIGHT, HAND_MAX_RIGHT)],
					 [max(sg * a, sg * b) + 0.0 for sg, a, b in zip(signs, HAND_MIN_RIGHT, HAND_MAX_RIGHT)]),
		}
		for side in ("left", "right"):
			lo, hi = self.hand_limits[side]
			for name, q in self.hand_poses[side].items():
				for i, (v, l, h) in enumerate(zip(q, lo, hi)):
					if v < l - 1e-9 or v > h + 1e-9:
						self.get_logger().warning(
							f"Postura '{name}' de la mano {side}: motor {i} = {v:+.3f} fuera de límites "
							f"[{l:+.3f}, {h:+.3f}]; se recortará")

		state_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
							history=HistoryPolicy.KEEP_LAST, depth=1)
		self.hand_pubs = {}
		for side in ("left", "right"):
			self.hand_pubs[side] = self.create_publisher(
				HandCmd, self.get_parameter(f"hand_cmd_topic_{side}").value, 10)
			self.create_subscription(
				HandState, self.get_parameter(f"hand_state_topic_{side}").value,
				lambda m, s=side: self._hand_state_callback(s, m), state_qos)
		self.create_timer(1.0 / hand_rate, self.hand_callback)
		self.get_logger().info("Manos listas: en modo 4, L1+L2 abre/cierra y L1+R1 puño/postura anterior (ambas manos)")

	def _hand_state_callback(self, side, msg):
		self.hand_state_q[side] = [float(m.q) for m in msg.motor_state[:HAND_MOTORS]]

	def _clamp_pose(self, side, q):
		lo, hi = self.hand_limits[side]
		return [min(max(v, l), h) for v, l, h in zip(q, lo, hi)]

	def _hand_start_pose(self):
		"""Postura de partida: la medida si hay telemetría; si no, la ABIERTA."""
		pose, missing = {}, False
		for side in ("left", "right"):
			q = self.hand_state_q[side]
			if q is None or len(q) < HAND_MOTORS:
				pose[side] = self._clamp_pose(side, self.hand_poses[side]["open"])
				missing = True
			else:
				pose[side] = self._clamp_pose(side, q)
		if missing:
			self.get_logger().warning("Sin telemetría de alguna mano: se asume la postura ABIERTA como punto de partida")
		return pose

	def _activate_hands(self):
		"""Primera pulsación de L2 o R1: empieza a publicar desde la postura actual de las manos."""
		self.hand_q = self._hand_start_pose()
		self.hand_start = {s: list(q) for s, q in self.hand_q.items()}
		self.hand_target = {s: list(q) for s, q in self.hand_q.items()}
		self.hand_progress = 1.0

		def dist(name):
			return sum((a - b) ** 2 for s in ("left", "right")
					for a, b in zip(self.hand_q[s], self.hand_poses[s][name]))
		self.l2_closed = dist("closed") < dist("open")       # L2 alterna respecto a la más cercana
		self.hand_enabled = True

	def _set_hand_target(self, target):
		"""Nueva transición lineal desde la consigna actual; dura lo que exija hand_max_speed."""
		target = {s: self._clamp_pose(s, target[s]) for s in ("left", "right")}
		self.hand_start = {s: list(self.hand_q[s]) for s in ("left", "right")}
		self.hand_target = target
		delta = max(abs(t - q) for s in ("left", "right")
					for t, q in zip(target[s], self.hand_q[s]))
		self.hand_duration = max(self.hand_min_time, delta / self.hand_max_speed)
		self.hand_progress = 0.0

	def _on_l2(self):
		"""L2: alterna ABIERTA <-> CERRADA. Anula el puño de R1 si estaba activo."""
		if self.hand_pubs is None:
			return
		if not self.hand_enabled:
			self._activate_hands()
		self.r1_active = False
		self.r1_saved = None
		self.l2_closed = not self.l2_closed
		name = "closed" if self.l2_closed else "open"
		self._set_hand_target({s: self.hand_poses[s][name] for s in ("left", "right")})
		self.get_logger().info("Manos (L2): " + ("CERRADA" if self.l2_closed else "ABIERTA"))

	def _on_r1(self):
		"""R1: alterna PUÑO <-> postura en la que estaban justo antes de pulsarlo."""
		if self.hand_pubs is None:
			return
		if not self.hand_enabled:
			self._activate_hands()
		if not self.r1_active:
			self.r1_saved = {s: list(self.hand_q[s]) for s in ("left", "right")}
			self.r1_active = True
			target = {s: self.hand_poses[s]["fist"] for s in ("left", "right")}
			self.get_logger().info("Manos (R1): PUÑO (se guarda la postura anterior)")
		else:
			target = self.r1_saved
			self.r1_active = False
			self.r1_saved = None
			self.get_logger().info("Manos (R1): vuelven a la postura anterior")
		self._set_hand_target(target)

	def _hand_cmd(self, q, mode=1, kp=None, kd=None):
		msg = HandCmd()
		msg.motor_cmd = [MotorCmd() for _ in range(HAND_MOTORS)]
		for i, cmd in enumerate(msg.motor_cmd):
			cmd.mode = mode
			cmd.q = float(q[i])
			cmd.dq = 0.0
			cmd.tau = 0.0
			cmd.kp = self.hand_kp if kp is None else kp
			cmd.kd = self.hand_kd if kd is None else kd
		return msg

	def hand_callback(self):
		if not self.hand_enabled:
			return
		now = time.monotonic()
		dt = 0.01 if self.last_hand_tick is None else min(max(now - self.last_hand_tick, 1e-3), 0.1)
		self.last_hand_tick = now
		self.hand_progress = min(1.0, self.hand_progress + dt / self.hand_duration)
		for side in ("left", "right"):
			a, b = self.hand_start[side], self.hand_target[side]
			self.hand_q[side] = [ai + (bi - ai) * self.hand_progress for ai, bi in zip(a, b)]
			self.hand_pubs[side].publish(self._hand_cmd(self.hand_q[side]))

	def release_hands(self):
		"""Al salir: dedos libres (mode 0, kp=kd=0), como en manos_agua.py. Solo si se llegaron a mandar."""
		if self.hand_pubs is None or not self.hand_enabled or not self.hand_free_on_exit:
			return
		try:
			for _ in range(3):
				for side in ("left", "right"):
					self.hand_pubs[side].publish(
						self._hand_cmd([0.0] * HAND_MOTORS, mode=0, kp=0.0, kd=0.0))
				time.sleep(0.03)
		except Exception as exc:
			self.get_logger().warning(f"No se pudo liberar las manos: {exc}")

	@staticmethod
	def _dpad(value):
		return 1.0 if value > 0.5 else (-1.0 if value < -0.5 else 0.0)

	@staticmethod
	def _read_axis(msg, index):
		return float(msg.axes[index]) if 0 <= index < len(msg.axes) else 0.0

	@staticmethod
	def _read_button(msg, index):
		return int(msg.buttons[index]) if 0 <= index < len(msg.buttons) else 0

	def _deadzone_axis(self, value):
		if abs(value) <= self.deadzone:
			return 0.0
		return float(np.sign(value) * (abs(value) - self.deadzone) / (1.0 - self.deadzone))

	def _zero_inputs(self):
		self.left_input = [0.0, 0.0, 0.0]
		self.right_input = [0.0, 0.0, 0.0]
		self.left_ang = [0.0, 0.0, 0.0]
		self.right_ang = [0.0, 0.0, 0.0]

	def joy_callback(self, msg):
		self.last_joy_time = time.monotonic()

		# Cambio de ctrl_arms con el touchpad: flanco de subida y solo en modo 2
		button = self._read_button(msg, self.ctrl_arms_button)
		if self.mode == self.arm_mode_value and button == 1 and self.prev_ctrl_button == 0:
			self.ctrl_arms = self.ctrl_arms % 4 + 1      # 1 -> 2 -> 3 ->4 -> 1
			self._zero_inputs()
			self.get_logger().info(f"ctrl_arms = {self.ctrl_arms}")
		self.prev_ctrl_button = button

		# Flanco de subida de L2 (se actualiza siempre, para no perder transiciones)
		l2 = self._read_button(msg, self.button_l2) if self.hand_pubs is not None else 0
		l2_edge = (l2 == 1 and self.prev_l2 == 0)
		self.prev_l2 = l2
		r1 = self._read_button(msg, self.button_r1) if self.hand_pubs is not None else 0
		r1_edge = (r1 == 1 and self.prev_r1 == 0)
		self.prev_r1 = r1

		# Seguridad: fuera de modo 2 o sin deadman (L1) -> todo a cero
		if self.mode != self.arm_mode_value or not self._read_button(msg, self.deadman_button):
			self._zero_inputs()
			return

				# Modo 4 + deadman (L1) + L2: alternar manos abierta <-> cerrada
		if self.ctrl_arms == 4:
			if l2_edge:
				self._on_l2()
			if r1_edge:
				self._on_r1()
		
		stick_left = [
			self._deadzone_axis(self._read_axis(msg, self.axis_left_vertical)),    # x
			self._deadzone_axis(self._read_axis(msg, self.axis_left_horizontal)),  # y
		]
		stick_right = [
			self._deadzone_axis(self._read_axis(msg, self.axis_right_vertical)),    # x
			self._deadzone_axis(self._read_axis(msg, self.axis_right_horizontal)),  # y
		]
		dpad_v = self._dpad(self._read_axis(msg, self.axis_dpad_vertical))      # arriba = +1
		dpad_h = self._dpad(self._read_axis(msg, self.axis_dpad_horizontal))    # izquierda = +1
		triangle = self._read_button(msg, self.button_triangle)
		cross = self._read_button(msg, self.button_x)
		square = self._read_button(msg, self.button_square)
		circle = self._read_button(msg, self.button_circle)

		self._zero_inputs()
		if self.ctrl_arms == 1:
			self.left_input = [stick_left[0], stick_left[1], dpad_v]
			self.right_input = [stick_right[0], stick_right[1], float(triangle - cross)]
			return

		wrist = [
			float(circle - square),    # pitch (cuadrado +, círculo -)
			float(cross - triangle),   # roll  (triángulo -, X +)
			dpad_h,                    # yaw   (cruceta izquierda +, derecha -)
		]
		if self.ctrl_arms == 2:
			self.left_input = [stick_left[0], stick_left[1], dpad_v]
			self.left_ang = wrist
		elif self.ctrl_arms == 3:
			self.right_input = [stick_right[0], stick_right[1], dpad_v]
			self.right_ang = wrist
		else:
			# Modo 4 (espejo): mandos del brazo izquierdo; el derecho es su imagen en el plano x-z
			self.left_input = [stick_left[0], stick_left[1], dpad_v]
			self.right_input = [stick_left[0], -stick_left[1], dpad_v]        # x y z iguales, y opuesta
			self.left_ang = wrist
			self.right_ang = [s * w for s, w in zip(self.mirror_angular_signs, wrist)]

	def mode_callback(self, msg):
		self.mode = msg.data
		if self.mode != self.arm_mode_value:
			self._zero_inputs()
			self.ctrl_arms = 1

	def _make_twist(self, linear, angular):
		twist = Twist()
		twist.linear.x = linear[0] * self.linear_speed
		twist.linear.y = linear[1] * self.linear_speed
		twist.linear.z = linear[2] * self.linear_speed* self.z_speed_scale
		twist.angular.x = angular[0] * self.angular_speed
		twist.angular.y = angular[1] * self.angular_speed
		twist.angular.z = angular[2] * self.angular_speed
		return twist

	def control_callback(self):
		active = (self.mode == self.arm_mode_value
				and time.monotonic() - self.last_joy_time <= 0.25)
		zero = [0.0, 0.0, 0.0]
		left_lin, left_ang = (self.left_input, self.left_ang) if active else (zero, zero)
		right_lin, right_ang = (self.right_input, self.right_ang) if active else (zero, zero)

		point = MultiDOFJointTrajectoryPoint()
		point.velocities = [self._make_twist(left_lin, left_ang),
							self._make_twist(right_lin, right_ang)]
		point.time_from_start = Duration(sec=0, nanosec=0)

		msg = MultiDOFJointTrajectory()
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.header.frame_id = self.frame_id
		msg.joint_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
		msg.points = [point]
		self.velocity_pub.publish(msg)
		self.ctrl_arms_pub.publish(Int8(data=self.ctrl_arms))


def main(args=None):
	rclpy.init(args=args)
	node = None
	try:
		node = ArmsTeleop()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		if node is not None:
			node.release_hands()
			node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == "__main__":
	main()

