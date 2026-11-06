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

		self.get_logger().info(
			"Publishing Cartesian arm velocities; hardware and IK are handled by hw_interface"
		)


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
			self.ctrl_arms = self.ctrl_arms % 3 + 1      # 1 -> 2 -> 3 -> 1
			self._zero_inputs()
			self.get_logger().info(f"ctrl_arms = {self.ctrl_arms}")
		self.prev_ctrl_button = button

		# Seguridad: fuera de modo 2 o sin deadman (L1) -> todo a cero
		if self.mode != self.arm_mode_value or not self._read_button(msg, self.deadman_button):
			self._zero_inputs()
			return

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
			float(triangle - cross),   # roll  (triángulo +, X -)
			float(square - circle),    # pitch (cuadrado +, círculo -)
			dpad_h,                    # yaw   (cruceta izquierda +, derecha -)
		]
		if self.ctrl_arms == 2:
			self.left_input = [stick_left[0], stick_left[1], dpad_v]
			self.left_ang = wrist
		else:
			self.right_input = [stick_right[0], stick_right[1], dpad_v]
			self.right_ang = wrist

	def mode_callback(self, msg):
		self.mode = msg.data
		if self.mode != self.arm_mode_value:
			self._zero_inputs()
			self.ctrl_arms = 1

	def _make_twist(self, linear, angular):
		twist = Twist()
		twist.linear.x = linear[0] * self.linear_speed
		twist.linear.y = linear[1] * self.linear_speed
		twist.linear.z = linear[2] * self.linear_speed
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
			node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == "__main__":
	main()

