#!/usr/bin/env python3

# -*- coding: utf-8 -*-
#
# Author: David Orbea <david.orbea@upm.es>
# Creation date: 2026-04-10
# Description: ROS2 node for teleoperation of the legs of Unitree G1 robot through ROS topic (using parameters)
#
# Copyright (c) 2026 David Orbea, Center for Automation and Robotics.
# Todos los derechos reservados.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
from sensor_msgs.msg import Joy
from rclpy.parameter import Parameter

rosRate=20

class locomotion_teleop(Node):
	def __init__(self):
		super().__init__("locomotion_teleop")
		
		#Parameters
		self.declare_parameters(
            namespace='',
            parameters=[
				('axis_linear.x',Parameter.Type.INTEGER),
				('axis_linear.y',Parameter.Type.INTEGER),
				('scale_linear.x',Parameter.Type.DOUBLE),
				('scale_linear.y',Parameter.Type.DOUBLE),
				('scale_linear_turbo.x',Parameter.Type.DOUBLE),
				('scale_linear_turbo.y',Parameter.Type.DOUBLE),
				('axis_angular.yaw',Parameter.Type.INTEGER),
				('scale_angular.yaw',Parameter.Type.DOUBLE),
				('scale_angular_turbo.yaw',Parameter.Type.DOUBLE),
				('enable_button',Parameter.Type.INTEGER),
				('enable_turbo_button',Parameter.Type.INTEGER),
				('mode_substract_button',Parameter.Type.INTEGER),
				('mode_add_button',Parameter.Type.INTEGER),
            ])

		#Subscriptors and publishers declarations
		self.create_subscription(Joy,"joy",self.callback_joy,1)
		self.m_pub=self.create_publisher(Twist,"cmd_vel",1)
		self.mode_pub=self.create_publisher(Int8,"mode",1)

		self.create_timer(1/rosRate,self.motor_pub)

		self.m_speed=Twist()

		self.m_pub.publish(self.m_speed)
		self.mode=0

		self.prev_mode_add=0
		self.prev_mode_substract=0

		self.joy_msg=Joy()

		#Parameters read
		self.axis_linear_x=self.get_parameter("axis_linear.x").get_parameter_value().integer_value
		self.axis_linear_y=self.get_parameter("axis_linear.y").get_parameter_value().integer_value
		self.scale_linear_x=self.get_parameter("scale_linear.x").get_parameter_value().double_value
		self.scale_linear_y=self.get_parameter("scale_linear.y").get_parameter_value().double_value
		self.scale_linear_turbo_x=self.get_parameter("scale_linear_turbo.x").get_parameter_value().double_value
		self.scale_linear_turbo_y=self.get_parameter("scale_linear_turbo.y").get_parameter_value().double_value
		self.axis_angular=self.get_parameter("axis_angular.yaw").get_parameter_value().integer_value
		self.scale_angular=self.get_parameter("scale_angular.yaw").get_parameter_value().double_value
		self.scale_angular_turbo=self.get_parameter("scale_angular_turbo.yaw").get_parameter_value().double_value

		self.enable_button=self.get_parameter("enable_button").get_parameter_value().integer_value
		self.enable_turbo_button=self.get_parameter("enable_turbo_button").get_parameter_value().integer_value
		self.mode_substract_button=self.get_parameter("mode_substract_button").get_parameter_value().integer_value
		self.mode_add_button=self.get_parameter("mode_add_button").get_parameter_value().integer_value

	def callback_joy(self,msg): #este es el que va a 500Hz
		self.joy_msg=msg

	
	def motor_pub(self, event=None): #este es el que va a 20Hz
		if not self.joy_msg.buttons or not self.joy_msg.axes:
			return
		
		self.enable=self.joy_msg.buttons[self.enable_button]
		self.enable_turbo=self.joy_msg.buttons[self.enable_turbo_button]

		#condicion para cambiar de modos

		mode_add=self.joy_msg.buttons[self.mode_add_button]
		mode_substract=self.joy_msg.buttons[self.mode_substract_button]

		#Deteccion de flancos
		if mode_substract==1 and self.prev_mode_substract==0:
			self.mode-=1
			#self.mode_pub.publish(Int8(data=self.mode))
		elif mode_add==1 and self.prev_mode_add==0:
			self.mode+=1
			#self.mode_pub.publish(Int8(data=self.mode))

		self.prev_mode_add = mode_add
		self.prev_mode_substract = mode_substract

		#calculo de velocidades
		if self.enable==1 and self.enable_turbo==0 and (self.mode==1 or self.mode==3):
			self.m_speed.linear.x=self.scale_linear_x*self.joy_msg.axes[self.axis_linear_x]
			self.m_speed.linear.y=self.scale_linear_y*self.joy_msg.axes[self.axis_linear_y]
			self.m_speed.angular.z=self.scale_angular*self.joy_msg.axes[self.axis_angular]
		elif self.enable==0 and self.enable_turbo==1 and (self.mode==1 or self.mode==3):
			self.m_speed.linear.x=self.scale_linear_turbo_x*self.joy_msg.axes[self.axis_linear_x]
			self.m_speed.linear.y=self.scale_linear_turbo_y*self.joy_msg.axes[self.axis_linear_y]
			self.m_speed.angular.z=self.scale_angular_turbo*self.joy_msg.axes[self.axis_angular]
		else:
			self.m_speed.linear.x=0.0
			self.m_speed.angular.z=0.0
		
		#condicion y publicacion de la velocidad
		if self.mode==1 or self.mode==3:
			self.m_pub.publish(self.m_speed)
			#self.mode=0
		
		self.mode_pub.publish(Int8(data=self.mode))

def main(args=None):
    rclpy.init()
    locomotion_teleop1=locomotion_teleop()
    try:
        rclpy.spin(locomotion_teleop1)
    except KeyboardInterrupt:
        print("Terminating Locomotion Teleop Node..")
        locomotion_teleop1.destroy_node()

if __name__=='__main__':
    main()