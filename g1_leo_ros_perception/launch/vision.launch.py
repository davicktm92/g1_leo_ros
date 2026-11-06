# -*- coding: utf-8 -*-
#
# Author: David Orbea <david.orbea@upm.es>
# Creation date: 2026-04-10
# Description: ROS2 launch file for the Unitree G1 robot
#
# Copyright (c) 2026 David Orbea, Center for Automation and Robotics.
# Todos los derechos reservados.
#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    #Nodo de vision para g1_leo
    vision_node = Node(
        package="g1_leo_ros_perception",
        executable="vision_grasping.py",
        name="g1_leo_vision",

        output="screen",
    )

    return LaunchDescription([
        vision_node,
    ])