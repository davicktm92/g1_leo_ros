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

    #Nodo de hardware interface para g1_leo
    manipulation_node = Node(
        package="g1_leo_ros_manipulation",
        executable="pick_drag.py",
        name="g1_leo_manipulation",

        output="screen",
        # Permite utilizar el Python del entorno virtual g1_leo
        #prefix="/home/pedro/venvs/g1_leo/bin/python3", #se añade para la compatibilidad con entornos virtuales de python, linkeado a donde esta t-fake

        # #parameters=[{
        #     "enable_hardware": True,
        # }],
    )

    return LaunchDescription([
        manipulation_node,
    ])
