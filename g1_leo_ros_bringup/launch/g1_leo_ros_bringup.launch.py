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
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    namespace = "g1"

    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("g1_leo_ros_hardware"),
                "launch",
                "hardware.launch.py",
            ])
        ),
    )

    locomotion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("g1_leo_ros_locomotion"),
                "launch",
                "loco_teleop.launch.py",
            ])
        ),
    )

    arms_teleop = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("g1_leo_ros_teleop"),
                "launch",
                "arms_teleop.launch.py",
            ])
        ),
    )

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("g1_leo_ros_perception"),
                "launch",
                "vision.launch.py",
            ])
        ),
    )

    return LaunchDescription([
        GroupAction([
            PushRosNamespace(namespace),
            hardware,
            locomotion,
            arms_teleop,
            perception,
        ]),
    ])