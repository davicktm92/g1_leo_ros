# -*- coding: utf-8 -*-
#
# Author: David Orbea <david.orbea@upm.es>
# Creation date: 2026-04-10
# Description: ROS2 launch file for teleoperation of the Mirokai robot
#
# Copyright (c) 2026 David Orbea, Center for Automation and Robotics.
# Todos los derechos reservados.

import os

from ament_index_python.packages import get_package_share_directory

import launch
import launch_ros.actions


def generate_launch_description():
    joy_config = launch.substitutions.LaunchConfiguration('joy_config')
    joy_dev = launch.substitutions.LaunchConfiguration('joy_dev')
    publish_stamped_twist = launch.substitutions.LaunchConfiguration('publish_stamped_twist')
    config_filepath = launch.substitutions.LaunchConfiguration('config_filepath')

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument('joy_topic', default_value='joy'),
        #launch.actions.DeclareLaunchArgument('joy_vel', default_value='cmd_vel'),
        launch.actions.DeclareLaunchArgument('joy_config', default_value='ps4'),
        launch.actions.DeclareLaunchArgument('joy_dev', default_value='0'),
        launch.actions.DeclareLaunchArgument('publish_stamped_twist', default_value='false'),
        launch.actions.DeclareLaunchArgument('config_filepath', default_value=[
            launch.substitutions.TextSubstitution(text=os.path.join(
                get_package_share_directory('g1_leo_ros_locomotion'), 'config', '')),
            joy_config, launch.substitutions.TextSubstitution(text='.config.yaml')]),

        launch_ros.actions.Node(
            package='joy', executable='joy_node', name='joy_node',
            parameters=[{
                'device_id': joy_dev,
                'deadzone': 0.05,
                'autorepeat_rate': 10.0,
            }],
            remappings=[('/joy', launch.substitutions.LaunchConfiguration('joy_topic'))]
            ),
        launch_ros.actions.Node(
            package='g1_leo_ros_locomotion', executable='loco_teleop.py',
            name='locomotion_teleop_node',
            parameters=[config_filepath, {'publish_stamped_twist': publish_stamped_twist}],
            #remappings=[('cmd_vel', launch.substitutions.LaunchConfiguration('joy_vel')),
            #            ('joy', launch.substitutions.LaunchConfiguration('joy_topic'))]
            ),
    ])
