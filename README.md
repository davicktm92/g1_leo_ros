# G1 LEO ROS

ROS 2 workspace for the Unitree G1 robot.

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3

## Installation

Clone the repository into a ROS 2 workspace and build:

```bash
cd ~/ws_leo
colcon build --symlink-install
source install/local_setup.bash
```

## Usage

Launch the system:

```bash
ros2 launch g1_leo_ros_bringup bringup.launch.py
```

## License

All rights reserved.
