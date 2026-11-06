cmake_minimum_required(VERSION 3.8)
project(g1_leo_ros_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_python_install_package(scripts)

install(PROGRAMS
  # scripts/head_stream.py
  # scripts/skills.py
  # scripts/tf_republish.py
  # scripts/teleop.py
  # scripts/wheel_odom_fix.py
  # scripts/skills_and_listen.py

  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch config
  DESTINATION
    share/${PROJECT_NAME}
)

ament_package()
