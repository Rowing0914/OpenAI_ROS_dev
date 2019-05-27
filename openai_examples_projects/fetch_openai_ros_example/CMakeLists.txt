cmake_minimum_required(VERSION 2.8.3)
project(fetch_openai_ros_example)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  openai_ros
  rospy
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES fetch_openai_ros_example
#  CATKIN_DEPENDS openai_ros rospy
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)