cmake_minimum_required(VERSION 2.8.3)
project(lmpc)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES
  DuckPose.msg
)

add_service_files(
  FILES
  GetMap.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES localization
   CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)