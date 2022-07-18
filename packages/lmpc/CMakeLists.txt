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
  Floats.msg
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

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/localize.py
  src/get_map.py
  src/blind_get_map.py
  src/blind_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )