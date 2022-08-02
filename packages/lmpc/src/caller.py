#!/usr/bin/env python3

# ROS
import rospy

from std_msgs.msg import Bool

MPC_TIME = 0.1

if __name__ == '__main__':
    rospy.init_node("caller")
    rate = rospy.Rate(1/MPC_TIME)
    pub = rospy.Publisher("/execute_controller", Bool, queue_size=1)
    while not rospy.is_shutdown():
        pub.publish(Bool(data=True))
        rate.sleep()
    
    