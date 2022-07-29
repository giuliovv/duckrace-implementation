#!/usr/bin/env python3

import os

import casadi as ca
import numpy as np
from scipy import spatial

# ROS
import rospy, rospkg
rp = rospkg.RosPack()

# Msgs
from lmpc.msg import DuckPose
from lmpc.msg import Floats
from duckietown_msgs.msg import WheelsCmdStamped

# Duckie
from dt_communication_utils import DTCommunicationGroup
from duckietown.dtros import DTROS, NodeType


lmpc_path = os.path.join(rp.get_path("lmpc"), "src", "LMPC.casadi")
mpc_path = os.path.join(rp.get_path("lmpc"), "src", "M.casadi")
LMPC = ca.Function.load(lmpc_path)
MPC = ca.Function.load(mpc_path)
delay = round(0.15/0.1)
u_delay0 = ca.DM(np.zeros((2, delay)))

group = DTCommunicationGroup('position', DuckPose)
group_map = DTCommunicationGroup('map', Floats)

VERBOSE = True
ROS_SUB = False

MAX_SPEED = 0.1

class GetMap():

    def __init__(self):
        self.map = None
        group_map.Subscriber(self.map_callback)

    def map_callback(self, msg, header):
        self.map = msg.data
        print("[Controller]: Got map.")
        rospy.loginfo("[Controller]: Got map.")

    def wait_for_map(self):
        print("[Controller]: Getting map...")
        rospy.loginfo("[Controller]: Getting map...")
        while self.map is None:
            rospy.sleep(0.1)
        group_map.shutdown()
        print("[Controller]: Map saved.")
        rospy.loginfo("[Controller]: Map saved.")
        return np.array(self.map).reshape(-1,2)

class TheController(DTROS):

    def __init__(self, node_name, track):
        print("[Controller]: Initializing...")

        # Duck name
        self.vehicle = os.environ['VEHICLE_NAME']

         # To estimate speed
        self.old_x = 0
        self.old_y = 0
        self.old_t = 0

        self.track = track

        # kdtree
        self.kdtree = spatial.KDTree(track)

        # UDP publisher
        if not ROS_SUB:
            self.subscriber = group.Subscriber(self.callback)
        
        # initialize the DTROS parent class
        super(TheController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Publisher to control the wheels
        self.pub = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        
        # Subscriber
        if ROS_SUB:
            self.subscriber = rospy.Subscriber("/watchtower00/localization",
                DuckPose, self.callback,  queue_size=1)

    def callback(self, ros_data, header):
        x = ros_data.x
        y = ros_data.y
        t = ros_data.t
        success = ros_data.success

        if VERBOSE:
            print(f"[Controller]: Got data, x: {x}, y: {y}, t: {t}")
            rospy.loginfo(f"[Controller]: Got data, x: {x}, y: {y}, t: {t}")

        if not success:
            return

        if ros_data.x == -1 or ros_data.y == -1:
            return

        # To estimate speed
        v = np.sqrt((x - self.old_x)**2+(y - self.old_y)**2)/0.1
        w = (t - self.old_t)/0.1

        self.old_x = x
        self.old_y = y
        self.old_t = t

        X = [x, y, t, v, w]

        _,idx = self.kdtree.query([x, y])
        distance = 10
        r = self.track[idx+distance, :]
        if VERBOSE:
            print(f"[Controller]: r: {r}")

        u = MPC(X, r, t, u_delay0,  1e3, 0, 0, 0)

        msg = WheelsCmdStamped()
        msg.vel_left = u[0]*MAX_SPEED
        msg.vel_right = u[1]*MAX_SPEED

        try:
            self.pub.publish(msg)
        except AttributeError:
            print("[Controller]: Publisher not initialized.")


    def on_shutdown(self):
        print("[Controller]: Shutdown.")
        rospy.loginfo("[Controller]: Shutdown.")
        self.pub.publish(WheelsCmdStamped(vel_left=0, vel_right=0))
        self.subscriber.shutdown()
        self.pub.unregister()

if __name__ == '__main__':
    take_map = GetMap()
    try:
        map = take_map.wait_for_map()
    except KeyboardInterrupt:
        print("[Controller]: Keyboard interrupt.")
        rospy.loginfo("[Controller]: Keyboard interrupt.")
        exit(0)
    node = TheController(track=map, node_name='controller')
    rospy.spin()