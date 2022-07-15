#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
import os, rospkg
rp = rospkg.RosPack()

from lmpc.msg import DuckPose
from lmpc.srv import GetMap
from duckietown_msgs.msg import WheelsCmdStamped

import casadi as ca
import numpy as np
from scipy import spatial


lmpc_path = os.path.join(rp.get_path("lmpc"), "src", "LMPC.casadi")
mpc_path = os.path.join(rp.get_path("lmpc"), "src", "M.casadi")
LMPC = ca.Function.load(lmpc_path)
MPC = ca.Function.load(mpc_path)
delay = round(0.15/0.1)
u_delay0 = ca.DM(np.zeros((2, delay)))

def get_map_client():
    rospy.loginfo("[Controller]: Waiting for map server...")
    print("[Controller]: Waiting for map server...")
    rospy.wait_for_service('duckwalker/get_map')
    try:
        get_map = rospy.ServiceProxy('duckwalker/get_map', GetMap)
        resp1 = get_map()
        rospy.loginfo("Got map.")
        print("Got map.")
        return np.array(resp1.data).reshape(-1, 2)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

class TheController(DTROS):

    def __init__(self, node_name, track):
        # Duck name
        self.vehicle = os.environ['VEHICLE_NAME']
        # kdtree
        self.kdtree = spatial.KDTree(track.T)
        # initialize the DTROS parent class
        super(TheController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        self.pub = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=10)
        # Subscriber
        self.subscriber = rospy.Subscriber("/watchtower00/localization",
            DuckPose, self.callback,  queue_size=1)
        self.track = track

        # To estimate speed
        self.old_x = 0
        self.old_y = 0
        self.old_t = 0

    def callback(self, ros_data):
        x = ros_data.data.x
        y = ros_data.data.y
        t = ros_data.data.t
        success = ros_data.data.success

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

        x = [x, y, t, v, w]

        _,idx = self.kdtree.query(np.array([x, y]).reshape(-1), workers=-1)
        r = self.track[idx, :]

        print(x)
        rospy.loginfo("Received position: '%s'" % x)

        u = MPC(x, r, t, u_delay0,  1e3, 5e-4, 1, 1e-3)

        msg = WheelsCmdStamped()
        msg.vel_left = u[0]
        msg.vel_right = u[1]

        self.pub.publish(msg)

if __name__ == '__main__':
    map = get_map_client()
    node = TheController(track=map, node_name='controller')
    rospy.spin()