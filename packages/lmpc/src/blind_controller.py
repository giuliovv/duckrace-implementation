#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
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

def model_F(dt=0.033):
    """
    Return the model casadi function.
    :param dt: the time step
    """
    up = 5
    wp = 4
    # parameters for forced dynamics
    u_alpha_r = 1.5
    u_alpha_l = 1.5
    w_alpha_r = 15  # modify this for trim
    w_alpha_l = 15
    # States
    x0 = ca.MX.sym('x')
    y0 = ca.MX.sym('y')
    th0 = ca.MX.sym('th')
    w0 = ca.MX.sym('w')
    v0 = ca.MX.sym('v')
    x = ca.vertcat(x0, y0, th0, v0, w0) # Always vertically concatenate the states --> [n_x,1]
    # Inputs
    wl = ca.MX.sym('wl')
    wr = ca.MX.sym('wr')
    u = ca.vertcat(wl, wr) # Always vertically concatenate the inputs --> [n_u,1]
    v1 = (1-up*dt)*v0 + u_alpha_r*dt*wr + u_alpha_l*dt*wl
    w1 = (1-wp*dt)*w0 + w_alpha_r*dt*wr - w_alpha_l*dt*wl
    x1 = x0 + v0*dt*np.cos(th0 + w0*dt/2)
    y1 = y0 + v0*dt*np.sin(th0 + w0*dt/2)
    # Cannot use atan2 because x1 and y1 are approximated while th1 is not
    theta1 = th0 + w0*dt
    dae = ca.vertcat(x1, y1, theta1, v1, w1)
    F = ca.Function('F',[x,u],[dae],['x','u'],['dae'])
    return F

def get_map_client():
    rospy.loginfo("Waiting for map server...")
    print("Waiting for map server...")
    rospy.wait_for_service('get_map')
    try:
        get_map = rospy.ServiceProxy('get_map', GetMap)
        resp1 = get_map()
        rospy.loginfo("Got map.")
        print("Got map.")
        return np.array(resp1.data).reshape(-1, 2)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

class TheController(DTROS):

    def __init__(self, node_name, track, x0=0, y0=0, t0=0, v0=0, w0=0):
        # Time
        frame_rate = 10
        self.rate = rospy.Rate(frame_rate)
        # Duck name
        self.vehicle = os.environ['VEHICLE_NAME']
        # kdtree
        self.kdtree = spatial.KDTree(track.T)
        # initialize the DTROS parent class
        super(TheController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        self.pub = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=10)
        self.track = track
        self.F = model_F(dt=frame_rate)
        self.x, self.y, self.t, self.v, self.w = x0, y0, t0, v0, w0

        self.duck_control()

        self.rate.sleep()

    def duck_control(self):
        x = self.x
        y = self.y
        t = self.t
        v = self.v
        w = self.w

        _,idx = self.kdtree.query(np.array([x, y]).reshape(-1), workers=-1)
        r = self.track[idx, :]

        print(x)
        rospy.loginfo("Received position: '%s'" % x)

        u = MPC(x, r, t, u_delay0,  1e3, 5e-4, 1, 1e-3)

        msg = WheelsCmdStamped()
        msg.vel_left = u[0]
        msg.vel_right = u[1]

        self.x, self.y, self.t, self.v, self.w = self.F([x,y,t,v,w], [u[0], u[1]])

        self.pub.publish(msg)

if __name__ == '__main__':
    map = get_map_client()
    node = TheController(track=map, node_name='controller')
    rospy.spin()