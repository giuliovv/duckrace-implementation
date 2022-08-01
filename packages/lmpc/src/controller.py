#!/usr/bin/env python3

from operator import mod
import os

import casadi as ca
import numpy as np
from scipy import spatial

# ROS
import rospy, rospkg
rp = rospkg.RosPack()

# Msgs
from std_msgs.msg import Bool
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
N = 5
Ts = 1
delay = round(0.15/Ts)
u_delay0 = ca.DM(np.zeros((2, delay)))

group = DTCommunicationGroup('my_position', DuckPose)
group_map = DTCommunicationGroup('my_map', Floats)

VERBOSE = False
SUB_ROS = True

MAX_SPEED = 1
MPC_TIME = 1/0.8

# Default value, will be updated after map retrieval
N_POINTS_MAP = 500

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
    # System dynamics (CAN BE NONLINEAR! ;))
    # x_long_dot_dot = -u1*v0 + u_alpha_r*wr + u_alpha_l*wl
    # w_dot_dot = -w1*w0 + w_alpha_r*wr - w_alpha_l*wl
    v1 = (1-up*dt)*v0 + u_alpha_r*dt*wr + u_alpha_l*dt*wl
    w1 = (1-wp*dt)*w0 + w_alpha_r*dt*wr - w_alpha_l*dt*wl
    x1 = x0 + v0*dt*np.cos(th0 + w0*dt/2)
    y1 = y0 + v0*dt*np.sin(th0 + w0*dt/2)
    # Cannot use atan2 because x1 and y1 are approximated while th1 is not
    theta1 = th0 + w0*dt
    dae = ca.vertcat(x1, y1, theta1, v1, w1)
    F = ca.Function('F',[x,u],[dae],['x','u'],['dae'])
    return F

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

        self.F = model_F(dt=MPC_TIME)

        # Position
        self.x = 0
        self.y = 0
        self.t = 0
        self.v = 0
        self.w = 0

        # To estimate speed
        self.old_x = 0
        self.old_y = 0
        self.old_t = 0
        self.old_v = 0
        self.old_w = 0

        self.last_u = [0,0]

        self.starting_time = 0
        self.localization_time = 0

        self.track = track

        # kdtree
        self.kdtree = spatial.KDTree(track)

        if SUB_ROS:
            # If subscribe to topic published by duckiebot
            self.subscriber = rospy.Subscriber("~/localization", DuckPose, self.callback)
        else:
            # UDP subscriber, if we use data from watchtower
            self.subscriber = group.Subscriber(lambda ros_data, header : self.callback(ros_data))
        
        # initialize the DTROS parent class
        super(TheController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Publisher to control the wheels
        self.pub = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        
        # Save the message from the watchtower every time it arrives but run the MPC every MPC_TIME seconds
        rospy.Subscriber("/execute_controller", Bool, self.control, queue_size=1)


    def callback(self, ros_data):
        curr_time = rospy.get_time()
        if VERBOSE:
            print(f"[Controller]: Received message after {curr_time-self.starting_time}s the MPC, and after {curr_time-self.localization_time} the last message.")

        if ros_data.success:
            stima_x, stima_y, stima_t, v, w = self.F([self.x, self.y, self.t, self.v, self.w], self.last_u).toarray()
            print("stima: ", stima_x, stima_y, np.rad2deg(stima_t))
            self.x = ros_data.x
            self.y = ros_data.y
            self.t = ros_data.t
            self.v = (self.last_u[0] + self.last_u[1])/2
            self.w = (self.last_u[1] - self.last_u[0])/10
            print("pose: ", ros_data.x, ros_data.y, np.rad2deg(ros_data.t))
        else:
            print("[Controller]: Position failed, using odometry...")
            x, y, t, v, w = self.F([self.x, self.y, self.t, self.v, self.w], self.last_u).toarray()
            self.x, self.y, self.t, self.v, self.w = x[0], y[0], t[0], v[0], w[0]

        self.old_x = self.x
        self.old_y = self.y
        self.old_t = self.t
        self.localization_time = curr_time
        

    def control(self, ros_data):
        if self.x == 0:
            return
        current_time = rospy.get_time()
        delta_time = current_time - self.starting_time
        if VERBOSE:
            print(f"[Controller]: Delta time: {delta_time}")
        self.starting_time = current_time
        x = self.x
        y = self.y
        t = self.t
        v = self.v
        w = self.w

        if True:
            print(f"[Controller]: Use MPC, x: {x}, y: {y}, t: {np.rad2deg(t)}, v: {v}, w: {w}")
            # rospy.loginfo(f"[Controller]: Got data, x: {x}, y: {y}, t: {np.rad2deg(t)}")

        X = ca.DM([x, y, t, v, w])

        _,idx = self.kdtree.query([x, y])
        distance = 10
        # r = self.track[(idx+distance)%N_POINTS_MAP:(idx+distance+N+1)%N_POINTS_MAP, :].T
        r = np.array([[1.5,1.5]]*(N+1)).T
        # r = np.array([[x,y]]*(N+1)).T
        if VERBOSE:
            print(f"[Controller]: r: {r.T}")

        next_r = self.track[(idx+distance+1)%N_POINTS_MAP:(idx+distance+N+2)%N_POINTS_MAP, :]
        # tr = np.arctan2(r[:,1]-next_r[:,1], r[:,0]-next_r[:,0])
        tr = 0

        u = MPC(X, r, 0, u_delay0,  1, 0, 0, 0)
        print("u: ", u)

        self.last_u = u

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
    map = take_map.wait_for_map()
    print(map)
    N_POINTS_MAP = map.shape[0]
    node = TheController(track=map, node_name='controller')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[Controller]: Keyboard interrupt.")
        exit(0)
    
    