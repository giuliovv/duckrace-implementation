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


group = DTCommunicationGroup('my_position', DuckPose)

MAX_SPEED = 0.5
MPC_TIME = 0.1

def old_model_F(dt=0.033):
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


def model_F(dt=0.033):
    """
    Return the model casadi function tuned according to the parameter found in the thesis.

    :param dt: the time step
    """
    # u1 = 7.662 # 5
    # u2 = 0.325
    # u3 = -0.050
    # w1 = 6.826 # 4
    # w2 = -4.929
    # w3 = -6.515
    u1 = 5
    u2 = 0
    u3 = 0
    w1 = 4
    w2 = 0
    w3 = 0
    # parameters for forced dynamics
    u_alpha_r = 1.5
    u_alpha_l = 1.5
    w_alpha_r = 15  # modify this for trim
    w_alpha_l = 15
   
    # u_alpha_r = 2.755
    # u_alpha_l = 2.741
    # w_alpha_r = 10.8 #14.663  # modify this for trim
    # w_alpha_l = 14.662
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
    
    # V =  [[wl], [wr]]
    V = ca.vertcat(wr, wl)

    f_dynamic = ca.vertcat(-u1 * v0 - u2 * w0 + u3 * w0 ** 2, -w1 * w0 - w2 * v0 - w3 * v0 * w0)
    # input Matrix
    B = ca.DM([[u_alpha_r, u_alpha_l], [w_alpha_r, -w_alpha_l]])
    # forced response
    f_forced = B@V
    # acceleration
    X_dot_dot = f_dynamic + f_forced
    
    v1 = v0 + X_dot_dot[0] * dt
    w1 = w0 + X_dot_dot[1] * dt
    x1 = x0 + v0*dt*np.cos(th0 + w0*dt/2)
    y1 = y0 + v0*dt*np.sin(th0 + w0*dt/2)
    # Cannot use atan2 because x1 and y1 are approximated while th1 is not
    theta1 = th0 + w0*dt
    dae = ca.vertcat(x1, y1, theta1, v1, w1)
    F = ca.Function('F',[x,u],[dae],['x','u'],['dae'])
    return F

class TheController(DTROS):

    def __init__(self, node_name):
        print("[Tester]: Initializing...")

        # Duck name
        self.vehicle = os.environ['VEHICLE_NAME']
        # initialize the DTROS parent class
        super(TheController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.F = model_F(dt=MPC_TIME)

        # Position
        self.x = None
        self.y = None
        self.t = None
        self.v = None
        self.w = None

        self.old_x = 0
        self.old_y = 0

        self.model_x = None
        self.model_y = None
        self.model_t = None
        self.model_v = None
        self.model_w = None

        self.last_u = [0,0]

        self.total_elapsed_time = 0
        self.call_N = 0
        self.control_N = 0

        self.starting_time = 0
        self.localization_time = 0

        self.positions = []
        self.model_positions = []
        self.inputs = []

        # UDP subscriber, if we use data from watchtower
        self.subscriber = group.Subscriber(lambda ros_data, header : self.callback(ros_data))
        

        # Publisher to control the wheels
        self.pub = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        
        # Save the message from the watchtower every time it arrives but run the MPC every MPC_TIME seconds
        rospy.Subscriber("/execute_controller", Bool, self.control, queue_size=1)


    def callback(self, ros_data):
        curr_time = rospy.get_time()
        elapsed_time = curr_time-self.localization_time
        print(f"[Controller]: Received message after {elapsed_time} the last message.")

        self.localization_time = curr_time
        self.total_elapsed_time += elapsed_time
        self.call_N += 1

        print("Average elapsed time: ",self.total_elapsed_time/self.call_N)

        self.x = ros_data.x
        self.y = ros_data.y
        self.t = ros_data.t
        self.v = np.sqrt((ros_data.x-self.old_x)**2+(ros_data.y-self.old_y)**2)/elapsed_time #(0.1*MAX_SPEED + 1*MAX_SPEED)/2
        self.w = (self.last_u[0]*MAX_SPEED - self.last_u[1]*MAX_SPEED)/0.1
        print("pose: ", ros_data.x, ros_data.y, np.rad2deg(ros_data.t))

        self.old_x = self.x
        self.old_y = self.y
        self.localization_time = curr_time
        

    def control(self, _):

        if not self.x:
            return
        if not self.model_x:
            self.model_x = self.x
            self.model_y = self.y
            self.model_t = self.t
            self.model_v = 0
            self.model_w = 0

        self.control_N += 1

        u = [0.6*MAX_SPEED, 1.*MAX_SPEED]

        model_x, model_y, model_t, model_v, model_w = self.F([self.model_x, self.model_y, self.model_t, self.model_v, self.model_w], u).toarray()
        self.model_x, self.model_y, self.model_t, self.model_v, self.model_w = model_x[0], model_y[0], model_t[0], model_v[0], model_w[0]
        
        self.model_positions.append([model_x, model_y, model_t, model_v, model_w])
        self.positions.append([self.x, self.y, self.t, self.v, self.w])
        self.inputs.append(u)

        msg = WheelsCmdStamped()
        msg.vel_left = u[0]
        msg.vel_right = u[1]

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
        print("Modello: ", self.model_positions)
        print("Positions: ",self.positions)
        print("Inputs: ", self.inputs)

if __name__ == '__main__':
    node = TheController(node_name='controller')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[Controller]: Keyboard interrupt.")
        exit(0)
    
    