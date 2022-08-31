#!/usr/bin/env python3

import os

import casadi as ca
import numpy as np
from scipy import spatial
from scipy.signal import lfilter
import pickle
import matplotlib.pyplot as plt

# ROS
import rospy, rospkg
rp = rospkg.RosPack()

# Msgs
from std_msgs.msg import Bool
from lmpc.msg import DuckPose
from lmpc.msg import Floats
from duckietown_msgs.msg import WheelsCmdStamped, BoolStamped

# Duckie
from dt_communication_utils import DTCommunicationGroup
from duckietown.dtros import DTROS, NodeType

VERBOSE = False
SUB_ROS = False

# If both the following are False uses open loop until it gets a new poisition from the camera
# If true does not use the camera but only the model
OPEN_LOOP = False
# If True, uses only the camera but not the model
FORCE_CLOSED_LOOP = False
if FORCE_CLOSED_LOOP == OPEN_LOOP == True:
    print("[Controller]: Warning both FORCE_CLOSED_LOOP and OPEN_LOOP are True")


map_path = os.path.join(rp.get_path("lmpc"), "src", "maps", "map_400_points.npy")
with open(map_path, 'rb') as f:
    map_data = np.load(f)

RIC_VER = False
MAX_SPEED = 1
MPC_TIME = 0.1
N = 10
NEW_PARAM=True

lmpc_path = os.path.join(rp.get_path("lmpc"), "src", "controllers", "LMPC.casadi")
if RIC_VER:
    mpc_path = os.path.join(rp.get_path("lmpc"), "src", "controllers", "M_ric_N10_manual_200p_speed05.casadi")
else:
    mpc_path = os.path.join(rp.get_path("lmpc"), "src", "controllers", "N10_max03_fittedspeed.casadi")
LMPC = ca.Function.load(lmpc_path)
MPC = ca.Function.load(mpc_path)
delay = round(0.15/MPC_TIME)
u_delay0 = ca.DM(np.zeros((2, delay)))

group = DTCommunicationGroup('my_position', DuckPose)
# group_map = DTCommunicationGroup('my_map', Floats)


# Default value, will be updated after map retrieval
N_POINTS_MAP = 400


if not NEW_PARAM:
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

if NEW_PARAM:
    def model_F(dt=0.033):
        """
        Return the model casadi function tuned according to the parameter found by estimation.

        :param dt: the time step
        """
        # IDEAL PARAMS
        # u1 = 5
        # u2 = 0
        # u3 = 0
        # w1 = 4
        # w2 = 0
        # w3 = 0
        # # parameters for forced dynamics
        # u_alpha_r = 1.5
        # u_alpha_l = 1.5
        # w_alpha_r = 15  # modify this for trim
        # w_alpha_l = 15

        # THESIS BY GUY
        # u1 = 7.662
        # u2 = 0.325
        # u3 = -0.050
        # w1 = 6.826
        # w2 = -4.929
        # w3 = -6.515
        # u_alpha_r = 2.755
        # u_alpha_l = 2.741
        # w_alpha_r = 10.8  # modify this for trim
        # w_alpha_l = 14.662

        # MANUAL
        # [7.662, 0.325, -0.050, 6.826, 0, 0, 2.755, 2.741, 14, 14.662]
        # u1 = 7.662
        # u2 = 0.325
        # u3 = -0.050
        # w1 = 6.826
        # w2 = 0
        # w3 = 0
        # u_alpha_r = 2.755
        # u_alpha_l = 2.741
        # w_alpha_r = 14
        # w_alpha_l = 14.662

        # FITTED wrt speed
        # [2.14148837,  0.12200042, -0.28237442,  1.3380637 ,  0.40072379, 1.30781483,  1.03762896,  0.62189597,  2.9650673 ,  2.89169198] 
        # u1 = 2.14148837
        # u2 = 0.12200042
        # u3 = -0.28237442
        # w1 = 1.3380637
        # w2 = 0.40072379
        # w3 = 1.30781483
        # u_alpha_r = 1.30781483
        # u_alpha_l = 1.03762896
        # w_alpha_r = 2.9650673
        # w_alpha_l = 2.89169198

        # Fitted wrt position
        u1 = 4.3123709
        u2 = 0.42117578
        u3 = 0. 
        w1 = 1.34991163
        w2 = 0.66724572
        w3 = 0.74908594
        u_alpha_r = 2.27306332
        u_alpha_l = 0.73258966
        w_alpha_r = 3.12010274
        w_alpha_l = 2.86162447

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

class GetMap():

    def __init__(self):
        self.map = None
        if map_data is None:
            group_map.Subscriber(self.map_callback)

    def map_callback(self, msg, header):
        self.map = msg.data
        print("[Controller]: Got map.")
        rospy.loginfo("[Controller]: Got map.")

    def wait_for_map(self):
        print("[Controller]: Getting map...")
        rospy.loginfo("[Controller]: Getting map...")
        if map_data is None:
            while self.map is None:
                rospy.sleep(0.1)
            group_map.shutdown()
        else:
            self.map = map_data
        map = np.array(self.map).reshape(-1,2)
        map = np.around(map, 2)
        # Cut map to number of points of interest:
        map = np.array([m for idx, m in enumerate(map) if idx%3 != 0])
        # map = map[::2] # if max speed = 0.6
        angles = np.zeros(map.shape[0])
        angles[:-1] = np.arctan2(map[1:,1]-map[:-1,1], map[1:,0]-map[:-1,0])
        angles[-1] = np.arctan2(map[0,1]-map[-1,1], map[0,0]-map[-1,0])
        # angles[angles < 0] += 2*np.pi
        angles = np.around(angles, 1)
        self.map = np.hstack((map, angles.reshape(-1,1)))
        print("[Controller]: Map saved.")
        rospy.loginfo("[Controller]: Map saved.")
        return self.map

class TheController(DTROS):

    def __init__(self, node_name, track):
        print("[Controller]: Initializing...")

        # Duck name
        self.vehicle = os.environ['VEHICLE_NAME']

        # initialize the DTROS parent class
        super(TheController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.F = model_F(dt=MPC_TIME)

        # Position
        self.x = None
        self.y = None
        self.t = -np.pi/2
        self.v = 0
        self.w = 0

        # To estimate speed
        self.old_x = 0
        self.old_y = 0
        self.old_t = 0
        self.old_v = 0
        self.old_w = 0

        self.n_samples_m_average = 3
        self.last_5_samples = np.zeros((self.n_samples_m_average, 3))
        self.last_5_samples_index = 0
        self.wait_to_start_idx = 0

        self.last_u = [0,0]

        self.starting_time = 0
        self.localization_time = 0

        self.track = track
        self.positions = []

        self.iteration = 0

        # kdtree
        self.kdtree = spatial.KDTree(track[:,:2])

        if SUB_ROS:
            # If subscribe to topic published by duckiebot
            self.subscriber = rospy.Subscriber("~/localization", DuckPose, self.callback)
        else:
            # UDP subscriber, if we use data from watchtower
            self.subscriber = group.Subscriber(lambda ros_data, header : self.callback(ros_data))
        

        # Publishers
        # Wheel control
        self.pub = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        # Emergency stop
        self.pub_e_stop = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/emergency_stop",BoolStamped,queue_size=1)
        
        # Save the message from the watchtower every time it arrives but run the MPC every MPC_TIME seconds
        rospy.Subscriber("/execute_controller", Bool, self.control, queue_size=1)


    def callback(self, ros_data):
        """
        Callback function for the localization subscriber.
        """
        if self.x and OPEN_LOOP:
            return
        curr_time = rospy.get_time()
        if VERBOSE:
            print(f"[Controller]: Received message after {curr_time-self.starting_time}s the MPC, and after {curr_time-self.localization_time} the last message.\n")
        if self.x:
            try:
                _, _, t, v, w = self.F([self.x, self.y, self.t, self.v, self.w], self.last_u).toarray()
                self.v = np.around(v[0], 3)
                self.w = np.around(w[0], 3)
                self.t = (t[0]+np.pi)%(2*np.pi)-np.pi
            except Exception:
                print("[Controller]: Error in F.", self.x, self.y, self.t, self.v, self.w, self.last_u)
        if ros_data.success:
            self.x = ros_data.x
            self.y = ros_data.y
            # self.t = ros_data.t
            if ros_data.t * self.t < 0: # a positive and a negative angle
                self.t = ros_data.t if min(ros_data.t, self.t) + np.pi/2 < max(ros_data.t, self.t) else self.t
            else: # same sign
                self.t = ros_data.t if np.abs(ros_data.t-self.t) < np.pi/2 else self.t
            print("\n[Controller]: Pose: ", ros_data.x, ros_data.y, np.rad2deg(ros_data.t), "\n")
        else:
            print("[Controller]: Position failed...")

        self.old_x = self.x
        self.old_y = self.y
        self.old_t = self.t
        self.localization_time = curr_time
        

    def control(self, ros_data):
        """
        Control function, callback for the /execute_controller topic.
        """
        if not self.x:
            return
        current_time = rospy.get_time()
        delta_time = current_time - self.starting_time
        if VERBOSE:
            print(f"[Controller]: Delta time: {delta_time}")
        
        # X0
        self.starting_time = current_time
        x = np.around(self.x, 2)
        y = np.around(self.y, 2)
        t = np.around(self.t, 1)
        v = self.v
        w = self.w

        # v = MAX_SPEED*0.6*(self.last_u[0] + self.last_u[1])/2

        # self.last_5_samples[self.last_5_samples_index] = [self.x, self.y, self.t]
        # self.last_5_samples_index = (self.last_5_samples_index + 1) % self.n_samples_m_average
        # self.x, self.y, self.t = self.last_5_samples.mean(axis=0)
        # self.wait_to_start_idx += 1

        # if self.wait_to_start_idx < self.n_samples_m_average:
        #     return

        self.positions.append([x, y, t, v, w])

        if True:
            print(f"\n[Controller]: Use MPC, x: {x}, y: {y}, t: {np.rad2deg(t)}, v: {v}, w: {w}\n")
            # rospy.loginfo(f"[Controller]: Got data, x: {x}, y: {y}, t: {np.rad2deg(t)}")

        X = ca.DM([x, y, t, v, w])

        # Reference
        # if self.iteration == 0:
        #     _,idx = self.kdtree.query([x, y], k=2)
        #     idx = max(idx) if (min(idx) != 0 or max(idx) == 1) else 0
        #     self.iteration = idx
        #     self.iteration += 1
        # else:
        #     idx = self.iteration
        #     self.iteration += 1
        _,idx = self.kdtree.query([x, y], k=2)
        idx = max(idx) if (min(idx) != 0 or max(idx) == 1) else 0
        idx = (idx+1) % N_POINTS_MAP # 1 is best
        if idx+N+1 < N_POINTS_MAP:
            r = self.track[idx:idx+N+1, :].T
        else:
            r = self.track[idx:, :]
            r = np.vstack((r, self.track[:idx+N+1-N_POINTS_MAP, :]))
            r = r.T
        
        # r = np.repeat(r[:,1].reshape(1, -1), N+1, axis=0).T
        # r = np.array([[ 1.5, 1.5, 0]]*(N+1)).T
        # r = np.array([[ 0.3, 2.5, 0]]*(N+1)).T
        # r = np.array([[x,y]]*(N+1)).T
        # r = np.repeat(self.track[(idx+1)%N_POINTS_MAP, :], N+1, axis=0).T
        if VERBOSE:
            print(f"[Controller]: r: {r.T}")

        tr = r[2,:]
        r = r[:2, :]

        #  Control
        if RIC_VER:
            # Riccardo version of the MPC
            weights_0 = [1e3, 1, 0, 1e-1]
            u = MPC(X, r, tr, u_delay0,  self.last_u, weights_0)*MAX_SPEED
        else:
            u = MPC(X, r, tr, u_delay0, 1e3, 1e-2, 0, 10)*MAX_SPEED
        u = np.around([u[0], u[1]], 3)
        print("[Controller]: u: ", u, "\n")

        self.last_u = u

        # Publish the message
        msg = WheelsCmdStamped()
        msg.vel_left = u[0]
        msg.vel_right = u[1]

        try:
            self.pub.publish(msg)
        except AttributeError:
            print("[Controller]: Publisher not initialized.")
            return

        # Update open loop model
        if (not FORCE_CLOSED_LOOP and np.abs(current_time - self.localization_time) > MPC_TIME) or OPEN_LOOP:
            try:
                x, y, t, v, w = self.F(ca.DM([self.x, self.y, self.t, self.v, self.w]), u).toarray()
            except Exception:
                print(x, y, t, v, w, self.last_u)
                raise
            self.x, self.y, self.t, self.v, self.w = x[0], y[0], t[0], v[0], w[0]


    def on_shutdown(self):
        print("[Controller]: Shutdown.")
        rospy.loginfo("[Controller]: Shutdown.")
        self.pub.publish(WheelsCmdStamped(vel_left=0, vel_right=0))
        self.subscriber.shutdown()
        self.pub.unregister()
        plt.plot(*self.track.T)
        plt.scatter(*np.array(self.positions)[:, :2].T)
        plt.show()
        plt.savefig("map.png")
        print(self.positions)
        with open('filename.pickle', 'wb') as handle:
            pickle.dump(self.positions, handle)

if __name__ == '__main__':
    take_map = GetMap()
    map = take_map.wait_for_map()
    N_POINTS_MAP = map.shape[0]
    print(f"[Controller]: Map has {N_POINTS_MAP} points.")
    node = TheController(track=map, node_name='controller')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[Controller]: Keyboard interrupt.")
        exit(0)
    
    