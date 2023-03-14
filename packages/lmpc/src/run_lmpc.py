#!/usr/bin/env python3

import os
import sys
from timeit import default_timer as timer

import casadi as ca
import numpy as np
from scipy import spatial
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ROS
import rospy, rospkg
rp = rospkg.RosPack()

# Msgs
from std_msgs.msg import Bool
from lmpc.msg import DuckPose
from lmpc.msg import Floats
from duckietown_msgs.msg import WheelsCmdStamped, BoolStamped
from sensor_msgs.msg import Imu

# Duckie
from dt_communication_utils import DTCommunicationGroup
from duckietown.dtros import DTROS, NodeType

np.set_printoptions(threshold=sys.maxsize)

VERBOSE = False
SUB_ROS = False

# If both the following are False uses open loop until it gets a new poisition from the camera
# If true does not use the camera but only the model
OPEN_LOOP = False
# If True, uses only the camera but not the model
FORCE_CLOSED_LOOP = False
if FORCE_CLOSED_LOOP and OPEN_LOOP:
    print("[LMPC]: Warning both FORCE_CLOSED_LOOP and OPEN_LOOP are True")

map_path = os.path.join(rp.get_path("lmpc"), "src", "maps", "map_400_fixed.npy")
with open(map_path, 'rb') as f:
    map_data = np.load(f)

MAX_SPEED = 1
MPC_TIME = 0.1
N_MPC = 10
N = 3
# K nearest neighbors
K = 3

# N iterations to consider
i_j = 2

lmpc_path = os.path.join(rp.get_path("lmpc"), "src", "controllers", f"LMPC_max03_noborder_noorientation_k{K}_ij{i_j}_N{N}.casadi")
mpc_path = os.path.join(rp.get_path("lmpc"), "src", "controllers", "N10_max03_fittedspeed.casadi") #N10_max03_fittedspeed.casadi
LMPC = ca.Function.load(lmpc_path)
MPC = ca.Function.load(mpc_path)
delay = round(0.15/MPC_TIME)
u_delay0 = ca.DM(np.zeros((2, delay)))

group = DTCommunicationGroup('my_position', DuckPose)
# group_map = DTCommunicationGroup('my_map', Floats)

# Default value, will be updated after map retrieval
N_POINTS_MAP = 200

def model_F(dt=MPC_TIME):
    """
    Return the model casadi function tuned according to the parameter found from the fitting.

    :param dt: the time step
    """
    # u1 = 4.3123709
    # u2 = 0.42117578
    # u3 = 0. 
    # w1 = 1.34991163
    # w2 = 0.66724572
    # w3 = 0.74908594
    # u_alpha_r = 2.27306332
    # u_alpha_l = 0.73258966
    # w_alpha_r = 3.12010274
    # w_alpha_l = 2.86162447

    # FITTED wrt speed
    # [2.14148837,  0.12200042, -0.28237442,  1.3380637 ,  0.40072379, 1.30781483,  1.03762896,  0.62189597,  2.9650673 ,  2.89169198] 
    u1 = 2.14148837
    u2 = 0.12200042
    u3 = -0.28237442
    w1 = 1.3380637
    w2 = 0.40072379
    w3 = 1.30781483
    u_alpha_r = 1.30781483
    u_alpha_l = 1.03762896
    w_alpha_r = 2.9650673
    w_alpha_l = 2.89169198

    # Normalized
    # u1 = 3.51843
    # u2 = -0.40282789
    # u3 = 0.2789241
    # w1 = 1.41903836
    # w2 = 0.16628764
    # w3 = 1.52162695
    # u_alpha_r = 0.54710851
    # u_alpha_l = -0.50963813
    # w_alpha_r = 1.08095586
    # w_alpha_l = 0.73014709

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

    # x1 = (x0-0.24)/1.68 + v0*dt*np.cos(th0 + w0*dt/2)
    # y1 = (y0-0.34)/3.18 + v0*dt*np.sin(th0 + w0*dt/2)
    # x1 = x1*1.68+0.24
    # y1 = y1*3.18+0.34
    # Cannot use atan2 because x1 and y1 are approximated while th1 is not
    theta1 = th0 + w0*dt
    dae = ca.vertcat(x1, y1, theta1, v1, w1)
    F = ca.Function('F',[x,u],[dae],['x','u'],['dae'])
    return F

def get_border(traj, distance=0.15):
    """
    Get the border of the trajectory.
    
    :param traj: the trajectory of the yellow line
    :param distance: the distance from the center

    :return: the borders of the track
    """
    inside_path = os.path.join(rp.get_path("lmpc"), "src", "maps", "inside_400_points.npy")
    outside_path = os.path.join(rp.get_path("lmpc"), "src", "maps", "outside_400_points.npy")
    with open(inside_path, 'rb') as f:
        borders_inside = np.load(f)
    with open(outside_path, 'rb') as f:
        borders_outside = np.load(f)
    return borders_inside, borders_outside

class GetMap():

    def __init__(self):
        self.map = None
        if map_data is None:
            group_map.Subscriber(self.map_callback)

    def map_callback(self, msg, header):
        self.map = msg.data
        print("[LMPC]: Got map.")
        rospy.loginfo("[LMPC]: Got map.")


    def _centered_moving_average(self, a, n=3):
        border = n//2
        for idx in range(border, len(a)-border):
            a[idx] = (a[idx-border]+a[idx+border])/2
        return a

    def _clean_angle_reference(self, ref_angle, n=4):
        """
        Dirty solution to clean the angle references
        Remove outliers and do central moving average
        """
        for idx in range(0, 50):
            if ref_angle[idx] < 0:
                ref_angle[idx] = np.pi
        for idx in range(60, 100):
            if ref_angle[idx] > 0:
                ref_angle[idx] = -np.pi
        return np.hstack([self._centered_moving_average(ref_angle[:50], 4), ref_angle[50:60], self._centered_moving_average(ref_angle[60:], 4)])

    def wait_for_map(self):
        """
        Wait until a message with the map description is received.
        """
        print("[LMPC]: Getting map...")
        rospy.loginfo("[LMPC]: Getting map...")
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
        # Compute angles wrt to the next point
        angles = np.zeros(map.shape[0])
        angles[:-1] = np.arctan2(map[1:,1]-map[:-1,1], map[1:,0]-map[:-1,0])
        angles[-1] = np.arctan2(map[0,1]-map[-1,1], map[0,0]-map[-1,0])
        angles = self._clean_angle_reference(angles)
        # angles[angles < 0] += 2*np.pi
        angles = np.around(angles, 1)
        self.map = np.hstack((map, angles.reshape(-1,1)))
        # Compute track borders
        inside, outside = get_border(self.map, distance=0.16)
        print("[Controller]: Map saved.")
        rospy.loginfo("[Controller]: Map saved.")
        return self.map, inside, outside

class TheController(DTROS):

    def __init__(self, node_name, track, inside, outside):
        print("[LMPC]: Initializing...")

        # Duck name
        self.vehicle = os.environ['VEHICLE_NAME']

        # initialize the DTROS parent class
        super(TheController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Track description
        self.track = track
        self.inside = inside
        self.outside = outside
        self.positions = []
        self.lmpc_references = []
        self.lmpc_references_points = []
        self.mpc_reference = []

        # Car model
        self.F = model_F(dt=MPC_TIME)

        # Position
        self.x = None
        self.y = None
        self.t = -np.pi/2
        self.v = 0
        self.w = 0

        self.imu_time = rospy.get_time()

        self.ideal_states = []
        self.camera_states = []

        self.last_u = [0,0]

        # LMPC initialization
        self.finish_line = None
        self.loop_n = 0
        self.iteration_n = 0
        self.Js = []
        self.l = None
        self.plain_loops = []
        self.loops_with_time = []
        self.kdins = spatial.KDTree(inside)
        self.ins_len = inside.shape[0]
        self.already_changed = False
        self.see_all = False

        self.X_log = np.empty((5,0))
        self.U_log = np.empty((2,0))
        self.X_log_origin = None
        self.all_points = None
        self.first_loop_len = None
        self.last_loop = None
        self.old_idx = None
        # Distance to point for convex hull inside borders
        self.more = 20

        # Moving average filter
        self.n_samples_m_average = 1
        self.last_n_samples = np.ones((self.n_samples_m_average, 1)) * (-np.pi/2)
        self.last_n_samples_index = 0
        self.wait_to_start_idx = 0
        self.old_t = -np.pi/2

        # Timing
        self.starting_time = 0
        self.localization_time = 0

        # kdtree
        self.kdtree = spatial.KDTree(track[:,:2])

        # Subscribers
        if SUB_ROS:
            # If subscribe to topic published by duckiebot
            self.subscriber = rospy.Subscriber("~/localization", DuckPose, self._cb_localization)
        else:
            # UDP subscriber, if we use data from watchtower
            self.subscriber = group.Subscriber(lambda ros_data, header : self._cb_localization(ros_data))

        self.imu_subscriber = rospy.Subscriber("~/imu_node/imu_data", Imu, self._cb_imu)
        

        # Publishers
        # Wheel control
        self.pub = rospy.Publisher(f"/duckvader/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        # Emergency stop
        self.pub_e_stop = rospy.Publisher(f"/duckvader/wheels_driver_node/emergency_stop",BoolStamped,queue_size=1)
        
        # Save the message from the watchtower every time it arrives but run the MPC every MPC_TIME seconds
        rospy.Subscriber("/execute_controller", Bool, self._control, queue_size=1)


    def _cb_localization(self, ros_data):
        """
        Callback function for the localization subscriber.
        """
        if self.x and OPEN_LOOP:
            return
        curr_time = rospy.get_time()
        if VERBOSE:
            print(f"[Controller]: Received message after {curr_time-self.starting_time}s the MPC, and after {curr_time-self.localization_time} the last message.")
        if self.x:
            try:
                _x, _y, t, _v, _w = self.F([self.x, self.y, self.t, self.v, self.w], self.last_u).toarray()
                # self.v = np.around(v[0], 3)
                # self.w = np.around(w[0], 3)
                self.t = (t[0]+np.pi)%(2*np.pi)-np.pi
                self.ideal_states.append([_x, _y, self.t, _v, _w])
            except Exception:
                print("[Controller]: Error in F.", self.x, self.y, self.t, self.v, self.w, self.last_u)
            # self.w = w[0]
        if ros_data.success:
            self.x = ros_data.x
            self.y = ros_data.y

            if ros_data.t * self.t < 0: # a positive and a negative angle
                self.t = ros_data.t if min(ros_data.t, self.t) + np.pi/2 < max(ros_data.t, self.t) else self.t
            else: # same sign
                self.t = ros_data.t if np.abs(ros_data.t-self.t) < np.pi/2 else self.t
            
            # Moving average
            # self.last_n_samples[self.last_n_samples_index] = self.t
            # self.t = sum(self.last_n_samples)/self.n_samples_m_average
            # self.last_n_samples_index = (self.last_n_samples_index+1)%self.n_samples_m_average
            alpha = 1
            self.t = alpha * self.t + (1 - alpha) * self.old_t

            self.camera_states.append([self.x, self.y, ros_data.t, self.v, self.w])
            if VERBOSE:
                print("\n[Controller]: Pose: ", ros_data.x, ros_data.y, np.rad2deg(ros_data.t))
        else:
            print("[Controller]: Position failed...")
        # self.old_x = self.x
        # self.old_y = self.y
        self.old_t = self.t
        self.localization_time = curr_time
        

    def _cb_imu(self, ros_data):
        """
        Callback function for the imu subscriber.
        """
        dt = rospy.get_time() - self.imu_time
        vx = ros_data.linear_acceleration.x * dt
        vy = ros_data.linear_acceleration.y * dt
        v = (vx**2 + vy**2)**0.5
        w = ros_data.angular_velocity.z

        self.v = np.around(v, 3)
        self.w = np.around(w, 3)

        self.imu_time = rospy.get_time()

    def _control(self, ros_data):
        """
        Control function, callback for the /execute_controller topic.
        """
        if not self.x:
            return
        current_time = rospy.get_time()
        delta_time = current_time - self.starting_time
        if VERBOSE:
            print(f"[Controller]: Delta time: {delta_time}")
        
        # Retrieve X0
        self.starting_time = current_time
        x = np.around(self.x, 2)
        y = np.around(self.y, 2)
        t = np.around(self.t, 1)
        v = self.v
        w = self.w
        self.positions.append([x, y, t, v, w])
        if VERBOSE:
            print(f"\n[Controller]: Use MPC, x: {x}, y: {y}, t: {np.rad2deg(t)}, v: {v}, w: {w}")

        # Complete state
        X = ca.DM([x, y, t, v, w])

        # Compute reference
        _,idx = self.kdtree.query([x, y], k=2)
        idx = max(idx) if (min(idx) != 0 or max(idx) == 1) else 0
        idx = (idx+1) % N_POINTS_MAP #if self.loop_n == 0 else (idx+2) % N_POINTS_MAP

        if self.old_idx is not None:
            # Idx must be always >= old_idx, only case is when we are at the last point
            if self.old_idx > idx and np.abs(self.old_idx-idx) < 50:
                idx = self.old_idx
        self.old_idx = idx

        # If it is just starting save the start line position
        if self.finish_line == None:
            self.finish_line = [self.track[idx, 0], self.track[idx, 1], -np.pi/2, idx]

        # Check if the index is close to the same of the finish line
        if idx >= self.finish_line[-1] and idx <= self.finish_line[-1]+5 and self.iteration_n >= 300:
            self.loop_n += 1
            self.iteration_n = 0
            print("\n\n\n\t########### [LMPC]: Loop n ", self.loop_n, " ###########\n\n\n")

            # If loop 0 has just ended we need to initialize the LMPC
            if self.loop_n == 1:
                # New state definition to consider the time to arrive,
                # all_points: [x, y, theta, v, w, steps to arrive]
                self.all_points = self.X_log
                self.first_loop_len = self.all_points.shape[1]
                self.all_points = np.vstack((self.all_points, np.arange(self.first_loop_len)[::-1]))
                self.last_iterations = np.hstack([self.all_points]*i_j)
                if idx+N > self.all_points.shape[1]:
                    idx = self.all_points.shape[1] - idx

                # Save first loop
                self.X_log_origin = self.X_log

            # At start of each LMPC loop reset the values
            if self.loop_n >= 1:
                # Compute last iteration distance to last point
                self.last_loop = self.X_log
                last_points = self.X_log
                print("From X_log: ", repr(last_points))
                last_points = np.vstack((last_points, np.arange(last_points.shape[1])[::-1]))

                # Delete duplicates on first 5 values of the state,
                # thus keeping the time to arrive
                # Delete them because sometimes the LMPC is not synced with the watchtower
                _, indices = np.unique(last_points[:-1], axis=1, return_index=True)
                sorted_indices = list(sorted(indices))
                last_points = last_points[:, sorted_indices]
                print("[LMPC]: Last points: ", repr(last_points))

                self.loops_with_time.append(last_points)

                # Reset as new loop is starting
                self.X_log = np.empty((5,0))
                self.U_log = np.empty((2,0))
                
                # Angle normalization
                X[2] = ca.mod(X[2]+ca.pi, 2*ca.pi)-ca.pi

                # Reset search tree to last loop
                self.kdtree = spatial.KDTree(self.last_loop[:2, :int((1/MPC_TIME)*10)].T)

                # Hide the last points of the track to avoid misdirections 
                self.last_iterations_filtered = self.last_iterations[np.vstack([self.last_iterations[-1] > K]*6)].reshape(6,-1)

                # Nearest neighbour for LMPC
                self.nbrs = NearestNeighbors(n_neighbors=K*i_j, algorithm='ball_tree').fit(self.last_iterations_filtered[:2].T)

                # Flag to update old states and prevent misdirections 
                self.already_changed = False
                self.see_all = False
            
            print("[LMPC]: Last time: ", [t.shape[1]*MPC_TIME for t in self.loops_with_time])
        # If it is far enough and using LMPC
        elif not self.see_all and self.iteration_n >= K+N+50 and np.abs(idx - self.finish_line[-1]) >= K+N and self.loop_n >= 1:
            # Stop hiding points in nbrs
            print("####################[LMPC]: Stop hiding points in nbrs")
            self.kdtree = spatial.KDTree(self.last_loop[:2, :].T)
            self.last_iterations_filtered = self.last_iterations
            self.nbrs = NearestNeighbors(n_neighbors=K*i_j, algorithm='ball_tree').fit(self.last_iterations[:2, :].T)
            self.see_all = True

        # If the car is about to complete the lap
        if X[1] > 2.5 and X[1] <= 3 and X[0] > 1 and not self.already_changed and self.loop_n >= 1:
            print("####################[LMPC]: Get ready to complete the lap")
            # The first points in the next iteration have time 0
            self.last_iterations_filtered[-1, :N+K*i_j] = 0

            # The first points in the next iteration are as in the first loop
            self.last_iterations_filtered[:-1, :N+1] = self.X_log_origin[:, :N+1]

            # After one iteration the angle is +2*pi 
            self.last_iterations_filtered[2, :N+K*i_j] += 2*np.pi

            # Do not update this values again during the current lap
            self.already_changed = True

        # Select MPC vs LMPC
        if self.loop_n == 0:
            self._mpc(X, idx)
        else:
            lmpc_timer_start = timer()
            self._lmpc(X, idx)
            lmpc_timer_end = timer()
            print("[LMPC]: Time: ", lmpc_timer_end - lmpc_timer_start)

        # Update open loop model
        # If there has not been a new message during the execution
        # of the control loop update the position open loop.
        # If there will be a message before the next execution of the
        # control loop this value will be overwritten
        if (not FORCE_CLOSED_LOOP and np.abs(current_time - self.localization_time) > MPC_TIME) or OPEN_LOOP:
            try:
                x, y, t, v, w = self.F(ca.DM([self.x, self.y, self.t, self.v, self.w]), self.last_u).toarray()
            except Exception:
                print(x, y, t, v, w, self.last_u)
                raise
            self.x, self.y, self.t, self.v, self.w = x[0], y[0], t[0], v[0], w[0]

        # Keep track of iteration number (=time/MPC_TIME)
        self.iteration_n += 1

    def _mpc(self, X, idx):
        """
        MPC function.

        :param X: Current state.
        :param idx: Index of the closest point in the track.
        :param current_time: Current time.
        """

        # Reference
        if idx+N_MPC+1 < N_POINTS_MAP:
            r = self.track[idx:idx+N_MPC+1, :].T
        else:
            r = self.track[idx:, :]
            r = np.vstack((r, self.track[:int((idx+N_MPC+1)%N_POINTS_MAP), :]))
            r = r.T
        
        if VERBOSE:
            print(f"[LMPC]: r: {r.T}")
        
        self.mpc_reference.append(r)

        # Angular reference
        tr = r[2,:]
        r = r[:2, :]

        #  Control
        u = MPC(X, r, tr, u_delay0, 1e3, 1e1, 0, 50)*MAX_SPEED # 1e3, 1e-1, 0, 50
        u = np.around([u[0], u[1]], 3)
        print("[LMPC]: u: ", u)
        self.last_u = u

        # Publish the message
        msg = WheelsCmdStamped()
        msg.vel_left = u[0]
        msg.vel_right = u[1]

        try:
            self.pub.publish(msg)
        except AttributeError:
            print("[LMPC]: Publisher not initialized.")
            return

        self.U_log = np.column_stack((self.U_log, u))
        self.X_log = np.column_stack((self.X_log, X))


    def _lmpc(self, X, idx):
        """
        LMPC function.
        
        :param X: Current state.
        :param idx: Index of the closest point in the track.
        """
        # TODO what is better?
        # distances, indices = self.nbrs.kneighbors([self.last_loop[:2, (idx+N)%self.last_loop.shape[1]].T])
        distances, indices = self.nbrs.kneighbors([self.track[(idx+N)%N_POINTS_MAP, :2].T])
        indices = indices.reshape(-1)

        self.lmpc_references_points.append(self.last_iterations_filtered[:2, indices])

        # print("[LMPC]: indices: ", self.last_iterations_filtered[:-1, indices])

        # Compute values for convex hull
        if self.iteration_n == 0:
            D = self.last_iterations_filtered[:-1, indices]
            J = self.last_iterations_filtered[-1, indices].reshape(-1)
        else:
            S = self.last_iterations_filtered[:, indices]@self.l
            distances, indices = self.nbrs.kneighbors(np.array(S[:2].T))
            indices = indices.reshape(-1)
            D = self.last_iterations_filtered[:-1, indices]
            J = self.last_iterations_filtered[-1, indices].reshape(-1)

        # print("[LMPC]: D: ", D)
        self.lmpc_references.append(D[:2, 0])
        
        self.Js.append(J)

        # Closest point to the current pose in inside line
        _, border_idx = self.kdins.query(np.array([X[0], X[1]]).reshape(-1))
        # Margins in track for convex hull
        margins = np.array([
            inside[border_idx],
            inside[(border_idx+N+self.more)%self.ins_len],
            outside[border_idx],
            outside[(border_idx+N+int(self.more/4))%self.ins_len],
            outside[(border_idx+N+self.more)%self.ins_len]]).T

        # Call the LMPC function
        u, self.l = LMPC(X, ca.DM(D[:, :]), ca.DM(J)/600, (np.arange(self.iteration_n, self.iteration_n+N-1)>=self.finish_line[2]).T, margins)

        # Approximate the input
        u = np.around([u[0], u[1]], 3)
        if VERBOSE:
            print("[LMPC]: u: ", u)
        self.last_u = u

        # Publish the message
        msg = WheelsCmdStamped()
        msg.vel_left = u[0]
        msg.vel_right = u[1]

        try:
            self.pub.publish(msg)
        except AttributeError:
            print("[LMPC]: Publisher not initialized.")
            return

        # Store last values
        self.U_log = np.column_stack((self.U_log, u))
        self.X_log = np.column_stack((self.X_log, X))


    def on_shutdown(self):
        print("[LMPC]: Shutdown.")
        rospy.loginfo("[LMPC]: Shutdown.")
        self.pub.publish(WheelsCmdStamped(vel_left=0, vel_right=0))
        self.subscriber.shutdown()
        self.pub.unregister()
        print(self.positions)
        # print(self.lmpc_references)
        # print("LMPC reference points: ", self.lmpc_references_points)
        print("MPC reference points: ", self.mpc_reference)
        # print("Ideal states: ", self.ideal_states)
        # print("Camera states: ", self.camera_states)

if __name__ == '__main__':
    take_map = GetMap()
    track, inside, outside = take_map.wait_for_map()
    N_POINTS_MAP = track.shape[0]
    print(f"[LMPC]: Map has {N_POINTS_MAP} points.")
    node = TheController(track=track, inside=inside, outside=outside, node_name='controller')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[LMPC]: Keyboard interrupt.")
        exit(0)
    
    