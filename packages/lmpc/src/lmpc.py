#!/usr/bin/env python3

import os

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
if FORCE_CLOSED_LOOP and OPEN_LOOP:
    print("[Controller]: Warning both FORCE_CLOSED_LOOP and OPEN_LOOP are True")

map_data = [1.8184616665769566,
 1.7982579980773814,
 1.814445615898867,
 1.835070397980356,
 1.8104615850039194,
 1.871868224486425,
 1.8065095738921149,
 1.9086514775955898,
 1.8025895825634515,
 1.94542015730785,
 1.7987016110179308,
 1.9821742636232056,
 1.7948456592555515,
 2.0189137965416566,
 1.7910217272763143,
 2.055638756063203,
 1.7872298150802197,
 2.092349142187844,
 1.7834699226672672,
 2.129044954915581,
 1.7797420500374563,
 2.165726194246414,
 1.7760461971907877,
 2.2023928601803413,
 1.7723823641272616,
 2.2390449527173644,
 1.7687625504934716,
 2.275682471857483,
 1.7654025543188132,
 2.312305417600697,
 1.7623822993909875,
 2.348913789947006,
 1.7597017857099932,
 2.385507588896411,
 1.7573610132758308,
 2.4220868144489103,
 1.7552436608833224,
 2.458651466604506,
 1.7516745253790593,
 2.4952015453631966,
 1.7461050972870846,
 2.531737050724982,
 1.7385353766073972,
 2.5682579826898637,
 1.7289653633399977,
 2.604764341257841,
 1.717395057484886,
 2.6412561264289125,
 1.7037265319557016,
 2.6775559816011794,
 1.6856151206072845,
 2.7094174614288833,
 1.6620638935674692,
 2.735035017508283,
 1.633072850836256,
 2.75440864983938,
 1.599852561072965,
 2.7675383584221716,
 1.5649420498343958,
 2.7744241432566596,
 1.5284217340651043,
 2.7750660043428432,
 1.4906232263066264,
 2.7706520256974856,
 1.4548764150940527,
 2.7731123572715157,
 1.422042264343294,
 2.7855316154149774,
 1.39212077405435,
 2.807909800127872,
 1.365111944227221,
 2.8365414184007576,
 1.3410157748619074,
 2.8655343707729575,
 1.3198322659584087,
 2.8948134637313583,
 1.3020743882210224,
 2.9252359408129283,
 1.2890035832531486,
 2.958909841819397,
 1.280677925441445,
 2.9959322169179736,
 1.274285541716859,
 3.033576535215555,
 1.266597062147585,
 3.0687114402575295,
 1.2576086827348836,
 3.1013332434989933,
 1.2473204034787548,
 3.1314419449399464,
 1.2357322243791993,
 3.1600346846447285,
 1.2227627371519545,
 3.190824700169356,
 1.2068867689200669,
 3.221916311902562,
 1.1875274120441848,
 3.252449149509488,
 1.164684666524308,
 3.2824232129901327,
 1.1383585323604368,
 3.3109063140065307,
 1.1085490095525707,
 3.331863043678908,
 1.0752560981007102,
 3.344210000627909,
 1.0389769420302248,
 3.3490185345189873,
 1.0022860205099002,
 3.351836670180849,
 0.9655508624961328,
 3.3534564356866206,
 0.928771467988923,
 3.353877831036301,
 0.8919478369882701,
 3.353100856229892,
 0.8550799694941746,
 3.3511255112673926,
 0.8181678655066364,
 3.347951796148803,
 0.7812115250256552,
 3.3435801824759266,
 0.7442109480512318,
 3.3395692164242137,
 0.7071661345833654,
 3.337372251664987,
 0.6700770846220565,
 3.336989288198245,
 0.6329437981673045,
 3.3379732507961344,
 0.5957662752191096,
 3.336396355153845,
 0.5585445157774727,
 3.331340533114618,
 0.5217387634705516,
 3.322565890839761,
 0.4881762349658126,
 3.3085987917210034,
 0.458339264793756,
 3.2891878274281097,
 0.4322278529543823,
 3.2643329979610822,
 0.4098419994476918,
 3.2349567814715874,
 0.3911817042736838,
 3.203645198943049,
 0.37624696743235897,
 3.1705525728674147,
 0.36503778892371663,
 3.135678903244683,
 0.3566555868738687,
 3.0996514504794526,
 0.3496821169604921,
 3.0634602284294346,
 0.34409984250339776,
 3.027117478678205,
 0.3399087635025859,
 2.9906232012257656,
 0.3371088799580563,
 2.953977396072117,
 0.335700191869809,
 2.9171800632172573,
 0.3356342249357377,
 2.880237080838028,
 0.3359855011013401,
 2.84326067589652,
 0.33640058317262195,
 2.8062937075209233,
 0.3368794711495834,
 2.769336175711238,
 0.3374221650322243,
 2.7323880804674627,
 0.33802866482054467,
 2.695449421789599,
 0.3386989705145445,
 2.6585201996776457,
 0.33943308211422396,
 2.621600414131603,
 0.3402309996195829,
 2.584690065151472,
 0.3410927230306213,
 2.5477891527372516,
 0.3420182523473392,
 2.510897676888942,
 0.3430075875697367,
 2.474015637606543,
 0.3440607286978137,
 2.4371430348900547,
 0.34517767573157004,
 2.4002798687394775,
 0.3463584286710061,
 2.3634260728768854,
 0.34760298751612156,
 2.326576495134589,
 0.3489113522669165,
 2.2897279015103424,
 0.35028352292339104,
 2.2528802920041464,
 0.35171949948554493,
 2.2160336666160005,
 0.3532192819533785,
 2.1791880253459044,
 0.35478287032689154,
 2.1423433681938584,
 0.35641026460608405,
 2.1054996951598635,
 0.3581014647909561,
 2.0686570062439182,
 0.35985647088150763,
 2.0318153014460236,
 0.36167528287773865,
 1.994974580766178,
 0.36355790077964917,
 1.958134844204384,
 0.36550432458723925,
 1.9212960917606392,
 0.3675145543005088,
 1.884458323434945,
 0.3695752886787955,
 1.8476213598843692,
 0.37162115027228537,
 1.8107843196134283,
 0.3736433390288919,
 1.7739470839694982,
 0.3756418549486151,
 1.7371096529525771,
 0.377616698031455,
 1.700272026562666,
 0.37956786827741157,
 1.6634342047997634,
 0.38149536568648473,
 1.6265961876638708,
 0.3833991902586746,
 1.5897579751549884,
 0.3852793419939812,
 1.552919567273114,
 0.38713582089240434,
 1.51608096401825,
 0.3889686269539442,
 1.4792421653903955,
 0.39077776017860066,
 1.4424031713895495,
 0.39256322056637394,
 1.4055639820157144,
 0.3943250081172637,
 1.3687245972688882,
 0.3960631228312702,
 1.3318850171490717,
 0.3977775647083934,
 1.2950452416562637,
 0.3994683337486333,
 1.2582052707904663,
 0.40113542995198975,
 1.2213651045516782,
 0.402778853318463,
 1.1845247429398986,
 0.40439860384805276,
 1.1476841859551292,
 0.4059946815407593,
 1.1108434335973696,
 0.4075670863965824,
 1.0740024858666186,
 0.4091158184155223,
 1.037161342762878,
 0.4106408775975787,
 1.0003200042861466,
 0.4121422639427519,
 0.9634784704364239,
 0.41361997745104173,
 0.9266367412137118,
 0.41507401812244815,
 0.8897948166180087,
 0.4165043859569713,
 0.8529526966493155,
 0.41834218037669463,
 0.8161377388527811,
 0.4227178569326946,
 0.7794851418182058,
 0.4299199732512953,
 0.7430132173963743,
 0.439948529332497,
 0.7067219655872851,
 0.4532831292447934,
 0.6714991253951897,
 0.472127654350495,
 0.6414240438785964,
 0.4967561727394789,
 0.6170040164279867,
 0.5271469856048516,
 0.5982235348276781,
 0.56019313091878,
 0.5828620420909425,
 0.5936851730056076,
 0.5729393213375975,
 0.6276231118653351,
 0.5797014482980485,
 0.662006947497961,
 0.5862792293072321,
 0.6968366799034857,
 0.5875344705138967,
 0.7319669124666851,
 0.5901671565481413,
 0.7671928825434768,
 0.6026510213614347,
 0.8025131478871934,
 0.6147064064909493,
 0.8379277084978358,
 0.6240991599848283,
 0.8734365643754018,
 0.6327572205239733,
 0.9090397155198928,
 0.6408363900155554,
 0.9447371619313095,
 0.6486751686952812,
 0.9805289036096501,
 0.6566865180631978,
 1.0164149405549154,
 0.664871434623591,
 1.0523952727671055,
 0.673229918376461,
 1.0884699002462215,
 0.6817619693218075,
 1.1246388229922613,
 0.6904675874596311,
 1.1608433830684222,
 0.6990956874079641,
 1.1969126753539103,
 0.7069147095540246,
 1.2328357636516845,
 0.7138778414908211,
 1.268272792233051,
 0.7208606232058599,
 1.3006564382433816,
 0.7344770194797308,
 1.3294567675669726,
 0.7560922522871314,
 1.3546717206084393,
 0.7856951892057981,
 1.3755275847449326,
 0.8191037973523142,
 1.3913965429176365,
 0.8529246312700559,
 1.4022844919325936,
 0.8871576909590215,
 1.4100180420171513,
 0.921802976419212,
 1.416047938701844,
 0.9568604876506281,
 1.420374181986672,
 0.9923302246532684,
 1.4229967718716354,
 1.028212187427133,
 1.4239157083567342,
 1.0645063759722233,
 1.423130991441968,
 1.1012127902885376,
 1.4208596396585518,
 1.138331430376077,
 1.4202989563627608,
 1.1758622962348417,
 1.4225313150767578,
 1.2137121631346215,
 1.4277892249670474,
 1.2508698248648298,
 1.436136185496135,
 1.287059115750714,
 1.4475721966640207,
 1.322280035792275,
 1.4620972584707044,
 1.3565325849895113,
 1.479711370916186,
 1.3898167633424234,
 1.5004168438231145,
 1.4219834137356264,
 1.5242657229802592,
 1.4496716721963667,
 1.5512795371968693,
 1.4714913129045228,
 1.5814582864729458,
 1.4874750315566028,
 1.6148019708084873,
 1.5003967357368033,
 1.6513105902034941,
 1.512033466448877,
 1.689684491822554,
 1.5234538809821292,
 1.7235972347332997,
 1.539859981774078,
 1.7522066979499331,
 1.5619442143035656,
 1.7755128814724555,
 1.5896300792096338,
 1.7935157853008667,
 1.6206087583816422,
 1.8062154094351661,
 1.6537985706077456,
 1.8136117538753538,
 1.6891995158879451,
 1.816561164403471,
 1.7263041926942144]

MAX_SPEED = 0.5
MPC_TIME = 0.1
N_MPC = 3
N = 2
# K nearest neighbors
K = 8

# N iterations to consider
i_j = 4

lmpc_path = os.path.join(rp.get_path("lmpc"), "src", "controllers", "LMPC.casadi")
mpc_path = os.path.join(rp.get_path("lmpc"), "src", "controllers", "M_01_N3_angle_max03_noretro.casadi")
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
    Return the model casadi function tuned according to the parameter found in the thesis.

    :param dt: the time step
    """
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

def get_border(traj, distance=0.15):
    """
    Get the border of the trajectory.
    
    :param traj: the trajectory of the yellow line
    :param distance: the distance from the center

    :return: the borders of the track
    """
    borders_inside = []
    borders_outside = []
    xm, ym = traj.mean(0)
    for idx, _ in enumerate(traj[:-1]):
        x0, y0 = traj[idx]
        x1, y1 = traj[idx+1]
        m = (y1 - y0) / (x1 - x0)
        xp, yp = (x1 + x0) / 2, (y1 + y0) / 2
        mp = -1 / m
        kp = yp - mp * xp
        a = 1+mp**2
        b = -2*xp+2*mp*(kp-yp)
        c = -distance**2+(kp-yp)**2+xp**2
        try:
            xs0, xs1 = np.roots([a,b,c])
        except np.linalg.LinAlgError:
            print("No solution")
            continue
        ys0, ys1 = mp*xs0+kp, mp*xs1+kp

        if (xs0-xm)**2+(ys0-ym)**2 < (xs1-xm)**2+(ys1-ym)**2:
            borders_inside.append([xs0, ys0])
            borders_outside.append([xs1, ys1])
        else:
            borders_inside.append([xs1, ys1])
            borders_outside.append([xs0, ys0])
    borders_inside = np.array(borders_inside)
    borders_outside = np.array(borders_outside)
    return borders_inside, borders_outside

class GetMap():

    def __init__(self):
        self.map = None
        if not map_data:
            group_map.Subscriber(self.map_callback)

    def map_callback(self, msg, header):
        self.map = msg.data
        print("[Controller]: Got map.")
        rospy.loginfo("[Controller]: Got map.")

    def wait_for_map(self):
        """
        Wait until a message with the map description is received.
        """
        print("[Controller]: Getting map...")
        rospy.loginfo("[Controller]: Getting map...")
        if not map_data:
            while self.map is None:
                rospy.sleep(0.1)
            group_map.shutdown()
        else:
            self.map = map_data
        map = np.array(self.map).reshape(-1,2)
        map = np.around(map, 2)
        # Cut map to number of points of interest:
        map = map[::10] # 5 appears to be the best number of points
        # Compute angles wrt to the next point
        angles = np.zeros(map.shape[0])
        angles[:-1] = np.arctan2(map[1:,1]-map[:-1,1], map[1:,0]-map[:-1,0])
        angles[-1] = np.arctan2(map[0,1]-map[-1,1], map[0,0]-map[-1,0])
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
        print("[Controller]: Initializing...")

        # Duck name
        self.vehicle = os.environ['VEHICLE_NAME']

        # initialize the DTROS parent class
        super(TheController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Track description
        self.track = track
        self.inside = inside
        self.outside = outside
        self.positions = []

        # Car model
        self.F = model_F(dt=MPC_TIME)

        # Position
        self.x = None
        self.y = None
        self.t = None
        self.v = 0
        self.w = 0

        self.last_u = [0,0]

        # LMPC initialization
        self.finish_line = None
        self.loop_n = 0
        self.iteration_n = 0
        self.Js = []
        self.plain_loops = []
        self.loops_with_time = []
        self.kdins = spatial.KDTree(inside)
        self.ins_len = inside.shape[0]
        self.already_changed = False

        self.X_log = np.empty((5,0))
        self.U_log = np.empty((2,0))
        self.X_log_origin = None
        self.all_points = None
        self.first_loop_len = None
        self.last_loop = None
        # Distance to point for convex hull inside borders
        self.more = 20

        # Moving average filter
        self.n_samples_m_average = 3
        self.last_5_samples = np.zeros((self.n_samples_m_average, 3))
        self.last_5_samples_index = 0
        self.wait_to_start_idx = 0

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
        

        # Publishers
        # Wheel control
        self.pub = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        # Emergency stop
        self.pub_e_stop = rospy.Publisher(f"/{self.vehicle}/wheels_driver_node/emergency_stop",BoolStamped,queue_size=1)
        
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
                _, _, _, v, w = self.F([self.x, self.y, self.t, self.v, self.w], self.last_u).toarray()
                self.v = np.around(v[0], 3)
                self.w = np.around(w[0], 3)
            except Exception:
                print("[Controller]: Error in F.", self.x, self.y, self.t, self.v, self.w, self.last_u)
            # self.w = w[0]
        if ros_data.success:
            self.x = ros_data.x
            self.y = ros_data.y
            self.t = ros_data.t
            print("\n[Controller]: Pose: ", ros_data.x, ros_data.y, np.rad2deg(ros_data.t))
        else:
            print("[Controller]: Position failed, using odometry...")
            x, y, t, v, w = self.F([self.x, self.y, self.t, self.v, self.w], self.last_u).toarray()
            self.x, self.y, self.t, self.v, self.w = x[0], y[0], t[0], v[0], w[0]

        # self.old_x = self.x
        # self.old_y = self.y
        # self.old_t = self.t
        self.localization_time = curr_time

        if self.finish_line == None:
            self.finish_line = [self.x, self.y, self.t]
        

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
        # TODO it may be needed to check if idx is always >= old idx
        _,idx = self.kdtree.query([x, y], k=2)
        idx = max(idx) if (min(idx) != 0 or max(idx) == 1) else 0

        # Check if finish line
        if self.track[idx, 1] <= self.finish_line[1] and self.iteration_n >= 200:
            self.loop_n += 1
            self.iteration_n = 0
            print("[LMPC]: Loop n ", self.loop_n)

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
                last_points = np.vstack((last_points, np.arange(last_points.shape[1])[::-1]))
                self.loops_with_time.append(last_points)

                # Reset as new loop is starting
                self.X_log = np.empty((5,0))
                self.U_log = np.empty((2,0))
                
                # Angle normalization
                X[2] = ca.mod(X[2]+ca.pi, 2*ca.pi)-ca.pi

                # Reset search tree to last loop
                self.kdtree = spatial.KDTree(self.last_loop[:2, :(1/MPC_TIME)*10].T)

                # Hide the last points of the track to avoid misdirections 
                self.last_iterations_filtered = self.last_iterations[np.vstack([self.last_iterations[-1] > K]*6)].reshape(6,-1)

                # Nearest neighbour for LMPC
                self.nbrs = NearestNeighbors(n_neighbors=K*i_j, algorithm='ball_tree').fit(self.last_iterations_filtered[:2].T)

                # Flag to update old states ad prevent misdirections 
                self.already_changed = False
            
            print("[LMPC]: Last time: ", [t.shape[1]*MPC_TIME for t in self.loops_with_time])
        # If it is far enough and using LMPC
        elif self.iteration_n == K+N and self.loop_n >= 1:
            # Stop hiding points in nbrs
            self.kdtree = spatial.KDTree(self.last_loop[:2, :].T)
            self.last_iterations_filtered = self.last_iterations
            self.nbrs = NearestNeighbors(n_neighbors=K*i_j, algorithm='ball_tree').fit(self.last_iterations[:2, :].T)

        # If the car is about to complete the lap
        # TODO update with autolab values
        if x[1] > 1 and x[1] < 1.7 and x[0] > 1.5 and not self.already_changed:
            # The first points in the next iteration have time 0
            self.last_iterations_filtered[-1, :N+K*i_j] = 0

            # The first points in the next iteration are as in the first loop
            self.last_iterations_filtered[:-1, :N+1] = self.X_log_orig[:, :N+1]

            # After one iteration the angle is +2*pi 
            self.last_iterations_filtered[2, :N+K*i_j] += 2*np.pi

            # Do not update this values again during the current lap
            self.already_changed = True

        # Select MPC vs LMPC
        if self.loop_n == 0:
            self._mpc(X, idx, current_time)
        else:
            self._lmpc(X, idx)

        # Update open loop model
        # If there has not been a new message during the execution
        # of the control loop update the position open loop.
        # If there will be a message before the next execution of the
        # control loop this value will be overwritten
        if (not FORCE_CLOSED_LOOP and np.abs(current_time - self.localization_time) > MPC_TIME) or OPEN_LOOP:
            try:
                x, y, t, v, w = self.F(ca.DM([self.x, self.y, self.t, self.v, self.w]), u).toarray()
            except Exception:
                print(x, y, t, v, w, self.last_u)
                raise
            self.x, self.y, self.t, self.v, self.w = x[0], y[0], t[0], v[0], w[0]

        # Keep track of iteration number (=time/MPC_TIME)
        self.iteration_n += 1

    def _mpc(self, X, idx, current_time):
        """
        MPC function.

        :param X: Current state.
        :param idx: Index of the closest point in the track.
        :param current_time: Current time.
        """

        # Reference
        if idx+N+1 < N_POINTS_MAP:
            r = self.track[idx:idx+N+1, :].T
        else:
            r = self.track[idx:, :]
            r = np.vstack((r, self.track[:idx+N+1-N_POINTS_MAP, :]))
            r = r.T
        
        if VERBOSE:
            print(f"[LMPC]: r: {r.T}")

        # Angular reference
        tr = r[2,:]
        r = r[:2, :]

        #  Control
        u = MPC(X, r, tr, u_delay0,  1, 0, 0, 0)*MAX_SPEED
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
            print("[Controller]: Publisher not initialized.")
            return

        self.X_log = np.column_stack((self.X_log, X))
        self.U_log = np.column_stack((self.U_log, u))


    def _lmpc(self, X, idx):
        """
        LMPC function.
        
        :param X: Current state.
        :param idx: Index of the closest point in the track.
        """
        distances, indices = self.nbrs.kneighbors([self.last_loop[:2, (idx+N)%self.last_loop.shape[1]].T])
        indices = indices.reshape(-1)

        # Compute values for convex hull
        if self.iteration_n == 0:
            D = self.last_iterations_filtered[:-1, indices]
            J = self.last_iterations_filtered[-1, indices].reshape(-1)
        else:
            S = self.last_iterations_filtered[:, indices]@l
            distances, indices = self.nbrs.kneighbors(np.array(S[:2].T))
            indices = indices.reshape(-1)
            D = self.last_iterations_filtered[:-1, indices]
            J = self.last_iterations_filtered[-1, indices].reshape(-1)
        
        self.Js.append(J)

        # Closest point to the current pose in inside line
        _, border_idx = self.kdins.query(np.array([X[0], X[1]]).reshape(-1), workers=-1)
        # Margins in track for convex hull
        margins = np.array([
            inside[border_idx],
            inside[(border_idx+N+self.more)%self.ins_len],
            outside[border_idx],
            outside[(border_idx+N+int(self.more/4))%self.ins_len],
            outside[(border_idx+N+self.more)%self.ins_len]]).T

        # Call the LMPC function
        u, l = LMPC(X, ca.DM(D[:, :]), ca.DM(J)/600, (np.arange(self.iteration_n, self.iteration_n+N-1)>=self.finish_line.t).T, margins)

        # Approximate the input
        u = np.around([u[0], u[1]], 3)
        print("[LMPC]: u: ", u)
        self.last_u = u

        # Store last values
        U_log = np.column_stack((U_log, u))
        X_log = np.column_stack((X_log, X))


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
    track, inside, outside = take_map.wait_for_map()
    N_POINTS_MAP = map.shape[0]
    print(f"[Controller]: Map has {N_POINTS_MAP} points.")
    node = TheController(track=track, inside=inside, outside=outside, node_name='controller')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[Controller]: Keyboard interrupt.")
        exit(0)
    
    