#!/usr/bin/env python
"""Detects duckiebots from the watchtower using their unique colors.

Publish on watchtower00/localization a PointStamped with the car coordinates.

Edited version of the original example by Simon Halle.
"""
__author__ =  'Giulio Vaccari <giulio.vaccari at mail.polimi.it>'
__version__=  '0.1'
__license__ = 'MIT'
# Python libs
import sys, os

# numpy and scipy
import numpy as np

# OpenCV
import cv2

# Ros libraries
import rospy
import tf

from duckietown.dtros import DTROS, NodeType

# Ros Messages
from sensor_msgs.msg import CameraInfo, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

from image_geometry import PinholeCameraModel

from lmpc.msg import DuckPose
from lmpc.srv import GetMap

VERBOSE=False
PLOT=False
PUB_RECT=False

def get_map_client():
    rospy.loginfo("[Localize]: Waiting for map server...")
    print("[Localize]: Waiting for map server...")
    rospy.wait_for_service('get_map')
    try:
        get_map = rospy.ServiceProxy('get_map', GetMap)
        resp1 = get_map()
        rospy.loginfo("[Localize]: Got map.")
        print("[Localize]: Got map.")
        return np.array(resp1.data).reshape(-1, 2)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def get_car(img):
    """
    Extract the car from the image.

    :param img: image
    :return: front coord, left coord, theta
    """
    # scale_percent = 60
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_color1_blue = np.array([50, 200, 90])
    hsv_color2_blue = np.array([100, 300, 300])
    mask_blue = cv2.inRange(img_hsv, hsv_color1_blue, hsv_color2_blue)

    hsv_color1_pink = np.array([150, 50, 200])
    hsv_color2_pink = np.array([200, 100, 250])
    mask_pink = cv2.inRange(img_hsv, hsv_color1_pink, hsv_color2_pink)
    
    back_coo = np.argwhere(mask_pink==255).mean(axis=0)[::-1]
    front_coo = np.argwhere(mask_blue==255).mean(axis=0)[::-1]
    
    x_center = (front_coo[0] + back_coo[0])/2
    y_center = (front_coo[1] + back_coo[1])/2
    angle = np.arctan2(-front_coo[1]+back_coo[1], -front_coo[0]+back_coo[0])
    
    return x_center, y_center, angle

class ImageFeature(DTROS):

    def __init__(self, node_name):
        """
        Initialize the ImageFeature class.
        """

        # Duck name
        self.vehicle = os.environ['VEHICLE_NAME']

        # initialize the DTROS parent class
        # https://github.com/duckietown/dt-ros-commons/blob/daffy/packages/duckietown/include/duckietown/dtros/constants.py
        super(ImageFeature, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)

        self.rate = rospy.Rate(10)

        self.rectify_alpha = rospy.get_param("~rectify_alpha", 0.0)
        # camera info
        self._camera_parameters = None
        self._mapx, self._mapy = None, None

        # subscribed Topic
        # https://stackoverflow.com/questions/33559200/ros-image-subscriber-lag?rq=1
        self._cinfo_sub = rospy.Subscriber("/watchtower00/camera_node/camera_info",
            CameraInfo, self._cinfo_cb, queue_size=1)
        self.image_subscriber = rospy.Subscriber("/watchtower00/camera_node/image/compressed",
            CompressedImage, self.callback, queue_size=1, buff_size=2**24)

        # Publisher
        self.coordinates_pub = rospy.Publisher("/localization", Odometry, queue_size=1)
        if PUB_RECT:
            self.image_pub = rospy.Publisher("/watchtower00/image_rectified/compressed", CompressedImage, queue_size=1)
        # self.coordinates_pub_sync = rospy.Publisher("/watchtower00/localization_sync", Odometry, queue_size=1)
        # self.image_sync = rospy.Publisher("/watchtower00/image_sync", CompressedImage, queue_size=1)
        self.odom_broadcaster = tf.TransformBroadcaster()
        
        # Ts
        # self.image_subscriber_msg = message_filters.Subscriber("/watchtower00/camera_node/image/compressed", CompressedImage)
        # self.localization_subscriber_msg = message_filters.Subscriber("/watchtower00/localization", Odometry)

        # ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber_msg, self.localization_subscriber_msg], 20, 10)
        # ts.registerCallback(self.callback_sync)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")
        self.rate.sleep()

    def callback_sync(self, image_msg, localization_msg):
        self.coordinates_pub_sync.publish(localization_msg)
        self.image_sync.publish(image_msg)

    def _cinfo_cb(self, msg):
        """
        Callback for the camera_info topic, first step to rectification.

        :param msg: camera_info message
        """
        if VERBOSE :
            print("subscribed to /camera_info")
        # create mapx and mapy
        H, W = msg.height, msg.width
        # create new camera info
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)
        # find optimal rectified pinhole camera
        rect_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_model.K, self.camera_model.D, (W, H), self.rectify_alpha
        )
        # store new camera parameters
        self._camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])
        # create rectification map
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            self.camera_model.K, self.camera_model.D, None, rect_K, (W, H), cv2.CV_32FC1
        )
        try:
            self._cinfo_sub.shutdown()
        except BaseException:
            pass

    def callback(self, ros_data):
        """
        Callback function for subscribed topic.
        Get image and extract position and orientation of Duckiebot.
        Publish duckiebot position and orientation on /watchtower00/localization.

        :param ros_data: received image

        :type ros_data: sensor_msgs.msg.CompressedImage
        """
        if self._camera_parameters is None:
            return
        # make sure we have a rectification map available
        if self._mapx is None or self._mapy is None:
            return

        if VERBOSE :
            print(f'received image of type: "{ros_data.format}"' )

        # To CV
        np_arr = np.frombuffer(ros_data.data, 'u1')
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Rectify
        image_np = cv2.remap(image_np, self._mapx, self._mapy, cv2.INTER_NEAREST)
        if PUB_RECT:
            #### Create CompressedIamge ####
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
            # Publish new image
            self.image_pub.publish(msg)
        # Img has origin on top left, after the interpolation it will be rotated of 90 degrees, need to prevent that
        image_np = cv2.flip(image_np, 0)

        localized = False

        try:
            x, y, theta = get_car(image_np)
            if PLOT:
                print(x, y, theta)
                cv2.circle(image_np, [int(x),int(y)], 20, [0,0,255], -1)
                cv2.arrowedLine(image_np, [int(x),int(y)], [int(x-50*np.cos(theta)),int(y-50*np.sin(theta))], [0,255,0], 5)
            localized = True
        except ValueError:
            print("No lines found.")

        if PLOT:
            cv2.imshow('cv_img', image_np)
            cv2.waitKey(2)

        # Rotate and remove offset
        if rospy.has_param('scale_x'):
            scale_x = rospy.get_param('scale_x')
            x = x*scale_x
        else:
            raise ValueError("scale_x not found")
        if rospy.has_param('scale_y'):
            scale_y = rospy.get_param('scale_y')
            y = y*scale_y
        else:
            raise ValueError("scale_y not found")
        if rospy.has_param('offset_x'):
            offset_x = rospy.get_param('offset_x')
            x -= offset_x
        else:
            raise ValueError("offset_x not found")
        if rospy.has_param('offset_y'):
            offset_y = rospy.get_param('offset_y')
            y -= offset_y
        else:
            raise ValueError("offset_y not found")

        # DuckPose (not published)
        pose = DuckPose()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "/localization"
        if localized:
            pose.x = x
            pose.y = y
            pose.theta = theta
            pose.success = True
        else:
            pose.x = -1
            pose.y = -1
            pose.theta = -1
            pose.success = False


        # Odometry:
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        self.odom_broadcaster.sendTransform(
            (x, y, 0.),
            odom_quat,
            rospy.Time.now(),
            "base_link",
            "odom"
        )

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"

        # set the position
        odom.pose.pose = Pose(Point(x, y, 0.), Quaternion(*odom_quat))

        # set the velocity
        odom.child_frame_id = "base_link"
        odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

        self.coordinates_pub.publish(odom)


def main(args):
    '''Initializes and cleanup ros node'''
    map = get_map_client()
    # rospy.init_node('image_feature', anonymous=True)
    ic = ImageFeature(node_name='localizer')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down localization")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)