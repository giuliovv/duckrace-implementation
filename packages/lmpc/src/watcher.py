#!/usr/bin/env python

"""Detects duckiebots from the watchtower using their unique colors.

Publish on watchtower00/localization a DuckPose with the car coordinates.
Publish using dt_communication.
"""
__author__ =  'Giulio Vaccari <giulio.vaccari at mail.polimi.it>'
__version__=  '0.1'
__license__ = 'MIT'
# Python libs
import sys, os, copy

# numpy and scipy
import numpy as np

# OpenCV
import cv2

# Ros libraries
import rospy
import tf
from image_geometry import PinholeCameraModel

# Ros Messages
from sensor_msgs.msg import CameraInfo, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import CameraInfo, CompressedImage
from localization.msg import DuckPose

# Duckie stuff
from dt_communication_utils import DTCommunicationGroup
from duckietown.dtros import DTROS, NodeType

# from std_msgs.msg import String
group_img = DTCommunicationGroup('image', CompressedImage)
group_info = DTCommunicationGroup('camera_info', CameraInfo)

VERBOSE=False
PUB_RECT=False
PUB_ROS=False

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

    hsv_color1_blue = np.array([80, 180, 110])
    hsv_color2_blue = np.array([200, 250, 250])
    mask_blue = cv2.inRange(img_hsv, hsv_color1_blue, hsv_color2_blue)

    hsv_color1_pink = np.array([150, 50, 100])
    hsv_color2_pink = np.array([200, 100, 250])
    mask_pink = cv2.inRange(img_hsv, hsv_color1_pink, hsv_color2_pink)
    
    back_coo = np.argwhere(mask_pink==255).mean(axis=0)[::-1]
    front_coo = np.argwhere(mask_blue==255).mean(axis=0)[::-1]
    
    x_center = (front_coo[0] + back_coo[0])/2
    y_center = (front_coo[1] + back_coo[1])/2

    if np.isnan(x_center):
        print(front_coo[0], back_coo[0])
    # In the angle computation x and y are inverted because of the image coordinate system
    angle = np.arctan2(-front_coo[0]+back_coo[0], -front_coo[1]+back_coo[1])
    
    return x_center, y_center, angle

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

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


        rospy.loginfo('[Watcher]: Waiting for parameter server...')
        while not rospy.has_param('scale_x'):
            self.rate.sleep()
        self.scale_x = rospy.get_param('scale_x', 0.00345041662607739)
        self.scale_y = rospy.get_param('scale_y', 0.005417244522218992)
        rospy.loginfo('[Watcher]: Got params.')

        self.coordinates_dt_publish = rospy.Publisher("~/localization", DuckPose, queue_size=1)

        # subscribed Topic
        group_info.Subscriber(self._cinfo_cb)
        group_img.Subscriber(self._img_cb)

        # Publisher
        if PUB_ROS:
            self.coordinates_pub = rospy.Publisher("~/localization_odom", Odometry, queue_size=1)
            self.odom_broadcaster = tf.TransformBroadcaster()
        if PUB_RECT:
            self.image_pub = rospy.Publisher("~/image_rectified/compressed", CompressedImage, queue_size=1)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")
        # self.rate.sleep()

    def _cinfo_cb(self, msg, header):
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

    def _img_cb(self, ros_data, header):
        """
        Callback function for subscribed topic.
        Get image and extract position and orientation of Duckiebot.
        Publish duckiebot position and orientation on /watchtower00/localization.

        :param ros_data: received image

        :type ros_data: sensor_msgs.msg.CompressedImage
        """
        # make sure we have a rectification map available
        if self._camera_parameters is None or self._mapx is None or self._mapy is None:
            return

        # To CV
        np_arr = np.frombuffer(ros_data.data, 'u1')
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Rectify
        remapped = cv2.remap(image_np, self._mapx, self._mapy, cv2.INTER_NEAREST)
        # White balance
        img = white_balance(remapped)
        # Cut image
        W, H = img.shape[1], img.shape[0]
        img = img[int(H*0.15):int(H*0.78), int(W*0.2):int(W*0.8)]
        # MEMO: Img has origin on top left, after the interpolation it will be rotated of 90 degrees, no need to rotate it back

        localized = False

        try:
            x, y, theta = get_car(img)
            localized = True
        except ValueError:
            localized = False
            print("No lines found.")
        
        if np.isnan(x) or np.isnan(y) or np.isnan(theta):
            print("No lines found.")
            localized = False

        # Because of the different methods between map creation and localization x and y are flipped
        x, y = y, x

        if VERBOSE:
            print("Pixel: x: ", x, "y: ", y)


        if PUB_RECT:
            # NB It is correct wrt to the map, not the image thus we need to flip along y
            img_to_be_pulished = copy.deepcopy(img)
            #### Create CompressedImage ####
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            img_to_be_pulished = cv2.rotate(img_to_be_pulished, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.circle(img_to_be_pulished,(int(x), int(img.shape[1]-y)), 25, (0,255,0))
            msg.data = np.array(cv2.imencode('.jpg', img_to_be_pulished)[1]).tobytes()
            # Publish new image
            self.image_pub.publish(msg)

        # Resize and remove offset
        x = x*self.scale_x
        y = y*self.scale_y
        # offset_x = rospy.get_param('offset_x', 0.3598180360213129)
        # x -= offset_x
        # offset_y = rospy.get_param('offset_y', 0.07411439522846053)
        # y -= offset_y

        # if not rospy.has_param('offset_x'):
        #     print("[STREAM_TO_BOT] params not found")

        # DuckPose
        pose = DuckPose()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "watchtower00/localization"
        pose.x = x
        pose.y = y
        pose.theta = theta
        pose.success = localized

        print(f"[Watcher]: publishing x:{x}, y:{y}, theta:{np.rad2deg(theta)}")
        self.coordinates_dt_publish.publish(pose)

        # Odometry:
        if PUB_ROS:
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

def main():
    '''Initializes and cleanup ros node'''
    print("[Watcher]: Starting...")
    ic = ImageFeature(node_name='watcher')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("[Watcher]: Shutting down...")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()