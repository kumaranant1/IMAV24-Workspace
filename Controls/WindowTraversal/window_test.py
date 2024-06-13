#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import sqrt
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

bridge = CvBridge()

# HSV color range for detecting red color
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Load calibration data
calib_data_path = "calibration.npz"
calib_data = np.load(calib_data_path)
camera_matrix = calib_data["camera_matrix"]
distortion_coeffs = calib_data["dist_coeffs"]

MARKER_SIZE = 100
current_state = State()
current_pose = PoseStamped()

# Callback to update the current state
def state_cb(msg):
    global current_state
    current_state = msg

# Callback to update the current pose
def pose_cb(msg):
    global current_pose
    current_pose = msg

# Function to calculate the rotation matrix and translation vector
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    return R, t, nada

# Function to move the drone to the center of the window
def move_drone_to_window_center(tVec, pub):
    global current_pose
    x_tolerance = 0.1
    y_tolerance = 0.1
    z_tolerance = 0.1

    new_pose = PoseStamped()
    new_pose.header = Header()
    new_pose.header.stamp = rospy.Time.now()
    new_pose.pose = current_pose.pose

    if abs(tVec[0][0][0]) > x_tolerance:
        new_pose.pose.position.x -= tVec[0][0][0] / 100

    if abs(tVec[0][0][1]) > y_tolerance:
        new_pose.pose.position.y -= tVec[0][0][1] / 100

    if abs(tVec[0][0][2]) > z_tolerance:
        new_pose.pose.position.z -= tVec[0][0][2] / 100

    pub.publish(new_pose)

# Main processing function
def check():
    global cam_feed, current_pose
    cam_feed = None

    rospy.init_node('offb_node_py')

    rospy.Subscriber("/webcam/image_raw", Image, guessCallback)
    rospy.Subscriber("/mavros/state", State, state_cb)
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb)

    local_pos_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    rate = rospy.Rate(20)

    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    pose = PoseStamped()
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 2

    for i in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    while not rospy.is_shutdown():
        if current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
            if set_mode_client.call(offb_set_mode).mode_sent:
                rospy.loginfo("OFFBOARD enabled")
            last_req = rospy.Time.now()
        else:
            if not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                if arming_client.call(arm_cmd).success:
                    rospy.loginfo("Vehicle armed")
                last_req = rospy.Time.now()

        if cam_feed is not None:
            frame = cv2.flip(cam_feed, -1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 | mask2
            contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            canvas = np.zeros_like(frame)

            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    parent_index = hierarchy[0][i][3]
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    epsilon_p = 0.04 * cv2.arcLength(contours[parent_index], True)
                    approx_p = cv2.approxPolyDP(contours[parent_index], epsilon_p, True)

                    if len(approx) == 4 and len(approx_p) == 4:
                        contour_rect = cv2.boundingRect(contour)
                        parent_rect = cv2.boundingRect(contours[parent_index])
                        if (contour_rect[0] > parent_rect[0] and
                            contour_rect[1] > parent_rect[1] and
                            contour_rect[0] + contour_rect[2] < parent_rect[0] + parent_rect[2] and
                            contour_rect[1] + contour_rect[3] < parent_rect[1] + parent_rect[3]):

                            area = cv2.contourArea(contour)
                            if area > 10000:
                                cv2.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)
                                approx = reorder(approx)
                                approx_p = reorder(approx_p)
                                rVec, tVec, _ = my_estimatePoseSingleMarkers(
                                    (approx.astype(np.float32),), MARKER_SIZE, camera_matrix, distortion_coeffs)
                                try:
                                    distance = sqrt(tVec[0][0][0] ** 2 + tVec[0][0][2] ** 2 + tVec[0][0][1] ** 2)
                                except:
                                    distance = 0

                                move_drone_to_window_center(tVec, local_pos_pub)

            processed_frame = cv2.addWeighted(frame, 1, canvas, 1, 0)
            cv2.imshow("processed_frame", processed_frame)
            cv2.waitKey(1)

        local_pos_pub.publish(pose)
        rate.sleep()

    cv2.destroyAllWindows()

def guessCallback(data):
    global cam_feed
    cam_feed = bridge.imgmsg_to_cv2(data, "bgr8")

if __name__ == '__main__':
    try:
        check()
    except rospy.ROSInterruptException:
        pass
