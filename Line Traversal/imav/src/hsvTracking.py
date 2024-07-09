#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import atan2, pow, sqrt, degrees, radians, sin, cos
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, TwistStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandTOL, CommandTOLRequest
from mavros_msgs.srv import CommandLong, CommandLongRequest
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from tf.transformations import euler_from_quaternion


# The following code does not utilizes PID for velocity control, it's experimental and may not work
# due to some missing or manipulated code scripts that are done by us only for purpose of experiment
################################################################################################################################################
# define global variables

current_yaw = 0.0 # this just intialized to store the current yaw of drone 
CENTRE_TOP = 100
CENTRE_BOTTOM = 380
CENTRE_LEFT = 165
CENTRE_RIGHT = 315

BOTTOM_CENTRE = np.array([480, 320])
VERT_VEL = 1 # tunable
YAW_VEL = 0.1 # tunable
HORI_VEL = 0.25 # tunable
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LOWER_BOUND = 350
minThreshold = 100
boundaryPoints = []
bottomPoints = []
EPSILON = 1e-8
for i in range(640):
    boundaryPoints.append([LOWER_BOUND, i])
    bottomPoints.append([LOWER_BOUND, i])
    boundaryPoints.append([0, i])
for i in range(480):
    boundaryPoints.append([i, 0])
    boundaryPoints.append([i, 640])
boundaryPoints = np.array(boundaryPoints)
bottomPoints = np.array(bottomPoints)

previousPos = TwistStamped()
previousPos.twist.linear.x = 0
previousPos.twist.linear.y = 0
currentPos = TwistStamped()
centre = (-1, -1)
################################################################################################################################################

# Function for Line tracking 
def lineTraversal(img : np.ndarray)->tuple:
    '''
    This function is used to find the 
    '''
    frame = img
    # print(frame.shape)
    # lumin = frame[:,:,0]*0.114 + frame[:,:,0]*0.587 + frame[:,:,0]*0.299 
    # lumin /= 255
    # grayScale = cv2.cvtColor(frame[:LOWER_BOUND, :, :], cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", grayScale)
    # cv2.imshow("Lum0", lumin)
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvFrame, np.array([55, 104, 85]), np.array([179, 255, 255]))
    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contourNum = 0 
    contourIntersection = np.zeros(len(contours))
    maxContour = 0 
    maxIndex = 0
    contours = np.array(contours)
    global centre
    if (len(contours)>=1):
        for contour in contours:
            contour = np.squeeze(contour, axis=1)
            contInt = np.intersect1d(contour, boundaryPoints)
            contourIntersection[contourNum] = contInt.shape[0]
            if (contourNum == 0):
                maxContour = contourIntersection[contourNum]
                maxIndex = contourNum
            elif (contourIntersection[contourNum] > maxContour):
                maxContour = contourIntersection[contourNum]
                maxIndex = contourNum
            contourNum += 1
        cv2.drawContours(frame, contours, maxIndex, GREEN, 4)
        centroid = np.mean(np.array(contours[maxIndex]), axis=0)
        centre = (int(centroid[0][0]), int(centroid[0][1]))
        cv2.circle(frame, centre, 4, RED, -1)
    else:
        centre = (-1, -1)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    cv2.waitKey(3)
    return centre

################################################################################################################################################

# the following code & showImage() is for HSV Tracking it will open tracker bars to set correct value for colors we want to track
def nothing(x)->None:
    pass
# cv2.namedWindow("Trackbars", 1)
# cv2.createTrackbar("Hue Lower", "Trackbars", 0, 179, nothing)
# cv2.createTrackbar("Hue Upper", "Trackbars", 179, 179, nothing)
# cv2.createTrackbar("Saturation Lower", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("Saturation Upper", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("Value Lower", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("Value Upper", "Trackbars", 255, 255, nothing)
def showImage(img)->None:
    frame = img
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_low = cv2.getTrackbarPos("Hue Lower", "Trackbars")
    hue_high = cv2.getTrackbarPos("Hue Upper", "Trackbars")
    sat_low = cv2.getTrackbarPos("Saturation Lower", "Trackbars")
    sat_high = cv2.getTrackbarPos("Saturation Upper", "Trackbars")
    val_low = cv2.getTrackbarPos("Value Lower", "Trackbars")
    val_high = cv2.getTrackbarPos("Value Upper", "Trackbars")
#or theta < -0.3:
    # Define the HSV range based on trackbar positions
    lower_range = np.array([hue_low, sat_low, val_low])
    upper_range = np.array([hue_high, sat_high, val_high])

    # Threshold the HSV image to get the mask
    mask = cv2.inRange(hsv, lower_range, upper_range)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Image", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    cv2.waitKey(3)
    
#######################################################0.0043667#########################################################################################

# Image function for processing the images
def image_callback(msg : Image)->np.ndarray:
    try:
        # Convert ROS Image message to OpenCV format
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # showImage(cv_image) # -> it's for getting HSV values for hsvTracking
        lineTraversal(cv_image)
        return cv_image

    except Exception as e:
        print(e)

################################################################################################################################################

# Function for generating the messages for cmd_velocity
def cmd_vel(vel_linx , vel_liny, vel_linz, vel_angx, vel_angy, vel_angz):
    vel_msg = TwistStamped()
    vel_msg.header = Header()
    desired_velocity = np.array([vel_linx, vel_liny, vel_linz])
    rotation_matrix = np.array([[np.cos(current_yaw), -np.sin(current_yaw), 0],
                                [np.sin(current_yaw), np.cos(current_yaw), 0],
                                [0, 0, 1]])
    transformed_velocity = np.dot(rotation_matrix, desired_velocity)
    vel_msg.header.stamp = rospy.Time.now()
    vel_msg.twist.linear.x = transformed_velocity[0]
    vel_msg.twist.linear.y = transformed_velocity[1]
    vel_msg.twist.linear.z = transformed_velocity[2]
    vel_msg.twist.angular.x = vel_angx
    vel_msg.twist.angular.y = vel_angy
    vel_msg.twist.angular.z = vel_angz
    return vel_msg

################################################################################################################################################

# Callback funcion for getting pose of Drone
def orientation_callback(msg : Pose)->None:
    global current_yaw
    orientation_q = msg.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, current_yaw) = euler_from_quaternion(orientation_list)

################################################################################################################################################

def inBox(point : tuple)->bool:
    if (point[0]<=CENTRE_RIGHT and point[0]>=CENTRE_LEFT and point[1]>=CENTRE_TOP and point[1]<=CENTRE_BOTTOM):
        return True

################################################################################################################################################

def retVel(point : float)->(float, float):
    global previousPos
    currentPos.twist.linear.x = point[0]
    currentPos.twist.linear.y = point[1]
    if(point[0] < 0 or point[1] < 0):
        return 0, 0
    xDiff = currentPos.twist.linear.x - previousPos.twist.linear.x
    yDiff = currentPos.twist.linear.y - previousPos.twist.linear.y
    # ratio = xDiff/yDiff
    if (currentPos.twist.linear.x < CENTRE_LEFT):
        horVel = HORI_VEL
    elif (currentPos.twist.linear.x > CENTRE_RIGHT):
        horVel = -HORI_VEL
    else:
        horVel = 0
    if (point[0] == -1 and point[1] == -1):
        vertVel = 0
    else:
        vertVel = VERT_VEL
    previousPos = currentPos
    return horVel, vertVel

###############################################################################################################################################
# Yaw angular velocity
# we are taking bottom absolute centre 
def retYaw(centre : tuple)->float:
    if(centre == (-1, -1)):
        return 0
    else:
        difference = np.array(centre)-BOTTOM_CENTRE
        theta = np.arctan(difference[1]/(difference[0]+EPSILON))
        if theta > 0.3:
            return YAW_VEL
        elif theta < -0.3:
            return -YAW_VEL
        return 0 


def main():
    # Initialize the ROS node
    rospy.init_node('line_traversal_node', anonymous=True)
    
    # Subscriber of Pose of Drone
    orientation_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, orientation_callback)
    
    # 
    pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
    # Subscriber for the image topic
    rospy.Subscriber("/webcam/image_raw", Image, image_callback)

    # Set the rate at which to publish messages
    rate = rospy.Rate(30)  # 30Hz
    
    # Main loop
    while not rospy.is_shutdown():
        # create the TwistStamped msg
        horVel, verVel = retVel(centre)
        yaw = retYaw(centre)
        # print(centre)
        vel_msg = cmd_vel(verVel, horVel, 0, 0, 0, yaw) # here set the required velocities to track
        # Publish the TwistStamped message
        pub.publish(vel_msg)
        # Process any other tasks here

        # Sleep to maintain the specified publishing rate
        rate.sleep()

if __name__=="__main__":
    main()