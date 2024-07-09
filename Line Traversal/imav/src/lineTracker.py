#!/usr/bin/env python3
from LFD.image_converter import Image_converter
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import sys
import rospy
import cv2
import numpy as np
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from std_msgs.msg import Int16  # For error/angle plot publishing
# from bebop_msgs.msg import CommonCommonStateBatteryStateChanged  # For battery percentage
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, TwistStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# import tf2_ros
# import tf2_geometry_msgs

class LineFollower(Image_converter):
    def __init__(self):
        super().__init__()
        self.pub_vel = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self.image_sub = rospy.Subscriber('/webcam/image_raw', Image, self.callback)
        self.orientation_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.orientation_callback)
        self.velocity = 0.3
        self.current_yaw = 0
    
    def orientation_callback(self, msg : Pose)->None:
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, self.current_yaw) = euler_from_quaternion(orientation_list)
    
    def callback(self, msg : Image)->np.ndarray:
        try:
            # Convert ROS Image message to OpenCV format
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print("SHit")
            print(e)
            
        # showImage(cv_image) # -> it's for getting HSV values for hsvTracking
        self.line_detect(cv_image)
        cv2.imshow("Image", cv_image)
        cv2.waitKey(1)
        return cv_image
    
        
    def line_detect(self, cv_image):
        # Create a mask
        # cv_image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        hsvFrame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvFrame,  np.array([55, 104, 85]), np.array([179, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=5)
        mask = cv2.dilate(mask, kernel, iterations=9)
        contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blk.sort(key=cv2.minAreaRect)
        if len(contours_blk) > 0 and cv2.contourArea(contours_blk[0]) > 5000:
            self.was_line = 1
            blackbox = cv2.minAreaRect(contours_blk[0])
            (x_min, y_min), (w_min, h_min), angle = blackbox
            if angle < -45:
                angle = 90 + angle
            if w_min < h_min and angle > 0:
                angle = (90 - angle) * -1
            if w_min > h_min and angle < 0:
                angle = (90 + angle) * -1

            setpoint = cv_image.shape[1] / 2
            error = int(x_min - setpoint)
            self.error.append(error)
            self.angle.append(angle)
            normal_error = float(error) / setpoint

            if error > 0:
                self.line_side = 1  # line in right
            elif error <= 0:
                self.line_side = -1  # line in left

            self.integral = float(self.integral + normal_error)
            self.derivative = normal_error - self.last_error
            self.last_error = normal_error


            error_corr = -1 * (self.Kp * normal_error + self.Ki * self.integral + self.kd * self.derivative)  # PID controler
            # print("error_corr:  ", error_corr, "\nP", normal_error * self.Kp, "\nI", self.integral* self.Ki, "\nD", self.kd * self.derivative)

            angle = int(angle)
            normal_ang = float(angle) / 90

            self.integral_ang = float(self.integral_ang + angle)
            self.derivative_ang = angle - self.last_ang
            self.last_ang = angle

            ang_corr = -1 * (self.Kp_ang * angle + self.Ki_ang * self.integral_ang + self.kd_ang * self.derivative_ang)  # PID controler

            box = cv2.boxPoints(blackbox)
            box = np.int0(box)

            cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 3)

            cv2.putText(cv_image, "Angle: " + str(angle), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

            cv2.putText(cv_image, "Error: " + str(error), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.line(cv_image, (int(x_min), 200), (int(x_min), 250), (255, 0, 0), 3)
            
            desired_velocity = np.array([self.velocity, error_corr, 0])
            rotation_matrix = np.array([[np.cos(self.current_yaw), -np.sin(self.current_yaw), 0],
                                [np.sin(self.current_yaw), np.cos(self.current_yaw), 0],
                                [0, 0, 1]])
            transformed_velocity = np.dot(rotation_matrix, desired_velocity)
            
            cmd_vel = TwistStamped()
            cmd_vel.header.stamp = rospy.Time.now()
            cmd_vel.twist.linear.x = transformed_velocity[0]
            cmd_vel.twist.linear.y = transformed_velocity[1]
            cmd_vel.twist.linear.z = transformed_velocity[2]
            cmd_vel.twist.angular.x = 0
            cmd_vel.twist.angular.y = 0
            cmd_vel.twist.angular.z = ang_corr
            
            self.pub_vel.publish(cmd_vel)
            # print("angVal: ", twist.angular.z)

        if len(contours_blk) == 0 and self.was_line == 1 and self.line_back == 1:
            cmd = TwistStamped()
            cmd.header = Header()
            cmd.header.stamp = rospy.Time.now()
            if self.line_side == 1:  # line at the right
                cmd.twist.linear.y = -0.05
                self.pub_vel.publish(cmd)
            if self.line_side == -1:  # line at the left
                cmd.twist.linear.y = 0.05
                self.pub_vel.publish(cmd)
                
        # cv2.imshow("mask", mask)
        # cv2.waitKey(1) & 0xFF

def main():
    rospy.init_node('line_traversal_node', anonymous=True)
    ic = LineFollower()
    time.sleep(3)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()