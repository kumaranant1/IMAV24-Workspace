#!/usr/bin/env python3
from __future__ import print_function


import sys
import rospy
import cv2
import numpy as np
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from std_msgs.msg import Int16  # For error/angle plot publishing
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



class Image_converter(object):

    def __init__(self):
        self.bridge = CvBridge()
        self.Kp = 0.112                 # Ku=0.14 T=6. PID: p=0.084,i=0.028,d=0.063. PD: p=0.112, d=0.084/1. P: p=0.07
        self.Ki = 0
        self.kd = 1
        self.integral = 0
        self.derivative = 0
        self.last_error = 0
        self.Kp_ang = 0.02             # Ku=0.04 T=2. PID: p=0.024,i=0.024,d=0.006. PD: p=0.032, d=0.008. P: p=0.02/0.01
        self.Ki_ang = 0
        self.kd_ang = 0
        self.integral_ang = 0
        self.derivative_ang = 0
        self.last_ang = 0
        self.was_line = 0
        self.line_side = 0
        self.line_back = 1
        self.landed = 0
        self.takeoffed = 0
        self.error = []
        self.angle = []
        self.fly_time = 0.0
        self.start = 0.0
        self.stop = 0.0
        self.velocity = 0.06



    # The following function is redefined for child class <LineFollower>
    def line_detect(self, cv_image):
        # Create a mask
        # cv_image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(cv_image, (130, 130, 130), (255, 255, 255))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=5)
        mask = cv2.dilate(mask, kernel, iterations=9)
        _, contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                angle = 90 + angle

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


            twist = Twist()
            twist.linear.x = self.velocity
            twist.linear.y = error_corr
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = ang_corr
            self.pub_vel.publish(twist)
            # print("angVal: ", twist.angular.z)

            ang = Int16()
            ang.data = angle
            self.pub_angle.publish(ang)

            err = Int16()
            err.data = error
            self.pub_error.publish(err)

        if len(contours_blk) == 0 and self.was_line == 1 and self.line_back == 1:
            twist = Twist()
            if self.line_side == 1:  # line at the right
                twist.linear.y = -0.05
                self.pub_vel.publish(twist)
            if self.line_side == -1:  # line at the left
                twist.linear.y = 0.05
                self.pub_vel.publish(twist)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(1) & 0xFF

    def isTakeoff(self, data):
        time.sleep(3.5)
        self.start = time.time()
        self.takeoffed = 1
        self.landed = 0

    def isLand(self, data):
        self.landed = 1
        self.takeoffed = 0
        self.stop = time.time()
        self.errorPlot()

    def errorPlot(self):
        meanError = np.mean(self.error)
        stdError = np.std(self.error)
        meanAngle = np.mean(self.angle)
        stdAngle = np.std(self.angle)
        self.fly_time = self.stop - self.start
        print("""
      /*---------------------------------------------
              meanError: %f[px],   stdError: %f[px]
              meanAngle: %f[deg],   stdAngle: %f[deg]
              Time: %f[sec], Velocity: %f[percent]
      ---------------------------------------------*/
      """ %(meanError, stdError, meanAngle, stdAngle, self.fly_time, self.velocity))

    # Zoom-in the image
    def zoom(self, cv_image, scale):
        height, width, _ = cv_image.shape
        # print(width, 'x', height)
        # prepare the crop
        centerX, centerY = int(height / 2), int(width / 2)
        radiusX, radiusY = int(scale * height / 100), int(scale * width / 100)

        minX, maxX = centerX - radiusX, centerX + radiusX
        minY, maxY = centerY - radiusY, centerY + radiusY

        cv_image = cv_image[minX:maxX, minY:maxY]
        cv_image = cv2.resize(cv_image, (width, height))

        return cv_image


    # Image processing @ 10 FPS
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv_image = self.zoom(cv_image, scale=20)
        cv_image = cv2.add(cv_image, np.array([-50.0]))
        # height, width, _ = cv_image.shape
        # print(width, 'x', height)
        if self.takeoffed and (not self.landed):
            self.line_detect(cv_image)
            self.land_detect(cv_image)


        cv2.putText(cv_image, "battery: " + str(self.battery) + "%", (570, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Image window", cv_image)
        # cv2.imshow("mask", mask)
        cv2.waitKey(1) & 0xFF


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    time.sleep(3)
    ic.cam_down()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
