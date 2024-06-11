#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import atan2, pow, sqrt, degrees, radians, sin, cos
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandTOL, CommandTOLRequest
from mavros_msgs.srv import CommandLong, CommandLongRequest
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Virtual Red Window Detection.avi', fourcc, 20.0, (640, 480))

#pub1 = rospy.Publisher('/result', Image, queue_size=10)
bridge = CvBridge()

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])



#Load calibration data
calib_data_path = "calibration.npz"
calib_data = np.load(calib_data_path)

camera_matrix = calib_data["camera_matrix"]
distortion_coeffs = calib_data["dist_coeffs"]

#print("Camera Matrix:\n", camera_matrix)



def sign_marker(x):
    if x==0:
        m,n=-1,1
    elif x == 1:
        m, n = 1, 1
    elif x == 2:
        m, n = 1, -1
    elif x == 3:
        m, n = -1, -1

    return (m,n)

#to get least greatest integer
def absl(x):
    x=x//1
    try:
        return abs(int(x))
    except:
        return 0

#to have constant rule, which corner will represent which index
def reorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2), np.int32)
    add=myPoints.sum(1)

    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[2] = myPoints[np.argmax(add)]
    diff=np.diff(myPoints, axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[3]=myPoints[np.argmax(diff)]

    return myPointsNew

# Function to check if a contour is a quadrilateral (polygon with four sides)
def is_quadrilateral(contour):
    return len(contour) == 4



def reshape_np(arr_n):
    arr_new=np.zeros((1,1,3), dtype=np.float64)
    arr_new[0][0][0]=arr_n[0][0]
    arr_new[0][0][1]=arr_n[1][0]
    arr_new[0][0][2]=arr_n[2][0]
    return arr_new




#it is for a square right now, will make it for rectangles soon considering 2 dimensiosn - length and breadth
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    rvecs = reshape_np(R)
    tvecs = reshape_np(t)
    return rvecs, tvecs, nada
frame_count=0

MARKER_SIZE=100
param_markers = None	
	
def check():
	frame_count=0
	global cam_feed
	cam_feed = None
	rospy.init_node('cam_feed_node')
	rospy.Subscriber("/webcam/image_raw", Image, guessCallback)
	#rospy.sleep(1)
	
	while not rospy.is_shutdown():
		#result = compareImage(imgMsg,hidden_image)
		if cam_feed is not None:
			
			frame = cv2.flip(cam_feed, -1)
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        	
			mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
			mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
			red_mask = mask1 | mask2          


			contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			# Create a black canvas to draw contours
			canvas = np.zeros_like(frame)

			# Loop through each contour with hierarchy information
			for i, contour in enumerate(contours):
				# Check if contour has hierarchy information
				if hierarchy[0][i][3] != -1:
					# Get the parent contour index
					parent_index = hierarchy[0][i][3]

					epsilon = 0.04 * cv2.arcLength(contour, True)
					approx = cv2.approxPolyDP(contour, epsilon, True)

					epsilon_p = 0.04 * cv2.arcLength(contours[parent_index], True)
					approx_p = cv2.approxPolyDP(contours[parent_index], epsilon_p, True)

					# Check if both contour and its parent are quadrilaterals
					if len(approx) == 4 and len(approx_p) == 4:
						# Get the bounding rectangle of the contour and its parent
						contour_rect = cv2.boundingRect(contour)
						parent_rect = cv2.boundingRect(contours[parent_index])

						# Check if contour is inside the parent contour (rectangle inside rectangle)
						if (contour_rect[0] > parent_rect[0] and
								contour_rect[1] > parent_rect[1] and
								contour_rect[0] + contour_rect[2] < parent_rect[0] + parent_rect[2] and
								contour_rect[1] + contour_rect[3] < parent_rect[1] + parent_rect[3]):

							# Filter contours based on area
							area = cv2.contourArea(contour)
							if area > 10000:
								cv2.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)
								object_width_cm = 19.2  # Width of the rectangle in cm (change according to your object)
								#distance = estimate_distance_to_object(camera_matrix, distortion_coeffs, approx.astype(np.float32),object_width_cm)
								if True:
									approx = reorder(approx)
									approx_p = reorder(approx_p)
									rVec, tVec, _ = my_estimatePoseSingleMarkers(
										(approx.astype(np.float32),), MARKER_SIZE, camera_matrix, distortion_coeffs)

								#using tVec to calculate distance of window from camera
								try:
									distance = np.sqrt(tVec[0][0][0] ** 2 + tVec[0][0][2] ** 2 + tVec[0][0][1] ** 2)
								except:
									distance = 0
								#cv2.putText(frame, str(absl(distance)), (int(approx[0][0][0]), int(approx[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
								cv2.putText(frame, str(int(tVec[0][0][0]))+","+ str(int(tVec[0][0][1]))+ ","+str(int(tVec[0][0][2])) , (int(np.mean(approx[:,0,0])),int(np.mean(approx[:,0,1]))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
								point = cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rVec[0], tVec[0], 4, 4)

								rot_mat,_ = cv2.Rodrigues(rVec[0][0])
								#print(tVec, rVec)

								#iterating through each of the 4 corners
								for i in range(4):

									#marking corners according to centre of the square
									(m,n)= sign_marker(i)
									corner_marker = np.array([m*MARKER_SIZE / 2, n*MARKER_SIZE / 2, 0])

									# Transform corner from marker coordinate system to camera coordinate system
									corner_camera = np.dot(rot_mat, corner_marker ) + tVec


									cv2.putText(frame, str(int(corner_camera[0][0][0])) + "," + str(int(corner_camera[0][0][1])) + "," + str(
										int(corner_camera[0][0][2])),(int(approx[i, 0, 0]), int(approx[i, 0, 1])),
												cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

									

			# Combine the original frame with the highlighted contours
			processed_frame = cv2.addWeighted(frame, 1, canvas, 1, 0)



			cv2.imshow("processed_frame", processed_frame)
                        
                        
			cv2.waitKey(1)
			#print(cam_feed.shape)
			frame_count+=1
			if frame_count%10==0:
				#continue
				out.write(processed_frame)
			
			
	
	out.release()
	cv2.destroyAllWindows()

def guessCallback(data):
	global cam_feed
	cam_feed = bridge.imgmsg_to_cv2(data,"bgr8")

		
	
if __name__ == '__main__':
	try:
		check()
	except rospy.ROSInterruptException:
		pass











