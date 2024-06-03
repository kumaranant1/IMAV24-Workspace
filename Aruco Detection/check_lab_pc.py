import cv2 as cv
from cv2 import aruco
import numpy as np
import pyrealsense2
from realsense_depth import *
import argparse
# load in the calibration data
calib_data_path = "./calib_data/MultiMatrix.npz"

def reshape_np(arr_n):
    arr_new=np.zeros((1,1,3), dtype=np.float64)
    arr_new[0][0][0]=arr_n[0][0]
    arr_new[0][0][1]=arr_n[1][0]
    arr_new[0][0][2]=arr_n[2][0]
    return arr_new
print(reshape_np(np.array([[-2.91844982],[-0.16555862],[ 0.76643604]])))

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
        nada, R, t = cv.solvePnP(marker_points, c, mtx, distortion, False, cv.SOLVEPNP_IPPE_SQUARE)
        '''rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)'''
    #trash=reshape_np(nada)
    rvecs = reshape_np(R)
    tvecs = reshape_np(t)
    '''rvecs=reshape_np(np.array(rvecs))
    tvecs=reshape_np(np.array(tvecs))
    trash=reshape_np(np.array(trash))'''

    return rvecs, tvecs, nada


calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--size", type=float, default=14.2, help="Minimum confidence level of detection")
args = vars(ap.parse_args())
#a=float(input('marker size? (side kength in cm): '))
MARKER_SIZE = args["size"] # centimeters (measure your printed marker size)

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

param_markers = aruco.DetectorParameters()

# cap = cv.VideoCapture(0)
dc = DepthCamera()
while True:
    # ret, frame = cap.read()
    ret, depth_frame, frame = dc.get_frame()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dt=aruco.ArucoDetector(marker_dict,param_markers)
    marker_corners, marker_IDs, reject = dt.detectMarkers(
        gray_frame
    )
    if marker_corners:
        rVec, tVec, _ = my_estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )

        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
            frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Calculating the distance
            try:
                distance = np.sqrt(tVec[i][0][0] ** 2 + tVec[i][0][2] ** 2 + tVec[i][0][1] ** 2)
            except:
                distance = 0
            # Draw the pose of the marker2
            print(i)
            try:
                point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            except Exception as e:
                print(e)
            cv.putText(
                frame,
                f"id: {ids[0]} Dist: {distance}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            try:
                cv.putText(
                frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            except:
                pass
            print(type(rVec), rVec.shape, rVec)
            # print(ids, "  ", corners)
    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        cv.imwrite(f"image_9.png", frame)
# cap.release()
cv.destroyAllWindows()
