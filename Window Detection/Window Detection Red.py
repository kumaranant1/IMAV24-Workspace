import cv2
import numpy as np

# Define the range for red color in HSV
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])


#to get the +ve/-ve of x & y while determining co-ordinates of markers
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



# Initialize video capture from webcam (you can change 0 to another number for different camera)
cap = cv2.VideoCapture('Virtual_red_window.avi')




# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Virtual Red output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
#inp = cv2.VideoWriter('input_box.avi', fourcc, 20.0, (640, 480))


#set length of the window (not used yet as square is being used)
rectangle_width_cm = 19
rectangle_height_cm = 17




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


#loading the camra parameters
calib_data_path = "calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)

camera_matrix = calib_data["camMatrix"]
distortion_coeffs = calib_data["distCoef"]


'''def estimate_distance_to_object(camera_matrix, distortion_coeffs, corner_points, object_width_cm):
    # Undistort corner points using camera calibration data
    undistorted_points = cv2.undistortPoints(corner_points, camera_matrix, distortion_coeffs)

    # Assume the rectangle is parallel to the image plane, so distance is simply focal length * object_width / width_in_pixels
    distance = camera_matrix[0, 0] * object_width_cm / (corner_points[0][0][0] - corner_points[1][0][0])

    return distance


def perspective_transform(image, src_points, dst_points):
    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply perspective transformation to the image
    transformed_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    return transformed_image

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
'''

#a=float(input('marker size? (side kength in cm): '))
MARKER_SIZE = 34


param_markers = None




frame_c=0



while True:

    ret, frame = cap.read()
    frame_c+=1
    if frame_c%10!=0:
        continue

    if not ret:
        print("Error: Failed to capture frame.")
        break
    #inp.write(frame)
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for blue color in HSV
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    #red_mask = mask2

    # Find contours in the mask with hierarchy
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

                            # Get the coordinates of the vertex in marker coordinate system
                            '''marker_vertex_coords = approx

                                                # Convert to homogeneous coordinates
                                                marker_vertex_coords_homogeneous = np.append(approx[0], 1)
                                                #print(marker_vertex_coords_homogeneous.shape, marker_vertex_coords_homogeneous)
                                                # Transform vertex coordinates to camera coordinate system
                                                camera_coords = np.dot(cv2.Rodrigues(rVec[0])[0].flatten()[:3], marker_vertex_coords_homogeneous) + tVec[0].flatten()
                                                # Print camera coordinates relative to the vertex
                                                #print("Camera coordinates relative to ArUco marker vertex:", camera_coords)'''
                            '''cv2.putText(frame,
                                    str(corner_camera),
                                    (int(approx[1, 0, 0]), int(approx[1, 0, 1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)'''


    # Combine the original frame with the highlighted contours
    processed_frame = cv2.addWeighted(frame, 1, canvas, 1, 0)

    # Display both original and processed frames side by side
    side_by_side = np.hstack((frame, processed_frame))
    cv2.imshow('Detected', processed_frame)

    out.write(frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
