import os
import sys
print("CWD:", os.getcwd())

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import glob
from scr.detection.manual_corners import get_manual_corners


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

pattern_size = (9, 6) # number of intern corners are squares -1, so for our chessboard is 9, 6
square_size = 0.017 # lenght of the squares converted to meters 

# creating an empty 3d matrix of the chessboard 
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32) 
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) 
objp *= square_size # here we scale for the square size (unlike the tutorial)

# loading and converting the images to gray scale, as suggested in the tutorial
images = glob.glob("data/training_images/*.jpg")
print("Found images:", images)


for fname in images:
    img = cv2.imread(fname)          
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # if found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    else:
        print(f"{fname}: manual annotation needed")
        corners2 = get_manual_corners(img, pattern_size)  
        if corners2 is not None:
            corners2 = cv2.cornerSubPix(gray, corners2, (11,11), (-1,-1), criteria)  

    imgpoints.append(corners2)
    objpoints.append(objp)

    found = ret or (corners2 is not None)
    cv2.drawChessboardCorners(img, pattern_size, corners2, found)
    cv2.imshow('img', img)
    cv2.waitKey(500)
 
cv2.destroyAllWindows()

# calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# testing
test = cv2.imread("data/test_image/test.jpeg")
gray_t = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
ret_t, corners_t = cv2.findChessboardCorners(gray_t, pattern_size, None)

# checking that the test was succesfull
if not ret_t:
    print("Test image: automatic corner detection FAILED")

corners_t = cv2.cornerSubPix(gray_t, corners_t, (11,11), (-1,-1), criteria)

# visualizing the final output

temp_test = test.copy()
cv2.drawChessboardCorners(temp_test, pattern_size, corners_t, True)
cv2.imwrite("test_corners_detected.png", temp_test)


cv2.imshow("test corners detected", temp_test)
cv2.waitKey(0)
cv2.destroyAllWindows()