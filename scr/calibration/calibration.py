import cv2 
import numpy as np
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

pattern_size = (9, 6) # number of intern corners are squares -1, so for our chessboard is 9, 6
square_size = 0.021 # lenght of the squares converted to meters 

# creating an empty 3d matrix of the chessboard 
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32) 
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) 
objp *= square_size # here we scale for the square size (unlike the tutorial)

# loading and converting the images to gray scale, as suggested in the tutorial
images = glob.glob("data/training_images/*.jpg")

for fname in images:
    img = cv2.imread(fname)          
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # if found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)