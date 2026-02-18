# ON-LINE PHASE 

import numpy as np
import cv2

#loading the data
data_1 = np.load("run1_all_images.npz")
K_1 = data_1["cameraMatrix"]
dist_1 = data_1["distCoeffs"]

# detecting corners automatically for the test image 
pattern_size = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

img = cv2.imread("data/test_image/test.jpg")
copy = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

if not ret:
    raise RuntimeError("Test image: automatic corner detection failed") 

corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

# getting the camera extrinsic for run 1
square_size = 0.022
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# computing the axis lenght and building the axis 
axis_lenght = 3 * square_size
axis = np.float32([[0, 0, 0], [axis_lenght,0,0], [0,axis_lenght,0], [0,0,-axis_lenght]]).reshape(-1,3)

ok, rvec, tvec = cv2.solvePnP(objp, corners, K_1, dist_1)
# projecting the 3d points on the image
imgpts, jac = cv2.projectPoints(axis, rvec, tvec, K_1, dist_1)

def draw_cube(img, imgpts):
    # function got from the tutorial to draw the axis 
    imgpts = imgpts.reshape(-1, 2).astype(int)
    origin = tuple(imgpts[0])
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (255,0,0), 5)
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 5)
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (0,0,255), 5)
    return img

copy = draw_cube(img, imgpts)
cv2.imshow('drawn image',copy)
k = cv2.waitKey(0) & 0xFF




print("tvec:", tvec.ravel())


