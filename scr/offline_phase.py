# OFF-LINE PHASE
# manual corner detection, loop for automatic + manual corner detection, camera calibration with the three runs

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import glob

def get_manual_corners(image: np.ndarray, pattern_size: tuple[int, int]):

# This function implements the manual detection of the corners by clicking on the four external inner corners of the chessboard and performing a linear interpolation
# it returns all the interpolated corner grid as np.array


    original = image.copy() # in case the user wants to reset 
    instance = image.copy()

    clicked_points = []

    def click_event(event: int, x: int, y: int, flags: int, params):
    
    # This function stores the four clicks of the external inner corners. it was implemented modifying the tutorial by open cv 

        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4: # if there are 4 clickes the next clicks are ignored
            
            clicked_points.append((x,y))
            cv2.circle(instance, (x,y), 5, (0, 0, 255), -1) # drawing a little red circel in the click aqrea 

            cv2.imshow("manual corners", instance) # showind the image updated 

    cv2.namedWindow("manual corners", cv2.WINDOW_NORMAL) 
    cv2.imshow("manual corners", instance)

    cv2.setMouseCallback("manual corners", click_event) # this also follows the quoted tutorial 

    while True: # get all the four points
        
        key = cv2.waitKey(20) & 0xFF # waiting for the keyboard to press something

        if key == 27: # in case of ESC, we exit the program
            cv2.destroyWindow("manual corners")
            return None 

        if key == ord('r'): # in case the user presser 'r', we reset the image 
            clicked_points.clear()
            instance[:] = original # resetting each instance pixel to the original 
            cv2.imshow("manual corners", instance)
        
        if len(clicked_points) == 4: # all the corners have been pressed
            break
    
    cv2.destroyWindow("manual corners") # once we collected all the corners, we can kill the window

    # ordering the points

    pts = np.array(clicked_points, dtype=np.float32)
    
    s = pts.sum(axis=1) # summing x + y for each couple 
    diff = pts[:, 0] - pts[:, 1]  # x - y for each couple 

    tl_idx = np.argmin(s)
    br_idx = np.argmax(s)
    tr_idx = np.argmax(diff)
    bl_idx = np.argmin(diff)

    tl = pts[tl_idx]
    br = pts[br_idx]
    tr = pts[tr_idx]
    bl = pts[bl_idx]

    # pattern_size = (cols, rows)
    cols, rows = pattern_size

    # interpolling the grid
    rows, cols = pattern_size

    points = []

    for row in range(pattern_size[1]):        # top → bottom
        v = row / (pattern_size[1] - 1)

        left = tl * (1 - v) + bl * v
        right = tr * (1 - v) + br * v

        for col in range(pattern_size[0]):    # left → right
            u = col / (pattern_size[0] - 1)

            p = left * (1 - u) + right * u
            points.append(p)
    
    corners = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    corners = corners.reshape(-1, 1, 2)
    print("TL:", tl, "TR:", tr, "BR:", br, "BL:", bl)
    return corners 



# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

pattern_size = (9, 6) # number of intern corners are squares -1, so for our chessboard is 9, 6
square_size = 0.022 # lenght of the squares converted to meters 

# creating an empty 3d matrix of the chessboard 
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32) 
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) 
objp *= square_size # here we scale for the square size (unlike the tutorial)

# loading and converting the images to gray scale, as suggested in the tutorial
images = glob.glob("data/training_images/*.jpg")
print("Found images:", images)

# loop to detect the corners of the training images (automatically or manually)
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
        #print(corners2.shape)
        if corners2 is not None: 
            corners2 = cv2.cornerSubPix(gray, corners2, (11,11), (-1,-1), criteria)  
        if corners2 is None:
            print("FAILED:", fname)
            continue
        else:
            print("OK:", fname)
    print(corners2[0], corners2[-1])    
    imgpoints.append(corners2)
    objpoints.append(objp)

    found = ret or (corners2 is not None)
    cv2.drawChessboardCorners(img, pattern_size, corners2, found)
    cv2.imshow('img', img)
    cv2.waitKey(500)
 
cv2.destroyAllWindows()

# testing
test = cv2.imread("data/test_image/test.jpg")
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

run1_obj, run1_img = [], []
run2_obj, run2_img = [], []
run3_obj, run3_img = [], []

auto_counter = 0
manual_counter = 0

for fname, objp_i, imgp_i in zip(images, objpoints, imgpoints):

    is_manual = "_manual" in fname

    #Run 1: all images
    run1_obj.append(objp_i)
    run1_img.append(imgp_i)

    #Run 2: first 5 auto + first 5 manual
    if is_manual and manual_counter < 5:
        run2_obj.append(objp_i)
        run2_img.append(imgp_i)
        manual_counter += 1

    if not is_manual and auto_counter < 5:
        run2_obj.append(objp_i)
        run2_img.append(imgp_i)
        auto_counter += 1

#Run 3: first 5 auto
auto_counter = 0
for fname, objp_i, imgp_i in zip(images, objpoints, imgpoints):
    is_manual = "_manual" in fname
    if not is_manual and auto_counter < 5:
        run3_obj.append(objp_i)
        run3_img.append(imgp_i)
        auto_counter += 1


# This function calibrates the camera and also provides an estimation of the intrinsic values (e.g. the standard deviation) for choice task 4
def do_calibration(name, obj, img):

    ret, mtx, dist, rvecs, tvecs, std_intr, std_ext, per_view_err = cv2.calibrateCameraExtended(obj, img, gray.shape[::-1], None, None)

    print(f"\n{name}")
    print("Images used:", len(obj))
    print("Camera matrix:\n", mtx)
    print("RMS reprojection error:", ret)
    print("Distortion:\n", dist.ravel())

    print("\nStandard deviation of intrinsic parameters:")
    print("fx std:", std_intr[0])
    print("fy std:", std_intr[1])
    print("cx std:", std_intr[2])
    print("cy std:", std_intr[3])

    np.savez(f"{name}.npz",
             cameraMatrix=mtx,
             distCoeffs=dist,
             rvecs=rvecs,
             tvecs=tvecs,
             stdIntrinsics=std_intr)
    print(f"Saved calibration to {name}.npz")


# calibrate for the three runs 
do_calibration("run1_all_images", run1_obj, run1_img)
do_calibration("run2_5auto_5manual", run2_obj, run2_img)
do_calibration("run3_5auto_only", run3_obj, run3_img)

# now we are gonna implement choice task 4

def calib_for_choice_4(obj_list, img_list, image_size):
    # this function returns in order the fxl fy, cx, cy of the performed calibration
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_list, img_list, image_size, None, None)
    return rms, K[0,0], K[1,1], K[0,2], K[1,2]


def subset_confidence(objpoints_all, imgpoints_all, image_size, subset_size=10, trials=50, seed=0):
    rng = np.random.default_rng(seed)
    n = len(objpoints_all)

    fx_list, fy_list, cx_list, cy_list, rms_list = [], [], [], [], []

    for _ in range(trials):
        idx = rng.choice(n, size=subset_size, replace=False)
        obj_sub = [objpoints_all[i] for i in idx]
        img_sub = [imgpoints_all[i] for i in idx]

        rms, fx, fy, cx, cy = calib_for_choice_4(obj_sub, img_sub, image_size)
        rms_list.append(rms)
        fx_list.append(fx); fy_list.append(fy); cx_list.append(cx); cy_list.append(cy)

    def summarize(x):
        x = np.array(x)
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
        }

    return {"subset_size": subset_size, "trials": trials, "rms": summarize(rms_list), "fx": summarize(fx_list), "fy": summarize(fy_list), "cx": summarize(cx_list), "cy": summarize(cy_list),}

    print("Choice 4: subset confidence = \n")

conf_25 = subset_confidence(objpoints, imgpoints, gray.shape[::-1], subset_size=25, trials=40, seed=1)
conf_10 = subset_confidence(objpoints, imgpoints, gray.shape[::-1], subset_size=10, trials=60, seed=2)
conf_5  = subset_confidence(objpoints, imgpoints, gray.shape[::-1], subset_size=5,  trials=80, seed=3)

for conf in [conf_25, conf_10, conf_5]:
    print("\nSubset size:", conf["subset_size"], "Trials:", conf["trials"])
    for k in ["fx","fy","cx","cy"]:
        s = conf[k]
        print(f"{k}: mean={s['mean']:.2f}, std={s['std']:.2f}, 95%CI=[{s['ci2.5']:.2f},{s['ci97.5']:.2f}]")
