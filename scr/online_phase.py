# ON-LINE PHASE 

import numpy as np
import cv2

#loading the data
calibration_files = [
    "run1_all_images.npz",
    "run2_5auto_5manual.npz",
    "run3_5auto_only.npz"
]

for calib_file in calibration_files:
    print("\n=== Running:", calib_file, "===")

    data = np.load(calib_file)
    K_1 = data["cameraMatrix"]
    dist_1 = data["distCoeffs"]


    # detecting corners automatically for the test image 
    pattern_size = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img = cv2.imread("data/test_image/test.jpg")
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
    print("tvec:", tvec.ravel())

    # projecting the 3d points on the image
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, K_1, dist_1)

    def draw(img, imgpts):
        # function got from the tutorial to draw the axis 
        imgpts = imgpts.reshape(-1, 2).astype(int)
        origin = tuple(imgpts[0])
        img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (255,0,0), 5)
        img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 5)
        img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (0,0,255), 5)
        return img

    copy = draw(img, imgpts)
    cv2.imshow(f'drawn image {calib_file}',copy)
    k = cv2.waitKey(0) & 0xFF

    # now draweing the cube

    cube_size = 2 * square_size # we decide to do the cube of two swares as in the example output 

    # we follow the tutorial for this part

    cube = np.float32([[0,0,0], [0,cube_size,0], [cube_size,cube_size,0], [cube_size,0,0],
                    [0,0,-cube_size],[0,cube_size,-cube_size],[cube_size,cube_size,-cube_size],[cube_size,0,-cube_size] ])

    imgpts_cube, _ = cv2.projectPoints(cube, rvec, tvec, K_1, dist_1)

    def draw_cube(img, imgpts):
        # function to draw the cube on the correnspondent point
        imgpts = imgpts.reshape(-1, 2).astype(int)
        # bottom face
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)
        # pillars
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
        # # top face
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
        return img

    out = img.copy()
    out = draw(out, imgpts)
    out = draw_cube(out, imgpts_cube)

    cv2.imshow(f"axes + cube {calib_file}", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # now drawing a polygon on the top face of the cube

    top_face_img = imgpts_cube[4:8].reshape(-1, 2).astype(np.int32)

    top_face_obj = cube[4:8]
    top_center_obj = top_face_obj.mean(axis=0) # get a central point of the top of the cube 

    R, _ = cv2.Rodrigues(rvec)  
    Xw = top_center_obj.reshape(3, 1) 
    Xc = R @ Xw + tvec

    dist_m = float(np.linalg.norm(Xc))

    print("top-center camera coords:", Xc.ravel())
    print("distance (m):", dist_m)

    V = int(np.clip(255 * (1.0 - dist_m / 4.0), 0, 255))
    n_w = np.array([0, 0, 1]).reshape(3,1)
    n_c = R @ n_w

    z_cam = np.array([0,0,1]).reshape(3,1)

    dot = (n_c.T @ z_cam).item()
    cos_theta = dot / (np.linalg.norm(n_c) * np.linalg.norm(z_cam))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    cos_theta = abs(cos_theta)   # robustness
    theta = float(np.degrees(np.arccos(cos_theta)))


    H = int(np.clip(179 * (1.0 - theta / 45.0), 0, 179))

    S = 255

    hsv_color = np.uint8([[[H, S, V]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0,0]

    cv2.fillConvexPoly(out, top_face_img, tuple(int(c) for c in bgr_color))


    center_img, _ = cv2.projectPoints(top_center_obj.reshape(1,3), rvec, tvec, K_1, dist_1)
    center_img = center_img.reshape(2).astype(int)

    cv2.circle(out, tuple(center_img), 6, (0,0,255), -1)

    cv2.putText(out,
                f"{dist_m:.2f} m",
                (center_img[0] + 10, center_img[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2)

    cv2.imshow(f"final_{calib_file}", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    






