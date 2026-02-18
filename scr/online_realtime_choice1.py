import cv2
import numpy as np

CALIB_FILE = "run1_all_images.npz"
CAMERA_INDEX = 0

def draw_cube(img, imgpts):
    imgpts = imgpts.reshape(-1, 2).astype(int)

    #bottom
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0,255,0), 2)

    #pillars
    for i, j in zip(range(4), range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0), 2)

    #top
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0,0,255), 2)

    return img


def color_top_face(img, cube, imgpts, rvec, tvec, K, dist):

    top_face_img = imgpts[4:8].reshape(-1,2).astype(np.int32)
    top_face_obj = cube[4:8]
    top_center_obj = top_face_obj.mean(axis=0)

    R, _ = cv2.Rodrigues(rvec)
    Xc = R @ top_center_obj.reshape(3,1) + tvec
    dist_m = float(np.linalg.norm(Xc))

    V = int(np.clip(255 * (1.0 - dist_m / 4.0), 0, 255))

    n_w = np.array([0,0,1]).reshape(3,1)
    n_c = R @ n_w

    z_cam = np.array([0,0,1]).reshape(3,1)

    dot = (n_c.T @ z_cam).item()
    cos_theta = dot / (np.linalg.norm(n_c) * np.linalg.norm(z_cam))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = float(np.degrees(np.arccos(abs(cos_theta))))

    H = int(np.clip(179 * (1.0 - theta / 45.0), 0, 179))
    S = 255

    hsv_color = np.uint8([[[H, S, V]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0,0]

    cv2.fillConvexPoly(img, top_face_img, tuple(int(c) for c in bgr_color))

    #draw center dot
    center_img, _ = cv2.projectPoints(
        top_center_obj.reshape(1,3),
        rvec, tvec, K, dist
    )
    center_img = center_img.reshape(2).astype(int)

    cv2.circle(img, tuple(center_img), 5, (0,0,255), -1)

    cv2.putText(
        img,
        f"{dist_m:.2f} m",
        (center_img[0]+10, center_img[1]-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    return img


def main():

    data = np.load(CALIB_FILE)
    K = data["cameraMatrix"]
    dist = data["distCoeffs"]

    pattern_size = (9,6)
    square_size = 0.022

    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size

    cube_size = 2 * square_size
    cube = np.float32([
        [0,0,0], [0,cube_size,0], [cube_size,cube_size,0], [cube_size,0,0],
        [0,0,-cube_size],[0,cube_size,-cube_size],
        [cube_size,cube_size,-cube_size],[cube_size,0,-cube_size]
    ])

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Camera not found")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(
        "choice1_output.avi",
        fourcc,
        20.0,
        (int(cap.get(3)), int(cap.get(4)))
    )

    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = frame.shape[:2]  # current webcam frame size
        calib_height, calib_width = 4096, 3072  # original training image resolution

        scale_x = w / calib_width
        scale_y = h / calib_height

        K_scaled = K.copy()
        K_scaled[0,0] *= scale_x  # fx
        K_scaled[1,1] *= scale_y  # fy
        K_scaled[0,2] *= scale_x  # cx
        K_scaled[1,2] *= scale_y  # cy

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1), criteria
            )

            ok, rvec, tvec = cv2.solvePnP(objp, corners, K_scaled, dist)

            if ok:
                imgpts, _ = cv2.projectPoints(cube, rvec, tvec, K_scaled, dist)
                frame = draw_cube(frame, imgpts)
                frame = color_top_face(frame, cube, imgpts, rvec, tvec, K_scaled, dist)

        cv2.imshow("Choice 1 - Realtime", frame)
        out_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
