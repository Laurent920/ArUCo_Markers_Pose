import numpy as np
import cv2
import os
import argparse


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Calculate the reprojection error for each frame."""
    errors = []
    for i in range(len(objpoints)):
        # Project object points to image points
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # Compute the error as the Euclidean distance
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)
    return errors


def calibrate(dirpath, square_size, width, height, visualize=False, use_video=None, reprojection_error_threshold=0.1):
    """ Apply camera calibration operation for images in the given directory path. """

    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (3D points in real-world space)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points
    objpoints = []
    imgpoints = []

    images = os.listdir(dirpath)
    video = None
    if use_video:
        for fname in images:
            if fname == use_video:
                video = cv2.VideoCapture(os.path.join(dirpath, fname))
                break

    i, j = 0, 0
    skip_frame = False
    img = None
    while True:
        if use_video:
            ret, img = video.read()
            if not ret:
                break
        else:
            if i >= len(images):
                break
            fname = images[i]
            img = cv2.imread(os.path.join(dirpath, fname))

        if use_video:
            if j >= 20:
                j = 0
                skip_frame = False
            if skip_frame:
                i, j = i + 1, j + 1
                continue
            skip_frame = True

        # Preprocess the image for better corner detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_equalized = clahe.apply(gray)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_equalized, (width, height), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray_equalized, corners, (11, 11), (-1, -1), criteria)

            if visualize:
                cv2.drawChessboardCorners(img, (width, height), corners2, ret)
                cv2.putText(img, f'frame nb: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.imshow('img', cv2.resize(img, (960, 520)))
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        if key == ord('y'):
                            objpoints.append(objp)
                            imgpoints.append(corners2)
                            break
                        else: 
                            break  
            else:
                # print(objp)
                objpoints.append(objp)
                imgpoints.append(corners2)
        
        i += 1

    print(f"Number of valid images: {len(imgpoints)}")

    # Initial calibration with additional flags
    flags = (
        cv2.CALIB_FIX_ASPECT_RATIO +
        cv2.CALIB_RATIONAL_MODEL +
        cv2.CALIB_ZERO_TANGENT_DIST
        # cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
    )
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)

    # Calculate reprojection errors
    reprojection_errors = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    mean_reprojection_error = np.mean(reprojection_errors)
    print(f"Initial mean reprojection error: {mean_reprojection_error:.4f}")

    for _ in range(5):
        # Filter out frames with high reprojection errors
        filtered_objpoints = []
        filtered_imgpoints = []
        for k, error in enumerate(reprojection_errors):
            if error <= reprojection_error_threshold:
                filtered_objpoints.append(objpoints[k])
                filtered_imgpoints.append(imgpoints[k])
            else:
                print(f"Frame {k} discarded due to high reprojection error: {error:.4f}")

        # Recalibrate using filtered data
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            filtered_objpoints, filtered_imgpoints, gray.shape[::-1], None, None, flags=flags
        )
        # Final reprojection errors
        new_reprojection_errors = calculate_reprojection_error(filtered_objpoints, filtered_imgpoints, rvecs, tvecs, mtx, dist)
        new_mean_reprojection_error = np.mean(new_reprojection_errors)
        print(f"New mean reprojection error: {new_mean_reprojection_error:.4f}")
        
        if new_mean_reprojection_error <= mean_reprojection_error:
            break
        else:
            mean_reprojection_error = new_mean_reprojection_error
            reprojection_errors = new_reprojection_errors
        
    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Path to folder containing checkerboard images/video for calibration")
    ap.add_argument("-uv", "--use_video", help="Video file in dir to use to calibrate the camera (uses every 20 frames)", default=None)
    ap.add_argument("-w", "--width", type=int, help="Width of checkerboard (default=7)", default=7)
    ap.add_argument("-t", "--height", type=int, help="Height of checkerboard (default=6)", default=6)
    ap.add_argument("-s", "--square_size", type=float, default=0.03, help="Length of one edge (in meters)")
    ap.add_argument("-v", "--visualize", type=str, default="False", help="To visualize each checkerboard image")
    ap.add_argument("-th", "--reprojection_error_threshold", type=float, default=0.1, help="Threshold for reprojection error")
    args = vars(ap.parse_args())

    dirpath = args['dir']
    use_video = args['use_video']
    square_size = args['square_size']
    width = args['width']
    height = args['height']
    reprojection_error_threshold = args['reprojection_error_threshold']

    visualize = args["visualize"].lower() == "true"

    ret, mtx, dist, rvecs, tvecs = calibrate(
        dirpath, square_size, visualize=visualize, width=width, height=height, 
        use_video=use_video, reprojection_error_threshold=reprojection_error_threshold
    )

    print("Camera Matrix:")
    print(mtx)
    print("Distortion Coefficients:")
    print(dist)

    np.save(f"{dirpath}/calibration_matrix", mtx)
    np.save(f"{dirpath}/distortion_coefficients", dist)
