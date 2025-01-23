'''
Sample Usage:-
python calibration.py --dir calibration_checkerboard/ --square_size 0.024
'''

from matplotlib import use
import numpy as np
import cv2
import os
import argparse


def calibrate(dirpath, square_size, width, height, visualize=False, use_video=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = os.listdir(dirpath)
    video:cv2.VideoCapture
    if use_video:
        for fname  in images:
            if os.path.splitext(fname)[1].lower() == ".mp4":
                video = cv2.VideoCapture(os.path.join(dirpath, fname))
                break

    i, j = 0, 0
    skip_frame = False
    img:np.ndarray
    while True:
        if use_video:
            ret, img = video.read()
            if not ret:
                break 
        else:
            fname = images[i]
            img = cv2.imread(os.path.join(dirpath, fname))
            
        if use_video: # Skip every 20 frames
            if j >= 20:
                j = 0
                skip_frame = False
            if skip_frame:
                i, j = i+1, j+1
                continue
            skip_frame = True
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        print(i, ret)
        # If found, add object points, image points (after refining them)            
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

            if visualize:
                cv2.putText(img, f'frame nb: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.imshow('img',cv2.resize(img, (960, 540)))
                cv2.waitKey(0)
            
        i += 1

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Path to folder containing checkerboard images/video for calibration")
    ap.add_argument("-uv", "--use_video", help="Use first video file in dir to calibrate the camera (uses every 20 frames)", default=False)
    ap.add_argument("-w", "--width", type=int, help="Width of checkerboard (default=7)",  default=7)
    ap.add_argument("-t", "--height", type=int, help="Height of checkerboard (default=6)", default=6)
    ap.add_argument("-s", "--square_size", type=float, default=0.03, help="Length of one edge (in metres)")
    ap.add_argument("-v", "--visualize", type=str, default="False", help="To visualize each checkerboard image")
    args = vars(ap.parse_args())
    
    dirpath = args['dir']
    
    if args['use_video'].lower() == "true":
        use_video = True
    else:
        use_video = False
    square_size = args['square_size']
    square_size = 0.03
    print(square_size)
    # Interior size of the checkerboard 
    width = args['width']
    height = args['height']

    if args["visualize"].lower() == "true":
        visualize = True
    else:
        visualize = False
    ret, mtx, dist, rvecs, tvecs = calibrate(dirpath, square_size, visualize=visualize, width=width, height=height, use_video=use_video)

    print(mtx)
    print(dist)

    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)
