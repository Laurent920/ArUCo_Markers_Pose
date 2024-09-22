'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import os
import ffmpeg


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    marker_length = 0.05

    object_points = np.array([[-marker_length / 2, marker_length / 2, 0],
                              [marker_length / 2, marker_length / 2, 0],
                              [marker_length / 2, -marker_length / 2, 0],
                              [-marker_length / 2, -marker_length / 2, 0]])
    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            ret, rvec, tvec = cv2.solvePnP(object_points, corners[i], matrix_coefficients, distortion_coefficients)

            if ret:
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Draw axis
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.05)

                print((tvec[0]))
                x, y, z = tvec.flatten()

                cv2.putText(frame, f'x: {x:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f'y: {y:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f'z: {z:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

def convert_to_mp4(mkv_file):
    name, ext = os.path.splitext(mkv_file)
    out_name = name + ".mp4"
    print(out_name)
    a = ffmpeg.input(mkv_file).output(out_name)
    a.run()
    print("Finished converting {}".format(mkv_file))

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy", help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default="distortion_coefficients.npy", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="Type of ArUCo tag to detect")
    ap.add_argument("-v", "--video", type=str, default=0, help="Path to video (.mp4 file)")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    video_path = args["video"] 
    if video_path != 0:
        if video_path[0] == '\\':
            video_path = video_path[1:]
        
        name, ext = os.path.splitext(video_path)
        if ext == ".mp4":
            pass
        elif ext == ".mkv":
            convert_to_mp4(video_path)
        else:
            print(f"Wrong video file format .{ext} is not supported")
            sys.exit(0)
    
    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(video_path)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        output = pose_estimation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
