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
from datetime import datetime, timedelta


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

                # print((tvec[0]))
                x, y, z = tvec.flatten()

                cv2.putText(frame, f'x: {x:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f'y: {y:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f'z: {z:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame


def convert_to_mp4(mkv_file, out_name):
    try:
        # Check if the input file exists
        if not os.path.exists(mkv_file):
            raise FileNotFoundError(f"Input file '{mkv_file}' not found.")

        # Check if the output directory exists and create it if necessary
        out_dir = os.path.dirname(out_name)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Run the conversion and capture stdout and stderr
        infile = ffmpeg.input(mkv_file)
        outfile = infile.output(out_name, vcodec='libx264', acodec='aac', audio_bitrate='128k', ac=2, strict='-2',
                                movflags='faststart')

        stdout, stderr = outfile.run(capture_stdout=True, capture_stderr=True)

        print(f"Finished converting {mkv_file} to {out_name}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode('utf-8')}")
    except OSError as e:
        print(f"OS error: {e}")


def split_video_into_4():
    for date_folder in os.listdir('cleanData/'):
        path = f'cleanData/{date_folder}'
        for edmo in os.listdir(path):
            path = f'cleanData/{date_folder}/{edmo}/'
            for time_folder in os.listdir(path):
                path = f'cleanData/{date_folder}/{edmo}/{time_folder}/'
                for file in os.listdir(path):
                    if file == "synced_video_data.mp4":
                        file_path = f'cleanData/{date_folder}/{edmo}/{time_folder}/'
                        # ffmpeg.input(file_path).crop(0, 0, 1920, 1080).output(f'{path}top_left.mp4').run()
                        # ffmpeg.input(file_path).crop(1920, 0, 1920, 1080).output(f'{path}top_right.mp4').run()
                        ffmpeg.input(f'{file_path}/{file}').crop(0, 1080, 1920, 1080).output(f'{path}bottom_left.mp4').run()
                        ffmpeg.input(f'{file_path}/{file}').crop(1920, 1080, 1920, 1080).output(f'{path}bottom_right.mp4').run()


def create_data_folder():
    for file in os.listdir('Videos(mkv)/'):
        name, ext = os.path.splitext(file)
        if ext == '.mkv':
            date, time = name.split(' ')
            if not os.path.exists('Video data/' + date):
                os.makedirs('Video data/' + date)
            if not os.path.exists(f'Video data/{date}/{time}'):
                os.makedirs(f'Video data/{date}/{time}')


def toTime(t):
    try:
        return datetime.strptime(t, "%H:%M:%S")
    except ValueError as e:
        try:
            return datetime.strptime(t, "%H:%M:%S.%f")
        except ValueError as e:
            print(f'error: {e}')
            print(t)


def sync_video_data(video_folder, data_folder):
    video_path = f'{video_folder}/{os.listdir(video_folder)[0]}'

    video_time = toTime(video_folder.split('/')[-1].replace('-', ':'))
    data_time = toTime(data_folder.split('/')[-1].replace('.', ':'))
    if video_time < data_time:
        with open(f'{data_folder}/Motor0.log', 'r') as f:
            f = f.read()
            logs = f.split('\n')[:-1]
            last_log_time = logs[-1].split(',')[0]

            print(last_log_time)
            print(data_time)
            print(video_time)
            print(data_time-video_time)
            ffmpeg.input(video_path, ss=data_time-video_time).output(f'{data_folder}/synced_video_data.mp4', t=last_log_time).run(overwrite_output=True)
    else:
        print("Video was started later than EDMO session")



if __name__ == '__main__':
    # sync_video_data('Video data/2024-09-24/09-15-18', 'cleanData/2024.09.24/Kumoko/09.18.05')
    # exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy",
                    help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default="distortion_coefficients.npy",
                    help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100", help="Type of ArUCo tag to detect")
    ap.add_argument("-v", "--video", type=str, default=0, help="Path to video (.mp4 file)")
    ap.add_argument("-s", "--split", type=bool, default=False, help="Split the videos in the folder: Video data")
    args = vars(ap.parse_args())

    if args["split"]:
        split_video_into_4()
        exit(0)

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    video_path = args["video"]

    if video_path != 0:
        if not os.path.exists(video_path):
            print(f"File {video_path} not found")
            sys.exit(0)

        name, ext = os.path.splitext(video_path)
        if ext == ".mp4":
            pass
        elif ext == ".mkv":
            out_name = name + ".mp4"
            if not os.path.exists(out_name):
                print(f"creating file {out_name}")
                convert_to_mp4(video_path, out_name)
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

        cv2.imshow('Estimated Pose', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
