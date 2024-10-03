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
import shutil
import ffmpeg
from datetime import datetime

first_frame = True
M_cam_to_first_aruco = np.eye(4)  # Transformation matrix from the camera frame to the world frame (= position of the marker on the first video frame)
M_first_aruco_to_cam = np.eye(4)  #                                world frame to the camera frame
keep_mkv = False # Keep at False -> TODO implement logic for keeping mkv files


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, show_img=False):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    global first_frame
    global M_cam_to_first_aruco
    global M_first_aruco_to_cam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    pos_dict = {}
    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            marker_length = 0.04
            if ids[i] == 98:
                marker_length = 0.09
            object_points = np.array([[-marker_length / 2, marker_length / 2, 0],
                                      [marker_length / 2, marker_length / 2, 0],
                                      [marker_length / 2, -marker_length / 2, 0],
                                      [-marker_length / 2, -marker_length / 2, 0]])
            
            # Estimate pose of each marker and return the values rvec and tvec
            ret, rvec, tvec = cv2.solvePnP(object_points, corners[i], matrix_coefficients, distortion_coefficients)
            if ret:
                x, y, z = tvec.flatten()
                # Convert rvec to a rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                M_cam_to_aruco = np.eye(4)
                M_cam_to_aruco[:3, :3] = R
                M_cam_to_aruco[:3, 3] = [x, y, z]

            
                if not show_img:
                    if first_frame:
                        M_cam_to_first_aruco = M_cam_to_aruco

                        M_first_aruco_to_cam = np.linalg.inv(M_cam_to_first_aruco)
                        first_frame = False
                        pos_dict[f'{ids[i]}'] = [np.array([0,0,0]), np.array([0,0,0])]
                    else:
                        M_marker_to_first_aruco = np.dot(M_first_aruco_to_cam, M_cam_to_aruco)

                        # Extract the relative rotation and translation vectors from the transformation matrix
                        R_rel = M_marker_to_first_aruco[:3, :3]
                        t_rel = M_marker_to_first_aruco[:3, 3]

                        # Convert the relative rotation matrix back to rvec
                        rvec_rel, _ = cv2.Rodrigues(R_rel)
                        pos_dict[f'{ids[i]}'] = [t_rel, rvec_rel.flatten()]
                else:
                    # Draw a square around the markers
                    cv2.aruco.drawDetectedMarkers(frame, corners)

                    # Draw axis
                    cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, marker_length)
                    a, b, c = rvec.flatten()
                    if ids[i] == 98:
                        cv2.putText(frame, f'x: {x:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        cv2.putText(frame, f'y: {y:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        cv2.putText(frame, f'z: {z:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        cv2.putText(frame, f'a: {a:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        cv2.putText(frame, f'b: {b:.2f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        cv2.putText(frame, f'c: {c:.2f}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    return frame if show_img else pos_dict
      

def rvec_to_quaternion(rvec):
    # TODO turn rvec into quaternion for smoother animation transition 
    
    # Calculate the angle of rotation (magnitude of rvec)
    angle = np.linalg.norm(rvec)
    
    # Avoid division by zero in case of zero rotation vector
    if angle == 0:
        return np.array([0, 0, 0, 1])  # No rotation, return the identity quaternion

    # Calculate quaternion components
    qw = np.cos(angle / 2)
    qx = np.sin(angle / 2) * (rvec[0] / angle)
    qy = np.sin(angle / 2) * (rvec[1] / angle)
    qz = np.sin(angle / 2) * (rvec[2] / angle)
    
    return np.array([qx, qy, qz, qw])      
              

def convert_to_mp4(mkv_file, out_name):
    try:
        # Check if the input file exists
        if not os.path.exists(mkv_file):
            raise FileNotFoundError(f"Input file '{mkv_file}' not found.")

        # Create output file directory if needed
        out_dir = os.path.dirname(out_name)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Run the conversion 
        infile = ffmpeg.input(mkv_file)
        outfile = infile.output(out_name, vcodec='libx264', acodec='aac', audio_bitrate='128k', ac=2, strict='-2', movflags='faststart')
        stdout, stderr = outfile.run(capture_stdout=True, capture_stderr=True)

        print(f"Finished converting {mkv_file} to {out_name}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ffmpeg.Error as e:
        if os.path.exists(out_name):
            os.remove(out_name)
        print(f"ffmpeg error: {e.stderr.decode('utf-8')}")


def split_video_into_4():
    for date_folder in os.listdir('cleanData/'):
        path = f'cleanData/{date_folder}'
        for edmo in os.listdir(path):
            path = f'cleanData/{date_folder}/{edmo}/'
            for time_folder in os.listdir(path):
                path = f'cleanData/{date_folder}/{edmo}/{time_folder}/'
                files = os.listdir(path)
                if 'bottom_left.mp4' in files or 'bottom_right.mp4' in files:
                    print("Found existing video splits -> aborting video splitting")
                    return 
                
                for file in files:
                    try:
                        if file == "synced_video_data.mp4":
                            file_path = f'cleanData/{date_folder}/{edmo}/{time_folder}/'
                            # ffmpeg.input(file_path).crop(0, 0, 1920, 1080).output(f'{path}top_left.mp4').run()
                            # ffmpeg.input(file_path).crop(1920, 0, 1920, 1080).output(f'{path}top_right.mp4').run()
                            ffmpeg.input(f'{file_path}/{file}').crop(0, 1080, 1920, 1080).output(f'{path}bottom_left.mp4').run()
                            ffmpeg.input(f'{file_path}/{file}').crop(1920, 1080, 1920, 1080).output(f'{path}bottom_right.mp4').run()
                    except ffmpeg.Error as e:
                        if os.path.exists(f'{path}bottom_left.mp4'):
                            os.remove('{path}bottom_left.mp4')
                        print(f"ffmpeg error: {e.stderr.decode('utf-8')}")


def create_data_folder():
    '''
    For all the videos in Videos(mkv) create the corresponding folders in Video data
    '''
    global keep_mkv
    for file in os.listdir('Videos(mkv)/'):
        name, ext = os.path.splitext(file)
        if ext == '.mkv':
            date, time = name.split(' ')
            if not os.path.exists('Video data/' + date):
                os.makedirs('Video data/' + date)
            if not os.path.exists(f'Video data/{date}/{time}'):
                os.makedirs(f'Video data/{date}/{time}')
                source = f'Videos(mkv)/{file}'
                destination = f'Video data/{date}/{time}/'
                if keep_mkv:
                    shutil.move(source, destination)
                else:
                    convert_to_mp4(source, f'{destination}{name}.mp4')


def toTime(t):
    try:
        return datetime.strptime(t, "%H:%M:%S")
    except ValueError as e:
        try:
            return datetime.strptime(t, "%H:%M:%S.%f")
        except ValueError as e:
            print(f'error: {e} with time {t}')


def sync_video_data(video_folder, data_folder):
    '''
    Synchronize the timestamps of the videos in the video_folder with the motor and IMU data in the data_folder
    and write the synchronized video in the data folder as 'synced_video_data.mp4
    '''
    if 'synced_video_data.mp4' in os.listdir(data_folder):
        return
    video_path = f'{video_folder}/{os.listdir(video_folder)[0]}'

    video_time = toTime(video_folder.split('/')[-1].replace('-', ':'))
    data_time = toTime(data_folder.split('/')[-1].replace('.', ':'))
    if video_time < data_time:
        with open(f'{data_folder}/Motor0.log', 'r') as f:
            f = f.read()
            logs = f.split('\n')[:-1]
            last_log_time = logs[-1].split(',')[0]

            # print(last_log_time)
            # print(data_time)
            # print(video_time)
            # print(data_time - video_time)
            ffmpeg.input(video_path, ss=data_time - video_time).output(f'{data_folder}/synced_video_data.mp4',
                                                                       t=last_log_time).run(overwrite_output=True)
    else:
        print("Video was started later than EDMO session")
        

def preprocess_video(date):
    if not date:
        return False
        
    create_data_folder()
    video_path = 'Video data/{date}/'
    data_path = 'cleanData/{date}/Kumoko/'
    if not os.path.exists(video_path) or not os.path.exists(data_path):
        print(f"Folder {video_path} or folder {data_path} not found")
        sys.exit(0)
    
    
    if len(os.listdir(video_path)) != len(os.listdir(data_path)):
        print("The number of videos in {video_path} doesn't match the number of log sessions in {data_path},\
                please check that they correspond!")
        x = input('To continue anyways type cont:')
        if x != 'cont':
            print("aborting data preprocessing")
            return True
        
    for v, d in zip(os.listdir(video_path), os.listdir(data_path)):
        sync_video_data(f'{video_path}/{v}', f'{data_path}/{d}')
    split_video_into_4()
    print("Finished preprocessing the videos")
    return True
        
def write_log(path, dict_all_pos):
    dir_path = os.path.dirname(path)
    for id, data_list in dict_all_pos.items():
        path = f'{dir_path}/marker_{id}.log'
        with open(path, 'w') as f:
            f.write('frame index, translation vector, rotation vector\n')
            for index, pose in data_list:
                f.write(f'{index}, {pose[0]}, {pose[1]}\n')            


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy",
                    help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default="distortion_coefficients.npy",
                    help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100", help="Type of ArUCo tag to detect")
    ap.add_argument("-v", "--video", type=str, default=0, help="Path to video (.mp4 file)")
    ap.add_argument("-date_folder", "--date_clean_sync_split", type=str, default=None, help="Turn mkv videos of the specified date to mp4, sync them and split them in the cleanData folder")
    ap.add_argument("-s", "--show", type=bool, default=False, help="Show output frame")
    args = vars(ap.parse_args())

    date = args["date_clean_sync_split"]
    stop_program = preprocess_video(date)
    if stop_program:
        sys.exit(0)
        
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    video_path = args["video"]

    if video_path != 0:
        if not os.path.exists(video_path):
            print(f"File {video_path} not found")
            sys.exit(0)

        name, ext = os.path.splitext(video_path)
        if ext != ".mp4" and ext != ".mkv":
            print(f"Wrong video file format .{ext} is not supported")
            sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    show = args['show']
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)

    index = 1
    dict_all_pos = {}
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Improve detection by applying smoothing
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        output = pose_estimation(frame, aruco_dict_type, k, d, show_img=show)

        if show:
            scaled_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('Estimated Pose', scaled_frame)
        else:
            if len(output) > 0:
                for id, v in output.items():
                    if id not in dict_all_pos:
                        dict_all_pos[id] = []    
                    dict_all_pos[id].append([index, v])
                    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        index += 1
    video.release()
    cv2.destroyAllWindows()
    write_log(video_path, dict_all_pos)
    