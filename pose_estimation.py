'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''
import cv2
import sys
from utils import *
import argparse
import os
import shutil
import ffmpeg
from tqdm import tqdm

first_frame = True
M_cam_to_first_aruco = np.eye(4)  # Transformation matrix from the camera frame to the world frame (= position of the marker on the first video frame)
M_first_aruco_to_cam = np.eye(4)  #                                world frame to the camera frame
keep_mkv = True  # Keep at False -> TODO implement logic for keeping mkv files
ext = '.mp4'


def get_aruco_pose(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, valid_tags, show_img=False):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    show = True: frame - The frame with the axis drawn on it
    show = False: pos_dict - Dictionary of aruco ids with their tvec and rvec
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
            if ids[i] not in valid_tags:
                return frame if show_img else pos_dict
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
                    if first_frame and ids[i] == 98:
                        # Record the transformation matrices
                        M_cam_to_first_aruco = M_cam_to_aruco
                        M_first_aruco_to_cam = np.linalg.inv(M_cam_to_first_aruco)

                        pos_dict[f'{ids[i]}'] = [np.array([0, 0, 0]), np.array([0, 0, 0])]
                        first_frame = False
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
        outfile = infile.output(out_name, vcodec='libx264', acodec='aac', audio_bitrate='128k', ac=2, strict='-2',
                                movflags='faststart')
        stdout, stderr = outfile.run(capture_stdout=True, capture_stderr=True)

        print(f"Finished converting {mkv_file} to {out_name}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ffmpeg.Error as e:
        if os.path.exists(out_name):
            os.remove(out_name)
        print(f"ffmpeg error: {e.stderr.decode('utf-8')}")


def split_video_into_4():
    global ext
    for date_folder in os.listdir('cleanData/'):
        path = f'cleanData/{date_folder}'
        for EDMO in os.listdir(path):
            path = f'cleanData/{date_folder}/{EDMO}/'
            for time_folder in os.listdir(path):
                path = f'cleanData/{date_folder}/{EDMO}/{time_folder}/'
                files = os.listdir(path)
                if f'bottom_left{ext}' in files or f'bottom_right{ext}' in files:
                    print(f"Found existing video splits in {path}-> skipping to the next one")
                    continue

                for file in files:
                    try:
                        if file == f"synced_video_data{ext}":
                            file_path = f'cleanData/{date_folder}/{EDMO}/{time_folder}/'
                            # ffmpeg.input(file_path).crop(0, 0, 1920, 1080).output(f'{path}top_left{ext}').run()
                            # ffmpeg.input(file_path).crop(1920, 0, 1920, 1080).output(f'{path}top_right{ext}').run()
                            ffmpeg.input(f'{file_path}/{file}').crop(0, 1080, 1920, 1080).output(
                                f'{path}bottom_left{ext}').run()
                            ffmpeg.input(f'{file_path}/{file}').crop(1920, 1080, 1920, 1080).output(
                                f'{path}bottom_right{ext}').run()
                    except ffmpeg.Error as e:
                        if os.path.exists(f'{path}bottom_left{ext}'):
                            os.remove(f'{path}bottom_left{ext}')
                        print(f"ffmpeg error: {e.stderr.decode('utf-8')}")


def create_data_folder():
    '''
    For all the videos in Videos(mkv) create the corresponding folders in Video data
    '''
    global keep_mkv
    for file in os.listdir('Videos(mkv)/'):
        name, ext = os.path.splitext(file)
        if ext == '.mkv':
            date, time = name.replace('-', '.').split(' ')
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
        else:
            print("Wrong format file in Videos(mkv)/")


def sync_video_data(video_folder, data_folder):
    '''
    Synchronize the timestamps of the videos in the video_folder with the motor and IMU data in the data_folder
    and write the synchronized video in the data folder as 'synced_video_data'
    '''
    global ext
    if f'synced_video_data{ext}' in os.listdir(data_folder):
        return
    video_path = f'{video_folder}/{os.listdir(video_folder)[0]}'

    video_time = toTime(video_folder.split('/')[-1].replace('.', ':'))
    data_time = toTime(data_folder.split('/')[-1].replace('.', ':'))
    if video_time < data_time:
        with open(f'{data_folder}/Motor0.log', 'r') as f:
            f = f.read()
            logs = f.split('\n')[:-1]
            last_log_time = logs[-1].split(',')[0]

            ffmpeg.input(video_path, ss=data_time - video_time).output(f'{data_folder}/synced_video_data{ext}',
                                                                       t=last_log_time).run(overwrite_output=True)
    else:
        print("Video was started later than EDMO session")


def preprocess_video(date, EDMO_name):
    date = date.replace('-', '.')

    # Create data folders
    print("Creating folders")
    create_data_folder()

    # Verify all folders are existing and the number of elements match
    video_path = f'Video data/{date}/'
    data_path = f'cleanData/{date}/{EDMO_name}/'
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

    # Synchronize the videos with the data logs
    print("Synchronizing videos with data")
    for v, data in tqdm(zip(os.listdir(video_path), os.listdir(data_path))):
        sync_video_data(f'{video_path}{v}', f'{data_path}{data}')

    # Split the videos to only keep the relevant parts
    split_video_into_4()

    print("Finished preprocessing the videos")
    return True


def pose_estimation(k, d, aruco_dict_type, video_path, date, EDMO_name, valid_tags, show):
    global keep_mkv
    global ext
    if keep_mkv:
        ext = '.mkv'

    if date:
        stop_program = preprocess_video(date, EDMO_name)

    if video_path != 0:
        if not os.path.exists(video_path):
            print(f"File {video_path} not found")
            sys.exit(0)

        name, ext = os.path.splitext(video_path)
        if ext != ".mp4" and ext != ".mkv":
            print(f"Wrong video file format .{ext} is not supported")
            sys.exit(0)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f'fps: {fps}')

    index = 1
    dict_all_pos = {}
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Improve detection by applying smoothing
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        output = get_aruco_pose(frame, aruco_dict_type, k, d, valid_tags, show_img=show)

        if show:
            scaled_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('Estimated Pose', scaled_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            if len(output) > 0:
                for tag_id, v in output.items():
                    if tag_id not in dict_all_pos:
                        dict_all_pos[tag_id] = []
                    dict_all_pos[tag_id].append([index, v])
        index += 1
    video.release()
    cv2.destroyAllWindows()
    write_log(video_path, dict_all_pos)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy", help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default="distortion_coefficients.npy", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100", help="Type of ArUCo tag to detect")
    ap.add_argument("-v", "--video", type=str, default=0, help="Path to video (.mp4 file)")
    ap.add_argument("-p", "--preprocess_date", type=str, default=None, help="Date of the videos in Videos(mkv) to preprocess")
    ap.add_argument("-edmo", "--EDMO_name", type=str, default='Kumoko', help="Name of the EDMO robot")
    ap.add_argument("-s", "--show", type=bool, default=False, help="Show output frame")
    args = vars(ap.parse_args())

    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)
    aruco_dict_type = ARUCO_DICT[args["type"]]

    video_path = args['video']
    date = args['preprocess_date']
    EDMO_name = args['EDMO_name']
    show = args['show']

    valid_tags = []
    try:
        with open(f'tags/{EDMO_name}.txt', 'r') as f:
            content = f.read().split(' ')
            for tag_id in content:
                valid_tags.append(int(tag_id))
    except FileNotFoundError:
        print(f"Error: The file tags/{EDMO_name}.txt is missing.")

    pose_estimation(k, d, aruco_dict_type, video_path, date, EDMO_name, valid_tags, show)
