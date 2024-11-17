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
from datetime import datetime


class Aruco_pose():
    keep_mkv = True  
    ext = '.mp4'
    camera_id = 0
    
    mat_width = 110
    mat_length = 170 
    
    def __init__(self, 
                 video_path:str, 
                 EDMO_name:str,
                 show:bool=False,
                 calibration_matrix_path:str="calibration_matrix.npy", 
                 distortion_coefficients_path:str="distortion_coefficients.npy",
                 aruco_dict_type:str="DICT_4X4_100") -> None:
        
        self.k = np.load(calibration_matrix_path)
        self.d = np.load(distortion_coefficients_path)
        self.aruco_dict_type = aruco_dict_type
        self.show = show
        if video_path[0] != '/':
            video_path = '/' + video_path
        self.video_path = os.getcwd() + video_path
        self.EDMO_name = EDMO_name
        
        if self.keep_mkv:
            self.ext = '.mkv'
        
        # Get the valid Aruco ids for the EDMO (format: {leg0 leg1 leg2 leg3 middle corner0 corner1 corner2 corner3})
        self.valid_tags = []
        try:
            with open(f'tags/{EDMO_name}.txt', 'r') as f:
                content = f.read().split(' ')
                for tag_id in content:
                    self.valid_tags.append(int(tag_id))
        except FileNotFoundError:
            print(f"Error: The file tags/{EDMO_name}.txt is missing.")
            return
        
        self.first_frame = True
        self.M_cam_to_first_aruco = np.eye(4)  # Transformation matrix from the camera frame to the world frame (= position of the marker on the first video frame)
        self.M_first_aruco_to_cam = np.eye(4)  #                                world frame to the camera frame


    def get_aruco_pose(self, frame):
        '''
        frame - Frame from the video stream
        aruco_dict_type - Type of Aruco dictionary used
        matrix_coefficients - Intrinsic matrix of the calibrated camera
        distortion_coefficients - Distortion coefficients associated with your camera
        valid_tags - The id of the Aruco markers we want to detect

        return: - A dictionary with the valid_tags as keys and its position if detected
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        # print(ids)
        pos_dict = {}
        # If markers are detected
        if len(corners) > 0:
            for i in range(0, len(ids)):
                if ids[i] not in self.valid_tags:
                    continue
                marker_length = 0.04
                if ids[i] == 4:
                    marker_length = 0.05
                if ids[i] == 98:
                    marker_length = 0.09
                object_points = np.array([[-marker_length / 2, marker_length / 2, 0],
                                        [marker_length / 2, marker_length / 2, 0],
                                        [marker_length / 2, -marker_length / 2, 0],
                                        [-marker_length / 2, -marker_length / 2, 0]])

                ret, rvec, tvec = cv2.solvePnP(object_points, corners[i], self.k, self.d)
                if ret:
                    x, y, z = tvec.flatten()
                    # Convert rvec to a rotation matrix
                    R, _ = cv2.Rodrigues(rvec)

                    M_cam_to_aruco = np.eye(4)
                    M_cam_to_aruco[:3, :3] = R
                    M_cam_to_aruco[:3, 3] = [x, y, z]

                    # Record the positions of the aruco markers with regard to the initial position
                    if self.first_frame:
                        if ids[i] == 4:
                            # Record the transformation matrices
                            self.M_cam_to_first_aruco = M_cam_to_aruco
                            self.M_first_aruco_to_cam = np.linalg.inv(self.M_cam_to_first_aruco)

                            pos_dict[f'{ids[i]}'] = [np.array([0, 0, 0]), np.array([0, 0, 0])]
                            self.first_frame = False
                        
                    else:
                        M_marker_to_first_aruco = np.dot(self.M_first_aruco_to_cam, M_cam_to_aruco)

                        # Extract the relative rotation and translation vectors from the transformation matrix
                        R_rel = M_marker_to_first_aruco[:3, :3]
                        t_rel = M_marker_to_first_aruco[:3, 3]

                        # Convert the relative rotation matrix back to rvec
                        rvec_rel, _ = cv2.Rodrigues(R_rel)
                        pos_dict[f'{ids[i]}'] = [t_rel, rvec_rel.flatten()]
                        
                    if self.show:
                        # Draw a square around the markers
                        cv2.aruco.drawDetectedMarkers(frame, corners)

                        # Draw axis
                        cv2.drawFrameAxes(frame, self.k, self.d, rvec, tvec, marker_length)
                        a, b, c = rvec.flatten()
                        if ids[i] == 4:
                            cv2.putText(frame, f'x: {x:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                            cv2.putText(frame, f'y: {y:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                            cv2.putText(frame, f'z: {z:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                            cv2.putText(frame, f'a: {a:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                            cv2.putText(frame, f'b: {b:.2f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                            cv2.putText(frame, f'c: {c:.2f}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        return pos_dict


    def convert_to_mp4(self, mkv_file, out_name):
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


    def split_video_into_4(self):
        for date_folder in os.listdir('cleanData/'):
            path = f'cleanData/{date_folder}'
            for EDMO in os.listdir(path):
                path = f'cleanData/{date_folder}/{EDMO}/'
                for time_folder in os.listdir(path):
                    path = f'cleanData/{date_folder}/{EDMO}/{time_folder}/'
                    files = os.listdir(path)
                    if f'bottom_left{self.ext}' in files or f'bottom_right{self.ext}' in files:
                        print(f"Found existing video splits in {path}-> skipping to the next one")
                        continue

                    for file in files:
                        try:
                            if file == f"synced_video_data{self.ext}":
                                file_path = f'cleanData/{date_folder}/{EDMO}/{time_folder}/'
                                # ffmpeg.input(file_path).crop(0, 0, 1920, 1080).output(f'{path}top_left{self.ext}').run()
                                # ffmpeg.input(file_path).crop(1920, 0, 1920, 1080).output(f'{path}top_right{self.ext}').run()
                                ffmpeg.input(f'{file_path}/{file}').crop(0, 1080, 1920, 1080).output(
                                    f'{path}bottom_left{self.ext}').run()
                                ffmpeg.input(f'{file_path}/{file}').crop(1920, 1080, 1920, 1080).output(
                                    f'{path}bottom_right{self.ext}').run()
                        except ffmpeg.Error as e:
                            if os.path.exists(f'{path}bottom_left{self.ext}'):
                                os.remove(f'{path}bottom_left{self.ext}')
                            print(f"ffmpeg error: {e.stderr.decode('utf-8')}")


    def create_data_folder(self):
        '''
        For all the videos in Videos(mkv) create the corresponding folders in Video data
        '''
        for file in os.listdir('Videos(mkv)/'):
            filename, ext = os.path.splitext(file)
            if ext == '.mkv':
                date, time = filename.replace('-', '.').split(' ')
                if not os.path.exists('Video data'):
                    os.makedirs(f'Video data')
                if not os.path.exists('Video data/' + date):
                    os.makedirs('Video data/' + date)
                if not os.path.exists(f'Video data/{date}/{time}'):
                    os.makedirs(f'Video data/{date}/{time}')
                    
                source = f'Videos(mkv)/{file}'
                destination = f'Video data/{date}/{time}/'
                if self.keep_mkv:
                    shutil.move(source, destination)
                else:
                    self.convert_to_mp4(source, f'{destination}{filename}.mp4')
            else:
                print("Wrong format file in Videos(mkv)/")


    def sync_video_data(self, video_folder, data_folder):
        '''
        Synchronize the timestamps of the videos in the video_folder with the motor and IMU data in the data_folder
        and write the synchronized video in the data folder as 'synced_video_data'
        '''
        if f'synced_video_data{self.ext}' in os.listdir(data_folder):
            return
        video_path = f'{video_folder}/{os.listdir(video_folder)[0]}'

        video_time = toTime(video_folder.split('/')[-1].replace('.', ':'))
        data_time = toTime(data_folder.split('/')[-1].replace('.', ':'))
        if video_time < data_time:
            with open(f'{data_folder}/Motor0.log', 'r') as f:
                f = f.read()
                logs = f.split('\n')[:-1]
                last_log_time = logs[-1].split(',')[0]

                ffmpeg.input(video_path, ss=data_time - video_time).output(f'{data_folder}/synced_video_data{self.ext}',
                                                                        t=last_log_time).run(overwrite_output=True)
        else:
            print("Video was started later than EDMO session")


    def preprocess_video(self):
        # Create data folders
        print("Creating folders")
        self.date = self.date.replace('-', '.')

        self.create_data_folder()

        # Verify all folders are existing and the number of elements match
        video_path = f'Video data/{self.date}/'
        data_path = f'cleanData/{self.date}/{self.EDMO_name}/'
        if not os.path.exists(video_path) or not os.path.exists(data_path):
            print(f"Folder {video_path} or folder {data_path} not found")
            sys.exit(0)
        if len(os.listdir(video_path)) != len(os.listdir(data_path)):
            print("The number of videos in {video_path} doesn't match the number of log sessions in {data_path}, please check that they correspond!")
            x = input('To continue anyways type cont:')
            if x != 'cont':
                print("aborting data preprocessing")
                sys.exit(0)

        # Synchronize the videos with the data logs
        print("Synchronizing videos with data")
        for v, data in zip(os.listdir(video_path), os.listdir(data_path)):
            self.sync_video_data(f'{video_path}{v}', f'{data_path}{data}')

        # Split the videos to only keep the relevant parts
        self.split_video_into_4()

        print("Finished preprocessing the videos")
        sys.exit(0)


    def pose_estimation(self):
        # if self.date:
        #     self.preprocess_video()
        video = cv2.VideoCapture(self.video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        print(f'fps: {fps}')
        use_video = self.video_path != self.camera_id
        if use_video:
            if not os.path.exists(self.video_path):
                print(f"File {self.video_path} not found when getting the video")
                sys.exit(0)

            name, ext = os.path.splitext(self.video_path)
            if ext.lower() != ".mp4" and ext.lower() != ".mkv":
                print(f"Wrong video file format {ext} is not supported")
                sys.exit(0)
        else: # Record the camera feed
            frame_width = int(video.get(3)) 
            frame_height = int(video.get(4)) 
            
            size = (frame_width, frame_height) 
            file_date = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            
            date, time = file_date.replace('-', '.').split(' ')
            if not os.path.exists('Video data'):
                os.makedirs(f'Video data')
            if not os.path.exists('Video data/' + date):
                os.makedirs(f'Video data/{date}')
            if not os.path.exists(f'Video data/{date}/{time}'):
                os.makedirs(f'Video data/{date}/{time}')    
            self.video_path = f'Video data/{date}/{time}/{file_date}.mkv'
            
            print(f'Video file will be stored at {self.video_path}')
            output_video = cv2.VideoWriter(self.video_path,  
                                    cv2.VideoWriter_fourcc(*'MJPG'), 
                                    fps, size) 

        frame_index = 1
        dict_all_pos = {}
        while True:
            ret, frame = video.read()
            if not ret:
                break 
            if not use_video:
                output_video.write(frame)
            
            # Improve detection by applying smoothing
            # frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
            output = self.get_aruco_pose(frame)

            if show or not use_video:
                scaled_frame = cv2.resize(frame, (960, 540))
                cv2.imshow('Estimated Pose', scaled_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
            if len(output) > 0:
                for tag_id, v in output.items():
                    if tag_id not in dict_all_pos:
                        dict_all_pos[tag_id] = []
                    dict_all_pos[tag_id].append([frame_index, v])
            frame_index += 1
        video.release()
        if not use_video:
            output_video.release()
        cv2.destroyAllWindows()
        write_log(self.video_path, dict_all_pos)


if __name__ == '__main__':
    # Example usage : python pose_estimation.py -v 'Videos(mkv)/GX010412.MP4'
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy", help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default="distortion_coefficients.npy", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100", help="Type of ArUCo tag to detect")
    ap.add_argument("-v", "--video", type=str, default=0, help="Path to video or uses laptop camera feed by defaut)")
    # ap.add_argument("-p", "--preprocess_video", type=str, default=None, help="Video to preprocess (i.e sync video and data, cut to )")
    ap.add_argument("-edmo", "--EDMO_name", type=str, default='Kumoko', help="Name of the EDMO robot")
    ap.add_argument("-s", "--show", type=bool, default=False, help="Show output frame")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)
    aruco_dict_type = ARUCO_DICT[args["type"]]

    video_path = args['video']
    # preprocess_video = args['preprocess_video']
    EDMO_name = args['EDMO_name']
    show = args['show']

    aruco_pose = Aruco_pose(video_path, EDMO_name, show=show, aruco_dict_type=aruco_dict_type)
    aruco_pose.pose_estimation()
