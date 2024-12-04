import sys
import os
from datetime import datetime
import ffmpeg
import shutil

import cv2
import numpy as np

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			# show the output image
	return image


def toTime(t):
	try:
		return datetime.strptime(t, "%H:%M:%S")
	except ValueError as e:
		try:
			return datetime.strptime(t, "%H:%M:%S.%f")
		except ValueError as e:
			print(f'error: {e} with time {t}')
      

def rvec_to_quaternion(rvec):
	# TODO turn rvec into quaternion for smoother animation transition

	# Calculate the angle of rotation (magnitude of rvec)
	angle = np.linalg.norm(rvec)

	# Avoid division by zero in case of zero rotation vector
	if angle == 0:
		return np.array([0, 0, 0, 1])  # No rotation, return the identity quaternion

	# Calculate quaternion components
	qw = np.cos(angle / 2)
	qx = np.sin(angle / 2) * (rvec[0][0] / angle)
	qy = np.sin(angle / 2) * (rvec[1][0] / angle)
	qz = np.sin(angle / 2) * (rvec[2][0] / angle)

	return np.array([qx, qy, qz, qw])


def write_log(path, dict_all_pos):
	'''
	Write in log file: frame index, translation vector, rotation vector
	for each frame the aruco markers are detected
	'''
	dir_path = os.path.dirname(path)
	for tag_id, data_list in dict_all_pos.items():
		path = f'{dir_path}/marker_{tag_id}.log'
		with open(path, 'w') as f:
			f.write('frame index, translation vector, rotation vector\n')
			for index, pose in data_list:
				f.write(f'{index}, {pose[0]}, {pose[1]}\n')


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


class Preprocess_4_split_video():
	def	__init__(self, date) -> None:
		'''
		This class is used to preprocess videos with 4 quadrants.
		The video must be put in the folder Videos folder and have be named in YYYY-MM-DD HH-MM-SS format
		and it will be compared with the corresponding date/time data the cleanData folder in order to 
		synchronize the time stamps and only keep one quadrant of the video 
		'''
		self.date = date
    
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
		For all the videos in Videos create the corresponding folders in Video data
		'''
		for file in os.listdir('Videos/'):
			filename, ext = os.path.splitext(file)
			if ext == '.mkv':
				date, time = filename.replace('-', '.').split(' ')
				if not os.path.exists('Video data'):
					os.makedirs(f'Video data')
				if not os.path.exists('Video data/' + date):
					os.makedirs('Video data/' + date)
				if not os.path.exists(f'Video data/{date}/{time}'):
					os.makedirs(f'Video data/{date}/{time}')
					
				source = f'Videos/{file}'
				destination = f'Video data/{date}/{time}/'
				if self.keep_mkv:
					shutil.move(source, destination)
				else:
					convert_to_mp4(source, f'{destination}{filename}.mp4')
			else:
				print("Wrong format file in Videos/")


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
				return

		# Synchronize the videos with the data logs
		print("Synchronizing videos with data")
		for v, data in zip(os.listdir(video_path), os.listdir(data_path)):
			self.sync_video_data(f'{video_path}{v}', f'{data_path}{data}')

		# Split the videos to only keep the relevant parts
		self.split_video_into_4()

		print("Finished preprocessing the videos")
