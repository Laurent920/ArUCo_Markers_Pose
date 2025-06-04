This repository is forked from https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python 
refer to that repository for basic functionalities

# ArUCo-Markers-Pose-Estimation-Generation-Python

This repository contains all the code you need to generate an ArucoTag,
detect ArucoTags in images and videos, and then use the detected tags
to estimate the pose of the object. In addition to this, I have also 
included the code required to obtain the calibration matrix for your 
camera.

## 1. ArUCo Marker Generation
The file `generate_aruco_tags.py` contains the code for ArUCo Marker Generation.
You need to specify the type of marker you want to generate.

The command for running is :-  
`python generate_aruco_tags.py --id 24 --type DICT_4X4_100 --output tags/`

You can find more details on other parameters using `python generate_aruco_tags.py --help`


## 2. Calibration
The file `calibration.py` contains the code necessary for calibrating your camera. This step 
has several pre-requisites. You need to have a folder containing a set of checkerboard images 
taken using your camera. Make sure that these checkerboard images are of different poses and 
orientation. You need to provide the path to this directory and the size of the square in metres. 
You can also change the shape of the checkerboard pattern using the parameters given. Make sure this
matches with your checkerboard pattern. 

This code will generate two numpy files `calibration_matrix.npy` and `distortion_coefficients.npy`. These files are required to execute the next step that involves pose estimation. 
Note that the calibration and distortion numpy files given in my repository is obtained specifically for my camera 
and might not work well for yours.   

The command for running is :-  
`python calibration.py --dir calibration_checkerboard/ --use_video "wide calib.MP4" --square_size 0.024`

You can find more details on other parameters using `python calibration.py --help`  


## 3. ArUCo Marker Detection
The file `detect_aruco_video.py` contains the code for detecting
ArUCo Markers in videos. You need to specify the path to
the video file and the type of marker you want to detect.

The command for running is :-  
**For inference using webcam feed**  
`python detect_aruco_video.py --type DICT_4X4_100 --camera True `  
**For inference using video file**   
`python detect_aruco_video.py --type DICT_4X4_100 --camera False --video Videos/test_video.mp4`  

You can find more details on other parameters using `python detect_aruco_images.py --help`
and `python detect_aruco_video.py --help`


## 4. Pose Estimation  
The file `pose_estimation.py` contains the code that performs pose estimation after detecting the 
ArUCo markers. This file will store the position of the arucos markers in a file named `marker_pose.log` in the directory of the video, this file contains the coordinates of all the detected aruco marker and their corresponding frames and this will be used by the file `pose_data.py`. To call this file your directory needs to be outside ArUCo_Markers_Pose and you need to provide the video like:

`python -m ArUCo_Markers_Pose.pose_estimation -v 'ArUCo_Markers_Pose/Videos/test_video.MP4' -s True -edmo 'Snake' -p ./ArUCo_Markers_Pose/`

You can find more details on other parameters using `python -m ArUCo_Markers_Pose.pose_estimation --help`  

## 5. Pose data
This file is used to compute the position of an EDMO over time (only Snake configuration is implemented for now), it uses the `marker_pose.log` file and 
returns a `Pose_data` object which contains the x, y, z positions of the EDMO along with some  
error values on the pose estimation. This can be called like:

`python -m ArUCo_Markers_Pose.pose_data --path "ArUCo_Markers_Pose/Videos/"`

### <ins>Notes</ins>
The `utils.py` contains the ArUCo Markers dictionary and the other utility function to display the detected markers.

Feel free to reach out to me in case of any issues.  
If you find this repo useful in any way please do star ⭐️ it so that others can reap it's benefits as well.

Happy Learning! Keep chasing your dreams!

## References
1. https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python
2. https://docs.opencv.org/4.x/d9/d6d/tutorial_table_of_content_aruco.html
3. https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

