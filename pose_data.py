import cv2
import sys
from utils import *
import argparse
import os
import shutil
import ffmpeg
from datetime import datetime


class Pose_data():
    def __init__(self) -> None:
        pass
    
    
    
if __name__ == '__main__':
    # Example usage : python pose_estimation.py -v 'Videos(mkv)/GX010412.MP4'
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy", help="Path to calibration matrix (numpy file)")
    args = vars(ap.parse_args())