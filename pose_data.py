import json
from utils import *
import argparse
import os
import re
from dataclasses import dataclass


@dataclass
class corner_pose:
    # in cm
    mat_width = 110
    mat_length = 170 
    
    border_size = 1.6
    marker_size = 4
    origin_value = (marker_size + border_size)/2 
    tag_poses:dict[str, tuple] = {}
    tag_poses[4] = (origin_value, origin_value)
    tag_poses[0] = (origin_value, mat_width-origin_value)
    tag_poses[2] = (mat_length-origin_value, mat_width-origin_value)
    tag_poses[3] = (mat_length-origin_value, origin_value)
    
    
class Pose_data():
    
    
    dict_all_pose: dict[str, dict[str, list[list, list]]]
    
    def __init__(self, dir_path:str):
        for filename in os.listdir(dir_path):
            pattern = r"^marker_pose*\.log$"
            if not re.match(pattern, filename):
                continue
            
            with open(filename, 'r') as f:
                dict_all_pose = json.load(f)
        
    
    
    
    
if __name__ == '__main__':
    # Example usage : python pose_estimation.py -v 'Videos(mkv)/GX010412.MP4'
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy", help="Path to calibration matrix (numpy file)")
    args = vars(ap.parse_args())