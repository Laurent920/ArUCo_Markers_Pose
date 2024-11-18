import json
from utils import *
import argparse
import os
import re

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