import json
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation


class Aruco_Const:
    # in cm
    mat_width = 110
    mat_length = 170 
    
    border_size = 1.6
    marker_size = 4
    origin_value = (marker_size + border_size)/2 
    x_dist = mat_length-2*origin_value
    y_dist = mat_width-2*origin_value


    tag_poses:dict[str, tuple] = {}
    tag_poses[4] = (origin_value, origin_value)
    tag_poses[0] = (origin_value, mat_width-origin_value)
    tag_poses[2] = (mat_length-origin_value, mat_width-origin_value)
    tag_poses[3] = (mat_length-origin_value, origin_value)
    
    # distance between middle tag (if existing) and the leg tags
    # tag_poses['snake 98']
    tag_poses['spider 98'] = 5.03

    # dist between tag 1-47
    # tag_poses['snake y']
    tag_poses['spider y'] = 10.06
    
    # dist between tag 5-24
    tag_poses['spider x'] = 10.06
    # spider marker dist = 60.60 mm = 6.06 cm
    #snake marker = 4x4
    #spider marker = 3x3
    
    
class Pose_data():
    def __init__(self, dir_path:str, edmo_type:str="Snake", dict_all_pos=None):
        self.dict_all_pos: dict[str, dict[str, list[list, list]]] = dict_all_pos
        self.edmo_type = edmo_type
        self.dir_path = dir_path
        for filename in os.listdir(dir_path):
            pattern = r"^marker_pose*\.log$"
            if not re.match(pattern, filename):
                continue
            
            with open(f'{dir_path}/{filename}', 'r') as f:
                self.dict_all_pos = json.load(f)
        
    def get_pose(self):
        if self.dict_all_pos is None:
            print(f'The file marker_pose.log is missing from the directory, run Aruco_pose first')
            return False
        if not self.dict_all_pos:
            print('Position dictionary is empty, make sure the aruco markers are captured properly')
            return False
        self.edmo_poses: dict[int, list[int]] = {}
        self.edmo_rots: dict[int, list[int]] = {}
        self.x, self.y, self.z, self.t = [], [], [], []
        self.nbFrames = None
        self.x_avg_error, self.y_avg_error, self.z_avg_error = 0, 0, 0
        self.rx_avg_error, self.ry_avg_error, self.rz_avg_error = 0, 0, 0
        self.avg_denom = 0
        
        tags = list(self.dict_all_pos.keys())
        self.nbFrames = max(max(int(key) for key in self.dict_all_pos[tag].keys()) for tag in tags) 
        print(f'nb of frames: {self.nbFrames}')
        
        for frame in range(1, self.nbFrames):
            marker_pose_per_frame = {}
            for tag in tags:
                frames_dict = self.dict_all_pos[tag]

                key = str(frame)
                if key in frames_dict:
                    marker_pose_per_frame[tag] = frames_dict[key]
            
            a = self.compute_error(marker_pose_per_frame)
            if a:
                print(f'Too big distance error on frame: {frame}')
                #  return
            pose_rot = self.compute_pose(marker_pose_per_frame) 
            if pose_rot:
                self.edmo_poses[frame] = pose_rot[0]
                self.edmo_rots[frame] = pose_rot[1]
        
        for i in range(1, self.nbFrames):
            if i in self.edmo_poses:
                pose = self.edmo_poses[i]
                self.x.append(pose[0])
                self.y.append(pose[1])
                self.z.append(pose[2]) 
                self.t.append(i)    
        with open(f"{self.dir_path}/error.log", "w") as f:
            f.write(f' x error: {self.x_avg_error}\n y error: {self.y_avg_error}\n z error: {self.z_avg_error}\n x rotation error: {self.rx_avg_error}\n y rotation error: {self.ry_avg_error}\n z rotation error: {self.rz_avg_error}\n ')
            f.write(f'average error over {self.avg_denom} arucos :')
            if self.avg_denom != 0:
                f.write(f' x error: {self.x_avg_error/self.avg_denom} m\n y error: {self.y_avg_error/self.avg_denom} m\n z error: {self.z_avg_error/self.avg_denom} m\n x rotation error: {self.rx_avg_error/self.avg_denom}\n y rotation error: {self.ry_avg_error/self.avg_denom}\n z rotation error: {self.rz_avg_error/self.avg_denom}')
            else:
                print(f"self.avg_denom == 0, no average computation")
        return True
      
         
    def get_pose_for_frames(self, frame_start, frame_end):
        if frame_start <= 0 or frame_end > self.nbFrames:
            print('Frame arguments for "get_pose_for_frames" are not valid !!')
            return None
        
        pose, rot = [], []
        for i in range(frame_start, frame_end):
            if i in self.edmo_poses and i in self.edmo_rots:
                pose.append(self.edmo_poses[i])
                rot.append(self.edmo_rots[i])        
        return [pose, rot]                
    
    
    def compute_pose(self, marker_pose_per_frame):
        if self.edmo_type == "Snake":
            tags = ("[1]", "[47]")
            return self.get_average_pose_rot(marker_pose_per_frame, tags)
        elif self.edmo_type == "Spider":
            tags = (("[1]", "[47]"), ("[5]", "[24]"))
            middle_pose = marker_pose_per_frame["[98]"] if "[98]" in marker_pose_per_frame else [[0,0,0], [0,0,0]]
            for tag_comb in tags:
                pose_rot = self.get_average_pose_rot(marker_pose_per_frame, tag_comb)
                if pose_rot:
                    middle_pose = [[middle_pose[i] + pose for i, pose in enumerate(pose_rot[j])] for j in range(len(pose_rot))]
            return middle_pose
        else:
            print(f"{self.edmo_type} is not recognized, use Snake or Spider")                    
            
            
    def get_average_pose_rot(self, marker_pose, tags):
        m1, m2 = tags[0], tags[1]
        if m1 in marker_pose and m2 in marker_pose:
            p1 = marker_pose[m1][0]
            p2 = marker_pose[m2][0]
            pose = [(c1+c2)/2 for c1, c2 in zip(p1, p2)]
            r1 = marker_pose[m1][1]
            r2 = marker_pose[m2][1]
            rot = [(c1+c2)/2 for c1, c2 in zip(r1, r2)]
            return [pose, rot]
        else:
            return None
        
        
    def compute_error(self, marker_pose_per_frame):
        tags = ["[0]","[2]","[3]","[4]"]
        x_dist, y_dist = Aruco_Const.x_dist/100, Aruco_Const.y_dist/100
        corners = {}
        for tag in tags:
            if tag in marker_pose_per_frame:
                corners[int(tag[1])] = marker_pose_per_frame[tag]
        # print(corners)
        for tag1 in corners:
            for tag2 in corners:
                if tag1 == tag2:
                    continue
                
                old_x = self.x_avg_error
                old_y = self.y_avg_error
                match (tag1, tag2):
                    case (0, 2) | (4, 3):
                        self.x_avg_error += abs(corners[tag2][0][0] - corners[tag1][0][0]) - x_dist
                        self.y_avg_error += (corners[tag2][0][1] - corners[tag1][0][1])
                    case (0, 3) | (2, 4):
                        self.x_avg_error += abs(corners[tag2][0][0] - corners[tag1][0][0]) - x_dist
                        self.y_avg_error += abs(corners[tag2][0][1] - corners[tag1][0][1]) - y_dist
                    case (4, 0) | (3, 2):
                        self.x_avg_error += (corners[tag2][0][0] - corners[tag1][0][0])
                        self.y_avg_error += abs(corners[tag2][0][1] - corners[tag1][0][1]) - y_dist
                    case _:
                        continue
                
                self.z_avg_error += (corners[tag2][0][2] - corners[tag1][0][2])
                self.rx_avg_error += (corners[tag2][1][0] - corners[tag1][1][0])
                self.ry_avg_error += (corners[tag2][1][1] - corners[tag1][1][1])
                self.rz_avg_error += (corners[tag2][1][2] - corners[tag1][1][2])
                self.avg_denom += 1
                
                threshold = 1
                if abs(old_x - self.x_avg_error) > threshold:
                    print(tag1, tag2)
                    print(f'x: {abs(old_x - self.x_avg_error)} > {threshold} m ')
                    if abs(old_y - self.y_avg_error) > threshold:
                        print(f'y: {abs(old_y - self.y_avg_error) } > {threshold} m ')
                        if (corners[tag2][0][2] - corners[tag1][0][2]) > threshold:
                            print(f'z: {(corners[tag2][0][2] - corners[tag1][0][2])} > {threshold} m ')
                    return True
                elif abs(old_y - self.y_avg_error) > threshold:
                    print(tag1, tag2)
                    print(f'y: {abs(old_y - self.y_avg_error) } > {threshold} m ')
                    if (corners[tag2][0][2] - corners[tag1][0][2]) > threshold:
                        print(f'z: {(corners[tag2][0][2] - corners[tag1][0][2])} > {threshold} m ')
                    return True
                elif (corners[tag2][0][2] - corners[tag1][0][2]) > threshold:
                    print(tag1, tag2)
                    print(f'z: {(corners[tag2][0][2] - corners[tag1][0][2])} > {threshold} m ')
                    return True
                else:
                    return False
        return False
        
         
if __name__ == '__main__':
    # Example usage : python pose_data.py -v 'Videos/'
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", default=None, help="Path to the directory that contains the marker_pose.log file")
    ap.add_argument("-edmo", "--EDMO_type", default='Snake', help="Type of EDMO (Spider, Snake, ...)")
    args = vars(ap.parse_args())
    
    path = "./Videos(mkv)/"
    path = "cleanData/2024.09.23/Snake/15.19.22/"
    path = "exploreData/Snake/2700-2879/"
    # path = "./"
    
    
    
    path = args["path"]
    EDMO_type = args['EDMO_type']
    
    pose_data = Pose_data(path, edmo_type=EDMO_type)
    pose_data.get_pose()
