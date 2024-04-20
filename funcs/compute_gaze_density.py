import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

import configs
from read.read_reading import read_raw_reading


def compute_and_save_gaze_density(distance_threshold=configs.text_height):
    reading_data = read_raw_reading("original", "_after_cluster")
    save_path_prefix = f"{os.getcwd()}/data/original_gaze_data/{configs.round_num}/{configs.exp_device}"
    save_path_list = os.listdir(save_path_prefix)
    save_path_list.sort()

    for subject_index in range(len(reading_data)):
        for text_index in range(len(reading_data[subject_index])):
            print(f"processing: subject_{subject_index}, text_{text_index}")

            df = reading_data[subject_index][text_index]
            gaze_x = df["gaze_x"].tolist()
            gaze_y = df["gaze_y"].tolist()
            gaze_x = [float(x) for x in gaze_x]
            gaze_y = [float(y) for y in gaze_y]
            gaze_coordinates = [[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))]
            gaze_coordinates = np.array(gaze_coordinates)

            tree = NearestNeighbors(n_neighbors=len(gaze_coordinates), algorithm='kd_tree').fit(gaze_coordinates)
            density_list = []
            distance_list, index_list = tree.kneighbors(gaze_coordinates)
            for gaze_index, gaze_coordinate in enumerate(gaze_coordinates):
                distance = distance_list[gaze_index]
                index = index_list[gaze_index]
                # 筛选那些距离小于distance_threshold的点。
                sub_index = 0
                while sub_index < len(distance):
                    if distance[sub_index] > distance_threshold:
                        break
                    sub_index += 1
                index_within_threshold = [index[i] for i in range(sub_index)]
                index_within_threshold.sort()

                # 筛选那些与当前点在时序上相邻的点。
                # 找到gaze_index在index_within_threshold中的位置
                gaze_index_within_threshold = index_within_threshold.index(gaze_index)
                density = 0
                # 向后检索。
                probe_index = gaze_index_within_threshold + 1
                while probe_index < len(index_within_threshold):
                    if index_within_threshold[probe_index] - index_within_threshold[probe_index - 1] > 1:
                        break
                    probe_index += 1
                    density += 1

                # 向前检索。
                probe_index = gaze_index_within_threshold - 1
                while probe_index >= 0:
                    if index_within_threshold[probe_index + 1] - index_within_threshold[probe_index] > 1:
                        break
                    probe_index -= 1
                    density += 1

                density_list.append(density)
            df["density"] = density_list
            # relative_density_list = [density / max(density_list) for density in density_list]
            # df["relative_density"] = relative_density_list

            save_path = f"{save_path_prefix}/{save_path_list[subject_index]}/reading_after_cluster/{text_index}.csv"
            df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return reading_data