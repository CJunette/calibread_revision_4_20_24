import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import configs
from read.read_reading import read_raw_reading
from read.read_text import read_text_sorted_mapping_and_group_with_para_id


def sort_and_save_reading_with_matrix_x(before_sort_reading_type="unsorted_reading", after_sort_reading_type="reading"):
    """
    unsorted_reading中的数据都是按照实验的iteration来排列的，现在需要将它转为按照文章顺序（matrix_x）排列。
    before_sort_reading_type: 修改前的目录。在这里我将所有被试原来的reading数据都更名为unsorted_reading了（避免覆盖原始数据）。一般情况下，文件夹下是不存在这个目录的，reading数据就被放在reading目录下。
    after_sort_reading_type: 修改后的目录。修改之后的文件需要被存放到这里。
    """
    file_prefix = f'{os.getcwd()}/data/original_gaze_data/{configs.round_num}/{configs.exp_device}'
    file_paths = os.listdir(file_prefix)
    file_paths.sort()

    raw_reading_list = []
    for file_index, file_path in enumerate(file_paths):
        raw_reading_list_1 = []
        reading_file_path_prefix = f"{file_prefix}/{file_path}/{before_sort_reading_type}"
        reading_file_paths = os.listdir(reading_file_path_prefix)
        reading_file_paths.sort(key=lambda x: int(x.split(".")[0]))
        for reading_file_index, reading_file_path in enumerate(reading_file_paths):
            reading_file = pd.read_csv(f"{reading_file_path_prefix}/{reading_file_path}")
            if "Unnamed: 0" in reading_file.columns:
                reading_file = reading_file.drop(columns=["Unnamed: 0"])
            raw_reading_list_1.append(reading_file)
        raw_reading_list.append(raw_reading_list_1)

    for subject_index in range(len(raw_reading_list)):
        sorted_list = [None for _ in range(len(raw_reading_list[subject_index]))]
        for text_index in range(len(raw_reading_list[subject_index])):
            print(f"processing: subject_{subject_index}, text_{text_index}")

            df = raw_reading_list[subject_index][text_index]
            matrix_x = df["matrix_x"].tolist()[0]
            sorted_list[int(matrix_x)] = df
        save_file_path = f"{file_prefix}/{file_paths[subject_index]}/{after_sort_reading_type}"
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        for text_index in range(len(sorted_list)):
            sorted_list[text_index].to_csv(f"{save_file_path}/{text_index}.csv", index=False, encoding="utf-8-sig")


def compress_gaze_cluster_and_save():
    reading_data = read_raw_reading("original", "_after_trim")

    compress_ratio = 1 / (configs.right_down_text_center[0] - configs.left_top_text_center[0]) * configs.text_width
    for subject_index in range(len(reading_data)):
        print(f"processing subject {subject_index}")
        gaze_x_list = []
        gaze_y_list = []
        compressed_gaze_x_list = []
        compressed_gaze_y_list = []
        gaze_info_list = []
        for text_index in range(len(reading_data[subject_index])):
            reading_df = reading_data[subject_index][text_index]
            gaze_x = reading_df["gaze_x"].tolist()
            gaze_y = reading_df["gaze_y"].tolist()
            compressed_gaze_x = [gaze_x[i] * compress_ratio for i in range(len(gaze_x))]
            compressed_gaze_y = [gaze_y[i] for i in range(len(gaze_y))]
            gaze_info = [{"text_index": text_index, "gaze_index": i} for i in range(len(gaze_x))]

            gaze_x_list.extend(gaze_x)
            gaze_y_list.extend(gaze_y)
            compressed_gaze_x_list.extend(compressed_gaze_x)
            compressed_gaze_y_list.extend(compressed_gaze_y)
            gaze_info_list.extend(gaze_info)

        # cluster compressed_gaze
        compressed_coordinates = [[compressed_gaze_x_list[i], compressed_gaze_y_list[i]] for i in range(len(compressed_gaze_x_list))]
        compressed_coordinates = np.array(compressed_coordinates)
        kmeans = KMeans(n_clusters=configs.row_num, random_state=configs.random_seed).fit(compressed_coordinates)
        labels = kmeans.labels_

        # create label column.
        for text_index in range(len(reading_data[subject_index])):
            reading_data[subject_index][text_index]["row_label"] = [-1 for _ in range(reading_data[subject_index][text_index].shape[0])]

        # 将label按照y的均值大小排序，重新更新一下label。
        df = pd.DataFrame(compressed_coordinates, columns=['x', 'y'])
        df['label'] = labels
        label_group = df.groupby('label')
        cluster_centers = df.groupby('label')['y'].mean().sort_values().index
        label_map = {old_label: new_label for new_label, old_label in enumerate(cluster_centers)}
        # 应用映射关系，更新 label
        df['new_label'] = df['label'].map(label_map)
        labels = df['new_label'].values.tolist()

        for label_index in range(len(labels)):
            text_index = gaze_info_list[label_index]["text_index"]
            gaze_index = gaze_info_list[label_index]["gaze_index"]
            reading_data[subject_index][text_index]["row_label"].iloc[gaze_index] = labels[label_index]

    file_path_prefix = f"{os.getcwd()}/data/original_gaze_data/{configs.round_num}/{configs.exp_device}"
    subject_name_list = os.listdir(file_path_prefix)
    subject_name_list.sort()

    for subject_index in range(len(subject_name_list)):
        file_path = f"{file_path_prefix}/{subject_name_list[subject_index]}/reading_after_cluster"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        for text_index in range(len(reading_data[subject_index])):
            reading_df = reading_data[subject_index][text_index]
            reading_df.to_csv(f"{file_path}/{text_index}.csv", index=False, encoding="utf-8-sig")

    return reading_data


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

