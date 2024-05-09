import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

import configs
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

from funcs.gradient_descent import gradient_descent_affine, gradient_descent_translate_rotate_shear_scale
from funcs.manual_calibrate_for_std import apply_transform_to_calibration, compute_distance_between_std_and_correction
from funcs.point_matching import point_matching_multi_process, random_select_for_supplement_point_pairs, random_select_for_gradient_descent
from funcs.util_functions import change_homogeneous_vector_to_2d_vector, change_2d_vector_to_homogeneous_vector


def _prepare_dicts_for_text_point(text_data, calibration_data):
    # 用一个dict来记录所有有效的文本点。
    text_point_dict = {}
    for row_index in range(len(calibration_data[2])):
        for col_index in range(len(calibration_data[2][row_index])):
            x = calibration_data[2][row_index][col_index]["point_x"]
            y = calibration_data[2][row_index][col_index]["point_y"]
            text_point_dict[(x, y)] = 0
    for text_index in range(len(text_data)):
        for index, row in text_data[text_index].iterrows():
            x = row["x"]
            y = row["y"]
            word = row["word"]
            if word != "blank_supplement" and (x, y) in text_point_dict:
                text_point_dict[(x, y)] += 1
    # 用一个dict来记录非blank_supplement，且至少有过一次文字的文本点。
    effective_text_point_dict = {}
    for key in text_point_dict:
        if text_point_dict[key] != 0:
            effective_text_point_dict[key] = text_point_dict[key]
    text_point_total_utilized_count = 0
    for key in effective_text_point_dict:
        text_point_total_utilized_count += effective_text_point_dict[key]
    # 用一个dict来记录blank_supplement的文本点。
    supplement_text_point_dict = {}
    for text_index in range(len(text_data)):
        for index, row in text_data[text_index].iterrows():
            x = row["x"]
            y = row["y"]
            word = row["word"]
            if word == "blank_supplement":
                supplement_text_point_dict[(x, y)] = 0

    return text_point_dict, effective_text_point_dict, supplement_text_point_dict, text_point_total_utilized_count


def _create_text_nearest_neighbor(text_data):
    # 按文本、行号来对text point分类，然后据此生成对应的nearest neighbor。
    row_nbrs_list = [[] for _ in range(len(text_data))]
    for text_index in range(len(text_data)):
        for row_index in range(configs.row_num):
            df = text_data[text_index]
            filtered_text_data_df = df[(df["row"] == float(row_index)) & (df["word"] != "blank_supplement")]
            if filtered_text_data_df.shape[0] > 0:
                filtered_text_coordinate = filtered_text_data_df[["x", "y"]].values.tolist()
                nbr = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(filtered_text_coordinate)
                row_nbrs_list[text_index].append(nbr)
            else:
                row_nbrs_list[text_index].append(None)

    # 生成一个所有text_point的nearest neighbor。
    total_nbrs_list = []
    for text_index in range(len(text_data)):
        df = text_data[text_index]
        filtered_text_data_df = df[df["word"] != "blank_supplement"]
        text_coordinate = filtered_text_data_df[["x", "y"]].values.tolist()
        nbr = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(text_coordinate)
        total_nbrs_list.append(nbr)

    return row_nbrs_list, total_nbrs_list


def _transform_using_centroid_and_outbound(gaze_point_list_1d, selected_gaze_point_list_1d, effective_text_point_dict, reading_data, selected_reading_data, subject_index, calibration_data):
    """
    将gaze data和text point先做缩放，然后基于重心对齐。
    """
    def rotating_calipers(points):
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        min_area_rect = None
        min_area = float('inf')

        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]

            # 计算边界向量
            edge = p2 - p1
            edge_angle = math.atan2(edge[1], edge[0])
            if abs(edge_angle) > (np.pi / 4):
                continue
            # 旋转点以使边界水平
            rot_matrix = np.array([[np.cos(edge_angle), np.sin(edge_angle)],
                                   [-np.sin(edge_angle), np.cos(edge_angle)]])
            rotated_points = np.dot(rot_matrix, hull_points.T).T

            # 计算旋转后点的边界
            min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
            min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])

            # 计算矩形的面积
            area = (max_x - min_x) * (max_y - min_y)

            if area < min_area:
                min_area = area
                min_area_rect = (min_x, max_x, min_y, max_y, -edge_angle)

        return min_area_rect

    # dbscan = DBSCAN(eps=32, min_samples=5)
    dbscan = DBSCAN(eps=20, min_samples=30)
    clusters = dbscan.fit_predict(gaze_point_list_1d)
    filtered_gaze_point_list_1d = gaze_point_list_1d[clusters != -1]

    # outer_rect = rotating_calipers(filtered_gaze_point_list_1d)
    outer_rect = [np.min(filtered_gaze_point_list_1d[:, 0]), np.max(filtered_gaze_point_list_1d[:, 0]), np.min(filtered_gaze_point_list_1d[:, 1]), np.max(filtered_gaze_point_list_1d[:, 1])]

    x_scale = (configs.right_down_text_center[0] - configs.left_top_text_center[0]) / (outer_rect[1] - outer_rect[0])
    y_scale = (configs.right_down_text_center[1] - configs.left_top_text_center[1]) / (outer_rect[3] - outer_rect[2])
    scale_matrix = np.array([[x_scale, 0, 0],
                             [0, y_scale, 0],
                             [0, 0, 1]])

    selected_gaze_point_list_1d_homogeneous = [change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in selected_gaze_point_list_1d]
    selected_gaze_point_list_1d_homogeneous = [np.dot(scale_matrix, gaze_point) for gaze_point in selected_gaze_point_list_1d_homogeneous]
    selected_gaze_point_list_1d = np.array([change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in selected_gaze_point_list_1d_homogeneous])

    # filtered_gaze_point_max_x = np.max(gaze_point_list_1d[clusters != -1][:, 0])
    # filtered_gaze_point_min_x = np.min(gaze_point_list_1d[clusters != -1][:, 0])
    # filtered_gaze_point_max_y = np.max(gaze_point_list_1d[clusters != -1][:, 1])
    # filtered_gaze_point_min_y = np.min(gaze_point_list_1d[clusters != -1][:, 1])
    # gaze_point_center = np.array([(filtered_gaze_point_max_x + filtered_gaze_point_min_x) / 2,
    #                               (filtered_gaze_point_max_y + filtered_gaze_point_min_y) / 2])

    gaze_point_center = np.array([(outer_rect[0] + outer_rect[1]) / 2, (outer_rect[2] + outer_rect[3]) / 2])

    text_point_center = np.array([0, 0])
    for key, value in effective_text_point_dict.items():
        text_point_center[0] += key[0]
        text_point_center[1] += key[1]
    text_point_center[0] /= len(effective_text_point_dict)
    text_point_center[1] /= len(effective_text_point_dict)
    translate_vector = np.array(text_point_center - gaze_point_center)
    translate_matrix = np.array([[1, 0, translate_vector[0]],
                                 [0, 1, translate_vector[1]],
                                 [0, 0, 1]])

    selected_gaze_point_list_1d_homogeneous = [change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in selected_gaze_point_list_1d]
    selected_gaze_point_list_1d_homogeneous = [np.dot(translate_matrix, gaze_point) for gaze_point in selected_gaze_point_list_1d_homogeneous]
    selected_gaze_point_list_1d = np.array([change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in selected_gaze_point_list_1d_homogeneous])

    for text_index in range(len(selected_reading_data)):
        gaze_x = selected_reading_data[text_index]["gaze_x"]
        gaze_y = selected_reading_data[text_index]["gaze_y"]
        gaze_homogeneous = [change_2d_vector_to_homogeneous_vector([gaze_x.iloc[i], gaze_y.iloc[i]]) for i in range(len(gaze_x))]
        gaze_homogeneous = [np.dot(translate_matrix, np.dot(scale_matrix, gaze_point)) for gaze_point in gaze_homogeneous]
        gaze_1d = [change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_homogeneous]
        selected_reading_data[text_index]["gaze_x"] = [gaze_1d[i][0] for i in range(len(gaze_1d))]
        selected_reading_data[text_index]["gaze_y"] = [gaze_1d[i][1] for i in range(len(gaze_1d))]

    # 把calibration的数据也做一下相同的变换。
    for row_index in range(len(calibration_data[1])):
        for col_index in range(len(calibration_data[1][row_index])):
            avg_gaze_x = calibration_data[1][row_index][col_index]["avg_gaze_x"]
            avg_gaze_y = calibration_data[1][row_index][col_index]["avg_gaze_y"]
            avg_gaze_homogeneous = change_2d_vector_to_homogeneous_vector([avg_gaze_x, avg_gaze_y])
            avg_gaze_homogeneous = np.dot(translate_matrix, np.dot(scale_matrix, avg_gaze_homogeneous))
            avg_gaze_2d = change_homogeneous_vector_to_2d_vector(avg_gaze_homogeneous)
            calibration_data[1][row_index][col_index]["avg_gaze_x"] = avg_gaze_2d[0]
            calibration_data[1][row_index][col_index]["avg_gaze_y"] = avg_gaze_2d[1]

            gaze_x = calibration_data[0][row_index][col_index]["gaze_x"]
            gaze_y = calibration_data[0][row_index][col_index]["gaze_y"]
            gaze_homogeneous = [change_2d_vector_to_homogeneous_vector([gaze_x[i], gaze_y[i]]) for i in range(len(gaze_x))]
            gaze_homogeneous = [np.dot(translate_matrix, np.dot(scale_matrix, gaze_point)) for gaze_point in gaze_homogeneous]
            gaze_2d = [change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_homogeneous]
            calibration_data[0][row_index][col_index]["gaze_x"] = [gaze_2d[i][0] for i in range(len(gaze_2d))]
            calibration_data[0][row_index][col_index]["gaze_y"] = [gaze_2d[i][1] for i in range(len(gaze_2d))]

    return selected_gaze_point_list_1d, effective_text_point_dict, selected_reading_data, calibration_data, scale_matrix, translate_matrix


# def point_matching_single_process(reading_data, gaze_point_list_1d, text_data, filtered_text_data_list,
#                                   total_nbrs_list, row_nbrs_list,
#                                   effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
#                                   distance_threshold):
#     np.random.seed(configs.random_seed)
#
#     # 1. 首先遍历所有的reading point，找到与其row label一致的、距离最近的text point，然后将这些匹配点加入到point_pair_list中。
#     point_pair_list = []
#     weight_list = []
#     row_label_list = []
#     for text_index in range(len(reading_data)):
#         if text_index in configs.training_index_list:
#             continue
#         reading_df = reading_data[text_index]
#         for row_index in range(configs.row_num):
#             filtered_reading_df = reading_df[reading_df["row_label"] == row_index]
#
#             point_pair_list_1 = []
#             weight_list_1 = []
#             row_label_list_1 = []
#             distance_list_1 = []
#             if row_nbrs_list[text_index][row_index] and filtered_reading_df.shape[0] != 0:
#                 filtered_reading_coordinates = filtered_reading_df[["gaze_x", "gaze_y"]].values.tolist()
#                 filtered_distances_of_row, filtered_indices_of_row = row_nbrs_list[text_index][row_index].kneighbors(filtered_reading_coordinates)
#                 filtered_text_data = filtered_text_data_list[text_index][row_index]
#                 filtered_text_coordinate = filtered_text_data[["x", "y"]].values.tolist()
#                 filtered_reading_density = filtered_reading_df["density"].values.tolist()
#
#                 text_coordinate = text_data[text_index][["x", "y"]].values.tolist()
#                 filtered_distance_of_all_text, filtered_indices_of_all_text = total_nbrs_list[text_index].kneighbors(filtered_reading_coordinates)
#
#                 for gaze_index in range(len(filtered_distances_of_row)):
#                     if filtered_distances_of_row[gaze_index][0] < distance_threshold:
#                         point_pair_list_1.append([filtered_reading_coordinates[gaze_index], filtered_text_coordinate[filtered_indices_of_row[gaze_index][0]]])
#                         prediction = filtered_text_data.iloc[filtered_indices_of_row[gaze_index][0]]["prediction"]
#                         density = filtered_reading_density[gaze_index]
#                         distance_list_1.append(filtered_distances_of_row[gaze_index][0])
#                         # weight = 1 / abs(density - prediction) * 5
#                         weight = configs.weight_divisor / (abs(density - prediction) + configs.weight_intercept)
#                         if configs.bool_weight:
#                             weight_list_1.append(weight)
#                         else:
#                             weight_list_1.append(1)
#                     else:
#                         point_pair_list_1.append([filtered_reading_coordinates[gaze_index], text_coordinate[filtered_indices_of_all_text[gaze_index][0]]])
#                         if text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["word"] == "blank_supplement":
#                             weight = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["penalty"]
#                         else:
#                             prediction = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["prediction"]
#                             density = filtered_reading_density[gaze_index]
#                             distance_list_1.append(filtered_distance_of_all_text[gaze_index][0])
#                             # weight = 1 / abs(density - prediction) * 5
#                             weight = configs.weight_divisor / (abs(density - prediction) + configs.weight_intercept)
#                         if configs.bool_weight:
#                             weight_list_1.append(weight)
#                         else:
#                             weight_list_1.append(1)
#                     row_label_list_1.append(row_index)
#
#             for gaze_index in range(len(point_pair_list_1)):
#                 text_x = point_pair_list_1[gaze_index][1][0]
#                 text_y = point_pair_list_1[gaze_index][1][1]
#                 if (text_x, text_y) in actual_text_point_dict:
#                     actual_text_point_dict[(text_x, text_y)] += 1
#                 if (text_x, text_y) in actual_supplement_text_point_dict:
#                     actual_supplement_text_point_dict[(text_x, text_y)] += 1
#
#             point_pair_list.extend(point_pair_list_1)
#             weight_list.extend(weight_list_1)
#             row_label_list.extend(row_label_list_1)
#
#     # 2. 接下来做的是确认有文字，但没有reading data的text_unit，并根据其最近的reading data，添加额外的点对。该添加点对不受文章序号限制。
#     # 生成一个所有reading point的nearest neighbor。
#     total_reading_nbrs = NearestNeighbors(n_neighbors=int(len(gaze_point_list_1d)/4), algorithm='kd_tree').fit(gaze_point_list_1d)
#     # 生成每个文本每个reading data的nearest neighbor。
#     reading_nbrs_list = []
#     for text_index in range(len(reading_data)):
#         reading_df = reading_data[text_index]
#         reading_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
#         reading_nbrs = NearestNeighbors(n_neighbors=int(len(reading_coordinates)/4), algorithm='kd_tree').fit(reading_coordinates)
#         reading_nbrs_list.append(reading_nbrs)
#
#     supplement_point_pair_list = []
#     supplement_weight_list = []
#     supplement_row_label_list = []
#     # 然后找出那些没有任何匹配的actual text point，将其与最近的阅读点匹配。
#     total_effective_text_point_num = sum(effective_text_point_dict.values())
#     point_pair_length = len(point_pair_list)
#     # iterate over actual_text_point_dict
#     for key, value in actual_text_point_dict.items():
#         if value == 0:
#             closet_point_num = int(point_pair_length * effective_text_point_dict[key] / total_effective_text_point_num)
#             cur_text_point = [float(key[0]), float(key[1])]
#             distances, indices = total_reading_nbrs.kneighbors([cur_text_point])
#             # 对于右下角的未被匹配的文本点，我们将其权重放大10倍。（没实施）
#             if (key[0] == configs.right_down_text_center[0] and (key[1] == configs.right_down_text_center[1] or key[1] == configs.right_down_text_center[1] - configs.text_width)) or \
#                     (key[0] == configs.right_down_text_center[0] - configs.text_height and key[1] == configs.right_down_text_center[1]):
#                 weight = configs.completion_weight * configs.right_down_corner_unmatched_ratio
#             else:
#                 weight = configs.completion_weight
#
#             for point_index in range(closet_point_num):
#                 current_point_index = indices[0][point_index]
#                 gaze_point = gaze_point_list_1d[current_point_index].tolist()
#                 # point_pair_list.append([gaze_point, cur_text_point])
#                 # weight_list.append(weight)
#                 # row_label_list.append(-1)
#                 supplement_point_pair_list.append([gaze_point, cur_text_point])
#                 supplement_weight_list.append(weight)
#                 supplement_row_label_list.append(-1)
#
#     # raw_point_pair_length = len(point_pair_list)
#
#     # 3. 对于横向最外侧的补充点或空格点（即左右侧紧贴近正文的点），都可以考虑额外添加一些匹配点对，添加的weight是负数。
#     # 这里单独为要添加boundary的point pair生成list，方便后续筛选。
#     left_point_pair_list = []
#     right_point_pair_list = []
#     left_weight_list = []
#     right_weight_list = []
#     left_row_label_list = []
#     right_row_label_list = []
#     bottom_point_pair_list = []
#     bottom_weight_list = []
#     bottom_row_label_list = []
#
#     for text_index in range(len(text_data)):
#         text_df = text_data[text_index]
#         row_list = text_df["row"].unique().tolist()
#
#         for row_index in range(len(row_list)):
#             # 对于最后一行，我们为这里的blank_supplement添加匹配。 # 这里给最后一行添加匹配的效果不好，所以去掉了。
#             if row_list[row_index] == 5.5:
#                 pass
#                 row_df = text_df[text_df["row"] == row_list[row_index]]
#                 for index in range(row_df.shape[0]):
#                     x = row_df.iloc[index]["x"]
#                     y = row_df.iloc[index]["y"]
#                     distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
#                     for point_index in range(len(indices[0])):
#                         if distances[0][point_index] < distance_threshold * configs.bottom_boundary_distance_threshold_ratio:
#                             gaze_point = reading_data[text_index].iloc[indices[0][point_index]][["gaze_x", "gaze_y"]].values.tolist()
#                             bottom_point_pair_list.append([gaze_point, [x, y]])
#                             bottom_weight_list.append(configs.empty_penalty * configs.bottom_boundary_ratio)
#                             bottom_row_label_list.append(-1)
#                         else:
#                             break
#             # 对于其它行，对左右两侧的blank_supplement添加匹配。
#             else:
#                 row_df = text_df[text_df["row"] == row_list[row_index]]
#                 if row_df[row_df["word"] != "blank_supplement"].shape[0] == 0:
#                     continue
#
#                 row_df = row_df.sort_values(by=["col"])
#                 for index in range(row_df.shape[0]):
#                     if index < row_df.shape[0] - 1:
#                         word = row_df.iloc[index]["word"]
#                         next_word = row_df.iloc[index + 1]["word"]
#                         if (word == "blank_supplement" or word.strip() == "") and (next_word != "blank_supplement" and next_word.strip() != ""):
#                             x = row_df.iloc[index]["x"]
#                             y = row_df.iloc[index]["y"]
#                             distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
#                             for point_index in range(len(indices[0])):
#                                 if distances[0][point_index] < distance_threshold * configs.left_boundary_distance_threshold_ratio:
#                                     gaze_point = reading_data[text_index].iloc[indices[0][point_index]][["gaze_x", "gaze_y"]].values.tolist()
#                                     # point_pair_list.append([gaze_point, [x, y]])
#                                     # weight_list.append(configs.empty_penalty * configs.left_boundary_ratio)
#                                     # row_label_list.append(-1)
#                                     left_point_pair_list.append([gaze_point, [x, y]])
#                                     left_weight_list.append(configs.empty_penalty * configs.left_boundary_ratio)
#                                     left_row_label_list.append(-1)
#                                 else:
#                                     break
#                             # 确保只对最左侧的空格点添加一次匹配。
#                             break
#
#                 row_df = row_df.sort_values(by=["col"], ascending=False)
#                 for index in range(row_df.shape[0]):
#                     if index < row_df.shape[0] - 1:
#                         word = row_df.iloc[index]["word"]
#                         next_word = row_df.iloc[index + 1]["word"]
#                         if (word == "blank_supplement" or word.strip() == "") and (next_word != "blank_supplement" and next_word.strip() != ""):
#                             x = row_df.iloc[index]["x"]
#                             y = row_df.iloc[index]["y"]
#                             distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
#                             for point_index in range(len(indices[0])):
#                                 if distances[0][point_index] < distance_threshold * configs.right_boundary_distance_threshold_ratio:
#                                     gaze_point = reading_data[text_index].iloc[indices[0][point_index]][["gaze_x", "gaze_y"]].values.tolist()
#                                     # point_pair_list.append([gaze_point, [x, y]])
#                                     # weight_list.append(configs.empty_penalty * configs.right_boundary_ratio)
#                                     # row_label_list.append(-1)
#                                     right_point_pair_list.append([gaze_point, [x, y]])
#                                     right_weight_list.append(configs.empty_penalty * configs.right_boundary_ratio)
#                                     right_row_label_list.append(-1)
#                                 else:
#                                     break
#                             # 确保只对最右侧的空格点添加一次匹配。
#                             break
#
#     # 4. 限制添加点的数量。
#     # 对于那些validation数量过少的情况，需要限制supplement point pair的数量，保证其与raw point pair的比例，避免出现过分的失调。这里的限制我修改过，看一下如果效果不好就还原回去。
#     supplement_point_pair_list, supplement_weight_list, supplement_row_label_list = random_select_for_supplement_point_pairs(point_pair_list, supplement_point_pair_list, supplement_weight_list, supplement_row_label_list, configs.supplement_select_ratio)
#
#     # 对于那些validation数量过少的情况，需要限制boundary point pair的数量，保证其与raw point pair的比例，避免出现过分的失调。这里的限制我修改过，看一下如果效果不好就还原回去。
#     left_point_pair_list, left_weight_list, left_row_label_list = random_select_for_supplement_point_pairs(point_pair_list, left_point_pair_list, left_weight_list, left_row_label_list, configs.boundary_select_ratio)
#     right_point_pair_list, right_weight_list, right_row_label_list = random_select_for_supplement_point_pairs(point_pair_list, right_point_pair_list, right_weight_list, right_row_label_list, configs.boundary_select_ratio)
#     bottom_point_pair_list, bottom_weight_list, bottom_row_label_list = random_select_for_supplement_point_pairs(point_pair_list, bottom_point_pair_list, bottom_weight_list, bottom_row_label_list, configs.boundary_select_ratio)
#
#     point_pair_list.extend(supplement_point_pair_list)
#     weight_list.extend(supplement_weight_list)
#     row_label_list.extend(supplement_row_label_list)
#
#     point_pair_list.extend(left_point_pair_list)
#     weight_list.extend(left_weight_list)
#     row_label_list.extend(left_row_label_list)
#
#     point_pair_list.extend(right_point_pair_list)
#     weight_list.extend(right_weight_list)
#     row_label_list.extend(right_row_label_list)
#
#     point_pair_list.extend(bottom_point_pair_list)
#     weight_list.extend(bottom_weight_list)
#     row_label_list.extend(bottom_row_label_list)
#
#     return point_pair_list, weight_list, row_label_list


def _get_int_point_pairs(point_pair):
    int_point_pair = []
    for pair in point_pair:
        int_point_pair.append([[int(pair[0][0]), int(pair[0][1])], [int(pair[1][0]), int(pair[1][1])]])
    return int_point_pair


def _prepare_boundary_points(text_data, distance_threshold):
    """
    这一函数用于准备point matching中step_3所需的边缘点数据。
    """
    def _add_data_to_list(x, y, weight, boundary_type, text_index, row_index, col_index, distance_threshold):
        coordinate_list.append([x, y])
        weight_list.append(weight)
        type_list.append(string_to_text[boundary_type])
        text_index_list.append(text_index)
        row_index_list.append(row_index)
        col_index_list.append(col_index)
        distance_threshold_list.append(distance_threshold)

    string_to_text = {"left": 0, "right": 1, "top": 2, "bottom": 3}
    coordinate_list = []
    type_list = []
    weight_list = []
    text_index_list = []
    row_index_list = []
    col_index_list = []
    distance_threshold_list = []
    total_point_num_list = []

    for text_index in range(len(text_data)):
        if text_index in configs.training_index_list:
            continue
        text_df = text_data[text_index]
        row_list = text_df["row"].unique().tolist()

        for row_index in range(len(row_list)):
            total_point_num = 0
            row = row_list[row_index]
            row_df = text_df[text_df["row"] == row]
            if row_df[row_df["word"] != "blank_supplement"].shape[0] == 0:
                # 只有top和bottom的情况。
                if int(row) != row and row > 0:
                    # bottom
                    for index in range(row_df.shape[0]):
                        col = row_df.iloc[index]["col"]
                        x = row_df.iloc[index]["x"]
                        y = row_df.iloc[index]["y"]
                        boundary_type = "bottom"
                        weight = configs.empty_penalty * configs.bottom_boundary_ratio
                        distance_threshold_bottom = distance_threshold * configs.bottom_boundary_distance_threshold_ratio
                        _add_data_to_list(x, y, weight, boundary_type, text_index, row, col, distance_threshold_bottom)
                        total_point_num += 1
                elif int(row) != row and row < 0:
                    # top
                    for index in range(row_df.shape[0]):
                        col = row_df.iloc[index]["col"]
                        x = row_df.iloc[index]["x"]
                        y = row_df.iloc[index]["y"]
                        boundary_type = "top"
                        weight = row_df.iloc[index]["penalty"]
                        distance_threshold_top = distance_threshold * configs.top_boundary_distance_threshold_ratio
                        _add_data_to_list(x, y, weight, boundary_type, text_index, row, col, distance_threshold_top)
                        total_point_num += 1
                total_point_num_list.append(total_point_num)
            else:
                # 只有left和right的情况。
                half_row_num = int((configs.row_num - 1) / 2)
                total_point_num = 0
                # left的情况。
                row_df = row_df.sort_values(by=["col"])
                for index in range(row_df.shape[0]):
                    if index < row_df.shape[0]:
                        word = row_df.iloc[index]["word"]
                        if (word == "blank_supplement" or word.strip() == "") and row_df.iloc[index]["col"] < configs.col_num / 2:
                            col = row_df.iloc[index]["col"]
                            x = row_df.iloc[index]["x"]
                            y = row_df.iloc[index]["y"]
                            boundary_type = "left"
                            if row_index <= 0:
                                weight = row_df.iloc[index]["penalty"]  # 为了让左上角的数据不出问题，我将左上角blank_supplement的penalty都设置为了1（而非负数）。
                            else:
                                weight = configs.empty_penalty * configs.left_boundary_ratio
                            distance_threshold_left = distance_threshold * configs.left_boundary_distance_threshold_ratio
                            _add_data_to_list(x, y, weight, boundary_type, text_index, row, col, distance_threshold_left)
                            total_point_num += 1
                # right的情况。
                row_df = row_df.sort_values(by=["col"], ascending=False)  # 注意，这里有一个ascending=False。
                for index in range(row_df.shape[0]):
                    if index < row_df.shape[0] - 1:
                        word = row_df.iloc[index]["word"]
                        if (word == "blank_supplement" or word.strip() == "") and row_df.iloc[index]["col"] >= configs.col_num / 2:
                            col = row_df.iloc[index]["col"]
                            x = row_df.iloc[index]["x"]
                            y = row_df.iloc[index]["y"]
                            boundary_type = "right"
                            if row_index > half_row_num:
                                weight = configs.empty_penalty * (configs.right_boundary_ratio + row_index - half_row_num * configs.right_boundary_ratio_derivative)
                            else:
                                weight = configs.empty_penalty * configs.right_boundary_ratio
                            if row_index > half_row_num:
                                distance_threshold_right = distance_threshold * (configs.right_boundary_distance_threshold_ratio + configs.right_boundary_distance_threshold_ratio_derivative * (row_index - half_row_num))
                            else:
                                distance_threshold_right = distance_threshold * configs.right_boundary_distance_threshold_ratio
                            _add_data_to_list(x, y, weight, boundary_type, text_index, row, col, distance_threshold_right)
                            total_point_num += 1
                total_point_num_list.append(total_point_num)
    return coordinate_list, weight_list, type_list, text_index_list, row_index_list, col_index_list, distance_threshold_list, total_point_num_list


def _prepare_static_text_and_reading(reading_data, filtered_text_data_list, text_data):
    """
    这一函数用于准备后续point matching的step_1中需要的数据。
    """
    filtered_reading_density_list = []
    filtered_text_coordinate_list = []
    filtered_text_penalty_list = []
    filtered_text_prediction_list = []

    full_text_coordinate_list = []
    full_text_penalty_list = []
    full_text_prediction_list = []

    text_index_list = []
    row_index_list = []
    gaze_index_list = []

    for text_index in range(len(reading_data)):
        if text_index in configs.training_index_list:
            continue
        reading_df = reading_data[text_index]
        text_df = text_data[text_index]
        text_df = text_df[text_df["word"] != "blank_supplement"]

        full_text_coordinates = text_df[["x", "y"]].values.tolist() # 保证这里只有文字，没有blank_supplement。
        full_text_prediction = text_df["prediction"].values.tolist()
        full_text_penalty = text_df["penalty"].values.tolist()
        for row_index in range(configs.row_num):
            filtered_reading_df = reading_df[reading_df["row_label"] == row_index]
            filtered_reading_density = filtered_reading_df["density"].values.tolist()
            filtered_text_data = filtered_text_data_list[text_index][row_index]
            filtered_text_coordinate = filtered_text_data[["x", "y"]].values.tolist()
            filtered_text_prediction = filtered_text_data["prediction"].values.tolist()
            filtered_text_penalty = filtered_text_data["penalty"].values.tolist()

            filtered_reading_length = len(filtered_reading_density)
            if len(filtered_reading_density) == 0:
                continue
            else:
                if len(filtered_text_coordinate) == 0:
                    continue
                filtered_reading_density_list.extend(filtered_reading_density)

                filtered_text_coordinate_list.extend([torch.tensor(filtered_text_coordinate)] * filtered_reading_length)
                filtered_text_prediction_list.extend([torch.tensor(filtered_text_prediction)] * filtered_reading_length)
                filtered_text_penalty_list.extend([torch.tensor(filtered_text_penalty)] * filtered_reading_length)

                full_text_coordinate_list.extend([torch.tensor(full_text_coordinates)] * filtered_reading_length)
                full_text_prediction_list.extend([torch.tensor(full_text_prediction)] * filtered_reading_length)
                full_text_penalty_list.extend([torch.tensor(full_text_penalty)] * filtered_reading_length)
                text_index_list.extend([text_index] * filtered_reading_length)
                row_index_list.extend([row_index] * filtered_reading_length)
                gaze_index_list.extend([i for i in range(filtered_reading_length)])

    return (filtered_reading_density_list, filtered_text_coordinate_list, filtered_text_prediction_list, filtered_text_penalty_list,
            full_text_coordinate_list, full_text_prediction_list, full_text_penalty_list, text_index_list, row_index_list, gaze_index_list)


def calibrate_with_simple_linear_and_weight(model_index, subject_index, text_data, reading_data, calibration_data, max_iteration=100, distance_threshold=64):
    np.random.seed(configs.random_seed)
    # reading_data = reading_data.copy()
    selected_reading_data = []
    for text_index in range(len(reading_data)):
        df = reading_data[text_index].iloc[::5, :].copy()
        df.reset_index(drop=True, inplace=True)
        selected_reading_data.append(df)

    # 之前的代码中，centroid对齐永远是25个reading_data一起对齐。这样必然造成问题，因此这里提前把reading_data中没必要的哪些数据都去掉。
    for reading_index in configs.training_index_list:
        selected_reading_data[reading_index].drop(index=selected_reading_data[reading_index].index, inplace=True)

    log_path = f"gradient_descent_log/{configs.file_index}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # create a json file to log
    log_file_path = f"{log_path}/calibrate_with_simple_linear_and_weight-model_{model_index}-subject_{subject_index}.json"
    with open(log_file_path, "w") as log_file:
        log_file.write("[\n")
        json.dump({"validation_index_list": np.setdiff1d(np.arange(configs.passage_num), configs.training_index_list).tolist(),
                   "model_index": model_index, "subject_index": subject_index,
                   "bool_weight": configs.bool_weight, "bool_text_weight": configs.bool_text_weight,
                   "weight_divisor": configs.weight_divisor, "weight_intercept": configs.weight_intercept,
                   "location_penalty": configs.location_penalty, "punctuation_penalty": configs.punctuation_penalty,
                   "empty_penalty": configs.empty_penalty, "completion_weight": configs.completion_weight,
                   "right_down_corner_unmatched_ratio": configs.right_down_corner_unmatched_ratio,
                   "left_boundary_ratio": configs.left_boundary_ratio, "right_boundary_ratio": configs.right_boundary_ratio,
                   "top_boundary_ratio": configs.top_boundary_ratio, "bottom_boundary_ratio": configs.bottom_boundary_ratio,
                   "left_boundary_distance_threshold_ratio": configs.left_boundary_distance_threshold_ratio,
                   "right_boundary_distance_threshold_ratio": configs.right_boundary_distance_threshold_ratio,
                   "top_boundary_distance_threshold_ratio": configs.top_boundary_distance_threshold_ratio,
                   "bottom_boundary_distance_threshold_ratio": configs.bottom_boundary_distance_threshold_ratio,
                   "right_boundary_distance_threshold_ratio_derivative": configs.right_boundary_distance_threshold_ratio_derivative,
                   "right_boundary_ratio_derivative": configs.right_boundary_ratio_derivative,
                   "random_select_ratio_for_point_pair": configs.random_select_ratio_for_point_pair,
                   "last_iteration_ratio": configs.last_iteration_ratio,
                   "punctuation_ratio": configs.punctuation_ratio, "boundary_select_ratio": configs.boundary_select_ratio,
                   "supplement_select_ratio": configs.supplement_select_ratio, "gradient_descent_iteration_threshold": configs.gradient_descent_iteration_threshold,
                   "max_iteration": max_iteration
                   }, log_file, indent=4)
        log_file.write(",\n")

    # 获取1d的gaze point list。
    total_gaze_point_num = 0
    selected_gaze_point_list_1d = []
    selected_gaze_point_info_list_1d = []
    for text_index in range(len(selected_reading_data)):
        coordinates = selected_reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
        selected_gaze_point_list_1d.extend(coordinates)
        selected_gaze_row = selected_reading_data[text_index]["row_label"].values.tolist()
        selected_gaze_info = [(text_index, selected_gaze_row[i], i) for i in range(len(selected_gaze_row))]
        selected_gaze_point_info_list_1d.extend(selected_gaze_info)
        total_gaze_point_num += len(coordinates)
    selected_gaze_point_list_1d = np.array(selected_gaze_point_list_1d)
    selected_gaze_point_info_list_1d = np.array(selected_gaze_point_info_list_1d)

    gaze_point_list_1d = []
    for text_index in range(len(reading_data)):
        coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
        gaze_point_list_1d.extend(coordinates)
        total_gaze_point_num += len(coordinates)
    gaze_point_list_1d = np.array(gaze_point_list_1d)

    # 过滤出不包含blank_supplement的text data。
    filtered_text_data_list = [[] for _ in range(len(text_data))]
    for text_index in range(len(text_data)):
        # row_list = text_data[text_index]["row"].unique().tolist()
        # row_list = [row_index for row_index in row_list if row_index <= configs.row_num - 1]
        # row_list.sort()
        # for row_index in row_list:
        #     df = text_data[text_index]
        #     filtered_text_data_df = df[df["row"] == float(row_index)]
        #     filtered_text_data_df = filtered_text_data_df[(df["word"] != "blank_supplement") | (df["penalty"] > 0)]
        #     filtered_text_data_list[text_index].append(filtered_text_data_df)
        for row_index in range(configs.row_num):
            df = text_data[text_index]
            filtered_text_data_df = df[(df["row"] == float(row_index)) & (df["word"] != "blank_supplement")]
            filtered_text_data_list[text_index].append(filtered_text_data_df)

    # 生成3个dict。text_point_dict用来记录所有有效的文本点；effective_text_point_dict用来记录非blank_supplement，且至少有过一次文字的文本点。supplement_text_point_dict用来记录blank_supplement的文本点。
    text_point_dict, effective_text_point_dict, supplement_text_point_dict, text_point_total_utilized_count = _prepare_dicts_for_text_point(text_data, calibration_data)
    # 生成：按文本、行号来对text point分类得到的nearest neighbor；按文本、行号来对text point分类，且去除blank_supplement的nearest neighbor；以及所有text_point的nearest neighbor。
    # row_nbrs_list只包括每行文本的text_unit的nearest neighbor；total_nbrs_list包括所有文本的text_unit及boundary unit的nearest neighbor。这两个nearest neighbor都是为了方便后面reading data找到需要匹配的text_unit和boundary_unit。
    row_nbrs_list, total_nbrs_list = _create_text_nearest_neighbor(text_data)
    # 将gaze data和text point先做缩放，然后基于重心对齐。这里中心对齐还需要使用之前完整的text_data。
    selected_gaze_point_list_1d, effective_text_point_dict, selected_reading_data, calibration_data, scale_matrix, translate_matrix = (
        _transform_using_centroid_and_outbound(gaze_point_list_1d, selected_gaze_point_list_1d, effective_text_point_dict, reading_data, selected_reading_data, subject_index, calibration_data))

    static_text_and_reading = _prepare_static_text_and_reading(selected_reading_data, filtered_text_data_list, text_data)
    static_boundary = _prepare_boundary_points(text_data, distance_threshold)

    with open(log_file_path, "a") as log_file:
        json.dump({"scale_matrix": scale_matrix.tolist(), "translate_matrix": translate_matrix.tolist()}, log_file, indent=4)
        log_file.write(",\n")

    total_transform_matrix = np.eye(3)
    avg_error_list = []
    last_iteration_num_list = []
    last_iteration_num = 100000
    gd_error_list = []
    learning_rate_list = []
    last_grad_norm = 10000
    last_selected_point_pair_info_list = []
    last_selected_point_pair_list = []
    last_weight_list = []
    for iteration_index in range(max_iteration):
        print("iteration_index: ", iteration_index)
        # 每次迭代前，创建一个类似effective_text_point_dict的字典，用于记录每个文本点被阅读点覆盖的次数。
        actual_text_point_dict = effective_text_point_dict.copy()
        for key in actual_text_point_dict:
            actual_text_point_dict[key] = 0

        actual_supplement_text_point_dict = supplement_text_point_dict.copy()
        point_pair_list, weight_list, info_list = point_matching_multi_process(selected_reading_data, selected_gaze_point_list_1d, selected_gaze_point_info_list_1d,
                                                                               text_data, filtered_text_data_list,
                                                                               total_nbrs_list, row_nbrs_list,
                                                                               effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                                                                               static_text_and_reading, static_boundary,
                                                                               iteration_index, distance_threshold)

        random_selected_point_pair_list, random_selected_weight_list, random_selected_info_list = random_select_for_gradient_descent(iteration_index, point_pair_list, weight_list, info_list, configs.random_select_ratio_for_point_pair)

        # 将前后两次不同的匹配点保存到random_selected_point_pair_differ中。
        if len(last_selected_point_pair_info_list) == 0:
            random_selected_point_pair_differ = random_selected_point_pair_list
            random_selected_weight_differ = random_selected_weight_list
        else:
            random_selected_point_pair_differ = []
            random_selected_weight_differ = []
            for select_index in range(len(random_selected_info_list)):
                if random_selected_info_list[select_index] not in last_selected_point_pair_info_list:
                    random_selected_point_pair_differ.append(random_selected_point_pair_list[select_index])
                    random_selected_weight_differ.append(random_selected_weight_list[select_index])

        # 将上一轮的匹配点加入到这一轮的匹配点中。
        if len(last_selected_point_pair_info_list) == 0:
            supplement_last_point_pair_list = []
            supplement_last_weight_list = []
        else:
            select_index = np.random.choice(len(last_selected_point_pair_list), int(configs.last_iteration_ratio * len(last_selected_point_pair_list)), replace=False)
            supplement_last_point_pair_list = [last_selected_point_pair_list[i] for i in select_index]
            supplement_last_weight_list = [last_weight_list[i] for i in select_index]
        random_selected_point_pair_list.extend(supplement_last_point_pair_list)
        random_selected_weight_list.extend(supplement_last_weight_list)

        # 更新数据。
        last_selected_point_pair_info_list = random_selected_info_list
        last_weight_list = random_selected_weight_list
        last_selected_point_pair_list = random_selected_point_pair_list

        # learning_rate = 0.1
        # if iteration_index < max_iteration / 2:
        #     learning_rate = 0.1
        # else:
        #     learning_rate = 0.1 - (iteration_index - max_iteration / 2) / (max_iteration / 2) * 0.09
        learning_rate = configs.learning_rate_in_gradient_descent
        learning_rate_list.append(learning_rate)

        # transform_matrix, gd_error, last_iteration_num, last_grad_norm = gradient_descent_affine(random_selected_point_pair_list, random_selected_weight_list,
        #                                                                                          learning_rate=learning_rate, last_iteration_num=last_iteration_num,
        #                                                                                          max_iterations=2000, stop_grad_norm=1, grad_clip_value=1e8)

        transform_matrix, parameters, gd_error, last_iteration_num, last_grad_norm = gradient_descent_translate_rotate_shear_scale(random_selected_point_pair_list, random_selected_weight_list,
                                                                                                                                   learning_rate=learning_rate, last_iteration_num=last_iteration_num,
                                                                                                                                   max_iterations=1000, stop_grad_norm=1, grad_clip_value=1e8)

        gd_error_list.append(gd_error)
        # update total_transform_matrix
        total_transform_matrix = np.dot(transform_matrix, total_transform_matrix)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = apply_transform_to_calibration(calibration_data, total_transform_matrix)

        # update selected_gaze_point_list_1d
        selected_gaze_point_list_1d = [change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in selected_gaze_point_list_1d]
        selected_gaze_point_list_1d = [np.dot(transform_matrix, gaze_point) for gaze_point in selected_gaze_point_list_1d]
        selected_gaze_point_list_1d = np.array([change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in selected_gaze_point_list_1d])
        # update reading_data
        for text_index in range(len(selected_reading_data)):
            if text_index in configs.training_index_list:
                continue
            gaze_coordinates = selected_reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
            gaze_coordinates = [change_2d_vector_to_homogeneous_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [np.dot(transform_matrix, gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            selected_reading_data[text_index][["gaze_x", "gaze_y"]] = gaze_coordinates

        distance_list, avg_distance = compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        avg_error_list.append(avg_distance)
        print(f"average distance: {avg_distance}, last iteration num: {last_iteration_num}")
        last_iteration_num_list.append(last_iteration_num)

        int_random_selected_point_pair_differ = _get_int_point_pairs(random_selected_point_pair_differ)
        int_random_selected_point_pair = _get_int_point_pairs(random_selected_point_pair_list)

        with open(log_file_path, "a") as log_file:
            json.dump({"iteration_index": iteration_index, "avg_error": avg_distance, "last_iteration_num": last_iteration_num,
                       "last_gd_error": gd_error, "learning_rate": learning_rate, "transform_matrix": transform_matrix.tolist(),
                       "theta": parameters[0], "tx": parameters[1], "ty": parameters[2], "sx": parameters[3], "sy": parameters[4], "shx": parameters[5], "shy": parameters[6],
                       "different_point_pair": int_random_selected_point_pair_differ, "different_weight": random_selected_weight_differ,
                       "full_point_pair": int_random_selected_point_pair, "full_weight": random_selected_weight_list}, log_file, indent=2)
            log_file.write(",\n")

        if avg_distance > 5000:
            break

    with open(log_file_path, "a") as log_file:
        json.dump({"finish": "finish"}, log_file, indent=4)
        log_file.write("\n]")

    return avg_error_list

