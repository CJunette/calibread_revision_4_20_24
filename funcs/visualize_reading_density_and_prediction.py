import math
import time

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

import configs
from evaluate_result.evaluate_inherent import compute_transform_matrix
from funcs.manual_calibrate_for_std import apply_transform_to_calibration
from funcs.util_functions import change_2d_vector_to_homogeneous_vector, change_homogeneous_vector_to_2d_vector
from read.read_calibration import read_calibration
from read.read_reading import read_text_density, read_raw_reading
from read.read_text import read_text_sorted_mapping_and_group_with_para_id


def match_gaze_with_text(reading_list, text_list):
    for subject_index in range(len(reading_list)):
        for text_index in range(len(reading_list[subject_index])):
            text_df = text_list[text_index]
            text_coordinates = text_df[["x", "y"]].values
            reading_df = reading_list[subject_index][text_index]
            gaze_coordinates = reading_df[["gaze_x", "gaze_y"]].values

            text_coordinates_tensor = torch.tensor(text_coordinates, dtype=torch.float32, device=configs.gpu_device_id)
            gaze_coordinates_tensor = torch.tensor(gaze_coordinates, dtype=torch.float32, device=configs.gpu_device_id)
            # compute distance between each gaze coordinates to each text coordinates
            distance_matrix = torch.cdist(gaze_coordinates_tensor, text_coordinates_tensor)
            nearest_text_index = torch.argmin(distance_matrix, dim=1)
            nearest_text_points = text_coordinates_tensor[nearest_text_index].cpu().numpy()
            nearest_text_distances = torch.min(distance_matrix, dim=1).values.cpu().numpy()

            reading_list[subject_index][text_index]["nearest_text_point_x"] = [nearest_text_point[0] for nearest_text_point in nearest_text_points]
            reading_list[subject_index][text_index]["nearest_text_point_y"] = [nearest_text_point[1] for nearest_text_point in nearest_text_points]
            reading_list[subject_index][text_index]["nearest_text_distance"] = nearest_text_distances
    return reading_list


def get_reading_density_and_prediction(model_index=1):
    """
    该函数的目的是将reading density和prediction在每篇文本和文本内的空间位置上进行对齐。
    """
    # TODO 这里生成的text_point_reading_density_dict可能需要把subject的信息也添加进去。

    reading_list = read_raw_reading("original", "_after_cluster")
    text_list = read_text_sorted_mapping_and_group_with_para_id(f"_with_prediction_{model_index}")
    calibration_data = read_calibration()
    transform_matrix_list = compute_transform_matrix(calibration_data)
    # 将original gaze data转化为经过calibration的gaze data。
    for subject_index in range(len(reading_list)):
        transform_matrix = transform_matrix_list[subject_index]
        for text_index in range(len(reading_list)):
            df = reading_list[subject_index][text_index]
            gaze_coordinates = df[["gaze_x", "gaze_y"]].values
            gaze_coordinates_after_translation = [np.dot(transform_matrix, change_2d_vector_to_homogeneous_vector(gaze_coordinate)) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates_after_translation = [change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates_after_translation]
            reading_list[subject_index][text_index]["gaze_x"] = [gaze_coordinate[0] for gaze_coordinate in gaze_coordinates_after_translation]
            reading_list[subject_index][text_index]["gaze_y"] = [gaze_coordinate[1] for gaze_coordinate in gaze_coordinates_after_translation]

    # 将每个gaze data对应的text point标记到reading data中。
    reading_list = match_gaze_with_text(reading_list, text_list)

    text_point_reading_density_list = []
    for text_index in range(len(reading_list[0])):
        for subject_index in range(len(reading_list)):
            df = reading_list[subject_index][text_index]
            df = df[df["nearest_text_distance"] < 64]  # 这里的64是text_height，作为distance threshold。
            nearest_text_point_x = df["nearest_text_point_x"].values
            nearest_text_point_y = df["nearest_text_point_y"].values
            density = df["density"].values

            for gaze_index in range(len(nearest_text_point_x)):
                text_point = (text_index, nearest_text_point_x[gaze_index], nearest_text_point_y[gaze_index])
                text_point_reading_density_list.append((text_point, density[gaze_index]))
    # 将text_point_reading_density_list按照text_point进行转为字典。
    text_point_reading_density_dict = {}
    for text_point, density in text_point_reading_density_list:
        if text_point not in text_point_reading_density_dict:
            text_point_reading_density_dict[text_point] = []
        text_point_reading_density_dict[text_point].append(density)

    text_point_prediction_list = []
    for text_index in range(len(reading_list[0])):
        for word_index in range(len(text_list[text_index])):
            text_point = (text_index, text_list[text_index].iloc[word_index]["x"], text_list[text_index].iloc[word_index]["y"])
            prediction = text_list[text_index].iloc[word_index]["prediction"]
            text_point_prediction_list.append((text_point, prediction))
    text_point_prediction_dict = {}
    for text_point, prediction in text_point_prediction_list:
        if text_point not in text_point_prediction_dict:
            text_point_prediction_dict[text_point] = []
        text_point_prediction_dict[text_point].append(prediction)

    # TODO 这里text_point_prediction_list和text_point_reading_density_dict的长度不一，似乎有点问题。

    return text_point_reading_density_dict, text_point_prediction_dict


def visual_reading_density_and_prediction_correlation(model_index=1):
    """
    该函数的目的是将reading density和prediction在每篇文本和文本内的空间位置上进行对齐，然后可视化两者的关系（density=f(prediction)）。
    """
    text_point_reading_density_dict, text_point_prediction_dict = get_reading_density_and_prediction(model_index)

    # plot prediction and density
    reading_density_list = []
    prediction_list = []
    for text_point in text_point_reading_density_dict:
        reading_densities = text_point_reading_density_dict[text_point]
        predictions = text_point_prediction_dict[text_point] * len(reading_densities)
        reading_density_list.extend(reading_densities)
        prediction_list.extend(predictions)
    # prediction_list = [2 * predict for predict in prediction_list]
    # random pick 1/10 of the data
    random_indices = np.random.choice(len(reading_density_list), len(reading_density_list) // 10, replace=False)
    reading_density_list = [reading_density_list[index] for index in random_indices]
    prediction_list = [prediction_list[index] for index in random_indices]
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_aspect('equal')
    ax.scatter(reading_density_list, prediction_list, s=20, alpha=0.01)

    plt.xlabel("reading density")
    plt.ylabel("prediction")
    plt.show()


def visual_reading_density_and_prediction_on_canvas(model_index=1):
    """
    该函数的目的是将reading density和prediction在每篇文本和文本内的空间位置上进行对齐，然后将两者都可视化在每个位置上。
    """
    text_point_reading_density_dict, text_point_prediction_dict = get_reading_density_and_prediction(model_index)

    prediction_alpha_ratio = 0.05
    # fine the 3/4 point of text_point_prediction_dict values
    prediction_value_list = []
    prediction_coordinate_list = []
    for key, value in text_point_prediction_dict.items():
        prediction_value_list.extend(value)
        prediction_coordinate_list.extend([[key[1], key[2]]])
    min_prediction_value = min(prediction_value_list)
    prediction_value_list = [prediction_value + abs(min_prediction_value) + 0.001 for prediction_value in prediction_value_list]
    prediction_value_3_4 = np.sort(prediction_value_list, axis=0)[int(3 * len(prediction_value_list) / 4)]
    prediction_alpha_list = [min(1, prediction_value / prediction_value_3_4) * prediction_alpha_ratio for prediction_value in prediction_value_list]

    density_alpha_ratio = 0.01
    density_value_list = []
    density_coordinate_list = []
    for key, value in text_point_reading_density_dict.items():
        value = [v + 1 for v in value] # 这里+1是为了让density=0的点也能显示出来。
        density_value_list.extend(value)
        density_coordinate_list.extend([[key[1], key[2]]] * len(value))
    density_random_select_indices = np.random.choice(len(density_value_list), len(density_value_list) // 10, replace=False)
    density_value_list = [density_value_list[index] for index in density_random_select_indices]
    density_coordinate_list = [density_coordinate_list[index] for index in density_random_select_indices]
    density_value_3_4 = np.sort(density_value_list, axis=0)[int(3 * len(density_value_list) / 4)]
    density_alpha_list = [min(1, density_value / density_value_3_4) * density_alpha_ratio for density_value in density_value_list]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0, configs.screen_width)
    ax.set_ylim(configs.screen_height, 0)

    radius = 5
    time_1 = time.time()
    prediction_circle_list = []
    for prediction_index in range(len(prediction_coordinate_list)):
        circle = plt.Circle((prediction_coordinate_list[prediction_index][0], prediction_coordinate_list[prediction_index][1]), radius=radius)
        prediction_circle_list.append(circle)
    time_2 = time.time()
    density_circle_list = []
    for density_index in range(len(density_coordinate_list)):
        circle = plt.Circle((density_coordinate_list[density_index][0] + radius * 2, density_coordinate_list[density_index][1]), radius=radius)
        density_circle_list.append(circle)
    time_3 = time.time()
    collection_prediction = PatchCollection(prediction_circle_list, color='red', alpha=prediction_alpha_list)
    time_4 = time.time()
    collection_density = PatchCollection(density_circle_list, color='blue', alpha=density_alpha_list)
    time_5 = time.time()
    ax.add_collection(collection_prediction)
    ax.add_collection(collection_density)
    time_6 = time.time()
    print(f"time_2 - time_1: {time_2 - time_1}")
    print(f"time_3 - time_2: {time_3 - time_2}")
    print(f"time_4 - time_3: {time_4 - time_3}")
    print(f"time_5 - time_4: {time_5 - time_4}")
    print(f"time_6 - time_5: {time_6 - time_5}")

    plt.show()



















