import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from funcs.process_text_before_calibration import preprocess_text_in_batch
from funcs.util_functions import change_2d_vector_to_homogeneous_vector, change_homogeneous_vector_to_2d_vector
from read.read_calibrate_and_grad_descent import read_calibrate_and_grad_descent, read_all_subject_calibrate_and_grad_descent
from read.read_calibration import read_calibration
from read.read_reading import read_raw_reading


def _visualize_calibrate_points(ax, calibration_data, transform_matrix, point_color):
    std_calibration_point_list = []
    for row_index in range(len(calibration_data)):
        for col_index in range(len(calibration_data[row_index])):
            calibration_point = calibration_data[row_index][col_index]
            calibration_point = [calibration_point["avg_gaze_x"], calibration_point["avg_gaze_y"]]
            std_calibration_point_list.append(calibration_point)

    std_calibration_point_list = [change_2d_vector_to_homogeneous_vector(cali_point) for cali_point in std_calibration_point_list]
    std_calibration_point_list = [np.dot(transform_matrix, cali_point) for cali_point in std_calibration_point_list]
    std_calibration_point_list = np.array([change_homogeneous_vector_to_2d_vector(cali_point) for cali_point in std_calibration_point_list])

    ax.scatter(std_calibration_point_list[:, 0], std_calibration_point_list[:, 1], c=point_color, s=25)
    # for calibration_point_after_transform in std_calibration_point_list:
    #     ax.scatter(calibration_point_after_transform[0], calibration_point_after_transform[1], c=point_color, s=25)


def _visualize_std_calibrate_points(ax, calibration_data):
    calibration_point_x_list = []
    calibration_point_y_list = []
    for row_index in range(len(calibration_data)):
        for col_index in range(len(calibration_data[row_index])):
            calibration_point = calibration_data[row_index][col_index]
            # ax.scatter(calibration_point["point_x"], calibration_point["point_y"], c="black", s=25)
            calibration_point_x_list.append(calibration_point["point_x"])
            calibration_point_y_list.append(calibration_point["point_y"])
    ax.scatter(calibration_point_x_list, calibration_point_y_list, c="black", s=25)


def _visualize_reading_points(ax, reading_data, transform_matrix, point_color):
    gaze_x_list = reading_data["gaze_x"].tolist()
    gaze_y_list = reading_data["gaze_y"].tolist()
    gaze_points = [np.array([gaze_x_list[i], gaze_y_list[i]]) for i in range(len(gaze_x_list))]
    gaze_points = np.array(gaze_points)

    gaze_point_list = [change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_points]
    gaze_point_list = [np.dot(transform_matrix, gaze_point) for gaze_point in gaze_point_list]
    gaze_point_list = np.array([change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list])

    ax.scatter(gaze_points[:, 0], gaze_points[:, 1], c=point_color, s=1)
    # for gaze_point_after_transform in gaze_point_list:
    #     ax.scatter(gaze_point_after_transform[0], gaze_point_after_transform[1], c=point_color, s=1)


def _visualize_result(reading_data, calibration_data, transform_matrix_last_iter, transform_matrix,
                      file_index, model_index, subject_index, iteration_index):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1920)
    ax.set_ylim(1200, 0)
    ax.set_aspect("equal")

    # 生成标准校准点。
    _visualize_std_calibrate_points(ax, calibration_data[2])
    # _visualize_reading_points(ax, reading_data, transform_matrix_last_iter, "green")
    # _visualize_reading_points(ax, reading_data, transform_matrix, "orange")
    _visualize_calibrate_points(ax, calibration_data[1], transform_matrix_last_iter, "green")
    _visualize_calibrate_points(ax, calibration_data[1], transform_matrix, "orange")

    save_path_prefix = f"{os.getcwd()}/image/calibrate_process/{file_index}-{model_index}-{subject_index}"
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    plt.savefig(f"{save_path_prefix}/{iteration_index}.png")
    plt.clf()
    plt.close()


def visualize_cali_grad_process(file_index, model_index, subject_index):
    # text_data_list = preprocess_text_in_batch()
    reading_data_list = read_raw_reading("original", "_after_cluster")
    calibration_data_list = read_calibration()

    # text_data = text_data_list[model_index]
    reading_data = reading_data_list[subject_index]
    calibration_data = calibration_data_list[subject_index]

    all_reading_data = pd.concat(reading_data, ignore_index=True)

    calibrate_process = read_calibrate_and_grad_descent(file_index, model_index, subject_index)
    # calibrate_process = cali_grad_list[subject_index]

    affine_matrix = np.eye(3)
    scale_matrix = calibrate_process[1]["scale_matrix"]
    translate_matrix = calibrate_process[1]["translate_matrix"]
    affine_matrix = np.dot(scale_matrix, affine_matrix)
    affine_matrix = np.dot(translate_matrix, affine_matrix)

    _visualize_result(all_reading_data, calibration_data, np.eye(3), affine_matrix, file_index, model_index, subject_index, 0)

    for index in range(2, len(calibrate_process) - 1):
        print(f"processing: {index - 1}")
        affine_matrix_last_iter = np.eye(3)
        for row_index in range(len(affine_matrix_last_iter)):
            for col_index in range(len(affine_matrix_last_iter[row_index])):
                affine_matrix_last_iter[row_index][col_index] = affine_matrix[row_index][col_index]

        transform_matrix = calibrate_process[index]["transform_matrix"]
        affine_matrix = np.dot(transform_matrix, affine_matrix)

        _visualize_result(all_reading_data, calibration_data, affine_matrix_last_iter, affine_matrix, file_index, model_index, subject_index, index - 1)


def visualize_all_subject_cali_grad_process(file_index, model_index):
    cali_grad_list = read_all_subject_calibrate_and_grad_descent(file_index, model_index)
    for subject_index in range(len(cali_grad_list)):
        visualize_cali_grad_process(file_index, model_index, subject_index)


def visualize_cali_grad_result(file_index, model_index, subject_index):
    # cali_grad = cali_grad_list[subject_index]
    cali_grad = read_calibrate_and_grad_descent(file_index, model_index, subject_index)

    avg_error_list = []
    last_gd_error_list = []
    for cali_grad_index, cali_grad_dict in enumerate(cali_grad):
        if cali_grad_index < 2:
            continue
        if cali_grad_index == len(cali_grad) - 1:
            break
        avg_error = cali_grad_dict["avg_error"]
        last_gd_error = cali_grad_dict["last_gd_error"]
        avg_error_list.append(avg_error)
        last_gd_error_list.append(last_gd_error)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(avg_error_list, label="avg_error", color="green")
    # 添加第二个坐标轴，显示last_gd_error。
    ax2 = ax.twinx()
    ax2.plot(last_gd_error_list, label="last_gd_error", color="blue")
    ax.legend()
    save_path_prefix = f"{os.getcwd()}/image/calibrate_and_grad_descent_result/file_{file_index}-model_{model_index}"
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    save_path = f"{save_path_prefix}/{subject_index}.png"
    plt.savefig(save_path)


def visualize_all_subject_cali_grad_result(file_index, model_index):
    cali_grad_list = read_all_subject_calibrate_and_grad_descent(file_index, model_index)
    for subject_index in range(len(cali_grad_list)):
        visualize_cali_grad_result(file_index, model_index, subject_index)



