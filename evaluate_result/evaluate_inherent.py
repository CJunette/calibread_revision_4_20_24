import cv2
import numpy as np

from evaluate_result.evaluate_seven_points import get_paired_points_of_std_cali_from_cali_dict
from funcs.manual_calibrate_for_std import apply_transform_to_calibration, compute_distance_between_std_and_correction
from read.read_calibration import read_calibration_with_certain_corners, read_calibration


def compute_transform_matrix(calibration_data):
    transform_matrix_list = []
    for subject_index in range(len(calibration_data)):
        calibration_avg_list = calibration_data[subject_index][1]
        calibration_point_list = calibration_data[subject_index][2]
        point_pairs = get_paired_points_of_std_cali_from_cali_dict(calibration_avg_list, calibration_point_list)

        # 使用梯度下降算法求解仿射变换矩阵。
        # transform_matrix = get_affine_transform_matrix_gradient_descent(point_pairs)

        # 使用cv2求解仿射变换矩阵。
        source_points = point_pairs[:, 0, :]
        target_points = point_pairs[:, 1, :]
        source_points = source_points.astype(np.float32)
        target_points = target_points.astype(np.float32)
        transform_matrix, _ = cv2.estimateAffine2D(source_points, target_points)
        transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 1])))
        transform_matrix_list.append(transform_matrix)

    return transform_matrix_list


def evaluate_inherent():
    calibration_data = read_calibration()
    error_list = []
    transform_matrix_list = compute_transform_matrix(calibration_data)

    for subject_index in range(len(calibration_data)):
        transform_matrix = transform_matrix_list[subject_index]

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = apply_transform_to_calibration(calibration_data[subject_index], transform_matrix)

        distance_list, avg_distance = compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        print(avg_distance)
        error_list.append(avg_distance)
    return error_list

