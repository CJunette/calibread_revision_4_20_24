import cv2
import numpy as np

from funcs.manual_calibrate_for_std import apply_transform_to_calibration, compute_distance_between_std_and_correction
from read.read_calibration import read_calibration, read_calibration_with_certain_corners


def _get_paired_points_of_std_cali_from_cali_dict(avg_gaze_list, calibration_point_list):
    '''
    从ReadData.read_calibration_data()得到的数据中，对avg_gaze与calibration_point进行配对。
    :param avg_gaze_list:
    :param calibration_point_list:
    :return:
    '''
    point_pairs = []

    for row_index in range(len(avg_gaze_list)):
        for col_index in range(len(avg_gaze_list[row_index])):
            avg_calibration_point_dict = avg_gaze_list[row_index][col_index]
            calibration_point_dict = calibration_point_list[row_index][col_index]
            avg_point = [avg_calibration_point_dict["avg_gaze_x"], avg_calibration_point_dict["avg_gaze_y"]]
            calibration_point = [calibration_point_dict["point_x"], calibration_point_dict["point_y"]]
            point_pairs.append([avg_point, calibration_point])

    return np.array(point_pairs)


def evaluate_seven_points_for_all_subjects(left_top_index=0, right_top_index=0, left_bottom_index=0, right_bottom_index=0):
    corner_index_dict = {(0, 0): 0, (0, 29): 0, (5, 0): 0, (5, 29): 0}
    calibration_data = read_calibration_with_certain_corners(corner_index_dict)
    # calibration_data = read_calibration()
    error_list = []

    top_row = 0
    middle_row = 2
    bottom_row = 5
    left_col = 0
    middle_col = 14
    right_col = 29

    for subject_index in range(len(calibration_data)):
        calibration_avg_list = calibration_data[subject_index][1]
        calibration_point_list = calibration_data[subject_index][2]

        calibration_avg_list = [[calibration_avg_list[top_row][left_col], calibration_avg_list[top_row][middle_col], calibration_avg_list[top_row][right_col]],
                                [calibration_avg_list[middle_row][middle_col]],
                                [calibration_avg_list[bottom_row][left_col], calibration_avg_list[bottom_row][middle_col], calibration_avg_list[bottom_row][right_col]]]
        calibration_point_list = [[calibration_point_list[top_row][left_col], calibration_point_list[top_row][middle_col], calibration_point_list[top_row][right_col]],
                                  [calibration_point_list[middle_row][middle_col]],
                                  [calibration_point_list[bottom_row][left_col], calibration_point_list[bottom_row][middle_col], calibration_point_list[bottom_row][right_col]]]

        point_pairs = _get_paired_points_of_std_cali_from_cali_dict(calibration_avg_list, calibration_point_list)

        # 使用梯度下降算法求解仿射变换矩阵。
        # transform_matrix = get_affine_transform_matrix_gradient_descent(point_pairs)

        # 使用cv2求解仿射变换矩阵。
        source_points = point_pairs[:, 0, :]
        target_points = point_pairs[:, 1, :]
        source_points = source_points.astype(np.float32)
        target_points = target_points.astype(np.float32)
        transform_matrix, _ = cv2.estimateAffine2D(source_points, target_points, ransacReprojThreshold=0.1)
        transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 1])))

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = apply_transform_to_calibration(calibration_data[subject_index], transform_matrix)

        distance_list, avg_distance = compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        print(avg_distance)
        error_list.append(avg_distance)
    return error_list





