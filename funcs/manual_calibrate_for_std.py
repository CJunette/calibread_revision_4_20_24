import numpy as np
from funcs.util_functions import change_homogeneous_vector_to_2d_vector, change_2d_vector_to_homogeneous_vector


def apply_transform_to_calibration(subject_index, calibration_data, transform_matrix):
    gaze_list = calibration_data[0]
    avg_gaze_list = calibration_data[1]
    calibration_point_list = calibration_data[2]

    gaze_coordinates_before_translation_list = []
    avg_gaze_coordinate_before_translation_list = []
    gaze_coordinates_after_translation_list = []
    avg_gaze_coordinate_after_translation_list = []
    calibration_point_list_modified = []
    for row_index in range(len(gaze_list)):
        for col_index in range(len(gaze_list[row_index])):
            gaze_dict = gaze_list[row_index][col_index]
            gaze_x_list = gaze_dict["gaze_x"]
            gaze_y_list = gaze_dict["gaze_y"]
            gaze_coordinates = [np.array([gaze_x_list[i], gaze_y_list[i]]) for i in range(len(gaze_x_list))]
            gaze_coordinates_before_translation_list.append(np.array(gaze_coordinates))

            gaze_coordinates_after_translation = [np.dot(transform_matrix, change_2d_vector_to_homogeneous_vector(gaze_coordinate)) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates_after_translation = [change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates_after_translation]
            gaze_coordinates_after_translation_list.append(np.array(gaze_coordinates_after_translation))

            avg_gaze_dict = avg_gaze_list[row_index][col_index]
            avg_gaze_x = avg_gaze_dict["avg_gaze_x"]
            avg_gaze_y = avg_gaze_dict["avg_gaze_y"]
            avg_gaze_coordinate = np.array([avg_gaze_x, avg_gaze_y])
            avg_gaze_coordinate_before_translation_list.append(avg_gaze_coordinate)
            avg_gaze_coordinate_after_translation = np.dot(transform_matrix, change_2d_vector_to_homogeneous_vector(avg_gaze_coordinate))
            avg_gaze_coordinate_after_translation = change_homogeneous_vector_to_2d_vector(avg_gaze_coordinate_after_translation)
            avg_gaze_coordinate_after_translation_list.append(avg_gaze_coordinate_after_translation)

            calibration_point_dict = calibration_point_list[row_index][col_index]
            calibration_point_x = calibration_point_dict["point_x"]
            calibration_point_y = calibration_point_dict["point_y"]
            calibration_point = np.array([calibration_point_x, calibration_point_y])
            calibration_point_list_modified.append(calibration_point)

    gaze_coordinates_before_translation_list = np.array(gaze_coordinates_before_translation_list, dtype=object)
    gaze_coordinates_before_translation_list = np.concatenate(gaze_coordinates_before_translation_list, axis=0)

    gaze_coordinates_after_translation_list = np.array(gaze_coordinates_after_translation_list, dtype=object)
    gaze_coordinates_after_translation_list = np.concatenate(gaze_coordinates_after_translation_list, axis=0)

    avg_gaze_coordinate_before_translation_list = np.array(avg_gaze_coordinate_before_translation_list)
    avg_gaze_coordinate_after_translation_list = np.array(avg_gaze_coordinate_after_translation_list)

    calibration_point_list_modified = np.array(calibration_point_list_modified)

    return gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
           avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
           calibration_point_list_modified


def compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list: list, calibration_point_list_modified: list):
    point_pair_list = []
    for index in range(len(avg_gaze_coordinate_after_translation_list)):
        point_pair = [np.array(avg_gaze_coordinate_after_translation_list[index]), np.array(calibration_point_list_modified[index])]
        point_pair_list.append(point_pair)

    distance_list = []
    for point_pair_index, point_pair in enumerate(point_pair_list):
        std_point = point_pair[0]
        correction_point = point_pair[1]
        distance = np.linalg.norm(std_point - correction_point)  # TODO 这里计算distance的方法可以再做定义。
        distance_list.append(distance)
    avg_distance = np.mean(distance_list)
    return distance_list, avg_distance

