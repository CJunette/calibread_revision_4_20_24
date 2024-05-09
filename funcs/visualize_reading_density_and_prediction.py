import numpy as np

from evaluate_result.evaluate_inherent import compute_transform_matrix
from funcs.manual_calibrate_for_std import apply_transform_to_calibration
from funcs.util_functions import change_2d_vector_to_homogeneous_vector, change_homogeneous_vector_to_2d_vector
from read.read_calibration import read_calibration
from read.read_reading import read_text_density, read_raw_reading
from read.read_text import read_text_sorted_mapping_and_group_with_para_id


def visual_reading_density_and_prediction(model_index=1):
    reading_list = read_raw_reading("original", "_after_cluster")
    prediction_list = read_text_sorted_mapping_and_group_with_para_id(f"_with_prediction_{model_index}")
    calibration_data = read_calibration()
    transform_matrix_list = compute_transform_matrix(calibration_data)

    for subject_index in range(len(calibration_data)):
        transform_matrix = transform_matrix_list[subject_index]
        for text_index in range(len(reading_list)):
            df = reading_list[subject_index][text_index]
            gaze_coordinates = df[["gaze_x", "gaze_y"]].values
            gaze_coordinates_after_translation = [np.dot(transform_matrix, change_2d_vector_to_homogeneous_vector(gaze_coordinate)) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates_after_translation = [change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates_after_translation]
            reading_list[subject_index][text_index]["gaze_x"] = [gaze_coordinate[0] for gaze_coordinate in gaze_coordinates_after_translation]
            reading_list[subject_index][text_index]["gaze_y"] = [gaze_coordinate[1] for gaze_coordinate in gaze_coordinates_after_translation]


    # for subject_index in range(len(reading_list)):


        print()

