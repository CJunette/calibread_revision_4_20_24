import configs
from funcs.calibrate_with_simple_linear_and_weight import calibrate_with_simple_linear_and_weight
from funcs.process_text_before_calibration import preprocess_text_in_batch
from read.read_calibration import read_calibration
from read.read_reading import read_raw_reading


def calibrate_all_subjects(text_data_list, model_index, reading_data_list, calibration_data_list, calibrate_mode, max_iteration=100, distance_threshold=64):
    text_data = text_data_list[model_index]
    # for subject_index in range(len(reading_data_list)):
    for subject_index in range(3, 4):
        if calibrate_mode == "simple_linear_weight":
            calibrate_with_simple_linear_and_weight(model_index, subject_index, text_data, reading_data_list[subject_index], calibration_data_list[subject_index], max_iteration, distance_threshold)


def calibrate_in_batch(calibrate_mode):
    text_data_list = preprocess_text_in_batch()
    reading_data_list = read_raw_reading("original", "_after_cluster")
    calibration_data_list = read_calibration()

    # for model_index in range(len(text_data_list)):
    for model_index in range(1, 2):
        if model_index == 0:
            configs.bool_weight = False
        else:
            configs.bool_weight = True
        calibrate_all_subjects(text_data_list, model_index, reading_data_list, calibration_data_list, calibrate_mode)


