import numpy as np

import configs
from funcs.calibrate_with_simple_linear_and_weight import calibrate_with_simple_linear_and_weight
from funcs.process_text_before_calibration import preprocess_text_in_batch
from read.read_calibration import read_calibration
from read.read_reading import read_raw_reading


def calibrate_all_subjects(text_data_list, model_index, reading_data_list, calibration_data_list, calibrate_mode, max_iteration=100, distance_threshold=64):
    text_data = text_data_list[model_index]
    for subject_index in range(len(reading_data_list)):
    # for subject_index in range(2, len(reading_data_list)):
        if calibrate_mode == "simple_linear_weight":
            calibrate_with_simple_linear_and_weight(model_index, subject_index, text_data, reading_data_list[subject_index], calibration_data_list[subject_index], max_iteration, distance_threshold)


def calibrate_in_batch(calibrate_mode, model_indices, validation_num=25, random_seed=0):
    """
    使用configs.file_index来指定文件名，记得修改。
    """
    np.random.seed(random_seed)

    text_data_list = preprocess_text_in_batch()
    reading_data_list = read_raw_reading("original", "_after_cluster")
    calibration_data_list = read_calibration()

    total_index_list = np.array([i for i in range(configs.passage_num)])
    training_index_list = np.array(configs.training_index_list)
    validation_index_list = np.setdiff1d(total_index_list, training_index_list)
    validation_indices = np.random.choice(validation_index_list, validation_num, replace=False)
    training_replace_indices = np.setdiff1d(total_index_list, validation_indices).tolist()
    configs.training_index_list = training_replace_indices

    # for model_index in range(len(text_data_list)):
    for model_index in model_indices:
        calibrate_all_subjects(text_data_list, model_index, reading_data_list, calibration_data_list, calibrate_mode)


def calibrate_in_batch_for_different_training_num(calibrate_mode, start_file_index, start_num, end_num, random_seed=0, model_index=1):
    """
    使用start_file_index来指定文件名（第一个文件的编号）。使用start_index和end_index作为validation的数量。
    start_num和end_num分别代表起始的validation数量和结束的validation数量。
    start_num需要从1开始。
    """
    np.random.seed(random_seed)

    text_data_list = preprocess_text_in_batch()
    reading_data_list = read_raw_reading("original", "_after_cluster")
    calibration_data_list = read_calibration()

    total_index_list = np.array([i for i in range(configs.passage_num)])
    training_index_list = np.array(configs.training_index_list)
    validation_index_list = np.setdiff1d(total_index_list, training_index_list)

    print(f"start_num: {start_num}, end_num: {end_num}")
    # for validation_num in range(1, len(validation_index_list)):
    for validation_num in range(start_num, end_num):
        validation_indices = np.random.choice(validation_index_list, validation_num, replace=False)
        training_replace_indices = np.setdiff1d(total_index_list, validation_indices).tolist()
        configs.training_index_list = training_replace_indices
        configs.file_index = start_file_index - 1 + validation_num

        calibrate_all_subjects(text_data_list, model_index, reading_data_list, calibration_data_list, calibrate_mode)


def calibrate_in_batch_for_different_model(calibrate_mode, start_file_index):
    text_data_list = preprocess_text_in_batch()
    reading_data_list = read_raw_reading("original", "_after_cluster")
    calibration_data_list = read_calibration()

    for model_index in range(len(text_data_list)):
        if model_index == 0:
            configs.bool_weight = False
        else:
            configs.bool_weight = True
        configs.file_index = start_file_index + model_index
        calibrate_all_subjects(text_data_list, model_index, reading_data_list, calibration_data_list, calibrate_mode)
