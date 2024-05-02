import multiprocessing

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import configs
from funcs.calibrate_in_batch import calibrate_all_subjects
from funcs.manual_calibrate_for_std import apply_transform_to_calibration, compute_distance_between_std_and_correction
from funcs.process_text_before_calibration import preprocess_text_in_batch
from read.read_calibrate_and_grad_descent import read_calibrate_and_grad_descent, read_calibrate_and_grad_descent_process_attribute, read_all_subject_calibrate_and_grad_descent, \
    read_calibrate_and_grad_descent_start_matrix_attribute
from read.read_calibration import read_calibration
from read.read_reading import read_raw_reading


def cluster_transform_matrix(file_index, model_index, subject_index):
    transform_matrix_list = read_calibrate_and_grad_descent_process_attribute(file_index, model_index, subject_index, "transform_matrix")
    transform_matrix_list = [np.array(transform_matrix) for transform_matrix in transform_matrix_list]
    accumulated_transform_matrix_list = [transform_matrix_list[0]]
    for index in range(1, len(transform_matrix_list)):
        accumulated_transform_matrix_list.append(np.dot(transform_matrix_list[index], transform_matrix_list[0]))

    transform_matrix_list_1d = [transform_matrix_list[i].flatten() for i in range(len(transform_matrix_list))]
    accumulated_transform_matrix_list_1d = [accumulated_transform_matrix_list[i].flatten() for i in range(len(accumulated_transform_matrix_list))]
    error_list = read_calibrate_and_grad_descent_process_attribute(file_index, model_index, subject_index, "avg_error")

    # kmeans = KMeans(n_clusters=3, random_state=0).fit(accumulated_transform_matrix_list_1d)
    # rgb_list = ["r", "g", "b"]
    # color_list = [rgb_list[label] for label in kmeans.labels_]
    # plt.plot(error_list, color="#AAAAAA")
    # for index, transform_matrix in enumerate(transform_matrix_list):
    #     plt.scatter(index, error_list[index], c=color_list[index], s=20)
    # for matrix_index in range(len(transform_matrix_list)):
    #     print(matrix_index, kmeans.labels_[matrix_index], accumulated_transform_matrix_list_1d[matrix_index].tolist())

    clf = IsolationForest(contamination=0.2, random_state=configs.random_seed)
    # 训练模型并预测异常值
    pred = clf.fit_predict(transform_matrix_list_1d)

    # 找出离群矩阵的索引
    outlier_indices = np.where(pred == -1)
    # 获取离群矩阵
    outlier_matrices = np.array(transform_matrix_list_1d)[outlier_indices]
    plt.plot(error_list, color="#AAAAAA")
    for index, transform_matrix in enumerate(transform_matrix_list_1d):
        if index in outlier_indices[0]:
            color = "red"
        else:
            color = "blue"
        plt.scatter(index, error_list[index], c=color, s=20)

    plt.show()


def _compute_accumulated_transform_matrix_list(transform_matrix_list, start_translate_matrix, start_scale_matrix):
    """
    第index步的实际变换矩阵，为之前所有矩阵的累积。变换矩阵在最初会乘以一个初始的平移矩阵和缩放矩阵。
    """
    start_matrix = np.dot(start_translate_matrix, start_scale_matrix)
    accumulated_transform_matrix_list = [np.dot(transform_matrix_list[0], start_matrix)]
    for index in range(1, len(transform_matrix_list)):
        accumulated_transform_matrix_list.append(np.dot(transform_matrix_list[index], accumulated_transform_matrix_list[-1]))

    return accumulated_transform_matrix_list


def read_transform_start_translate_and_start_scale_matrix(file_index, model_index, subject_index):
    # print(f"reading data of subject {subject_index}")
    transform_matrix_list_1 = read_calibrate_and_grad_descent_process_attribute(file_index, model_index, subject_index, "transform_matrix")
    start_translate_matrix = read_calibrate_and_grad_descent_start_matrix_attribute(file_index, model_index, subject_index, "translate_matrix")
    start_scale_matrix = read_calibrate_and_grad_descent_start_matrix_attribute(file_index, model_index, subject_index, "scale_matrix")

    transform_matrix_list_1 = [np.array(transform_matrix) for transform_matrix in transform_matrix_list_1]
    start_translate_matrix = np.array(start_translate_matrix)
    start_scale_matrix = np.array(start_scale_matrix)

    return transform_matrix_list_1, start_translate_matrix, start_scale_matrix


def compute_final_transform_matrix_of_all_subjects(file_index, model_index):
    """
    目前只考虑iteration切割的情况，只取某个被试的特定iteration之后的累积变换矩阵，然后求其均值，最后确认该矩阵对应的accuracy error。
    """
    cali_grad_list_all = read_all_subject_calibrate_and_grad_descent(file_index, model_index)

    transform_matrix_list = []
    start_translate_matrix_list = []
    start_scale_matrix_list = []
    args_list = []
    for subject_index in range(len(cali_grad_list_all)):
        args_list.append((file_index, model_index, subject_index))
        # print(f"reading data of subject {subject_index}")
        # transform_matrix_list_1 = read_calibrate_and_grad_descent_process_attribute(file_index, model_index, subject_index, "transform_matrix")
        # start_translate_matrix = read_calibrate_and_grad_descent_start_matrix_attribute(file_index, model_index, subject_index, "translate_matrix")
        # start_scale_matrix = read_calibrate_and_grad_descent_start_matrix_attribute(file_index, model_index, subject_index, "scale_matrix")

        # transform_matrix_list_1 = [np.array(transform_matrix) for transform_matrix in transform_matrix_list_1]
        # start_translate_matrix = np.array(start_translate_matrix)
        # start_scale_matrix = np.array(start_scale_matrix)

        # transform_matrix_list.append(transform_matrix_list_1)
        # start_translate_matrix_list.append(start_translate_matrix)
        # start_scale_matrix_list.append(start_scale_matrix)

    with multiprocessing.Pool(configs.number_of_process) as pool:
        result_list = pool.starmap(read_transform_start_translate_and_start_scale_matrix, args_list)

    for subject_index in range(len(result_list)):
        transform_matrix_list_1 = result_list[subject_index][0]
        start_translate_matrix = result_list[subject_index][1]
        start_scale_matrix = result_list[subject_index][2]

        transform_matrix_list.append(transform_matrix_list_1)
        start_translate_matrix_list.append(start_translate_matrix)
        start_scale_matrix_list.append(start_scale_matrix)

    accumulated_transform_matrix_list = []
    for subject_index in range(len(cali_grad_list_all)):
        print(f"computing accumulated transform matrix list of subject {subject_index}")
        accumulated_transform_matrix_list_1 = _compute_accumulated_transform_matrix_list(transform_matrix_list[subject_index], start_translate_matrix_list[subject_index], start_scale_matrix_list[subject_index])
        accumulated_transform_matrix_list.append(accumulated_transform_matrix_list_1)

    accumulated_transform_matrix_avg_list = []
    for subject_index in range(len(accumulated_transform_matrix_list)):
        print(f"computing accumulated transform matrix avg of subject {subject_index}")
        if len(accumulated_transform_matrix_list[subject_index]) < configs.final_transform_matrix_iteration_end - configs.final_transform_matrix_iteration_start:
            accumulated_transform_matrix_list_1 = accumulated_transform_matrix_list[subject_index]
        else:
            accumulated_transform_matrix_list_1 = accumulated_transform_matrix_list[subject_index][configs.final_transform_matrix_iteration_start:configs.final_transform_matrix_iteration_end]
        accumulated_transform_matrix_avg = np.mean(accumulated_transform_matrix_list_1, axis=0)
        accumulated_transform_matrix_avg_list.append(accumulated_transform_matrix_avg)

    return accumulated_transform_matrix_avg_list


def compute_accuracy_error_for_all_subjects(file_index, model_index):
    accumulated_transform_matrix_avg_list = compute_final_transform_matrix_of_all_subjects(file_index, model_index)
    calibration_data = read_calibration()

    error_list = []
    print(f"processing calibration data")
    for subject_index in range(len(calibration_data)):
        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = apply_transform_to_calibration(calibration_data[subject_index], accumulated_transform_matrix_avg_list[subject_index])
        distance_list, avg_distance = compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        error_list.append(avg_distance)
        print(avg_distance)
    return error_list


def evaluate_accuracy_error_for_centroid_alignment(file_index, model_index=1):
    """
    这里file_index填什么其实关系不大，因为centroid alignment不受任何参数影响，因此用任意file_index得到的结果都是一样的。
    """

    calibration_data = read_calibration()

    print(f"file_index: {file_index}")
    cali_grad_list_all = read_all_subject_calibrate_and_grad_descent(file_index, model_index)

    centroid_alignment_matrix_list = []
    args_list = []
    for subject_index in range(len(cali_grad_list_all)):
        args_list.append((file_index, model_index, subject_index))

    with multiprocessing.Pool(configs.number_of_process) as pool:
        result_list = pool.starmap(read_transform_start_translate_and_start_scale_matrix, args_list)

    for subject_index in range(len(result_list)):
        start_translate_matrix = result_list[subject_index][1]
        start_scale_matrix = result_list[subject_index][2]
        start_matrix = np.dot(start_translate_matrix, start_scale_matrix)
        centroid_alignment_matrix_list.append(start_matrix)

    error_list = []
    for subject_index in range(len(calibration_data)):
        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = apply_transform_to_calibration(calibration_data[subject_index], centroid_alignment_matrix_list[subject_index])
        distance_list, avg_distance = compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        error_list.append(avg_distance)
        print(avg_distance)
    print()















