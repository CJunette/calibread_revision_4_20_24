import os

import pandas as pd

import configs


def _get_gaze_of_matrix_x_and_matrix_y(calibration_df):
    calibration_gaze_x = calibration_df["gaze_x"].tolist()[1:]
    calibration_gaze_x = [x for x in calibration_gaze_x if x != "failed"]
    calibration_gaze_x = [float(x) for x in calibration_gaze_x]
    calibration_gaze_y = calibration_df["gaze_y"].tolist()[1:]
    calibration_gaze_y = [y for y in calibration_gaze_y if y != "failed"]
    calibration_gaze_y = [float(y) for y in calibration_gaze_y]
    calibration = {"gaze_x": calibration_gaze_x, "gaze_y": calibration_gaze_y}

    return calibration


def _compute_avg_gaze_list(gaze_list, matrix_x_uniques, matrix_y_uniques):
    avg_gaze_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
    for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
        for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
            avg_x = sum(gaze_list[matrix_y_index][matrix_x_index]["gaze_x"]) / len(gaze_list[matrix_y_index][matrix_x_index]["gaze_x"])
            avg_y = sum(gaze_list[matrix_y_index][matrix_x_index]["gaze_y"]) / len(gaze_list[matrix_y_index][matrix_x_index]["gaze_y"])
            avg_gaze_list[matrix_y_index][matrix_x_index] = {"avg_gaze_x": avg_x, "avg_gaze_y": avg_y}
    return avg_gaze_list


def _get_std_calibration_point_list(matrix_x_uniques, matrix_y_uniques):
    calibration_point_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
    for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
        for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
            point_x = configs.left_top_text_center[0] + configs.text_width * matrix_x
            point_y = configs.left_top_text_center[1] + configs.text_height * matrix_y
            calibration_point_list[matrix_y_index][matrix_x_index] = {"point_x": point_x, "point_y": point_y}
    return calibration_point_list


def read_calibration(mode="original_gaze_data"):
    '''
    :return:
    subject_list: within each subject, there are 3 lists
        calibration data list: 1st layer -> subject, 2nd layer -> calibration points, 3rd layer -> all calibration data [x_1, y_1], [x_2, y_2], ...
        calibration avg data list: 1st layer -> subject, 2nd layer -> calibration points, 3rd layer -> avg calibration data [avg_x, avg_y]
        calibration point list: 1st layer -> calibration points, 2nd layer -> [x, y]
    '''
    data_path_prefix = f"{os.getcwd()}/data/{mode}/{configs.round_num}/{configs.exp_device}"
    subject_path_list = os.listdir(data_path_prefix)
    subject_path_list.sort()

    subject_list = []
    for subject_index, subject_name in enumerate(subject_path_list):
        subject_path = f"{data_path_prefix}/{subject_name}/calibration.csv"
        pd_calibration_file = pd.read_csv(subject_path)

        matrix_x_uniques = pd_calibration_file["matrix_x"].unique()
        matrix_x_uniques.sort()
        matrix_y_uniques = pd_calibration_file["matrix_y"].unique()
        matrix_y_uniques.sort()
        gaze_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
        for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
            for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
                # calibration_df = pd_calibration_file[(pd_calibration_file["matrix_x"] == matrix_x) & (pd_calibration_file["matrix_y"] == matrix_y)]
                # calibration_gaze_x = calibration_df["gaze_x"].tolist()[1:]
                # calibration_gaze_x = [x for x in calibration_gaze_x if x != "failed"]
                # calibration_gaze_x = [float(x) for x in calibration_gaze_x]
                # calibration_gaze_y = calibration_df["gaze_y"].tolist()[1:]
                # calibration_gaze_y = [y for y in calibration_gaze_y if y != "failed"]
                # calibration_gaze_y = [float(y) for y in calibration_gaze_y]
                # calibration = {"gaze_x": calibration_gaze_x, "gaze_y": calibration_gaze_y}
                calibration_df = pd_calibration_file[(pd_calibration_file["matrix_x"] == matrix_x) & (pd_calibration_file["matrix_y"] == matrix_y)]
                calibration = _get_gaze_of_matrix_x_and_matrix_y(calibration_df)
                gaze_list[matrix_y_index][matrix_x_index] = calibration

        # avg_gaze_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
        # for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
        #     for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
        #         avg_x = sum(gaze_list[matrix_y_index][matrix_x_index]["gaze_x"]) / len(gaze_list[matrix_y_index][matrix_x_index]["gaze_x"])
        #         avg_y = sum(gaze_list[matrix_y_index][matrix_x_index]["gaze_y"]) / len(gaze_list[matrix_y_index][matrix_x_index]["gaze_y"])
        #         avg_gaze_list[matrix_y_index][matrix_x_index] = {"avg_gaze_x": avg_x, "avg_gaze_y": avg_y}
        avg_gaze_list = _compute_avg_gaze_list(gaze_list, matrix_x_uniques, matrix_y_uniques)

        # calibration_point_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
        # for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
        #     for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
        #         point_x = configs.left_top_text_center[0] + configs.text_width * matrix_x
        #         point_y = configs.left_top_text_center[1] + configs.text_height * matrix_y
        #         calibration_point_list[matrix_y_index][matrix_x_index] = {"point_x": point_x, "point_y": point_y}
        calibration_point_list = _get_std_calibration_point_list(matrix_x_uniques, matrix_y_uniques)

        subject_list.append([gaze_list, avg_gaze_list, calibration_point_list])

    return subject_list


def read_calibration_with_certain_corners(corner_index_dict):
    """
    只读取manual calibration中，四个角点的数据。
    """
    data_path_prefix = f"{os.getcwd()}/data/original_gaze_data/{configs.round_num}/{configs.exp_device}"
    subject_path_list = os.listdir(data_path_prefix)
    subject_path_list.sort()

    subject_list = []
    for subject_index, subject_name in enumerate(subject_path_list):
        subject_path = f"{data_path_prefix}/{subject_name}/calibration.csv"
        pd_calibration_file = pd.read_csv(subject_path)

        matrix_x_uniques = pd_calibration_file["matrix_x"].unique()
        matrix_x_uniques.sort()
        matrix_y_uniques = pd_calibration_file["matrix_y"].unique()
        matrix_y_uniques.sort()
        gaze_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
        for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
            for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
                calibration_df = pd_calibration_file[(pd_calibration_file["matrix_x"] == matrix_x) & (pd_calibration_file["matrix_y"] == matrix_y)]
                iter_list = calibration_df["iteration"].unique()
                iter_list.sort()
                if (matrix_y, matrix_x) in corner_index_dict:
                    iteration = corner_index_dict[(matrix_y, matrix_x)]
                    calibration_df = calibration_df[calibration_df["iteration"] == iter_list[iteration]]

                # calibration_gaze_x = calibration_df["gaze_x"].tolist()[1:]
                # calibration_gaze_x = [x for x in calibration_gaze_x if x != "failed"]
                # calibration_gaze_x = [float(x) for x in calibration_gaze_x]
                # calibration_gaze_y = calibration_df["gaze_y"].tolist()[1:]
                # calibration_gaze_y = [y for y in calibration_gaze_y if y != "failed"]
                # calibration_gaze_y = [float(y) for y in calibration_gaze_y]
                # calibration = {"gaze_x": calibration_gaze_x, "gaze_y": calibration_gaze_y}
                # gaze_list[matrix_y_index][matrix_x_index] = calibration

                calibration = _get_gaze_of_matrix_x_and_matrix_y(calibration_df)
                gaze_list[matrix_y_index][matrix_x_index] = calibration

        avg_gaze_list = _compute_avg_gaze_list(gaze_list, matrix_x_uniques, matrix_y_uniques)
        calibration_point_list = _get_std_calibration_point_list(matrix_x_uniques, matrix_y_uniques)

        subject_list.append([gaze_list, avg_gaze_list, calibration_point_list])

    return subject_list


