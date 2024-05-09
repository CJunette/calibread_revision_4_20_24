import os

import pandas as pd

import configs


def read_raw_reading(reading_mode, suffix=""):
    file_prefix = f"{os.getcwd()}/data/{reading_mode}_gaze_data/{configs.round_num}/{configs.exp_device}"
    file_paths = os.listdir(file_prefix)
    file_paths.sort()

    raw_reading_dfs_1 = []
    for file_index, file_path in enumerate(file_paths):
        raw_reading_dfs_2 = []
        reading_file_path_prefix = f"{file_prefix}/{file_path}/reading{suffix}"
        reading_file_paths = os.listdir(reading_file_path_prefix)
        reading_file_paths.sort(key=lambda x: int(x.split(".")[0]))
        for reading_file_index, reading_file_path in enumerate(reading_file_paths):
            reading_file = pd.read_csv(f"{reading_file_path_prefix}/{reading_file_path}")
            if "Unnamed: 0" in reading_file.columns:
                reading_file = reading_file.drop(columns=["Unnamed: 0"])
            raw_reading_dfs_2.append(reading_file)
        raw_reading_dfs_1.append(raw_reading_dfs_2)
    return raw_reading_dfs_1


def read_text_density():
    data_path_prefix = f"{os.getcwd()}/data/text_density/{configs.round_num}/{configs.exp_device}"
    subject_list = os.listdir(data_path_prefix)

    subject_density_list = []
    for subject_index in range(len(subject_list)):
        file_path = f"{data_path_prefix}/{subject_list[subject_index]}/text_density.csv"
        pd_density = pd.read_csv(file_path)

        text_index_list = pd_density["para_id"].unique().tolist()
        text_index_list.sort()
        text_density_list = []
        for text_index in range(len(text_index_list)):
            text_density_df = pd_density[pd_density["para_id"] == text_index]
            text_density_list.append(text_density_df)

        subject_density_list.append(text_density_list)

    return subject_density_list

