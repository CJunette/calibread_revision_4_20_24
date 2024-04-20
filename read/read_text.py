import os
import re

import pandas as pd

import configs


def read_text_sorted_mapping(suffix=""):
    file_path = f"{os.getcwd()}/data/text/{configs.round_num}/text_sorted_mapping{suffix}.csv"
    text_sorted_mapping = pd.read_csv(file_path)
    if "Unnamed: 0" in text_sorted_mapping.columns:
        text_sorted_mapping = text_sorted_mapping.drop(columns=["Unnamed: 0"])
    return text_sorted_mapping


def group_text_sorted_mapping_with_para_id(text_sorted_mapping):
    para_id_list = text_sorted_mapping["para_id"].unique()
    para_id_list.sort()

    text_list = []
    for para_id_index, para_id in enumerate(para_id_list):
        text_for_para_id = text_sorted_mapping[text_sorted_mapping["para_id"] == para_id]
        text_list.append(text_for_para_id)

    return text_list


def read_text_sorted_mapping_and_group_with_para_id(suffix=""):
    text_sorted_mapping = read_text_sorted_mapping(suffix)
    text_list = group_text_sorted_mapping_with_para_id(text_sorted_mapping)

    return text_list


def read_text_raw_txt():
    """
    读取exp_txt_sorted.txt文件，并将其转化为list返回。
    :return:
    """
    file_path = f"{os.getcwd()}/data/text/{configs.round_num}/exp_text_sorted.txt"

    text_list = []
    with open(file_path, "r", encoding='utf-8-sig') as file:
        file_lines = file.readlines()
        start_pattern = ".*jjxnb"
        line_index = 0
        while line_index < len(file_lines):
            cur_line = file_lines[line_index]
            match = re.match(start_pattern, cur_line)
            if match:
                probe_index = line_index + 1
                while probe_index < len(file_lines) and not re.match(start_pattern, file_lines[probe_index]):
                    probe_index += 1
                cur_text_list = file_lines[line_index + 1:probe_index]
                if cur_text_list[-1].strip() == "-----":
                    del cur_text_list[-1]

                '''
                思路一：按txt中，添加\n。
                '''
                # for cur_text_line in cur_text_list:
                #     if not cur_text_line.endswith("\n"):
                #         cur_text_line += "\n"
                '''
                思路二：去掉所有的\n。
                '''
                for cur_text_line_index in range(len(cur_text_list)):
                    cur_text_list[cur_text_line_index] = cur_text_list[cur_text_line_index].strip()

                cur_text = "".join(cur_text_list)
                text_list.append(cur_text)
                line_index = probe_index
            else:
                line_index += 1

    return text_list


def read_fine_tokens():
    """
    将所有文章的fine tokens读入，并返回list。
    :return:
    """
    file_path_prefix = f"{os.getcwd()}/data/text/{configs.round_num}/tokens/fine_tokens"

    file_list = os.listdir(file_path_prefix)
    file_list.sort(key=lambda x: int(x.split(".csv")[0]))

    token_df_list = []
    for file_index, file_path in enumerate(file_list):
        full_path = f"{file_path_prefix}/{file_path}"
        df = pd.read_csv(full_path, encoding="utf-8-sig")
        token_df_list.append(df)

    return token_df_list


def read_text_pkl():
    """
    将pkl文件读入，按para_id分类，并返回list。
    :return:
    """
    data_path = f"{os.getcwd()}/data/text/{configs.round_num}/text_sorted_mapping_for_model.pkl"
    pd_text_file = pd.read_pickle(data_path)
    if "Unnamed: 0" in pd_text_file.columns:
        pd_text_file.drop(columns=["Unnamed: 0"], inplace=True)
    # divide pd_text_file according to its para_id
    para_id_list = pd_text_file["para_id"].unique()
    para_id_list.sort()
    pd_text_file_list = []
    for para_id in para_id_list:
        pd_text_file_list.append(pd_text_file[pd_text_file["para_id"] == para_id])

    return pd_text_file_list


def read_text_sorted_mapping_with_constituency_depth(suffix):
    file_path = f"{os.getcwd()}/data/text/{configs.round_num}/text_sorted_mapping_with_constituency_depth_{suffix}.csv"
    text_sorted_mapping = pd.read_csv(file_path)
    if "Unnamed: 0" in text_sorted_mapping.columns:
        text_sorted_mapping = text_sorted_mapping.drop(columns=["Unnamed: 0"])

    text_list = group_text_sorted_mapping_with_para_id(text_sorted_mapping)

    return text_list



