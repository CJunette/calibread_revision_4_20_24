import os
import time

import pandas as pd
from hanlp_restful import HanLPClient
from phrasetree.tree import Tree

import configs
from funcs.split_text_with_punctuation import split_text_with_punctuation
from read.read_text import read_text_raw_txt, read_text_pkl, read_text_sorted_mapping_with_constituency_depth
from read.read_keys import read_hanlp_key
from read.read_text import read_text_sorted_mapping_and_group_with_para_id
from read.read_text import read_fine_tokens


def _add_depth_to_full_text_sorted_mapping(token_depth_list, text_sorted_mapping):
    """
    根据token，将depth添加到text_sorted_mapping中。
    :param token_depth_list:
    :param text_sorted_mapping:
    :return: text_sorted_mapping为更新depth参数后的dataframe。token_depth_list_merged为将英文合并后的token_depth_list。
    """
    text_sorted_mapping["constituency_depth"] = 0

    # 先对token_depth_list，基于token进行合并，将英文都合并成一个token，depth取其均值。
    token_depth_list_merged = []
    token_index = 0
    while token_index < len(token_depth_list):
        cur_token = token_depth_list[token_index]["token"]
        # 如果字符串中存在英文
        if cur_token.isascii():
            probe_index = token_index + 1
            while probe_index < len(token_depth_list) and token_depth_list[probe_index]["token"].isascii():
                probe_index += 1
            merged_token = "".join([token_depth_list[token_index + i]["token"] for i in range(probe_index - token_index)])
            merged_depth = sum([token_depth_list[token_index + i]["constituency_depth"] for i in range(probe_index - token_index)]) / (probe_index - token_index)
            token_depth_list_merged.append({"token": merged_token, "constituency_depth": merged_depth})
            token_index = probe_index
        else:
            token_depth_list_merged.append(token_depth_list[token_index])
            token_index += 1

    token_index = 0
    text_unit_index = 0
    while token_index < len(token_depth_list_merged):
        cur_token = token_depth_list_merged[token_index]["token"]
        # 跳过空字符串。
        while len(text_sorted_mapping.iloc[text_unit_index]["word"].strip()) == 0:
            text_unit_index += 1

        text_unit = text_sorted_mapping.iloc[text_unit_index]["word"]
        merged_text_unit = text_unit
        probe_index = text_unit_index + 1
        while probe_index < len(text_sorted_mapping) and merged_text_unit != cur_token:
            merged_text_unit += text_sorted_mapping.iloc[probe_index]["word"]
            probe_index += 1

        text_sorted_mapping_index = text_sorted_mapping.index
        for i in range(text_unit_index, probe_index):
            # 获取i在text_sorted_mapping中的index
            loc_index = text_sorted_mapping_index[i]
            text_sorted_mapping.loc[loc_index, "constituency_depth"] = token_depth_list_merged[token_index]["constituency_depth"]

        text_unit_index = probe_index
        token_index += 1

    return text_sorted_mapping, token_depth_list_merged


def _constituency_parsing_all_text(HanLP, test_str, sleep_time=1.0):
    if len(test_str.split()) == 0:
        return [{"token": test_str, "constituency_depth": 0}]
    elif len(test_str.split()) == 1 and test_str in configs.punctuation_list:
        return [{"token": test_str, "constituency_depth": 0}]

    doc = HanLP.parse(test_str, tasks=['pos', 'con'])
    # doc.pretty_print() # 用于可视化doc的成分句法结构。

    token_depth_list = []
    for sentence in doc["con"]:
        _traverse_tree(sentence, token_depth_list) # _traverse_tree得到的token和之前得到的fine token是不一样的。
    # print()
    time.sleep(sleep_time)
    return token_depth_list


def _traverse_tree(tree, token_depth_list, depth=0):
    """
    递归遍历树结构并打印每个节点的字符串及其深度
    """
    if not isinstance(tree[0], Tree):
        # print(tree, depth)
        token_depth_list.append({"token": tree[0], "constituency_depth": depth})
    else:
        for sub_tree in tree:
            _traverse_tree(sub_tree, token_depth_list, depth + 1)

    # # 如果树是一个字符串，打印该字符串及其深度。这里的代码是以tree的格式为list做处理的。
    # if isinstance(tree, str):
    #     return
    # elif isinstance(tree, list) and len(tree) == 1 and isinstance(tree[0], str):
    #     print(tree, depth - 2)
    # elif isinstance(tree, list) and len(tree) == 2 and isinstance(tree[0], str):
    #     traverse_tree(tree[1], depth + 1)
    # # 如果树是一个列表，遍历其子树
    # elif isinstance(tree, list):
    #     for subtree in tree:
    #         traverse_tree(subtree, depth)


def _traverse_tree_start_with_IP(tree, token_depth_list, depth=0):
    """
    递归遍历树结构并打印每个节点的字符串及其深度，但此处的遍历只计算到最近的IP的depth。
    """
    if not isinstance(tree[0], Tree):
        # print(tree, depth)
        token_depth_list.append({"token": tree[0], "constituency_depth": depth})
    else:
        for sub_tree in tree:
            if sub_tree._label.strip() == "IP":
                _traverse_tree(sub_tree, token_depth_list, 0)
            else:
                _traverse_tree(sub_tree, token_depth_list, depth + 1)


def _save_full_passage_depth(HanLP, raw_txt_list, text_sorted_mapping_list):
    # 将整个文本进行成分句法分析。
    full_passage_merged_depth_list = []
    for raw_text_index in range(len(raw_txt_list)):
        # for raw_text_index in range(5, 6):
        print(f"parsing full text {raw_text_index + 1} ...")
        token_depth_list = _constituency_parsing_all_text(HanLP, raw_txt_list[raw_text_index])
        text_sorted_mapping, token_depth_list_merged = _add_depth_to_full_text_sorted_mapping(token_depth_list, text_sorted_mapping_list[raw_text_index])
        text_sorted_mapping_list[raw_text_index] = text_sorted_mapping

        full_passage_merged_depth_list.append(token_depth_list_merged)

        save_path_prefix = f"{os.getcwd()}/data/text/{configs.round_num}/constituency_depth"
        if not os.path.exists(save_path_prefix):
            os.makedirs(save_path_prefix)
        save_path = f"{save_path_prefix}/full_passage_{raw_text_index}.csv"
        text_sorted_mapping.to_csv(save_path, index=False, encoding="utf-8-sig")


def _save_split_passage_depth(HanLP):
    # 将每句话按标点分开后，再做一次parsing。
    # 首先先将每句话按标点分开。
    split_text_sorted_mapping_list, split_sentence_unit_list = split_text_with_punctuation()
    split_passage_merged_depth_list = []
    for raw_text_index in range(len(split_sentence_unit_list)):
    # for raw_text_index in range(5, 6):
        split_text_sorted_mapping = split_text_sorted_mapping_list[raw_text_index]
        split_text_sorted_mapping["constituency_depth"] = 0
        split_passage_merged_depth_list_1 = []
        for sentence_unit_index in range(len(split_sentence_unit_list[raw_text_index])):
            print(f"parsing split text {raw_text_index + 1}, sentence_unit {sentence_unit_index + 1} / {len(split_sentence_unit_list[raw_text_index])} ...")
            token_depth_list = _constituency_parsing_all_text(HanLP, split_sentence_unit_list[raw_text_index][sentence_unit_index], 1.5)

            if len(token_depth_list) == 1:
                if len(token_depth_list[0]["token"].strip()) == 0 or token_depth_list[0]["token"] in configs.punctuation_list:
                    split_passage_merged_depth_list_1.extend(token_depth_list)
                    continue

            split_text_sorted_mapping_with_sentence_unit = split_text_sorted_mapping[split_text_sorted_mapping["sentence"] == split_sentence_unit_list[raw_text_index][sentence_unit_index]]
            split_text_sorted_mapping_with_sentence_unit, token_depth_list_merged = _add_depth_to_full_text_sorted_mapping(token_depth_list, split_text_sorted_mapping_with_sentence_unit)
            split_text_sorted_mapping[split_text_sorted_mapping["sentence"] == split_sentence_unit_list[raw_text_index][sentence_unit_index]] = split_text_sorted_mapping_with_sentence_unit
            split_passage_merged_depth_list_1.extend(token_depth_list_merged)

        split_passage_merged_depth_list.append(split_passage_merged_depth_list_1)

        save_path_prefix = f"{os.getcwd()}/data/text/{configs.round_num}/constituency_depth"
        if not os.path.exists(save_path_prefix):
            os.makedirs(save_path_prefix)
        save_path = f"{save_path_prefix}/split_passage_{raw_text_index}.csv"
        split_text_sorted_mapping_list[raw_text_index].to_csv(save_path, index=False, encoding="utf-8-sig")


def _combine_depth_list(prefix):
    file_path_prefix = f"{os.getcwd()}/data/text/{configs.round_num}/constituency_depth"
    file_list = os.listdir(file_path_prefix)
    file_list_new = [file_path for file_path in file_list if file_path.startswith(prefix)]
    file_list_new.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    df_list = []
    if len(file_list_new) < configs.passage_num:
        raise ValueError("The number of files is less than the passage number.")
    for file_path in file_list_new:
        df = pd.read_csv(f"{file_path_prefix}/{file_path}", encoding="utf-8-sig")
        df_list.append(df)

    new_df = pd.concat(df_list, ignore_index=True)
    save_path = f"{os.getcwd()}/data/text/{configs.round_num}/text_sorted_mapping_with_constituency_depth_{prefix}.csv"
    new_df.to_csv(save_path, index=False, encoding="utf-8-sig")


def add_constituency_parsing_to_text():
    """
    将每篇文章成分句法分析的结果（full_text和split_text），都保存到text/{configs.round_num}/constituency_depth文件夹下。
    full_text，即将该文章的所有句子合并成一个句子，然后做parsing。
    split_text，即将该文章的每个由标点分隔开的单元，单独做parsing。
    然后，将full_text和split_text的结果，分别合并，生成text_sorted_mapping_with_constituency_depth_full_passage.csv和text_sorted_mapping_with_constituency_depth_split_passage.csv。
    :return:
    """

    raw_txt_list = read_text_raw_txt()
    text_sorted_mapping_list = read_text_sorted_mapping_and_group_with_para_id()

    hanlp_key = read_hanlp_key()
    HanLP = HanLPClient("https://www.hanlp.com/api", auth=hanlp_key, language='zh')

    _save_full_passage_depth(HanLP, raw_txt_list, text_sorted_mapping_list)
    _combine_depth_list("full_passage")
    _save_split_passage_depth(HanLP)
    _combine_depth_list("split_passage")


def _add_constituency_parsing_to_text(text_sorted_mapping, text_sorted_mapping_with_constituency_depth, text_type):
    """
    将text_sorted_mapping_with_constituency_depth中的constituency_depth列，添加到text_sorted_mapping中。
    :param text_sorted_mapping:
    :param text_sorted_mapping_with_constituency_depth:
    :param text_type:
    :return:
    """

    for text_index in range(len(text_sorted_mapping)):
        constituency_depth_list = text_sorted_mapping_with_constituency_depth[text_index][f"constituency_depth"].tolist()
        text_sorted_mapping[text_index][f"{text_type}_constituency_depth"] = constituency_depth_list

    return text_sorted_mapping


def save_pkl_with_constituency_parsing():
    """
    读取text_sorted_mapping_for_model.pkl，如果其中不存在constituency_depth列，则将数据添加进去，并保存。
    :return:
    """

    text_sorted_mapping = read_text_pkl()

    text_sorted_mapping_with_full_text_constituency_depth = read_text_sorted_mapping_with_constituency_depth("full_passage")
    text_sorted_mapping_with_split_text_constituency_depth = read_text_sorted_mapping_with_constituency_depth("split_passage")

    text_sorted_mapping = _add_constituency_parsing_to_text(text_sorted_mapping, text_sorted_mapping_with_full_text_constituency_depth, "full_text")
    text_sorted_mapping = _add_constituency_parsing_to_text(text_sorted_mapping, text_sorted_mapping_with_split_text_constituency_depth, "split_text")

    df = pd.concat(text_sorted_mapping, ignore_index=True)
    file_name = f"{os.getcwd()}\\data\\text\\{configs.round_num}\\text_sorted_mapping_for_model.pkl"
    df.to_pickle(file_name)

