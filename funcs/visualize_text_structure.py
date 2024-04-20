import os.path

import matplotlib.pyplot as plt
from matplotlib import patches

import configs
from read.read_text import read_text_sorted_mapping_and_group_with_para_id

plt.rcParams['font.sans-serif'] = ['SimHei']


def _get_text_num(text_list, para_id):
    text = text_list[para_id]
    text_num = 0
    for text_unit_index in range(text.shape[0]):
        text_unit = text["word"].iloc[text_unit_index]
        if len(text_unit.strip()) == 0:
            continue
        text_num += 1

    return text_num


def _check_direction_in_boundary_text_num(filtered_text):
    '''
    如果该位置是行首或行尾（列首或列尾），或者该位置内容为空格，则返回True，否则返回False。
    :param filtered_text:
    :return:
    '''
    if filtered_text.shape[0] == 0 or len(filtered_text.iloc[0].strip()) == 0:
        return True
    else:
        return False


# def _get_boundary_text_num(text_list, para_id):
#     text = text_list[para_id]
#     boundary_text_num = 0
#     for text_unit_index in range(text.shape[0]):
#         text_unit = text["word"].iloc[text_unit_index]
#         if len(text_unit.strip()) == 0:
#             continue
#
#         # row = text["row"].iloc[text_unit_index]
#         # col = text["col"].iloc[text_unit_index]
#         # if row == configs.row_num or row == 0:
#         #     boundary_text_num += 1
#         # elif col == configs.col_num or col == 0:
#         #     boundary_text_num += 1
#         # else:
#         #     bool_check = False
#         #     # left
#         #     left_text = text["word"][(text["row"] == row) & (text["col"] == col - 1)]
#         #     bool_check = bool_check or _check_direction_in_boundary_text_num(left_text)
#         #     # right
#         #     right_text = text["word"][(text["row"] == row) & (text["col"] == col + 1)]
#         #     bool_check = bool_check or _check_direction_in_boundary_text_num(right_text)
#         #     # up
#         #     up_text = text["word"][(text["row"] == row - 1) & (text["col"] == col)]
#         #     bool_check = bool_check or _check_direction_in_boundary_text_num(up_text)
#         #     # down
#         #     down_text = text["word"][(text["row"] == row + 1) & (text["col"] == col)]
#         #     bool_check = bool_check or _check_direction_in_boundary_text_num(down_text)
#         #
#         #     if bool_check:
#         #         boundary_text_num += 1
#
#         boundary_num = _compute_boundary_num_given_text_unit(text, text_unit_index)
#         boundary_text_num += boundary_num
#
#     return boundary_text_num


def _compute_boundary_num_given_text_unit(text, text_unit_index):
    row = text["row"].iloc[text_unit_index]
    col = text["col"].iloc[text_unit_index]

    left_text = text["word"][(text["row"] == row) & (text["col"] == col - 1)]
    right_text = text["word"][(text["row"] == row) & (text["col"] == col + 1)]
    up_text = text["word"][(text["row"] == row - 1) & (text["col"] == col)]
    down_text = text["word"][(text["row"] == row + 1) & (text["col"] == col)]

    left_check = _check_direction_in_boundary_text_num(left_text)
    right_check = _check_direction_in_boundary_text_num(right_text)
    up_check = _check_direction_in_boundary_text_num(up_text)
    down_check = _check_direction_in_boundary_text_num(down_text)
    count_true = sum(int(value) for value in [left_check, right_check, up_check, down_check])

    # minus = 0
    # if row == 0:
    #     if col == 0:
    #         minus = 2
    #     elif col == configs.col_num:
    #         minus = 2
    #     else:
    #         minus = 1
    # elif row == configs.row_num:
    #     if col == 0:
    #         minus = 2
    #     elif col == configs.row_num:
    #         minus = 2
    #     else:
    #         minus = 1
    # else:
    #     if col == 0:
    #         minus = 1
    #     elif col == configs.row_num:
    #         minus = 1
    #     else:
    #         minus = 0
    # boundary_num = max(0, count_true - minus)
    # return boundary_num

    return count_true


def _get_boundary_num(text_list, para_id):
    text = text_list[para_id]
    boundary_num = 0
    boundary_list = []
    for text_unit_index in range(text.shape[0]):
        text_unit = text["word"].iloc[text_unit_index]
        if len(text_unit.strip()) == 0:
            continue

        boundary = _compute_boundary_num_given_text_unit(text, text_unit_index)
        boundary_list.append(boundary)
        boundary_num += boundary

    return boundary_num


def _visualize_text_given_para_id(text_list, para_id):
    text = text_list[para_id]
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.set_xlim(0, configs.screen_width)
    ax.set_ylim(configs.screen_height, 0)

    for text_unit_index in range(text.shape[0]):
        text_unit = text["word"].iloc[text_unit_index]
        if len(text_unit.strip()) == 0:
            continue
        text_unit_center_x = text["x"].iloc[text_unit_index]
        text_unit_center_y = text["y"].iloc[text_unit_index]

        boundary_num = _compute_boundary_num_given_text_unit(text, text_unit_index)
        face_color = "yellow"
        alpha = min(boundary_num, 3) / 4

        rect = patches.Rectangle((text_unit_center_x - configs.text_width / 2, text_unit_center_y - configs.text_height / 2),
                                 configs.text_width, configs.text_height, linewidth=1, edgecolor='#AAAAAA', facecolor=face_color, alpha=alpha)
        ax.add_patch(rect)
        ax.text(text_unit_center_x, text_unit_center_y, text_unit, fontsize=20, ha="center", va="center")

    text_num = _get_text_num(text_list, para_id)
    boundary_num = _get_boundary_num(text_list, para_id)

    ax.text(configs.screen_width / 2, configs.screen_height * 3 / 4, f"text num: {text_num}", fontsize=30)
    ax.text(configs.screen_width / 2, configs.screen_height * 3 / 4 + 50, f"boundary text num: {boundary_num}({boundary_num/text_num:.2f})", fontsize=30)

    save_prefix = "image/text_structure"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    save_path = f"{save_prefix}/para_id_{para_id}-text_num_{text_num}-boundary_num_{boundary_num}.png"
    # save_path = f"{save_prefix}/para_id_{para_id}-text_num_{text_num}-boundary_text_num_{boundary_text_num}-fine_boundary_text_num_{fine_boundary_text_num}.png"
    plt.savefig(save_path, dpi=200)
    plt.clf()
    plt.close()


def _visualize_text_num_hist(text_list):
    text_num_list = []
    boundary_num_list = []

    for para_id_index in range(len(text_list)):
        text_num = _get_text_num(text_list, para_id_index)
        text_num_list.append(text_num)
        boundary_num = _get_boundary_num(text_list, para_id_index)
        boundary_num_list.append(boundary_num)

    fig, ax = plt.subplots(1, 3, figsize=(16, 10))
    ax[0].hist(text_num_list, bins=8)
    ax[0].set_title('Text Num Distribution')

    ax[1].hist(boundary_num_list, bins=8)
    ax[1].set_title('Boundary Text Num Distribution')

    avg_boundary_num_list = [boundary_num_list[i] / text_num_list[i] for i in range(len(text_num_list))]
    ax[2].hist(avg_boundary_num_list, bins=8)
    ax[2].set_title('Avg Boundary Text Num Distribution')

    save_prefix = "image/text_structure"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    save_path = f"{save_prefix}/text_num_hist.png"
    plt.savefig(save_path)


def visualize_text_structure():
    text_list = read_text_sorted_mapping_and_group_with_para_id()

    for para_id_index in range(len(text_list)):
        _visualize_text_given_para_id(text_list, para_id_index)

    _visualize_text_num_hist(text_list)



