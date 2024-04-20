import pandas as pd

import configs
from read.read_model_config import read_model_config
from read.read_text import read_text_sorted_mapping_and_group_with_para_id


def _supplement_bound_points(para_id, start_col, end_col, start_row, offset_x=-1, offset_y=-0.5):
    new_point_list = []
    row = start_row + offset_y
    start_x = configs.left_top_text_center[0]
    start_y = configs.left_top_text_center[1] + start_row * configs.text_height
    y = start_y + offset_y * configs.text_height

    for index in range(start_col, end_col):
        x = start_x + offset_x * configs.text_width + index * configs.text_width
        col = offset_x + index
        new_point = {"para_id": para_id,
                     "x": x, "y": y,
                     "row": row, "col": col,
                     "word": "blank_supplement",
                     "prediction": 0}
        new_point_list.append(new_point)

    return new_point_list


def _add_boundary_points_to_text_data(text_sorted_mapping_with_prediction_list, model_index):
    """
    代码无法处理某行是空行，而前后又有文字的情况。
    """
    text_sorted_mapping_with_prediction = text_sorted_mapping_with_prediction_list[model_index]
    for text_index in range(len(text_sorted_mapping_with_prediction)):
        new_point_list = []
        df = text_sorted_mapping_with_prediction[text_index]

        row_list = df["row"].unique().tolist()
        row_list.sort()
        para_id = df["para_id"].tolist()[0]
        for row_index in range(len(row_list)):
            col_list = df[df["row"] == row_list[row_index]]["col"].unique().tolist()
            col_list.sort()

            new_points = []
            left_right_padding = 3
            up_down_padding = 1

            # 如果某一行内本身就存在一些空白点，则将其添加进去。如果没有空白点，则添加首尾的边界点。
            min_col = min(col_list)
            points = _supplement_bound_points(para_id=para_id, start_col=-left_right_padding, end_col=min_col, start_row=row_list[row_index], offset_x=0, offset_y=0)
            new_points.extend(points)
            max_col = max(col_list)
            points = _supplement_bound_points(para_id=para_id, start_col=max_col + 1, end_col=configs.col_num + left_right_padding, start_row=row_list[row_index], offset_x=0, offset_y=0)
            new_points.extend(points)

            # 除此之外，额外添加行与行之间、以及文字边缘的点。目前的设计是每一行添加其上方的点。
            # if row_index > 0:
            #     points = supplement_bound_points(para_id=para_id, start_col=-left_right_padding, end_col=configs.col_num + left_right_padding, start_row=row_list[row_index], offset_x=0, offset_y=-0.5)
            #     new_points.extend(points)

            if row_index == 0:
                for repeat_index in range(1, up_down_padding + 1):
                    points = _supplement_bound_points(para_id=para_id, start_col=-left_right_padding, end_col=configs.col_num + left_right_padding, start_row=row_list[row_index], offset_x=0,
                                                     offset_y=-0.5 * repeat_index)
                    new_points.extend(points)

            # 如果是最后一行，则还需要添加下方的点。
            if row_index == len(row_list) - 1:
                for repeat_index in range(1, up_down_padding + 1):
                    points = _supplement_bound_points(para_id=para_id, start_col=-left_right_padding, end_col=configs.col_num + left_right_padding, start_row=row_list[row_index], offset_x=0,
                                                     offset_y=0.5 * repeat_index)
                    new_points.extend(points)

            new_point_list.extend(new_points)

        # add new_point_list to df.
        df = pd.concat([df, pd.DataFrame(new_point_list)], ignore_index=True)
        text_sorted_mapping_with_prediction[text_index] = df
    return text_sorted_mapping_with_prediction


def _copy_boundary_points_to_text_data(text_sorted_mapping_with_prediction_list, model_index):
    """
    由于添加boundary的过程都是固定的，因此可以直接复制第一个model的结果。
    """
    text_sorted_mapping_with_prediction = text_sorted_mapping_with_prediction_list[model_index]
    text_sorted_mapping_with_prediction_0 = text_sorted_mapping_with_prediction_list[0]

    for text_index in range(len(text_sorted_mapping_with_prediction)):
        prediction_list = text_sorted_mapping_with_prediction[text_index]["prediction"].tolist()
        text_sorted_mapping_with_prediction[text_index] = text_sorted_mapping_with_prediction_0[text_index].copy()
        for index in range(text_sorted_mapping_with_prediction[text_index].shape[0] - len(prediction_list)):
            # 多出来的supplement点，prediction为0。
            prediction_list.append(0)
        text_sorted_mapping_with_prediction[text_index]["prediction"] = prediction_list

    return text_sorted_mapping_with_prediction


def _add_penalty_to_text_data(text_sorted_mapping_with_prediction_list, model_index):
    text_sorted_mapping_with_prediction = text_sorted_mapping_with_prediction_list[model_index]
    for text_index in range(len(text_sorted_mapping_with_prediction)):
        text_penalty = []
        df = text_sorted_mapping_with_prediction[text_index]

        blank_supplement_col_list = df[df["word"] == "blank_supplement"]["col"].unique().tolist()
        blank_supplement_col_below_zero_list = [blank_supplement_col_list[i] for i in range(len(blank_supplement_col_list)) if blank_supplement_col_list[i] < 0]
        left_attract_supplement_threshold = -1 if len(blank_supplement_col_below_zero_list) > 0 else 0

        for index, row in df.iterrows():
            word = row["word"]
            col_index = row["col"]
            row_index = row["row"]
            penalty = 1
            # 添加的点按空白处理。
            if word == "blank_supplement":
                # 对于最左上端的点，惩罚项为1，保证它能够吸引reading points。
                if row_index < 0 and (left_attract_supplement_threshold <= col_index < configs.col_num / 6):
                    penalty = 1
                elif row_index == 0 and (left_attract_supplement_threshold <= col_index < configs.col_num / 6):
                    penalty = 1
                # 对于其他点，惩罚项为负，保证他们会排斥reading points。
                else:
                    penalty = min(penalty, configs.empty_penalty)
            else:
                # 非添加的点分多种情况处理。惩罚项最终按更大的计算。
                # 1. 标点符号
                if word in configs.punctuation_list:
                    penalty = min(penalty, configs.punctuation_penalty)
                # 2. 空白
                if word.strip() == "":
                    penalty = min(penalty, configs.empty_penalty)
                # 3. 位置靠近行尾。
                # if df[(df["row"] == row_index) & (df["col"] == col_index + 1)]["word"].shape[0] > 0 and df[(df["row"] == row_index) & (df["col"] == col_index + 1)]["word"].tolist()[0] == "blank_supplement":
                #     penalty = min(penalty, configs.location_penalty)
                # 4. 位置靠近行首。
                # if (df[(df["row"] == row_index) & (df["col"] == col_index - 1)]["word"].shape[0] > 0 and
                #         (df[(df["row"] == row_index) & (df["col"] == col_index - 1)]["word"].tolist()[0] == "blank_supplement" or
                #          df[(df["row"] == row_index) & (df["col"] == col_index - 1)]["word"].tolist()[0].strip() == "")):
                #     penalty = min(penalty, configs.location_penalty)
            text_penalty.append(penalty)
        df["penalty"] = text_penalty
        text_sorted_mapping_with_prediction[text_index] = df
    return text_sorted_mapping_with_prediction


def _copy_penalty_to_text_data(text_sorted_mapping_with_prediction_list, model_index):
    """
    由于添加penalty的过程都是固定的，因此可以直接复制第一个model的结果。
    """
    text_sorted_mapping_with_prediction = text_sorted_mapping_with_prediction_list[model_index]
    text_sorted_mapping_with_prediction_0 = text_sorted_mapping_with_prediction_list[0]

    for text_index in range(len(text_sorted_mapping_with_prediction)):
        penalty_list = text_sorted_mapping_with_prediction_0[text_index]["penalty"].tolist()
        text_sorted_mapping_with_prediction[text_index]["penalty"] = penalty_list

    return text_sorted_mapping_with_prediction


def preprocess_text_in_batch():
    model_config = read_model_config()

    # 首先，批量读取不同prediction下的text_sorted_mapping。
    text_sorted_mapping_with_prediction_list = []
    for model_index in range(len(model_config)):
        model_config_row = model_config.iloc[model_index]
        model_name = model_config_row["model_index"]
        text_sorted_mapping_with_prediction = read_text_sorted_mapping_and_group_with_para_id(f"_with_prediction_{model_name}")
        text_sorted_mapping_with_prediction_list.append(text_sorted_mapping_with_prediction)

    # 对每个model的text_sorted_mapping_with_prediction，添加周边点。
    for model_index in range(len(model_config)):
        if model_index == 0:
            text_sorted_mapping_with_prediction_list[model_index] = _add_boundary_points_to_text_data(text_sorted_mapping_with_prediction_list, model_index)
        else:
            text_sorted_mapping_with_prediction_list[model_index] = _copy_boundary_points_to_text_data(text_sorted_mapping_with_prediction_list, model_index)

    # 对添加完周边点的text_sorted_mapping_with_prediction，进行权重修改。
    for model_index in range(len(model_config)):
        if model_index == 0:
            text_sorted_mapping_with_prediction_list[model_index] = _add_penalty_to_text_data(text_sorted_mapping_with_prediction_list, model_index)
        else:
            text_sorted_mapping_with_prediction_list[model_index] = _copy_penalty_to_text_data(text_sorted_mapping_with_prediction_list, model_index)

    return text_sorted_mapping_with_prediction_list



