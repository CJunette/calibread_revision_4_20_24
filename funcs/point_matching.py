import math
from collections import Counter
import numpy as np
import multiprocessing
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

import configs


def step_1_matching_among_all(text_index, row_index, gaze_index,
                              text_coordinate, text_data,
                              filtered_reading_coordinates, filtered_text_data, filtered_text_coordinate, filtered_reading_density,
                              filtered_indices_of_row, filtered_distances_of_row,
                              filtered_indices_of_all_text, filtered_distance_of_all_text,
                              distance_threshold, bool_weight):
    '''
    将gaze point与text point进行匹配。首先匹配本行的，如果本行没有，再匹配所有text_point（但此时的weight会根据距离做调整，距离越大weight越小）。
    '''
    if filtered_distances_of_row[gaze_index][0] < distance_threshold * configs.text_distance_threshold_ratio:
        point_pair = [filtered_reading_coordinates[gaze_index], filtered_text_coordinate[filtered_indices_of_row[gaze_index][0]]]
        prediction = filtered_text_data.iloc[filtered_indices_of_row[gaze_index][0]]["prediction"]
        penalty = filtered_text_data.iloc[filtered_indices_of_row[gaze_index][0]]["penalty"]
        density = filtered_reading_density[gaze_index]
        distance = filtered_distances_of_row[gaze_index][0]
        if bool_weight:
            if penalty > 0:
                if configs.bool_text_weight:
                    weight = configs.weight_divisor / (abs(density - prediction) + configs.weight_intercept)
                else:
                    weight = penalty
            else:
                weight = penalty
                # weight = configs.weight_divisor / (abs(density - prediction) + configs.weight_intercept)
        else:
            weight = 1
        data_type = "filtered"
    else:
        # 在point_matching过程中，因为一定要给reading point添加一个匹配的text point，所以可能会出现距离极长的匹配。
        # 这种匹配应该在weight进行额外的处理，下面用distance和distance_threshold的比值作为ratio，去修改weight。
        # 经过调整，filtered_indices_of_all_text中不会再有blank_supplement。
        distance = filtered_distance_of_all_text[gaze_index][0]
        # ratio = min(1, distance_threshold / distance)
        ratio = min(1, np.float_power(distance_threshold / 2 / distance, 2))
        point_pair = [filtered_reading_coordinates[gaze_index], text_coordinate[filtered_indices_of_all_text[gaze_index][0]]]
        if text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["word"] == "blank_supplement":
            # 修改后的代码中，这里应该不存在blank_supplement。
            weight = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["penalty"] * ratio
        else:
            prediction = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["prediction"]
            density = filtered_reading_density[gaze_index]
            weight = configs.weight_divisor / (abs(density - prediction) + configs.weight_intercept) * ratio
        if not bool_weight:
            weight = 1 * ratio # 这里我加了个ratio，避免没有bool_weight的时候，远距离的点对出问题。
        data_type = "full"

    return text_index, row_index, gaze_index, point_pair, weight, data_type


def step_1_matching_among_all_gpu(reading_data, filtered_text_data_list, text_data, static_text_and_reading, distance_threshold):
    (reading_density_list, filtered_text_coordinate_list, filtered_text_prediction_list, filtered_text_penalty_list,
     full_text_coordinate_list, full_text_prediction_list, full_text_penalty_list, text_index_list, row_index_list, gaze_index_list) = static_text_and_reading

    reading_density_tensor = torch.tensor(reading_density_list, dtype=torch.float32, device=configs.gpu_device_id)
    filtered_text_coordinate_tensor = pad_sequence(filtered_text_coordinate_list, batch_first=True, padding_value=configs.padding_value).to(dtype=torch.float32, device=configs.gpu_device_id)
    filtered_text_prediction_tensor = pad_sequence(filtered_text_prediction_list, batch_first=True, padding_value=configs.padding_value).to(dtype=torch.float32, device=configs.gpu_device_id)
    filtered_text_penalty_tensor = pad_sequence(filtered_text_penalty_list, batch_first=True, padding_value=configs.padding_value).to(dtype=torch.float32, device=configs.gpu_device_id)
    filtered_text_index_tensor = torch.tensor(text_index_list, dtype=torch.int64, device=configs.gpu_device_id)
    filtered_row_index_tensor = torch.tensor(row_index_list, dtype=torch.int64, device=configs.gpu_device_id)
    filtered_gaze_index_tensor = torch.tensor(gaze_index_list, dtype=torch.int64, device=configs.gpu_device_id)

    full_text_coordinate_tensor = pad_sequence(full_text_coordinate_list, batch_first=True, padding_value=configs.padding_value).to(dtype=torch.float32, device=configs.gpu_device_id)
    full_text_prediction_tensor = pad_sequence(full_text_prediction_list, batch_first=True, padding_value=configs.padding_value).to(dtype=torch.float32, device=configs.gpu_device_id)
    full_text_penalty_tensor = pad_sequence(full_text_penalty_list, batch_first=True, padding_value=configs.padding_value).to(dtype=torch.float32, device=configs.gpu_device_id)
    full_text_index_tensor = torch.tensor(text_index_list, dtype=torch.int64, device=configs.gpu_device_id)
    full_row_index_tensor = torch.tensor(row_index_list, dtype=torch.int64, device=configs.gpu_device_id)
    full_gaze_index_tensor = torch.tensor(gaze_index_list, dtype=torch.int64, device=configs.gpu_device_id)

    filtered_distance_threshold_tensor = torch.full((filtered_text_coordinate_tensor.size(0), filtered_text_coordinate_tensor.size(1)), distance_threshold, dtype=torch.float32, device=configs.gpu_device_id)
    full_distance_threshold_tensor = torch.full((full_text_coordinate_tensor.size(0), full_text_coordinate_tensor.size(1)), distance_threshold, dtype=torch.float32, device=configs.gpu_device_id)

    filtered_padding_mask = filtered_text_coordinate_tensor[:, :, 0] != configs.padding_value
    full_padding_mask = full_text_coordinate_tensor[:, :, 0] != configs.padding_value

    filtered_dim_1_length = filtered_text_coordinate_tensor.size(1)
    full_dim_1_length = full_text_coordinate_tensor.size(1)

    # 这里的大部分内容都可以通过外面的_prepare_static_text_and_reading计算，但要注意，这里每次还需要重新计算一次filtered_reading_coordinates。
    reading_coordinate_list = []
    for text_index in range(len(reading_data)):
        if text_index in configs.training_index_list:
            continue
        reading_df = reading_data[text_index]
        for row_index in range(configs.row_num):
            filtered_reading_df = reading_df[reading_df["row_label"] == row_index]
            filtered_reading_coordinates = filtered_reading_df[["gaze_x", "gaze_y"]].values.tolist()
            filtered_text_data = filtered_text_data_list[text_index][row_index]

            if len(filtered_reading_coordinates) == 0 or len(filtered_text_data) == 0:
                continue
            else:
                reading_coordinate_list.extend(filtered_reading_coordinates)

    reading_coordinates_tensor = torch.tensor(reading_coordinate_list, dtype=torch.float32, device=configs.gpu_device_id)
    reading_coordinates_tensor_filtered = reading_coordinates_tensor.unsqueeze(1).expand(-1, filtered_dim_1_length, -1)
    reading_coordinates_tensor_full = reading_coordinates_tensor.unsqueeze(1).expand(-1, full_dim_1_length, -1)

    filtered_distance_tensor = torch.norm(reading_coordinates_tensor_filtered - filtered_text_coordinate_tensor, dim=2)
    filtered_distance_mask = filtered_distance_tensor < filtered_distance_threshold_tensor
    filtered_minimal_distance_mask = filtered_distance_tensor == filtered_distance_tensor.min(dim=1)[0].unsqueeze(1)

    if configs.bool_weight:
        filtered_weight_from_formula = configs.weight_divisor / (torch.abs(reading_density_tensor.unsqueeze(1) - filtered_text_prediction_tensor * configs.weight_coeff) + configs.weight_intercept)
        # 创建一个tensor，只有filtered_text_penalty_tensor的值大于0时，对应位置的值为1，否则为0。
        if configs.bool_text_weight:
            filtered_weight = filtered_weight_from_formula * (filtered_text_penalty_tensor > 0) + filtered_text_penalty_tensor * (filtered_text_penalty_tensor <= 0)
        else:
            filtered_weight = filtered_text_penalty_tensor.clone()
    else:
        filtered_weight = torch.ones_like(filtered_distance_tensor)

    filtered_mask = filtered_padding_mask & filtered_distance_mask & filtered_minimal_distance_mask
    masked_filtered_text_coordinate_tensor = filtered_text_coordinate_tensor[filtered_mask]
    masked_text_index_tensor = filtered_text_index_tensor.unsqueeze(1).expand(-1, filtered_dim_1_length)[filtered_mask]
    masked_row_index_tensor = filtered_row_index_tensor.unsqueeze(1).expand(-1, filtered_dim_1_length)[filtered_mask]
    masked_gaze_index_tensor = filtered_gaze_index_tensor.unsqueeze(1).expand(-1, filtered_dim_1_length)[filtered_mask]
    masked_filtered_weight_tensor = filtered_weight[filtered_mask]
    masked_filtered_reading_coordinate_tensor = reading_coordinates_tensor.unsqueeze(1).expand(-1, filtered_dim_1_length, -1)[filtered_mask]

    # 找出那些filter_distance_mask中，整行都为False的行，作为一个新的mask。
    filtered_distance_all_false_mask = torch.all(~filtered_distance_mask, dim=1).unsqueeze(1).expand(-1, full_dim_1_length)
    full_distance_tensor = torch.norm(reading_coordinates_tensor_full - full_text_coordinate_tensor, dim=2)
    full_distance_ratio_tensor = distance_threshold / 2 / full_distance_tensor
    full_distance_ratio_tensor = torch.pow(full_distance_ratio_tensor, 2)
    full_distance_ratio_tensor = torch.min(full_distance_ratio_tensor, torch.ones_like(full_distance_ratio_tensor))

    if configs.bool_weight:
        full_weight_from_formula = configs.weight_divisor / (torch.abs(reading_density_tensor.unsqueeze(1) - full_text_prediction_tensor) + configs.weight_intercept)
        full_weight = full_weight_from_formula * full_distance_ratio_tensor
    else:
        # 这里我稍微做了一点修改，之前如果是没有bool_weight的情况下，结果都是1。这里我改成了1乘以距离构成的full_distance_ratio_tensor。
        full_weight = full_text_penalty_tensor * full_distance_ratio_tensor

    # 以full_distance_tensor中每行的最小值为mask。
    full_minimal_distance_mask = full_distance_tensor == full_distance_tensor.min(dim=1)[0].unsqueeze(1)
    full_mask = full_padding_mask & filtered_distance_all_false_mask & full_minimal_distance_mask

    masked_full_text_coordinate_tensor = full_text_coordinate_tensor[full_mask]
    masked_full_text_index_tensor = full_text_index_tensor.unsqueeze(1).expand(-1, full_dim_1_length)[full_mask]
    masked_full_row_index_tensor = full_row_index_tensor.unsqueeze(1).expand(-1, full_dim_1_length)[full_mask]
    masked_full_gaze_index_tensor = full_gaze_index_tensor.unsqueeze(1).expand(-1, full_dim_1_length)[full_mask]
    masked_full_weight_tensor = full_weight[full_mask]
    masked_full_reading_coordinate_tensor = reading_coordinates_tensor.unsqueeze(1).expand(-1, full_dim_1_length, -1)[full_mask]

    # 拼接filtered和full
    masked_reading_coordinates_tensor = torch.cat([masked_filtered_reading_coordinate_tensor, masked_full_reading_coordinate_tensor], dim=0)
    masked_text_coordinate_tensor = torch.cat([masked_filtered_text_coordinate_tensor, masked_full_text_coordinate_tensor], dim=0)
    masked_text_index_tensor = torch.cat([masked_text_index_tensor, masked_full_text_index_tensor], dim=0)
    masked_row_index_tensor = torch.cat([masked_row_index_tensor, masked_full_row_index_tensor], dim=0)
    masked_gaze_index_tensor = torch.cat([masked_gaze_index_tensor, masked_full_gaze_index_tensor], dim=0)
    masked_weight_tensor = torch.cat([masked_filtered_weight_tensor, masked_full_weight_tensor], dim=0)

    # 将masked_reading_coordinates_tensor(n, 2)和masked_text_coordinate_tensor(n, 2)拼接成一个(n, 2, 2)的tensor。
    masked_reading_coordinates_tensor = masked_reading_coordinates_tensor.unsqueeze(1)
    masked_text_coordinate_tensor = masked_text_coordinate_tensor.unsqueeze(1)
    masked_point_pair_tensor = torch.cat([masked_reading_coordinates_tensor, masked_text_coordinate_tensor], dim=1)

    point_pair_list = masked_point_pair_tensor.cpu().numpy().tolist()
    masked_text_index_list = masked_text_index_tensor.cpu().numpy().tolist()
    masked_row_index_list = masked_row_index_tensor.cpu().numpy().tolist()
    masked_gaze_index_list = masked_gaze_index_tensor.cpu().numpy().tolist()
    masked_weight_list = masked_weight_tensor.cpu().numpy().tolist()
    data_type_list = ["filtered"] * masked_filtered_text_coordinate_tensor.size(0) + ["full"] * masked_full_text_coordinate_tensor.size(0)

    # 将point_pair_list, masked_weight_list, masked_text_index_list, masked_row_index_list, masked_gaze_index_list, data_type_list这6个长度为n的list，转为1个n*6的list。
    result = list(zip(masked_text_index_list, masked_row_index_list, masked_gaze_index_list, point_pair_list, masked_weight_list, data_type_list))
    return result


def step_2_add_no_matching_text_point(gaze_point_list_1d, gaze_point_info_list_1d, reading_data, effective_text_point_dict, actual_text_point_dict, point_pair_list):
    total_reading_nbrs = NearestNeighbors(n_neighbors=int(len(gaze_point_list_1d) / 4), algorithm='kd_tree').fit(gaze_point_list_1d)
    # 生成每个文本每个reading data的nearest neighbor。
    reading_nbrs_list = []
    for text_index in range(len(reading_data)):
        if text_index in configs.training_index_list:
            reading_nbrs_list.append([])
            continue
        reading_df = reading_data[text_index]
        reading_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
        reading_nbrs = NearestNeighbors(n_neighbors=int(len(reading_coordinates) / 4), algorithm='kd_tree').fit(reading_coordinates)
        reading_nbrs_list.append(reading_nbrs)

    supplement_point_pair_list = []
    supplement_weight_list = []
    supplement_info_list = []
    # 然后找出那些没有任何匹配的actual text point，将其与最近的阅读点匹配。
    total_effective_text_point_num = sum(effective_text_point_dict.values())
    point_pair_length = len(point_pair_list)
    # iterate over actual_text_point_dict
    # TODO 这里注意下，最好也设个distance threshold，不然会导致结果很不稳定。我觉得150-200可能是个比较合适的值。
    for key, value in actual_text_point_dict.items():
        if value == 0:
            closet_point_num = int(point_pair_length * effective_text_point_dict[key] / total_effective_text_point_num)
            cur_text_point = [float(key[0]), float(key[1])]
            distances, indices = total_reading_nbrs.kneighbors([cur_text_point])
            # 对于右下角的未被匹配的文本点，我们将其权重放大10倍。（没实施）
            if (key[0] == configs.right_down_text_center[0] and (key[1] == configs.right_down_text_center[1] or key[1] == configs.right_down_text_center[1] - configs.text_width)) or \
                    (key[0] == configs.right_down_text_center[0] - configs.text_height and key[1] == configs.right_down_text_center[1]):
                weight = configs.completion_weight * configs.right_down_corner_unmatched_ratio
            else:
                weight = configs.completion_weight

            for point_index in range(closet_point_num):
                current_point_index = indices[0][point_index]
                gaze_point = gaze_point_list_1d[current_point_index].tolist()
                # point_pair_list.append([gaze_point, cur_text_point])
                # weight_list.append(weight)
                # row_label_list.append(-1)
                supplement_point_pair_list.append([gaze_point, cur_text_point])
                supplement_weight_list.append(weight)
                gaze_info = gaze_point_info_list_1d[current_point_index]
                gaze_info = np.append(gaze_info, np.array([int(key[0]), int(key[1])]))
                gaze_info = tuple(gaze_info)
                supplement_info_list.append(gaze_info)

    return total_reading_nbrs, reading_nbrs_list, supplement_point_pair_list, supplement_weight_list, supplement_info_list


def _add_closest_reading_point_to_boundary(reading_nbrs_list, reading_data,
                                           text_index, row_df, row_index, col_index, x, y,
                                           distance_threshold, distance_ratio, boundary_ratio,
                                           data_type_input):
    point_pair_list = []
    weight_list = []
    info_list = []
    data_type_list = []

    half_row_num = int((configs.row_num - 1) / 2)

    if data_type_input == "right" and row_index > half_row_num:
        distance_ratio += configs.right_boundary_distance_threshold_ratio_derivative * (row_index - half_row_num)

    distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
    for point_index in range(len(indices[0])):
        if distances[0][point_index] < distance_threshold * distance_ratio:
            gaze_index = indices[0][point_index]
            gaze_point = reading_data[text_index].iloc[gaze_index][["gaze_x", "gaze_y"]].values.tolist()
            # ratio = min(1, distance_threshold * distance_ratio / distances[0][point_index])

            # 对于不同方向的文字，我们会进行筛选。如对于bottom的文字，指考虑那些gaze_y < y的注视点，避免将注视点越推越远。
            if data_type_input == "top" and gaze_point[1] <= y:
                continue
            elif data_type_input == "bottom" and gaze_point[1] >= y:
                continue
            elif data_type_input == "left" and gaze_point[0] <= x:
                continue
            elif data_type_input == "right" and gaze_point[0] >= x:
                continue

            point_pair = [gaze_point, [x, y]]
            if data_type_input == "left":
                if row_index <= 0:
                    weight = row_df.iloc[col_index]["penalty"]  # 为了让左上角的数据不出问题，我将左上角blank_supplement的penalty都设置为了1（而非负数）。
                else:
                    weight = configs.empty_penalty * configs.left_boundary_ratio
            elif data_type_input == "right":
                # 对右下角的点，增加他们的权重。
                if row_index > half_row_num:
                    weight = configs.empty_penalty * (boundary_ratio + row_index - half_row_num * configs.right_boundary_ratio_derivative)
                else:
                    weight = configs.empty_penalty * boundary_ratio
            elif data_type_input == "top":
                weight = row_df.iloc[col_index]["penalty"]
            else:
                weight = configs.empty_penalty * boundary_ratio

            # weight = weight * ratio
            info = (text_index, row_index, gaze_index, int(x), int(y))
            data_type = data_type_input

            point_pair_list.append(point_pair)
            weight_list.append(weight)
            info_list.append(info)
            data_type_list.append(data_type)
        else:
            break

    return point_pair_list, weight_list, info_list, data_type_list


def step_3_add_boundary_points(text_df, text_index, row_list, row_index,
                               reading_data, reading_nbrs_list, distance_threshold):
    data_type_list = []
    point_pair_list = []
    weight_list = []
    info_list = []
    total_point_num = 0
    # 1. 对于最下面的点，也添加了左右控制；# 这里的最底层可能不一定是5.5，对于那些提前结束的点，bottom可能是4.5或更小的值。
    # 2. 对于-0.5，添加了上方控制。
    if int(row_list[row_index]) != row_list[row_index] and row_list[row_index] > 0:
        pass
        row_df = text_df[text_df["row"] == row_list[row_index]]
        for index in range(row_df.shape[0]):
            x = row_df.iloc[index]["x"]
            y = row_df.iloc[index]["y"]
            bottom_point_pair_list, bottom_weight_list, bottom_info_list, bottom_data_type_list = _add_closest_reading_point_to_boundary(reading_nbrs_list, reading_data,
                                                                                                                                         text_index, row_df, row_index, index, x, y,
                                                                                                                                         distance_threshold,
                                                                                                                                         configs.bottom_boundary_distance_threshold_ratio,
                                                                                                                                         configs.bottom_boundary_ratio,
                                                                                                                                         "bottom")
            point_pair_list.extend(bottom_point_pair_list)
            weight_list.extend(bottom_weight_list)
            info_list.extend(bottom_info_list)
            data_type_list.extend(bottom_data_type_list)
            total_point_num += 1
    elif int(row_list[row_index]) != row_list[row_index] and row_list[row_index] < 0:
        pass
        row_df = text_df[text_df["row"] == row_list[row_index]]
        for index in range(row_df.shape[0]):
            x = row_df.iloc[index]["x"]
            y = row_df.iloc[index]["y"]
            top_point_pair_list, top_weight_list, top_info_list, top_data_type_list = _add_closest_reading_point_to_boundary(reading_nbrs_list, reading_data,
                                                                                                                             text_index, row_df, row_index, index, x, y,
                                                                                                                             distance_threshold,
                                                                                                                             configs.top_boundary_distance_threshold_ratio,
                                                                                                                             configs.top_boundary_ratio,
                                                                                                                             "top")
            point_pair_list.extend(top_point_pair_list)
            weight_list.extend(top_weight_list)
            info_list.extend(top_info_list)
            data_type_list.extend(top_data_type_list)
            total_point_num += 1

    # 如果当前row并不包含任何文字（即完全是上下册的补充空白行），则直接跳过左右匹配。
    row_df = text_df[text_df["row"] == row_list[row_index]]
    if row_df[row_df["word"] != "blank_supplement"].shape[0] == 0:
        return data_type_list, point_pair_list, weight_list, info_list, total_point_num

    # 对于其它行，对左右两侧的blank_supplement添加匹配。
    row_df = row_df.sort_values(by=["col"])
    for index in range(row_df.shape[0]):
        # if index < row_df.shape[0] - 1:
        if index < row_df.shape[0]:
            word = row_df.iloc[index]["word"]
            # next_word = row_df.iloc[index + 1]["word"]
            # if (word == "blank_supplement" or word.strip() == "") and (next_word != "blank_supplement" and next_word.strip() != ""):
            if (word == "blank_supplement" or word.strip() == "") and row_df.iloc[index]["col"] < configs.col_num / 2:
                x = row_df.iloc[index]["x"]
                y = row_df.iloc[index]["y"]
                left_point_pair_list, left_weight_list, left_info_list, left_data_type_list = _add_closest_reading_point_to_boundary(reading_nbrs_list, reading_data,
                                                                                                                                     text_index, row_df, row_index, index, x, y,
                                                                                                                                     distance_threshold,
                                                                                                                                     configs.left_boundary_distance_threshold_ratio,
                                                                                                                                     configs.left_boundary_ratio,
                                                                                                                                     "left")
                point_pair_list.extend(left_point_pair_list)
                weight_list.extend(left_weight_list)
                info_list.extend(left_info_list)
                data_type_list.extend(left_data_type_list)
                total_point_num += 1

    row_df = row_df.sort_values(by=["col"], ascending=False)  # 注意，这里有一个ascending=False。
    for index in range(row_df.shape[0]):
        # if index < row_df.shape[0]:
        if index < row_df.shape[0] - 1:
            word = row_df.iloc[index]["word"]
            # next_word = row_df.iloc[index + 1]["word"]
            # if (word == "blank_supplement" or word.strip() == "") and (next_word != "blank_supplement" and next_word.strip() != ""):
            if (word == "blank_supplement" or word.strip() == "") and row_df.iloc[index]["col"] >= configs.col_num / 2:
                x = row_df.iloc[index]["x"]
                y = row_df.iloc[index]["y"]
                right_point_pair_list, right_weight_list, right_info_list, right_data_type_list = _add_closest_reading_point_to_boundary(reading_nbrs_list, reading_data,
                                                                                                                                         text_index, row_df, row_index, index, x, y,
                                                                                                                                         distance_threshold,
                                                                                                                                         configs.right_boundary_distance_threshold_ratio,
                                                                                                                                         configs.right_boundary_ratio,
                                                                                                                                         "right")
                point_pair_list.extend(right_point_pair_list)
                weight_list.extend(right_weight_list)
                info_list.extend(right_info_list)
                data_type_list.extend(right_data_type_list)
                total_point_num += 1

    return data_type_list, point_pair_list, weight_list, info_list, total_point_num


def step_3_add_boundary_points_new_single_step(reading_data, reading_nbr,
                                               text_index, row_index, x, y,
                                               weight, data_type_input,
                                               distance_threshold):
    point_pair_list = []
    weight_list = []
    info_list = []
    data_type_list = []

    distances, indices = reading_nbr.kneighbors([[x, y]])
    # for point_index in range(len(indices[0])):
    #     if distances[0][point_index] < distance_threshold:
    #         gaze_index = indices[0][point_index]
    #         gaze_point = reading_data[text_index].iloc[gaze_index][["gaze_x", "gaze_y"]].values.tolist()
    #         # ratio = min(1, distance_threshold * distance_ratio / distances[0][point_index])
    #
    #         # 对于不同方向的文字，我们会进行筛选。如对于bottom的文字，指考虑那些gaze_y < y的注视点，避免将注视点越推越远。
    #         if data_type_input == "top" and gaze_point[1] <= y:
    #             continue
    #         elif data_type_input == "bottom" and gaze_point[1] >= y:
    #             continue
    #         elif data_type_input == "left" and gaze_point[0] <= x:
    #             continue
    #         elif data_type_input == "right" and gaze_point[0] >= x:
    #             continue
    #
    #         point_pair = [gaze_point, [x, y]]
    #
    #         # weight = weight * ratio
    #         info = (text_index, row_index, gaze_index, int(x), int(y))
    #         data_type = data_type_input
    #
    #         point_pair_list.append(point_pair)
    #         weight_list.append(weight)
    #         info_list.append(info)
    #         data_type_list.append(data_type)
    #     else:
    #         break
    # return data_type_list, point_pair_list, weight_list, info_list

    if len(distances) == 0:
        return data_type_list, point_pair_list, weight_list, info_list

    distances_filter_index = np.where(distances[0] < distance_threshold)[0]
    if len(distances_filter_index) == 0:
        return data_type_list, point_pair_list, weight_list, info_list

    distances = distances[0][distances_filter_index]
    indices = indices[0][distances_filter_index]
    # 将indices中的元素排序之后的index输出，然后用这个index重新构造distance和indices。
    sort_indices = np.argsort(indices)
    distances = distances[sort_indices]
    indices = indices[sort_indices]

    # 使用indices中的index，选择gaze point
    gaze_point_list = np.array(reading_data[text_index][["gaze_x", "gaze_y"]].iloc[indices].values.tolist())
    if data_type_input == 0:
        location_filter_indices = np.where(gaze_point_list[:, 1] > y)[0]
    elif data_type_input == 1:
        location_filter_indices = np.where(gaze_point_list[:, 1] < y)[0]
    elif data_type_input == 2:
        location_filter_indices = np.where(gaze_point_list[:, 0] > x)[0]
    elif data_type_input == 3:
        location_filter_indices = np.where(gaze_point_list[:, 0] < x)[0]
    else:
        raise ValueError("data_type_input should be one of ['top', 'bottom', 'left', 'right']")
    gaze_point_list = gaze_point_list[location_filter_indices]
    indices = indices[location_filter_indices]
    point_pair_list = [[gaze_point_list[i].tolist(), [x, y]] for i in range(len(gaze_point_list))]
    weight_list = [weight for _ in range(len(gaze_point_list))]
    info_list = [(text_index, row_index, indices[i], int(x), int(y)) for i in range(len(gaze_point_list))]
    data_type_list = [data_type_input for _ in range(len(gaze_point_list))]
    return data_type_list, point_pair_list, weight_list, info_list


def step_3_add_boundary_points_new(boundary_coordinate_list, boundary_weight_list, boundary_type_list, boundary_text_index_list, boundary_row_index_list, distance_threshold_list,
                                   reading_data, reading_nbrs_list):
    args_list = []
    for boundary_index in range(len(boundary_coordinate_list)):
        text_index = boundary_text_index_list[boundary_index]
        row_index = boundary_row_index_list[boundary_index]
        weight_list_1 = boundary_weight_list[boundary_index]
        data_type_input = boundary_type_list[boundary_index]
        x, y = boundary_coordinate_list[boundary_index]
        reading_nbr = reading_nbrs_list[text_index]
        distance_threshold = distance_threshold_list[boundary_index]
        args_list.append((reading_data, reading_nbr, text_index, row_index, x, y, weight_list_1, data_type_input, distance_threshold))

    # result_list = []
    # for boundary_index in range(len(args_list)):
    #     data_type_list_1, point_pair_list_1, weight_list_1, info_list_1 = step_3_add_boundary_points_new_single_step(*args_list[boundary_index])
    #     result_list.append((data_type_list_1, point_pair_list_1, weight_list_1, info_list_1))
    #
    # return result_list

    with multiprocessing.Pool(processes=8) as pool:
        result = pool.starmap(step_3_add_boundary_points_new_single_step, args_list)
    return result


def step_3_add_boundary_points_gpu(boundary_coordinate_list, boundary_weight_list, boundary_type_list, boundary_text_index_list, boundary_row_index_list, boundary_distance_threshold_list,
                                   reading_data, reading_nbrs_list):
    """
    string_to_text = {"left": 0, "right": 1, "top": 2, "bottom": 3}
    这里跑出的结果会比之前多，因为之前的reading_nbrs在设置最近邻的时候，最近邻的数量是int(len(reading_coordinates) / 4)，而这个数量可能少于距离小于threshold的点的数量。
    """
    unique_gaze_dict = {}
    unique_text_index_list = list(set(boundary_text_index_list))
    unique_text_index_counter = Counter(boundary_text_index_list)
    unique_text_index_list.sort()
    for text_index in unique_text_index_list:
        gaze_list = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
        gaze_list = torch.tensor(gaze_list, dtype=torch.float32, device=configs.gpu_device_id)
        unique_gaze_dict[text_index] = gaze_list

    # count the sum of unique_text_index_counter
    sum_of_unique_text_index_counter = sum(unique_text_index_counter.values())
    gaze_for_boundary = []
    for index in range(len(unique_text_index_list)):
        text_index = unique_text_index_list[index]
        counter = unique_text_index_counter[text_index]
        # add unique_gaze_list[text_index] to gaze_for_boundary for counter times
        gaze_for_boundary.extend([unique_gaze_dict[text_index]] * counter)

    gaze_for_tensor = pad_sequence(gaze_for_boundary, batch_first=True, padding_value=configs.padding_value)
    # gaze_for_tensor是一个3434, 385, 2的tensor，我希望生成一个3434, 385的tensor，表示哪些位置是padding的，哪些不是
    mask_padding = gaze_for_tensor[:, :, 0] != configs.padding_value
    gaze_for_tensor = gaze_for_tensor.to(dtype=torch.float32, device=configs.gpu_device_id)
    dim_1_length = gaze_for_tensor.size(1)

    boundary_coordinate_tensor = torch.tensor(boundary_coordinate_list, dtype=torch.float32, device=configs.gpu_device_id).unsqueeze(1).expand(-1, dim_1_length, -1)

    distance = torch.norm(gaze_for_tensor - boundary_coordinate_tensor, dim=2)
    distance_threshold_tensor = torch.tensor(boundary_distance_threshold_list, dtype=torch.float32, device=configs.gpu_device_id).unsqueeze(1).expand(-1, dim_1_length)
    mask_distance = distance < distance_threshold_tensor

    boundary_weight_tensor = torch.tensor(boundary_weight_list, dtype=torch.float32, device=configs.gpu_device_id).unsqueeze(1).expand(-1, dim_1_length)
    boundary_text_index_tensor = torch.tensor(boundary_text_index_list, dtype=torch.int, device=configs.gpu_device_id).unsqueeze(1).expand(-1, dim_1_length)
    boundary_row_index_tensor = torch.tensor(boundary_row_index_list, dtype=torch.float32, device=configs.gpu_device_id).unsqueeze(1).expand(-1, dim_1_length)
    gaze_index_tensor = torch.arange(dim_1_length, dtype=torch.int, device=configs.gpu_device_id).unsqueeze(0).expand(len(boundary_coordinate_list), -1)

    boundary_type_tensor = torch.tensor(boundary_type_list, dtype=torch.int, device=configs.gpu_device_id).unsqueeze(1)
    boundary_type_tensor = boundary_type_tensor.expand(-1, dim_1_length)

    condition_left = (boundary_type_tensor == 0) & (gaze_for_tensor[:, :, 0] > boundary_coordinate_tensor[:, :, 0])
    condition_right = (boundary_type_tensor == 1) & (gaze_for_tensor[:, :, 0] < boundary_coordinate_tensor[:, :, 0])
    condition_top = (boundary_type_tensor == 2) & (gaze_for_tensor[:, :, 1] > boundary_coordinate_tensor[:, :, 1])
    condition_bottom = (boundary_type_tensor == 3) & (gaze_for_tensor[:, :, 1] < boundary_coordinate_tensor[:, :, 1])

    # 根据条件生成最终mask
    mask_type = condition_left | condition_right | condition_top | condition_bottom
    mask = mask_padding & mask_distance & mask_type

    filtered_gaze_tensor = gaze_for_tensor[mask]
    filtered_boundary_coordinate_tensor = boundary_coordinate_tensor[mask]
    filtered_weight_tensor = boundary_weight_tensor[mask]
    filtered_text_index_tensor = boundary_text_index_tensor[mask]
    filtered_row_index_tensor = boundary_row_index_tensor[mask]
    filtered_gaze_index_tensor = gaze_index_tensor[mask]
    # filtered_distance_tensor = distance[mask]
    filtered_type_tensor = boundary_type_tensor[mask]
    # filtered_distance_threshold_tensor = distance_threshold_tensor[mask]

    filtered_gaze_list = filtered_gaze_tensor.cpu().numpy().tolist()
    filtered_boundary_coordinate_list = filtered_boundary_coordinate_tensor.cpu().numpy().tolist()
    filtered_weight_list = filtered_weight_tensor.cpu().numpy().tolist()
    filtered_text_index_list = filtered_text_index_tensor.cpu().numpy().tolist()
    filtered_row_index_list = filtered_row_index_tensor.cpu().numpy().tolist()
    filtered_gaze_index_list = filtered_gaze_index_tensor.cpu().numpy().tolist()
    # filtered_distance_list = filtered_distance_tensor.cpu().numpy().tolist()
    filtered_type_list = filtered_type_tensor.cpu().numpy().tolist()
    # filtered_distance_threshold_list = filtered_distance_threshold_tensor.cpu().numpy().tolist()

    info_list = [(filtered_text_index_list[i], filtered_row_index_list[i], filtered_gaze_index_list[i], int(filtered_boundary_coordinate_list[i][0]), int(filtered_boundary_coordinate_list[i][1])) for i in range(len(filtered_gaze_list))]
    data_type_list = [filtered_type_list[i] for i in range(len(filtered_type_list))]
    point_pair_list = [[filtered_gaze_list[i], filtered_boundary_coordinate_list[i]] for i in range(len(filtered_gaze_list))]
    weight_list = [filtered_weight_list[i] for i in range(len(filtered_weight_list))]
    # distance_threshold_list = [filtered_distance_threshold_list[i] for i in range(len(filtered_distance_threshold_list))]

    return data_type_list, point_pair_list, weight_list, info_list


def random_select_for_supplement_point_pairs(point_pair_list, supplement_point_pair_list, supplement_weight_list, supplement_info_list, select_ratio):
    if len(configs.training_index_list) > 0:
        supplement_select_indices = np.random.choice(len(supplement_point_pair_list), min(len(supplement_point_pair_list), max(int(len(point_pair_list) * select_ratio), 1)), replace=False)
        supplement_point_pair_list = [supplement_point_pair_list[i] for i in supplement_select_indices]
        supplement_weight_list = [supplement_weight_list[i] for i in supplement_select_indices]
        supplement_info_list = [supplement_info_list[i] for i in supplement_select_indices]

    return supplement_point_pair_list, supplement_weight_list, supplement_info_list


def random_select_for_gradient_descent(iteration_index, point_pair_list, weight_list, info_list, ratio, n_cluster=8):
    """
    在Gradient Descent过程中，随机选择应该是区域均匀的选择，而不是随机选择。后者会导致点密度高的右上角在被选择完成后也密度更高，从而导致优化时右上角的权重更大。
    因此，我们先对点进行聚类，然后根据聚类的体积，来确定每个聚类内部需要选择的点数量，最后再在聚类内部进行选择。
    """
    if len(point_pair_list) == 0:
        return point_pair_list, weight_list, info_list
    elif ratio == 1:
        return point_pair_list, weight_list, info_list
    elif len(point_pair_list) < 400:  # 尽量保证有400个点可以用于训练。
        return point_pair_list, weight_list, info_list
    elif len(point_pair_list) * ratio < 400:
        ratio = 400 / len(point_pair_list)
    np.random.seed(iteration_index)

    # 把之前简单的随机选择去掉。
    # random_select_indices = np.random.choice(len(point_pair_list), int(configs.random_select_ratio_for_point_pair * len(point_pair_list)), replace=False)
    # random_selected_point_pair_list = [point_pair_list[i] for i in random_select_indices]
    # random_selected_weight_list = [weight_list[i] for i in random_select_indices]
    # random_selected_info_list = [info_list[i] for i in random_select_indices]

    gaze_point = [point_pair_list[i][0] for i in range(len(point_pair_list))]
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init=1, max_iter=10).fit(gaze_point)
    # kmeans = cuKMeans(n_clusters=n_cluster, random_state=0).fit(gaze_point)
    cluster_labels = kmeans.labels_

    cluster_gaze_point_list = [[] for _ in range(n_cluster)]
    cluster_point_pair_list = [[] for _ in range(n_cluster)]
    cluster_weight_list = [[] for _ in range(n_cluster)]
    cluster_info_list = [[] for _ in range(n_cluster)]

    for i in range(len(gaze_point)):
        cluster_gaze_point_list[cluster_labels[i]].append(gaze_point[i])
        cluster_point_pair_list[cluster_labels[i]].append(point_pair_list[i])
        cluster_weight_list[cluster_labels[i]].append(weight_list[i])
        cluster_info_list[cluster_labels[i]].append(info_list[i])

    # 就散每个cluster的gaze point构成的convex hull的面积
    cluster_volume_list = []
    for i in range(len(cluster_gaze_point_list)):
        points = np.array(cluster_gaze_point_list[i])
        try:
            hull = ConvexHull(points)
            volume = hull.volume
        except:
            volume = 1
        cluster_volume_list.append(volume)

    select_num_list = []
    target_num = int(len(gaze_point) * ratio)
    for i in range(len(cluster_gaze_point_list)):
        select_num_list.append(min(int(cluster_volume_list[i] / sum(cluster_volume_list) * target_num), len(cluster_gaze_point_list[i])))

    random_selected_point_pair_list = []
    random_selected_weight_list = []
    random_selected_info_list = []
    for i in range(len(cluster_gaze_point_list)):
        indices = np.random.choice(len(cluster_gaze_point_list[i]), select_num_list[i], replace=False)
        for index in indices:
            random_selected_point_pair_list.append(cluster_point_pair_list[i][index])
            random_selected_weight_list.append(cluster_weight_list[i][index])
            random_selected_info_list.append(cluster_info_list[i][index])
    # 将weight_list都转为float，避免当weight为0时出现的问题。
    random_selected_weight_list = np.array(random_selected_weight_list, dtype=np.float32).tolist()
    return random_selected_point_pair_list, random_selected_weight_list, random_selected_info_list


def step_4_select_supplement(iteration_index,
                             point_pair_list, weight_list, row_label_list,
                             supplement_point_pair_list, supplement_weight_list, supplement_info_list,
                             left_point_pair_list, left_weight_list, left_info_list,
                             right_point_pair_list, right_weight_list, right_info_list,
                             top_point_pair_list, top_weight_list, top_info_list,
                             bottom_point_pair_list, bottom_weight_list, bottom_info_list):
    # 对于那些validation数量过少的情况，需要限制supplement point pair的数量，保证其与raw point pair的比例，避免出现过分的失调。这里的限制我修改过，看一下如果效果不好就还原回去。
    supplement_point_pair_list, supplement_weight_list, supplement_info_list = random_select_for_supplement_point_pairs(point_pair_list, supplement_point_pair_list, supplement_weight_list,
                                                                                                                        supplement_info_list, configs.supplement_select_ratio)
    # 对于那些validation数量过少的情况，需要限制boundary point pair的数量，保证其与raw point pair的比例，避免出现过分的失调。这里的限制我修改过，看一下如果效果不好就还原回去。
    left_point_pair_list, left_weight_list, left_info_list = random_select_for_supplement_point_pairs(point_pair_list, left_point_pair_list, left_weight_list, left_info_list,
                                                                                                      configs.boundary_select_ratio)
    right_point_pair_list, right_weight_list, right_info_list = random_select_for_supplement_point_pairs(point_pair_list, right_point_pair_list, right_weight_list, right_info_list,
                                                                                                         configs.boundary_select_ratio)
    top_point_pair_list, top_weight_list, top_info_list = random_select_for_supplement_point_pairs(point_pair_list, top_point_pair_list, top_weight_list, top_info_list,
                                                                                                   configs.boundary_select_ratio)
    bottom_point_pair_list, bottom_weight_list, bottom_info_list = random_select_for_supplement_point_pairs(point_pair_list, bottom_point_pair_list, bottom_weight_list, bottom_info_list,
                                                                                                            configs.boundary_select_ratio)

    # supplement_point_pair_list, supplement_weight_list, supplement_info_list = random_select_for_gradient_descent(iteration_index, supplement_point_pair_list, supplement_weight_list, supplement_info_list, configs.supplement_select_ratio, 4)
    # left_point_pair_list, left_weight_list, left_info_list = random_select_for_gradient_descent(iteration_index, left_point_pair_list, left_weight_list, left_info_list, configs.boundary_select_ratio, 4)
    # right_point_pair_list, right_weight_list, right_info_list = random_select_for_gradient_descent(iteration_index, right_point_pair_list, right_weight_list, right_info_list, configs.boundary_select_ratio, 4)
    # top_point_pair_list, top_weight_list, top_info_list = random_select_for_gradient_descent(iteration_index, top_point_pair_list, top_weight_list, top_info_list, configs.boundary_select_ratio, 4)
    # bottom_point_pair_list, bottom_weight_list, bottom_info_list = random_select_for_gradient_descent(iteration_index, bottom_point_pair_list, bottom_weight_list, bottom_info_list, configs.boundary_select_ratio, 4)

    point_pair_list.extend(supplement_point_pair_list)
    weight_list.extend(supplement_weight_list)
    row_label_list.extend(supplement_info_list)

    point_pair_list.extend(left_point_pair_list)
    weight_list.extend(left_weight_list)
    row_label_list.extend(left_info_list)

    point_pair_list.extend(right_point_pair_list)
    weight_list.extend(right_weight_list)
    row_label_list.extend(right_info_list)

    point_pair_list.extend(top_point_pair_list)
    weight_list.extend(top_weight_list)
    row_label_list.extend(top_info_list)

    point_pair_list.extend(bottom_point_pair_list)
    weight_list.extend(bottom_weight_list)
    row_label_list.extend(bottom_info_list)

    return point_pair_list, weight_list, row_label_list


def point_matching_multi_process(reading_data, gaze_point_list_1d, gaze_point_info_list_1d,
                                 text_data, filtered_text_data_list,
                                 total_nbrs_list, row_nbrs_list,
                                 effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                                 static_text_and_reading, static_boundary,
                                 iteration_index, distance_threshold):

    (boundary_coordinate_list, boundary_weight_list,
     boundary_type_list, boundary_text_index_list,
     boundary_row_index_list, boundary_col_index_list,
     boundary_distance_threshold_list, total_boundary_point_num_list) = static_boundary

    (filtered_reading_density_list, filtered_text_coordinate_list, filtered_text_prediction_list, filtered_text_penalty_list,
     full_text_coordinate_list, full_text_prediction_list, full_text_penalty_list, text_index_list, row_index_list, gaze_index_list) = static_text_and_reading

    np.random.seed(configs.random_seed)

    with multiprocessing.Pool(configs.number_of_process) as pool:
        # 1. 首先遍历所有的reading point，找到与其row label一致的、距离最近的text point，然后将这些匹配点加入到point_pair_list中。
        # args_list = []
        # for text_index in range(len(reading_data)):
        #     if text_index in configs.training_index_list:
        #         continue
        #     reading_df = reading_data[text_index]
        #     for row_index in range(configs.row_num):
        #         filtered_reading_df = reading_df[reading_df["row_label"] == row_index]
        #
        #         if row_nbrs_list[text_index][row_index] and filtered_reading_df.shape[0] != 0:
        #             filtered_reading_coordinates = filtered_reading_df[["gaze_x", "gaze_y"]].values.tolist()
        #             filtered_distances_of_row, filtered_indices_of_row = row_nbrs_list[text_index][row_index].kneighbors(filtered_reading_coordinates)
        #             filtered_text_data = filtered_text_data_list[text_index][row_index]
        #             filtered_text_coordinate = filtered_text_data[["x", "y"]].values.tolist()
        #             filtered_reading_density = filtered_reading_df["density"].values.tolist()
        #
        #             text_coordinate = text_data[text_index][["x", "y"]].values.tolist()
        #             filtered_distance_of_all_text, filtered_indices_of_all_text = total_nbrs_list[text_index].kneighbors(filtered_reading_coordinates)
        #
        #             for gaze_index in range(len(filtered_distances_of_row)):
        #                 args_list.append((text_index, row_index, gaze_index,
        #                                   text_coordinate, text_data,
        #                                   filtered_reading_coordinates, filtered_text_data, filtered_text_coordinate, filtered_reading_density,
        #                                   filtered_indices_of_row, filtered_distances_of_row,
        #                                   filtered_indices_of_all_text, filtered_distance_of_all_text,
        #                                   distance_threshold, configs.bool_weight))
        #
        # results = pool.starmap(step_1_matching_among_all, args_list)

        results = step_1_matching_among_all_gpu(reading_data, filtered_text_data_list, text_data, static_text_and_reading, distance_threshold * configs.text_distance_threshold_ratio)

        info_list = [(results[i][0], results[i][1], results[i][2], int(results[i][3][1][0]), int(results[i][3][1][1])) for i in range(len(results))]
        point_pair_list = [results[i][3] for i in range(len(results))]
        weight_list = [results[i][4] for i in range(len(results))]

        for gaze_index in range(len(point_pair_list)):
            text_x = point_pair_list[gaze_index][1][0]
            text_y = point_pair_list[gaze_index][1][1]
            if (text_x, text_y) in actual_text_point_dict:
                actual_text_point_dict[(text_x, text_y)] += 1
            if (text_x, text_y) in actual_supplement_text_point_dict:
                actual_supplement_text_point_dict[(text_x, text_y)] += 1

        # 2. 接下来做的是确认有文字，但没有reading data的text_unit，并根据其最近的reading data，添加额外的点对。该添加点对不受文章序号限制。
        # 生成一个所有reading point的nearest neighbor。
        total_reading_nbrs, reading_nbrs_list, supplement_point_pair_list, supplement_weight_list, supplement_info_list = \
            step_2_add_no_matching_text_point(gaze_point_list_1d, gaze_point_info_list_1d, reading_data, effective_text_point_dict, actual_text_point_dict, point_pair_list)

        # 3. 对于横向最外侧的补充点或空格点（即左右侧紧贴近正文的点），都可以考虑额外添加一些匹配点对，添加的weight是负数。
        # 这里单独为要添加boundary的point pair生成list，方便后续筛选。

        # # 最初的筛选方法
        # args_list = []
        # for text_index in range(len(text_data)):
        #     if text_index in configs.training_index_list:
        #         continue
        #     text_df = text_data[text_index]
        #     row_list = text_df["row"].unique().tolist()
        #
        #     for row_index in range(len(row_list)):
        #         args_list.append((text_df, text_index, row_list, row_index,
        #                           reading_data, reading_nbrs_list, distance_threshold))
        #
        # results_raw = []
        # for arg_index in range(len(args_list)):
        #     results_raw.append(step_3_add_boundary_points(*args_list[arg_index]))
        # # 最初方法的多线程。
        # results_raw = pool.starmap(step_3_add_boundary_points, args_list)
        # results_raw = [result_raw for result_raw in results_raw if len(result_raw[0]) > 0]
        # boundary_point_num_list = [result_raw[4] for result_raw in results_raw]
        # total_boundary_point_num = sum(boundary_point_num_list)

        # # 最初方法的改进，但仍然是CPU版本。
        # results_raw = step_3_add_boundary_points_new(boundary_coordinate_list, boundary_weight_list, boundary_type_list, boundary_text_index_list, boundary_row_index_list, boundary_distance_threshold_list,
        #                                              reading_data, reading_nbrs_list)
        #
        # results_raw = [result_raw for result_raw in results_raw if len(result_raw[0]) > 0]
        # results = []
        # for result_index, result in enumerate(results_raw):
        #     for sub_result_index in range(len(result[0])):
        #         results.append((result[0][sub_result_index], result[1][sub_result_index], result[2][sub_result_index], result[3][sub_result_index]))

        # GPU改进版本。
        results_gpu_raw = step_3_add_boundary_points_gpu(boundary_coordinate_list, boundary_weight_list, boundary_type_list, boundary_text_index_list, boundary_row_index_list,
                                                         boundary_distance_threshold_list, reading_data, reading_nbrs_list)
        results = []
        for result_index in range(len(results_gpu_raw[0])):
            results.append((results_gpu_raw[0][result_index], results_gpu_raw[1][result_index], results_gpu_raw[2][result_index], results_gpu_raw[3][result_index]))

        left_point_pair_list = [results[i][1] for i in range(len(results)) if results[i][0] == 0] # 0 for left
        left_weight_list = [results[i][2] for i in range(len(results)) if results[i][0] == 0]
        left_info_list = [results[i][3] for i in range(len(results)) if results[i][0] == 0]
        right_point_pair_list = [results[i][1] for i in range(len(results)) if results[i][0] == 1] # 1 for right
        right_weight_list = [results[i][2] for i in range(len(results)) if results[i][0] == 1]
        right_info_list = [results[i][3] for i in range(len(results)) if results[i][0] == 1]
        top_point_pair_list = [results[i][1] for i in range(len(results)) if results[i][0] == 2] # 2 for top
        top_weight_list = [results[i][2] for i in range(len(results)) if results[i][0] == 2]
        top_info_list = [results[i][3] for i in range(len(results)) if results[i][0] == 2]
        bottom_point_pair_list = [results[i][1] for i in range(len(results)) if results[i][0] == 3] # 4 for bottom
        bottom_weight_list = [results[i][2] for i in range(len(results)) if results[i][0] == 3]
        bottom_info_list = [results[i][3] for i in range(len(results)) if results[i][0] == 3]

        # 4. 限制添加点的数量。
        point_pair_list, weight_list, info_list = step_4_select_supplement(iteration_index,
                                                                           point_pair_list, weight_list, info_list,
                                                                           supplement_point_pair_list, supplement_weight_list, supplement_info_list,
                                                                           left_point_pair_list, left_weight_list, left_info_list,
                                                                           right_point_pair_list, right_weight_list, right_info_list,
                                                                           top_point_pair_list, top_weight_list, top_info_list,
                                                                           bottom_point_pair_list, bottom_weight_list, bottom_info_list)

        return point_pair_list, weight_list, info_list
