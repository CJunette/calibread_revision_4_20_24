import math

import numpy as np
import multiprocessing
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
    if filtered_distances_of_row[gaze_index][0] < distance_threshold * configs.text_distance_threshold_ratio:
        point_pair = [filtered_reading_coordinates[gaze_index], filtered_text_coordinate[filtered_indices_of_row[gaze_index][0]]]
        prediction = filtered_text_data.iloc[filtered_indices_of_row[gaze_index][0]]["prediction"]
        penalty = filtered_text_data.iloc[filtered_indices_of_row[gaze_index][0]]["penalty"]
        density = filtered_reading_density[gaze_index]
        distance = filtered_distances_of_row[gaze_index][0]
        if bool_weight:
            if penalty > 0:
                weight = configs.weight_divisor / (abs(density - prediction) + configs.weight_intercept)
            else:
                weight = penalty
                # weight = configs.weight_divisor / (abs(density - prediction) + configs.weight_intercept)
        else:
            weight = 1
    else:
        # 在point_matching过程中，因为一定要给reading point添加一个匹配的text point，所以可能会出现距离极长的匹配。
        # 这种匹配应该在weight进行额外的处理，下面用distance和distance_threshold的比值作为ratio，去修改weight。
        # 经过调整，filtered_indices_of_all_text中不会再有blank_supplement。
        distance = filtered_distance_of_all_text[gaze_index][0]
        # ratio = min(1, distance_threshold / distance)
        ratio = min(1, np.float_power(distance_threshold / 2 / distance, 2))
        point_pair = [filtered_reading_coordinates[gaze_index], text_coordinate[filtered_indices_of_all_text[gaze_index][0]]]
        if text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["word"] == "blank_supplement":
            weight = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["penalty"] * ratio
        else:
            prediction = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["prediction"]
            density = filtered_reading_density[gaze_index]
            weight = configs.weight_divisor / (abs(density - prediction) + configs.weight_intercept) * ratio
        if not bool_weight:
            weight = 1

    return text_index, row_index, gaze_index, point_pair, weight


def step_2_add_no_matching_text_point(gaze_point_list_1d, gaze_point_info_list_1d, reading_data, effective_text_point_dict, actual_text_point_dict, point_pair_list):
    total_reading_nbrs = NearestNeighbors(n_neighbors=int(len(gaze_point_list_1d) / 4), algorithm='kd_tree').fit(gaze_point_list_1d)
    # 生成每个文本每个reading data的nearest neighbor。
    reading_nbrs_list = []
    for text_index in range(len(reading_data)):
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
                    weight = row_df.iloc[col_index]["penalty"] # 为了让左上角的数据不出问题，我将左上角blank_supplement的penalty都设置为了1（而非负数）。
                else:
                    weight = configs.empty_penalty * configs.left_boundary_ratio
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
                               reading_data,
                               reading_nbrs_list, distance_threshold):
    data_type_list = []
    point_pair_list = []
    weight_list = []
    info_list = []

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

    # 对于其它行，对左右两侧的blank_supplement添加匹配。
    row_df = text_df[text_df["row"] == row_list[row_index]]
    if row_df[row_df["word"] != "blank_supplement"].shape[0] == 0:
        return point_pair_list, weight_list, info_list, data_type_list

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

    row_df = row_df.sort_values(by=["col"], ascending=False) # 注意，这里有一个ascending=False。
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

    return data_type_list, point_pair_list, weight_list, info_list


def random_select_for_supplement_point_pairs(point_pair_list, supplement_point_pair_list, supplement_weight_list, supplement_info_list, select_ratio):
    if len(configs.training_index_list) > 16:
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

    np.random.seed(iteration_index)

    # 把之前简单的随机选择去掉。
    # random_select_indices = np.random.choice(len(point_pair_list), int(configs.random_select_ratio_for_point_pair * len(point_pair_list)), replace=False)
    # random_selected_point_pair_list = [point_pair_list[i] for i in random_select_indices]
    # random_selected_weight_list = [weight_list[i] for i in random_select_indices]
    # random_selected_info_list = [info_list[i] for i in random_select_indices]

    gaze_point = [point_pair_list[i][0] for i in range(len(point_pair_list))]
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init=10).fit(gaze_point)
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
        hull = ConvexHull(points)
        cluster_volume_list.append(hull.volume)

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
    return random_selected_point_pair_list, random_selected_weight_list, random_selected_info_list


def step_4_supplement_num(iteration_index,
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


def point_matching_multi_process(reading_data, gaze_point_list_1d, selected_gaze_point_info_list_1d,
                                 text_data, filtered_text_data_list,
                                 total_nbrs_list, row_nbrs_list,
                                 effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                                 iteration_index, distance_threshold):
    np.random.seed(configs.random_seed)

    # TODO 这里2个小思路：1，第二步中，对于blank_supplement，应该把最终的weight和距离相关，距离越小weight绝对值越大。（因为目前不同方向的blank_supplement不会将gaze推向更远处）
    #  2. 第一步中，似乎不应额外添加超过distance threshold随机选择所有点做点对的情况。匹配还是在行内进行。将blank_supplement都放到第二步去做。

    with multiprocessing.Pool(configs.number_of_process) as pool:
        # 1. 首先遍历所有的reading point，找到与其row label一致的、距离最近的text point，然后将这些匹配点加入到point_pair_list中。
        args_list = []
        for text_index in range(len(reading_data)):
            if text_index in configs.training_index_list:
                continue
            reading_df = reading_data[text_index]
            for row_index in range(configs.row_num):
                filtered_reading_df = reading_df[reading_df["row_label"] == row_index]

                if row_nbrs_list[text_index][row_index] and filtered_reading_df.shape[0] != 0:
                    filtered_reading_coordinates = filtered_reading_df[["gaze_x", "gaze_y"]].values.tolist()
                    filtered_distances_of_row, filtered_indices_of_row = row_nbrs_list[text_index][row_index].kneighbors(filtered_reading_coordinates)
                    filtered_text_data = filtered_text_data_list[text_index][row_index]
                    filtered_text_coordinate = filtered_text_data[["x", "y"]].values.tolist()
                    filtered_reading_density = filtered_reading_df["density"].values.tolist()

                    text_coordinate = text_data[text_index][["x", "y"]].values.tolist()
                    filtered_distance_of_all_text, filtered_indices_of_all_text = total_nbrs_list[text_index].kneighbors(filtered_reading_coordinates)

                    for gaze_index in range(len(filtered_distances_of_row)):
                        args_list.append((text_index, row_index, gaze_index,
                                          text_coordinate, text_data,
                                          filtered_reading_coordinates, filtered_text_data, filtered_text_coordinate, filtered_reading_density,
                                          filtered_indices_of_row, filtered_distances_of_row,
                                          filtered_indices_of_all_text, filtered_distance_of_all_text,
                                          distance_threshold, configs.bool_weight))

        results = pool.starmap(step_1_matching_among_all, args_list)

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
            step_2_add_no_matching_text_point(gaze_point_list_1d, selected_gaze_point_info_list_1d, reading_data, effective_text_point_dict, actual_text_point_dict, point_pair_list)

        # 3. 对于横向最外侧的补充点或空格点（即左右侧紧贴近正文的点），都可以考虑额外添加一些匹配点对，添加的weight是负数。
        # 这里单独为要添加boundary的point pair生成list，方便后续筛选。
        args_list = []
        for text_index in range(len(text_data)):
            text_df = text_data[text_index]
            row_list = text_df["row"].unique().tolist()

            for row_index in range(len(row_list)):
                args_list.append((text_df, text_index, row_list, row_index,
                                  reading_data,
                                  reading_nbrs_list, distance_threshold))

        results_raw = pool.starmap(step_3_add_boundary_points, args_list)
        results_raw = [result_raw for result_raw in results_raw if len(result_raw[0]) > 0]
        results = []
        for result_index, result in enumerate(results_raw):
            for sub_result_index in range(len(result[0])):
                results.append((result[0][sub_result_index], result[1][sub_result_index], result[2][sub_result_index], result[3][sub_result_index]))

        left_point_pair_list = [results[i][1] for i in range(len(results)) if results[i][0] == "left"]
        left_weight_list = [results[i][2] for i in range(len(results)) if results[i][0] == "left"]
        left_info_list = [results[i][3] for i in range(len(results)) if results[i][0] == "left"]
        right_point_pair_list = [results[i][1] for i in range(len(results)) if results[i][0] == "right"]
        right_weight_list = [results[i][2] for i in range(len(results)) if results[i][0] == "right"]
        right_info_list = [results[i][3] for i in range(len(results)) if results[i][0] == "right"]
        top_point_pair_list = [results[i][1] for i in range(len(results)) if results[i][0] == "top"]
        top_weight_list = [results[i][2] for i in range(len(results)) if results[i][0] == "top"]
        top_info_list = [results[i][3] for i in range(len(results)) if results[i][0] == "top"]
        bottom_point_pair_list = [results[i][1] for i in range(len(results)) if results[i][0] == "bottom"]
        bottom_weight_list = [results[i][2] for i in range(len(results)) if results[i][0] == "bottom"]
        bottom_info_list = [results[i][3] for i in range(len(results)) if results[i][0] == "bottom"]


        # 4. 限制添加点的数量。
        point_pair_list, weight_list, info_list = step_4_supplement_num(iteration_index,
                                                                        point_pair_list, weight_list, info_list,
                                                                        supplement_point_pair_list, supplement_weight_list, supplement_info_list,
                                                                        left_point_pair_list, left_weight_list, left_info_list,
                                                                        right_point_pair_list, right_weight_list, right_info_list,
                                                                        top_point_pair_list, top_weight_list, top_info_list,
                                                                        bottom_point_pair_list, bottom_weight_list, bottom_info_list)

        return point_pair_list, weight_list, info_list

























