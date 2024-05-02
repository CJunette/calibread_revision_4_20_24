import multiprocessing
import os

import numpy as np
import torch
import wandb
from torch import nn

from compute_weight.simple_fc_model import SimpleNet

import configs
from read.read_model_config import read_model_config
from read.read_reading import read_text_density
from read.read_text import read_text_pkl, read_text_sorted_mapping


def _prepare_text_features(text_data, bool_dict):
    """
    将不同的text特征拼接成一个特征向量
    :param text_data:
    :param bool_dict:
    :return:
    """

    bool_row = bool_dict["bool_row"]
    bool_col = bool_dict["bool_col"]
    bool_row_length = bool_dict["bool_row_length"]
    bool_word_index_in_sentence = bool_dict["bool_word_index_in_sentence"]
    bool_sentence_length = bool_dict["bool_sentence_length"]
    bool_col_within_token = bool_dict["bool_col_within_token"]
    bool_token_length = bool_dict["bool_token_length"]
    bool_full_text_constituency_depth = bool_dict["bool_full_text_constituency_depth"]
    bool_split_text_constituency_depth = bool_dict["bool_split_text_constituency_depth"]
    bool_sentence_embedding = bool_dict["bool_sentence_embedding"]
    bool_token_embedding = bool_dict["bool_token_embedding"]

    vector_list = []
    for text_index in range(len(text_data)):
        print(f"processing vector of text {text_index}")
        text_df = text_data[text_index]
        row_list = text_df["row"].tolist()
        col_list = text_df["col"].tolist()
        row_length_list = text_df["row_length"].tolist()
        word_index_in_sentence_list = text_df["word_index_in_sentence"].tolist()
        sentence_length_list = text_df["sentence_length"].tolist()
        col_within_token_list = text_df["col_within_token"].tolist()
        token_length_list = text_df["token_length"].tolist()
        full_text_constituency_depth_list = text_df["full_text_constituency_depth"].tolist()
        split_text_constituency_depth_list = text_df["split_text_constituency_depth"].tolist()
        sentence_embedding_list = text_df["sentence_embedding"].tolist()
        token_embedding_list = text_df["token_embedding"].tolist()
        text_index_list = [text_index for _ in range(len(row_list))]

        # 这里加text_index_list, row_list, col_list是为了后面方便做筛选。
        combined_lists = [text_index_list, row_list, col_list]

        if bool_row:
            combined_lists.append(row_list)
        if bool_col:
            combined_lists.append(col_list)
        if bool_row_length:
            combined_lists.append(row_length_list)
        if bool_word_index_in_sentence:
            combined_lists.append(word_index_in_sentence_list)
        if bool_sentence_length:
            combined_lists.append(sentence_length_list)
        if bool_col_within_token:
            combined_lists.append(col_within_token_list)
        if bool_token_length:
            combined_lists.append(token_length_list)
        if bool_full_text_constituency_depth:
            combined_lists.append(full_text_constituency_depth_list)
        if bool_split_text_constituency_depth:
            combined_lists.append(split_text_constituency_depth_list)
        if bool_sentence_embedding:
            combined_lists.append(sentence_embedding_list)
        if bool_token_embedding:
            combined_lists.append(token_embedding_list)

        zip_list = list(zip(*combined_lists))
        # 目前，zip后的每个元素中，embedding仍然是list，因此我们需要将其展开。
        for zip_index in range(len(zip_list)):
            zip_item = zip_list[zip_index]
            new_zip_item = []
            for sub_zip_index in range(len(zip_item)):
                if isinstance(zip_item[sub_zip_index], list):
                    new_zip_item.extend(zip_item[sub_zip_index])
                else:
                    new_zip_item.append(zip_item[sub_zip_index])
            zip_list[zip_index] = new_zip_item
        vector_list.extend(np.array(zip_list))
    vector_list = np.array(vector_list)

    return vector_list


def _prepare_density(density_data, text_data):
    density_list = []
    for subject_index in range(len(density_data)):
        density_list_1 = []
        for text_index in range(len(density_data[subject_index])):
            print(f"processing density of subject {subject_index}, text {text_index}")
            density_df = density_data[subject_index][text_index].copy()
            density_df = density_df[density_df["word"] != "blank_supplement"]
            text_df = text_data[text_index]
            text_df_first_index = text_df.index.tolist()[0]
            text_df = text_df[text_df["sentence"] != "/split"]
            text_df_indices = text_df.index.tolist()
            text_df_indices = [index - text_df_first_index for index in text_df_indices]
            density_df = density_df.iloc[text_df_indices]

            row_list = density_df["row"].tolist()
            col_list = density_df["col"].tolist()
            density = density_df["text_density"].tolist()
            relative_density = density_df["relative_text_density"].tolist()
            text_index_list = [text_index for _ in range(len(row_list))]
            subject_index_list = [subject_index for _ in range(len(row_list))]
            zip_list = list(zip(subject_index_list, text_index_list, row_list, col_list, density, relative_density))
            zip_list = np.array(zip_list)
            density_list_1.extend(zip_list)
        density_list.append(np.array(density_list_1))
    density_list = np.array(density_list)
    return density_list


def _get_text_vector_and_density(bool_dict):
    '''
    用于获得text的特征作为模型的输入向量。获取density的特征作为模型的输出向量。
    :param bool_dict: 一个dict，包含了是否需要哪些特征。
    :return: 返回的vector_list中，每个元素的前3个对象分别是text_index, row, col，第四个对象才是输入的feature构成的vector。
             返回的density_list中，每个元素的前4个对象分别是subject_index, text_index, row, col，第5个对象是density，第6个对象是relative_density。
             之所以要放这些index，是因为我们需要根据这些index来根据para_id作筛选，同时方便进行匹配。
    '''
    text_data = read_text_pkl()
    density_data = read_text_density()

    # 将text_data转为model的输入向量。
    vector_list = _prepare_text_features(text_data, bool_dict)

    # 将density data转为model的输出结果。
    density_list = _prepare_density(density_data, text_data)

    return vector_list, density_list, text_data, density_data


def _select_data_given_indices(index_list, vector_list, density_list):
    X = [vector_list[i][3:] for i in range(len(vector_list)) if vector_list[i][0] in index_list]
    X = X * len(density_list)
    X = np.array(X)
    X_info = [vector_list[i][:3] for i in range(len(vector_list)) if vector_list[i][0] in index_list]
    X_info = X_info * len(density_list)
    X_info = np.array(X_info)

    y = []
    density_index_list = []
    for density_index in range(len(density_list[0])):
        if int(density_list[0][density_index][1]) in index_list:
            density_index_list.append(density_index)
    for subject_index in range(len(density_list)):
        density_list_1 = density_list[subject_index][density_index_list][:, configs.model_density_index] # 这里的4是density，5是relative_density。
        y.extend(density_list_1)

    return X, y, X_info


def _prepare_data_for_training(bool_dict):
    vector_list, density_list, text_data, density_data = _get_text_vector_and_density(bool_dict)

    training_index_list = configs.training_index_list
    X_train, y_train, X_train_info = _select_data_given_indices(training_index_list, vector_list, density_list)

    validation_index_list = np.setdiff1d(np.array([i for i in range(configs.passage_num)]), np.array(training_index_list))
    X_val, y_val, X_val_info = _select_data_given_indices(validation_index_list, vector_list, density_list)

    return X_train, y_train, X_val, y_val, X_train_info, X_val_info, vector_list, density_list, text_data, density_data


def _prepare_data_for_computing_weight(bool_dict):
    """
    这里返回的y_val只包括第一个subject的density。（后续其实不会实际用到这个值，为了代码的复用性，所以输出了它）
    """
    vector_list, density_list, text_data, density_data = _get_text_vector_and_density(bool_dict)
    index_list = np.array([i for i in range(configs.passage_num)])

    X_val = [vector_list[i][3:] for i in range(len(vector_list)) if vector_list[i][0] in index_list]
    X_val = np.array(X_val)
    X_info = [vector_list[i][:3] for i in range(len(vector_list)) if vector_list[i][0] in index_list]
    X_info = np.array(X_info)

    y_val = []
    density_index_list = []
    for density_index in range(len(density_list[0])):
        if int(density_list[0][density_index][1]) in index_list:
            density_index_list.append(density_index)

    density_list_1 = density_list[0][density_index_list][:, configs.model_density_index]  # 这里的4是density，5是relative_density。
    y_val.extend(density_list_1)

    return X_val, y_val, X_info, vector_list, density_list, text_data, density_data


def _create_model(model_type, input_size):
    model = model_type(input_size, hidden_layers_1=64, hidden_layers_2=128, hidden_layer_3=256, hidden_layer_4=512).to('cuda')
    # 定义优化器和损失函数
    if configs.model_density_index == 4:
        learning_rate = configs.learning_rate
        epoch_num = configs.epoch_num
    else:
        learning_rate = configs.learning_rate / 2
        epoch_num = configs.epoch_num * 2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    return model, optimizer, criterion, epoch_num


def _train_model(model, X, y, optimizer, criterion, model_prefix, model_name, epochs):
    # 转换数据为PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to('cuda')
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to('cuda')

    wandb.init(
        project="calibread_major_revision",
        name=f"{model_name}-{model_prefix}",
        config={
            "learning_rate": configs.learning_rate,
            "architecture": "SimpleNet",
            "epochs": epochs,
        }
    )

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


def _evaluate_model(model, X, y, criterion):
    X_tensor = torch.tensor(X, dtype=torch.float32).to('cuda')
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to('cuda')
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        print(f'Validation loss: {loss.item()}')

    return y_pred


def _save_model(model, model_prefix, model_name):
    if model_name != "":
        model_save_prefix = f"model/{model_prefix}"
        if not os.path.exists(model_save_prefix):
            os.makedirs(model_save_prefix)
        torch.save(model.state_dict(), f"{model_save_prefix}/{model_name}-{model_prefix}.pth")


def train_and_save_model(model_type, model_prefix, bool_dict=None):
    if bool_dict is None:
        bool_dict = dict(bool_row=True, bool_col=True, bool_row_length=True,
                         bool_word_index_in_sentence=True, bool_sentence_length=True,
                         bool_col_within_token=True, bool_token_length=True,
                         bool_full_text_constituency_depth=True, bool_split_text_constituency_depth=True,
                         bool_sentence_embedding=True, bool_token_embedding=True, model_index=0)
    model_name = bool_dict["model_index"]
    print(f"training model {model_name} ...")
    # 准备数据
    X_train, y_train, X_val, y_val, X_train_info, X_val_info, vector_list, density_list, text_data, density_data = _prepare_data_for_training(bool_dict)
    input_size = X_train.shape[1]

    # 创建模型
    model, optimizer, criterion, epoch_num = _create_model(model_type, input_size)

    # 训练模型
    _train_model(model, X_train, y_train, optimizer, criterion, model_prefix, model_name, epoch_num)
    wandb.finish()

    # 评估模型
    y_pred = _evaluate_model(model, X_val, y_val, criterion)

    # 保存模型
    _save_model(model, model_prefix, model_name)


def train_and_save_model_in_batch(model_type, model_prefix):
    # 读取不同的model config。model_config中存储    model_config = read_model_config()的数据包括model的序号和model对应的特征的布尔值。
    # 将model config的dataframe转化为字典。具体如下：
    # 假设csv文件为，
    # index, bool_a, bool_b
    # 0, True, True
    # 1, False, False
    # 转化后结果为bool_list = [{"bool_a": True, "bool_b": True}, {"bool_a": False, "bool_b": False}]。
    model_config = read_model_config()
    bool_dict_list = model_config.to_dict(orient="records")

    for bool_dict_list_index in range(len(bool_dict_list)):
        # if bool_dict_list_index == 0:
        if bool_dict_list[bool_dict_list_index]["model_index"] == 0:
        # if bool_dict_list[bool_dict_list_index]["model_index"] != 8 and bool_dict_list[bool_dict_list_index]["model_index"] != 9: # 更新8和9两个模型时候用的。
            # index为0对应没有任何特征（即不需要linear model），所以跳过。
            continue
        bool_dict = bool_dict_list[bool_dict_list_index]
        train_and_save_model(model_type, model_prefix, bool_dict)


def compute_weight_and_save_to_text(model_type, model_prefix, bool_dict):
    text_sorted_mapping = read_text_sorted_mapping()
    text_sorted_mapping["prediction"] = 0

    model_name = bool_dict["model_index"]
    print(f"validating model {model_name} ...")
    # 准备数据
    X_val, y_val, X_info, vector_list, density_list, text_data, density_data = _prepare_data_for_computing_weight(bool_dict)
    input_size = X_val.shape[1]

    # 读取模型。
    checkpoint = torch.load(f"{os.getcwd()}/model/{model_prefix}/{model_name}-{model_prefix}.pth")
    model = model_type(input_size).to('cuda')
    model.load_state_dict(checkpoint)
    model.eval()

    # 评估模型
    y_pred = _evaluate_model(model, X_val, y_val, nn.MSELoss())

    prediction_list = []
    for val_index in range(len(X_info)):
        # 似乎不需要使用逐一匹配的方法，这样效率有点低。之前输出的时候，y_pred和text_sorted_mapping的顺序应该是一致的。
        # text_index = int(X_info[val_index][0])
        # row = int(X_info[val_index][1])
        # col = int(X_info[val_index][2])
        # text_sorted_mapping.loc[(text_sorted_mapping["para_id"] == text_index) & (text_sorted_mapping["row"] == row) & (text_sorted_mapping["col"] == col), "prediction"] = y_pred[val_index].item()

        # 这边不如直接用一个list把所有输出结果都接住，然后再一次性赋值给text_sorted_mapping。
        prediction_list.append(y_pred[val_index].item())

    text_sorted_mapping["prediction"] = prediction_list

    save_path = f"{os.getcwd()}/data/text/{configs.round_num}/text_sorted_mapping_with_prediction_{model_name}.csv"
    text_sorted_mapping.to_csv(save_path, index=False, encoding="utf-8-sig")


def compute_weight_and_save_to_text_in_batch(model_type, model_prefix):
    """
    开头部分代码与train_and_save_model_in_batch相同。
    """
    model_config = read_model_config()
    bool_dict_list = model_config.to_dict(orient="records")

    for bool_dict_list_index in range(len(bool_dict_list)):
        if bool_dict_list[bool_dict_list_index]["model_index"] != 7 and bool_dict_list[bool_dict_list_index]["model_index"] != 8 and bool_dict_list[bool_dict_list_index]["model_index"] != 9: # 更新8和9两个模型时候用的。
            continue
        if bool_dict_list[bool_dict_list_index]["model_index"] == 0:
        # if bool_dict_list_index == 0:
            text_sorted_mapping = read_text_sorted_mapping()
            text_sorted_mapping["prediction"] = 0
            save_path = f"{os.getcwd()}/data/text/{configs.round_num}/text_sorted_mapping_with_prediction_0.csv"
            text_sorted_mapping.to_csv(save_path, index=False, encoding="utf-8-sig")

        bool_dict = bool_dict_list[bool_dict_list_index]
        compute_weight_and_save_to_text(model_type, model_prefix, bool_dict)






















