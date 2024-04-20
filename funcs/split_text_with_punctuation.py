import pandas as pd

import configs
from read.read_text import read_text_sorted_mapping_and_group_with_para_id


def _check_split_given_word(word):
    if word.strip() == "" or word in configs.punctuation_list:
        return True
    else:
        return False


def split_text_with_punctuation():
    '''
    该函数的目的是将text_sorted_mapping中的每行文字都按照标点符号进行分割，将得到的sentence_unit保存到dataframe中，同时将每篇文章的unique sentence_unit返回，便于后续获取成分句法分析。
    :return:
    '''
    text_sorted_mapping = read_text_sorted_mapping_and_group_with_para_id()
    sentence_unit_list_1 = []
    new_text_sorted_mapping_1 = []

    for text_index in range(0, len(text_sorted_mapping)):
        text_df = text_sorted_mapping[text_index]
        rows = text_df["row"].unique().tolist()
        rows.sort()
        sentence_unit_list_2 = []
        new_text_sorted_mapping_2 = []

        for row_index in range(0, len(rows)):
            row = rows[row_index]
            row_df = text_df[text_df["row"] == row].reset_index(drop=True)
            row_df_split = row_df[row_df["word"].apply(_check_split_given_word)]
            split_indices = row_df_split.index.tolist()

            sentence_list = ["" for _ in range(0, row_df.shape[0])]
            unique_sentence_list = ["" for _ in range(0, row_df.shape[0])] # 这个似乎最后没用上，是之前代码的遗留。
            word_index_in_sentence = [0 for _ in range(0, row_df.shape[0])]
            sentence_length = [0 for _ in range(0, row_df.shape[0])]

            # 如果len(split_indices) == 0，代表根本没有标点，此时可以直接把整个row_df作为一个sentence。
            if len(split_indices) == 0:
                word_list = row_df["word"].tolist()
                word_str = "".join(word_list)
                for index in range(0, row_df.shape[0]):
                    sentence_list[index] = word_str
                    word_index_in_sentence[index] = index
                    unique_sentence_list[index] = word_str
                    sentence_length[index] = row_df.shape[0]

            else:
                # 否则，首先标记split_indices中的标点，然后再处理其他的word。
                for index in range(len(split_indices)):
                    split_index = split_indices[index]
                    word_str = row_df["word"].iloc[split_index]
                    sentence_list[split_index] = word_str
                    word_index_in_sentence[split_index] = -1
                    unique_sentence_list[split_index] = word_str
                    sentence_length[split_index] = 1

                index = 0
                while index < len(split_indices):
                    split_index = split_indices[index]
                    # 遇到连续的标点或空格时，跳过。
                    if index + 1 < len(split_indices) and split_index + 1 == split_indices[index + 1]:
                        index += 2
                        continue

                    if index == 0:
                        start_index = 0
                    else:
                        start_index = split_indices[index - 1] + 1
                    end_index = split_index

                    word_list = row_df["word"].tolist()[start_index:end_index]
                    word_str = "".join(word_list)
                    for text_unit_index in range(start_index, end_index):
                        sentence_list[text_unit_index] = word_str
                        word_index_in_sentence[text_unit_index] = text_unit_index - start_index
                        unique_sentence_list[text_unit_index] = word_str
                        sentence_length[text_unit_index] = end_index - start_index

                    index += 1

                # 处理最后一个标点后的word（如果确实存在）。
                if split_indices[-1] < row_df.shape[0] - 1:
                    start_index = split_indices[-1] + 1
                    end_index = row_df.shape[0]
                    word_list = row_df["word"].tolist()[start_index:end_index]
                    word_str = "".join(word_list)
                    for text_unit_index in range(start_index, end_index):
                        sentence_list[text_unit_index] = word_str
                        word_index_in_sentence[text_unit_index] = text_unit_index - start_index
                        unique_sentence_list[text_unit_index] = word_str
                        sentence_length[text_unit_index] = end_index - start_index

            row_df["sentence"] = sentence_list
            row_df["word_index_in_sentence"] = word_index_in_sentence
            row_df["sentence_length"] = sentence_length

            # 将row_df保存。
            new_text_sorted_mapping_2.append(row_df)

            # 将unique_sentence保存。
            unique_sentences = row_df["sentence"].unique().tolist()
            sentence_unit_list_2.extend(unique_sentences)
            sentence_unit_list_2 = list(set(sentence_unit_list_2))

        sentence_unit_list_1.append(sentence_unit_list_2)
        new_text_sorted_mapping = pd.concat(new_text_sorted_mapping_2, ignore_index=True).reset_index(drop=True)
        new_text_sorted_mapping_1.append(new_text_sorted_mapping)

    return new_text_sorted_mapping_1, sentence_unit_list_1







