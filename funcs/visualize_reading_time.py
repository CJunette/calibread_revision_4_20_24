from read.read_text import read_text_sorted_mapping
from read.read_reading import read_raw_reading
from matplotlib import pyplot as plt


def _get_text_unit_num():
    texts = read_text_sorted_mapping()
    texts = texts[texts["word"].apply(lambda x: len(x.strip()) != 0)]

    text_num = len(texts["para_id"].unique())
    text_unit_num_list = []

    for text_index in range(text_num):
        text = texts[texts["para_id"] == text_index]
        text_unit_num = len(text)
        text_unit_num_list.append(text_unit_num)

    return text_unit_num_list


def _get_reading_time():
    modified_raw_reading_df = read_raw_reading("modified")

    reading_time_list_1 = []
    for user_index in range(len(modified_raw_reading_df)):
        reading_time_list_2 = []
        for text_index in range(len(modified_raw_reading_df[user_index])):
            reading_df = modified_raw_reading_df[user_index][text_index]
            reading_duration = reading_df['time'].iloc[-1] - reading_df['time'].iloc[0]
            reading_time_list_2.append(reading_duration)
        reading_time_list_1.append(reading_time_list_2)

    return reading_time_list_1


def visualize_reading_time():
    '''
    该函数被最终调用，实现可视化。
    用于可视化阅读时间的分布。结果展示了阅读时间和平均阅读时间（阅读时间除以text_unit数量）的整体分布，以及每个用户的阅读时间分布。
    :return:
    '''
    text_num_list = _get_text_unit_num()
    reading_time_list = _get_reading_time()

    avg_reading_time_list = []
    for user_index in range(len(reading_time_list)):
        avg_reading_time_list_1 = []
        for text_index in range(len(reading_time_list[user_index])):
            avg_reading_time_list_1.append(reading_time_list[user_index][text_index] / text_num_list[text_index])
        avg_reading_time_list.append(avg_reading_time_list_1)

    total_reading_time_list = []
    for user_index in range(len(reading_time_list)):
        total_reading_time_list.extend(reading_time_list[user_index])

    total_avg_reading_time_list = []
    for user_index in range(len(avg_reading_time_list)):
        total_avg_reading_time_list.extend(avg_reading_time_list[user_index])

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    bin_num = 50

    ax[0, 0].hist(total_reading_time_list, bins=bin_num)
    ax[0, 0].set_title('Total Reading Time Distribution')
    ax[0, 0].set_xlabel('Reading Time (s)')
    ax[0, 0].set_ylabel('Frequency')
    total_mean = sum(total_reading_time_list) / len(total_reading_time_list)
    total_std = (sum([(x - total_mean) ** 2 for x in total_reading_time_list]) / len(total_reading_time_list)) ** 0.5
    ax[0, 0].text(0.7, 0.7, f"Mean: {total_mean:.2f}\nStd: {total_std:.2f}", transform=ax[0, 0].transAxes)

    ax[0, 1].hist(total_avg_reading_time_list, bins=bin_num)
    ax[0, 1].set_title('Average Reading Time Distribution')
    ax[0, 1].set_xlabel('Reading Time (s)')
    ax[0, 1].set_ylabel('Frequency')
    avg_mean = sum(total_avg_reading_time_list) / len(total_avg_reading_time_list)
    avg_std = (sum([(x - avg_mean) ** 2 for x in total_avg_reading_time_list]) / len(total_avg_reading_time_list)) ** 0.5
    ax[0, 1].text(0.7, 0.7, f"Mean: {avg_mean:.2f}\nStd: {avg_std:.2f}", transform=ax[0, 1].transAxes)

    for user_index in range(len(reading_time_list)):
        ax[1, 0].hist(reading_time_list[user_index], bins=bin_num, alpha=0.2)
    ax[1, 0].set_title('Total Reading Time Distribution by User')
    ax[1, 0].set_xlabel('Reading Time (s)')

    for user_index in range(len(avg_reading_time_list)):
        ax[1, 1].hist(avg_reading_time_list[user_index], bins=bin_num, alpha=0.2)
    ax[1, 1].set_title('Average Reading Time Distribution by User')
    ax[1, 1].set_xlabel('Reading Time (s)')

    plt.show()




