from matplotlib import pyplot as plt

import configs
from read.read_reading import read_raw_reading
from read.read_text import read_text_sorted_mapping_and_group_with_para_id


def visualize_text_and_reading():
    plt.rcParams['font.sans-serif'] = ['SimHei']

    reading_data_list = read_raw_reading("original", "_after_cluster")
    text_data_list = read_text_sorted_mapping_and_group_with_para_id()

    for subject_index in range(2, len(reading_data_list)):
        for text_index in range(len(text_data_list)):
            reading_data = reading_data_list[subject_index][text_index]
            text_data = text_data_list[text_index]

            fig = plt.figure(figsize=(18, 7.5))
            ax = fig.add_subplot(111)
            ax.set_xlim(0, configs.screen_width)
            ax.set_ylim(configs.screen_height, 0)

            for index, row in text_data.iterrows():
                if row["word"] == " ":
                    continue
                if row["word"] == "blank_supplement":
                    ax.text(row["x"], row["y"], "·", fontsize=25, color=[0.7, 0.7, 0.7], zorder=1)
                else:
                    ax.text(row["x"], row["y"], row["word"], fontsize=25, color=[0.7, 0.7, 0.7], zorder=1)

            for index, row in reading_data.iterrows():
                x = row["gaze_x"]
                y = row["gaze_y"]
                ax.scatter(x, y, s=20, c=[0.9, 0.3, 0.3], alpha=1, zorder=2)

            plt.show()
            plt.close()

