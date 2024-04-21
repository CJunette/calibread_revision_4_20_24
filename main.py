from compute_weight.compute_weight_nn_model import train_and_save_model, train_and_save_model_in_batch, compute_weight_and_save_to_text_in_batch
from funcs.calibrate_in_batch import calibrate_in_batch
from funcs.compute_gaze_density import compute_and_save_gaze_density
from funcs.visualize_calibrate_and_gradient_descent import visualize_cali_grad_process, visualize_all_subject_cali_grad_result, visualize_all_subject_cali_grad_process, visualize_cali_grad_result
from funcs.visualize_reading_time import visualize_reading_time
from funcs.visualize_text_structure import visualize_text_structure
from funcs.constituency_parsing import add_constituency_parsing_to_text, save_pkl_with_constituency_parsing
from compute_weight.simple_fc_model import SimpleNet


if __name__ == '__main__':
    # visualize_reading_time() # 用于可视化阅读的时间分布。
    # visualize_text_structure() # 用于查看文本的结构性好坏。

    # TODO 确认一下text_density和reading_density之间的分布差异。

    # add_constituency_parsing_to_text() # 将句子的成分句法分析结果保存成单独的文件。
    # save_pkl_with_constituency_parsing() # 将text_sorted_mapping_for_model.pkl读取，把成分句法分析结果加入其中，然后重新保存。

    # train_and_save_model_in_batch(SimpleNet, "simple_linear_net") # 根据不同输入搭配，训练简单的神经网络模型，然后保存。
    # compute_weight_and_save_to_text_in_batch(SimpleNet, "simple_linear_net") # 通过之前训练的神将网络模型，计算不同输入搭配下的weight，然后保存到text_sorted_mapping中。

    # compute_and_save_gaze_density() # 读取reading_after_cluster中的数据，重新计算gaze density并保存。

    # TODO 目前有2个思路，一是把每个iter中变化的point pair标记出来；二是在每次iter中，保留一部分上次的点，再进行梯度下降。
    calibrate_in_batch("simple_linear_weight")

    # visualize_cali_grad_process(15, 1, 0)
    # visualize_cali_grad_process(15, 1, 1)
    # visualize_all_subject_cali_grad_process(10, 1)
    # visualize_cali_grad_result(13, 1, 0)
    # visualize_all_subject_cali_grad_result(12, 1)
