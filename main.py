from compute_weight.compute_weight_nn_model import train_and_save_model_in_batch, compute_weight_and_save_to_text_in_batch
from evaluate_result.evaluate_final_transform_matrix import compute_accuracy_error_for_all_subjects, evaluate_accuracy_error_for_centroid_alignment
from evaluate_result.evaluate_seven_points import evaluate_seven_points_for_all_subjects
from funcs.calibrate_in_batch import calibrate_in_batch, calibrate_in_batch_for_different_training_num
from funcs.compute_gaze_density import compute_and_save_gaze_density
from funcs.constituency_parsing import add_constituency_parsing_to_text, save_pkl_with_constituency_parsing
from funcs.manual_calibrate_for_std import evaluate_non_calibrate_for_all_subjects
from funcs.visualize_calibrate_and_gradient_descent import visualize_all_subject_cali_grad_result, visualize_all_subject_cali_grad_process, visualize_cali_grad_process, visualize_cali_grad_result
from funcs.visualize_reading_time import visualize_reading_time
from funcs.visualize_text_structure import visualize_text_structure
from compute_weight.simple_fc_model import SimpleNet

if __name__ == '__main__':
    # visualize_reading_time() # 用于可视化阅读的时间分布。
    # visualize_text_structure() # 用于查看文本的结构性好坏。

    # TODO 确认一下text_density和reading_density之间的分布差异。

    # add_constituency_parsing_to_text() # 将句子的成分句法分析结果保存成单独的文件。
    # save_pkl_with_constituency_parsing() # 将text_sorted_mapping_for_model.pkl读取，把成分句法分析结果加入其中，然后重新保存。
    #
    # train_and_save_model_in_batch(SimpleNet, "simple_linear_net") # 根据不同输入搭配，训练简单的神经网络模型，然后保存。
    # compute_weight_and_save_to_text_in_batch(SimpleNet, "simple_linear_net") # 通过之前训练的神将网络模型，计算不同输入搭配下的weight，然后保存到text_sorted_mapping中。

    # compute_and_save_gaze_density() # 读取reading_after_cluster中的数据，重新计算gaze density并保存。

    # calibrate_in_batch("simple_linear_weight", [0]) # 一般情况下校准，使用model_index=1，保证simple_linear_model中的所有参数都被使用。
    # calibrate_in_batch_for_different_training_num("simple_linear_weight", 148, 1, 4, 0)

    # calibrate_in_batch("simple_linear_weight", [0], 9) # 不带simple_linear_model的校准，使用model_index=0，保证simple_linear_model中的所有参数都不被使用。

    # visualize_cali_grad_process(67, 1, 0)
    # visualize_cali_grad_process(55, 1, 9)
    # visualize_all_subject_cali_grad_process(67, 1)
    # visualize_cali_grad_result(42, 1, 0)
    # visualize_all_subject_cali_grad_result(56, 1)

    # TODO 读取log文件，然后对其中不同iteration的transform matrix进行聚类，然后看看聚类的结果。
    # cluster_transform_matrix(48, 1, 2)

    # 先不考虑聚类，先简单地按照iteration做截取，然后看看将剩余transform_matrix取均值后的accuracy_error。
    # compute_accuracy_error_for_all_subjects(153, 0)
    # 评价7点校准法的accuracy_error。
    # evaluate_seven_points_for_all_subjects()
    # 评价未校准时的accuracy_error。
    # evaluate_non_calibrate_for_all_subjects()
    # 评价只有中心点对齐时的accuracy_error。
    # evaluate_accuracy_error_for_centroid_alignment(73, 1)





