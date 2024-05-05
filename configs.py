import numpy as np

round_num = "round_5"
exp_device = "tobii"

file_index = 210

random_seed = 0
number_of_process = 4
gpu_device_id = "cuda:1"

screen_width = 1920
screen_height = 1200

text_width = 40
text_height = 64
row_num = 6
col_num = 30
left_right_padding = 3
up_down_padding = 1

left_top_text_center = [380, 272]
right_down_text_center = [1540, 592]

passage_num = 40
punctuation_list = {'\'', '\"', '!', '?', '.', '/', '\\', '-', '，', ':', '：', '。', '……', '！', '？', '——', '（', '）', '【', '】', '“', '”', '’', '‘', '：', '；', '《', '》', '、', '—', '～', '·', '「', '」', '『', '』', '…'}

model_density_index = 4 # 这里的4是density，5是relative_density。
learning_rate = 0.01
epoch_num = 300

location_penalty = 1
punctuation_penalty = -0.001
empty_penalty = -0.001

bool_weight = True
bool_text_weight = True
weight_divisor = 5
weight_intercept = 0.5
completion_weight = 5
right_down_corner_unmatched_ratio = 1
left_boundary_ratio = 1600
right_boundary_ratio = 1200
top_boundary_ratio = 800
bottom_boundary_ratio = 100
left_boundary_distance_threshold_ratio = 1.25
right_boundary_distance_threshold_ratio = 1.25
top_boundary_distance_threshold_ratio = 0.75
bottom_boundary_distance_threshold_ratio = 0.75
right_boundary_distance_threshold_ratio_derivative = 0.5
right_boundary_ratio_derivative = 400
random_select_ratio_for_point_pair = 0.1
last_iteration_ratio = 0.25
punctuation_ratio = 0.1
boundary_select_ratio = 0.45
supplement_select_ratio = 0.1
gradient_descent_iteration_threshold = 500
max_iteration = 100
text_distance_threshold_ratio = 1
learning_rate_in_gradient_descent = 0.01

theta_error_threshold = np.pi / 72
scale_error_threshold = 0.15
shear_error_threshold = 0.15
theta_error_ratio = 100
scale_ratio = 100
shear_ratio = 100

final_transform_matrix_iteration_start = 20
final_transform_matrix_iteration_end = 100

subject_start = None
subject_end = None

training_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


