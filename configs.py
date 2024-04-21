round_num = "round_5"
exp_device = "tobii"

file_index = 16

random_seed = 0
number_of_process = 8

screen_width = 1920
screen_height = 1200

text_width = 40
text_height = 64
row_num = 6
col_num = 30

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
weight_divisor = 5
weight_intercept = 0.01
completion_weight = 4
right_down_corner_unmatched_ratio = 1
left_boundary_ratio = 1250
right_boundary_ratio = 3500
bottom_boundary_ratio = 750
left_boundary_distance_threshold_ratio = 1.25
right_boundary_distance_threshold_ratio = 1
bottom_boundary_distance_threshold_ratio = 0.25
boundary_select_ratio = 0.2
supplement_select_ratio = 1
random_select_ratio_for_point_pair = 0.1
gradient_descent_iteration_threshold = 500

training_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


