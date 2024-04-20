import json
import os


def read_calibrate_and_grad_descent(file_index, model_index, subject_index):
    calibrate_process_path = f"{os.getcwd()}/gradient_descent_log/{file_index}/calibrate_with_simple_linear_and_weight-model_{model_index}-subject_{subject_index}.json"
    with open(calibrate_process_path, "r") as json_file:
        calibrate_process = json.load(json_file)

        return calibrate_process


def read_all_subject_calibrate_and_grad_descent(file_index, model_index):
    cali_grad_list = []

    cali_grad_path_prefix = f"{os.getcwd()}/gradient_descent_log/{file_index}"
    cali_grad_path_list = os.listdir(cali_grad_path_prefix)

    cali_grad_path_prefix = f"calibrate_with_simple_linear_and_weight-model_{model_index}-subject_"
    cali_grad_path_list = [path for path in cali_grad_path_list if path.startswith(cali_grad_path_prefix)]
    cali_grad_path_list.sort(key=lambda x: int(x.split("subject_")[-1].split(".")[0]))

    for subject_index in range(len(cali_grad_path_list)):
        calibrate_process = read_calibrate_and_grad_descent(file_index, model_index, subject_index)
        cali_grad_list.append(calibrate_process)

    return cali_grad_list

