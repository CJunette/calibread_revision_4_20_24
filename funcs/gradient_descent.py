import numpy as np
import torch
from funcs.util_functions import change_homogeneous_vector_to_2d_vector, change_2d_vector_to_homogeneous_vector
import configs


def gradient_descent_with_torch(point_pairs, weight, last_iteration_num, learning_rate=2e-1, max_iterations=2000, stop_grad_norm=5, grad_clip_value=0):
    torch.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    source_points = np.array([change_2d_vector_to_homogeneous_vector(point_pair[0]) for point_pair in point_pairs])
    target_points = np.array([change_2d_vector_to_homogeneous_vector(point_pair[1]) for point_pair in point_pairs])
    source_points = torch.tensor(source_points, dtype=torch.float32, requires_grad=False).cuda(0)
    target_points = torch.tensor(target_points, dtype=torch.float32, requires_grad=False).cuda(0)
    # target_points = target_points[:, :2] / target_points[:, 2:]

    weight_tensor = torch.tensor(weight, dtype=torch.float32, requires_grad=False).unsqueeze(-1).cuda(0)

    matrix_00 = 1 + 1e-5
    matrix_01 = 1e-5
    matrix_02 = 1e-5
    matrix_10 = 1e-5
    matrix_11 = 1 + 1e-5
    matrix_12 = 1e-5
    matrix_20 = 0
    matrix_21 = 0
    matrix_22 = 1

    # transform_matrix = torch.tensor([[matrix_00, matrix_01, matrix_02],
    #                                  [matrix_10, matrix_11, matrix_12],
    #                                  [matrix_20, matrix_21, matrix_22]], dtype=torch.float32, requires_grad=True).cuda(0)
    transform_matrix_raw = torch.tensor([[matrix_00, matrix_01, matrix_02],
                                         [matrix_10, matrix_11, matrix_12]], dtype=torch.float32, requires_grad=True).cuda(0)
    transform_matrix_raw = torch.nn.Parameter(transform_matrix_raw)  # 将transform_matrix转换为一个可以优化的参数

    optimizer = torch.optim.Adam([transform_matrix_raw], lr=learning_rate)
    # optimizer = torch.optim.SGD([transform_matrix_raw], lr=learning_rate)
    last_error = 1000000
    grad_norm_list = []
    grad_norm_derivative_list = [0]
    square_scale_tensor = torch.tensor([1, 1, 1e5], dtype=torch.float32, requires_grad=False).cuda(0)

    final_transform_matrix_raw = None
    least_error = 1000000
    least_grad_norm = 0
    least_index = 0 # 用least index来标记最终是否是主动收敛退出。
    for iteration_index in range(max_iterations):
        optimizer.zero_grad()
        transform_matrix = torch.concat([transform_matrix_raw, torch.tensor([[0, 0, 1]], dtype=torch.float32, requires_grad=False).cuda(0)], dim=0)
        transformed_points = torch.matmul(transform_matrix, source_points.transpose(0, 1)).transpose(0, 1)
        # source_points_after_transform = source_points_after_transform[:, :2] / source_points_after_transform[:, 2:]
        # square = square * square_scale_tensor
        # error = torch.mean(torch.square(transformed_points - target_points) * weight_tensor)
        distance = transformed_points[:, :2] - target_points[:, :2]
        error = torch.mean(torch.square(distance) * weight_tensor)
        # penalty = torch.sum((transformed_points[:, 2] - target_points[:, 2]) ** 2) * 100
        # error += penalty
        error.backward()
        if grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(transform_matrix_raw, grad_clip_value)
        optimizer.step()

        # print(f"iteration: {iteration_index}, error: {error}, grad_norm: {torch.norm(transform_matrix_raw.grad)}")
        last_iteration_num = iteration_index
        last_error = error
        if len(grad_norm_list) > 0:
            grad_norm_derivative_list.append(torch.norm(transform_matrix_raw.grad).cpu().detach() - grad_norm_list[-1])
        grad_norm_list.append(torch.norm(transform_matrix_raw.grad).cpu())

        if (final_transform_matrix_raw is None
                or (abs(error) < 1000 and error * torch.norm(transform_matrix_raw.grad) < least_error * least_grad_norm)):
            least_error = error
            final_transform_matrix_raw = transform_matrix_raw.clone()
            least_grad_norm = torch.norm(transform_matrix_raw.grad)
            least_index = iteration_index

        bool_stop = True
        # # 检查过去50项，保证他们都小于stop_grad_norm。
        # for grad_norm in grad_norm_list[-50:]:
        #     if grad_norm > stop_grad_norm:
        #         bool_stop = False
        #         break

        if grad_clip_value > 0 and torch.norm(transform_matrix_raw.grad) > grad_clip_value * 0.95:
            bool_stop = False
        else:
            for grad_norm_derivative in grad_norm_derivative_list[-50:]:
                if abs(grad_norm_derivative) > stop_grad_norm:
                    bool_stop = False
                    break

        if last_iteration_num > 1000 and bool_stop:
            last_error = error
            least_index = max_iterations
            break
        else:
            last_error = error

    if least_index == max_iterations:
        transform_matrix = torch.concat([transform_matrix_raw, torch.tensor([[0, 0, 1]], dtype=torch.float32, requires_grad=False).cuda(0)], dim=0)
        transform_matrix = transform_matrix.cpu().detach().numpy()
        grad_norm = torch.norm(transform_matrix_raw.grad)
    else:
        transform_matrix = torch.concat([final_transform_matrix_raw, torch.tensor([[0, 0, 1]], dtype=torch.float32, requires_grad=False).cuda(0)], dim=0)
        transform_matrix = transform_matrix.cpu().detach().numpy()
        grad_norm = least_grad_norm
        last_iteration_num = least_index
    return transform_matrix, last_error, last_iteration_num, grad_norm

