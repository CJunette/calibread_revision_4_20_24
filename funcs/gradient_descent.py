import numpy as np
import torch
from funcs.util_functions import change_homogeneous_vector_to_2d_vector, change_2d_vector_to_homogeneous_vector
import configs


def gradient_descent_affine(point_pairs, weight, last_iteration_num, learning_rate=2e-1, max_iterations=2000, stop_grad_norm=5, grad_clip_value=0):
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
        error_value = error.cpu().detach().numpy().item()
        # penalty = torch.sum((transformed_points[:, 2] - target_points[:, 2]) ** 2) * 100
        # error += penalty
        error.backward()
        if grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(transform_matrix_raw, grad_clip_value)
        optimizer.step()

        # print(f"iteration: {iteration_index}, error: {error}, grad_norm: {torch.norm(transform_matrix_raw.grad)}")
        last_iteration_num = iteration_index
        # last_error = error_value
        if len(grad_norm_list) > 0:
            grad_norm_derivative_list.append(torch.norm(transform_matrix_raw.grad).cpu().detach() - grad_norm_list[-1])
        grad_norm_list.append(torch.norm(transform_matrix_raw.grad).cpu())

        if (final_transform_matrix_raw is None
                or (abs(error_value) < 1000 and error_value * torch.norm(transform_matrix_raw.grad) < least_error * least_grad_norm)):
            least_error = error_value
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

        if last_iteration_num > configs.gradient_descent_iteration_threshold and bool_stop:
            last_error = error_value
            least_index = max_iterations
            break
        else:
            last_error = error_value

    if least_index == max_iterations:
        transform_matrix = torch.concat([transform_matrix_raw, torch.tensor([[0, 0, 1]], dtype=torch.float32, requires_grad=False).cuda(0)], dim=0)
        transform_matrix = transform_matrix.cpu().detach().numpy()
        grad_norm = torch.norm(transform_matrix_raw.grad)
    elif least_index == 0:
        # 如果没有做任何迭代，那么返回一个单位矩阵。
        transform_matrix = np.eye(3)
        grad_norm = 0
        last_iteration_num = least_index
    else:
        transform_matrix = torch.concat([final_transform_matrix_raw, torch.tensor([[0, 0, 1]], dtype=torch.float32, requires_grad=False).cuda(0)], dim=0)
        transform_matrix = transform_matrix.cpu().detach().numpy()
        grad_norm = least_grad_norm
        last_iteration_num = least_index
    return transform_matrix, last_error, last_iteration_num, grad_norm


def gradient_descent_translate_rotate_shear_scale(point_pairs, weight, last_iteration_num, learning_rate=2e-1, max_iterations=2000, stop_grad_norm=5, grad_clip_value=0):
    torch.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    source_points = np.array([change_2d_vector_to_homogeneous_vector(point_pair[0]) for point_pair in point_pairs])
    target_points = np.array([change_2d_vector_to_homogeneous_vector(point_pair[1]) for point_pair in point_pairs])
    source_points = torch.tensor(source_points, dtype=torch.float32, requires_grad=False).cuda(0)
    target_points = torch.tensor(target_points, dtype=torch.float32, requires_grad=False).cuda(0)

    weight_tensor = torch.tensor(weight, dtype=torch.float32, requires_grad=False).unsqueeze(-1).cuda(0)

    theta = torch.tensor(0, dtype=torch.float32, requires_grad=True, device='cuda:0')
    tx = torch.tensor([0], dtype=torch.float32, requires_grad=True, device='cuda:0')
    ty = torch.tensor([0], dtype=torch.float32, requires_grad=True, device='cuda:0')
    sx = torch.tensor([1], dtype=torch.float32, requires_grad=True, device='cuda:0')
    sy = torch.tensor([1], dtype=torch.float32, requires_grad=True, device='cuda:0')
    shx = torch.tensor([0], dtype=torch.float32, requires_grad=True, device='cuda:0')
    shy = torch.tensor([0], dtype=torch.float32, requires_grad=True, device='cuda:0')

    theta = torch.nn.Parameter(theta)
    tx = torch.nn.Parameter(tx)
    ty = torch.nn.Parameter(ty)
    sx = torch.nn.Parameter(sx)
    sy = torch.nn.Parameter(sy)
    shx = torch.nn.Parameter(shx)
    shy = torch.nn.Parameter(shy)

    optimizer = torch.optim.Adam([theta, tx, ty, sx, sy, shx, shy], lr=learning_rate)

    # optimizer = torch.optim.SGD([transform_matrix_raw], lr=learning_rate)
    last_error = 1000000
    grad_norm_list = []
    grad_norm_derivative_list = [0]

    final_transform_matrix_raw = None
    least_error = 1000000
    least_grad_norm = 0
    least_index = 0 # 用least index来标记最终是否是主动收敛退出。
    for iteration_index in range(max_iterations):
        optimizer.zero_grad()

        translate_matrix_row_0 = torch.concat([torch.tensor([1, 0], dtype=torch.float32, requires_grad=False, device='cuda:0'), tx], dim=0).unsqueeze(0)
        translate_matrix_row_1 = torch.concat([torch.tensor([0, 1], dtype=torch.float32, requires_grad=False, device='cuda:0'), ty], dim=0).unsqueeze(0)
        translate_matrix = torch.concat([translate_matrix_row_0, translate_matrix_row_1, torch.tensor([0, 0, 1], dtype=torch.float32, requires_grad=False, device='cuda:0').unsqueeze(0)], dim=0)

        rotate_matrix_row_0 = torch.concat([torch.cos(theta).unsqueeze(0), (-torch.sin(theta)).unsqueeze(0), torch.tensor([0], dtype=torch.float32, requires_grad=False, device='cuda:0')], dim=0).unsqueeze(0)
        rotate_matrix_row_1 = torch.concat([torch.sin(theta).unsqueeze(0), torch.cos(theta).unsqueeze(0), torch.tensor([0], dtype=torch.float32, requires_grad=False, device='cuda:0')], dim=0).unsqueeze(0)
        rotate_matrix = torch.concat([rotate_matrix_row_0, rotate_matrix_row_1, torch.tensor([0, 0, 1], dtype=torch.float32, requires_grad=False, device='cuda:0').unsqueeze(0)], dim=0)

        scale_matrix_row_0 = torch.concat([sx, torch.tensor([0, 0], dtype=torch.float32, requires_grad=False, device='cuda:0')], dim=0).unsqueeze(0)
        scale_matrix_row_1 = torch.concat([torch.tensor([0], dtype=torch.float32, requires_grad=False, device='cuda:0'), sy, torch.tensor([0], dtype=torch.float32, requires_grad=False, device='cuda:0')], dim=0).unsqueeze(0)
        scale_matrix = torch.concat([scale_matrix_row_0, scale_matrix_row_1, torch.tensor([0, 0, 1], dtype=torch.float32, requires_grad=False, device='cuda:0').unsqueeze(0)], dim=0)

        shear_matrix_row_0 = torch.concat([torch.tensor([1], dtype=torch.float32, requires_grad=False, device='cuda:0'), shx, torch.tensor([0], dtype=torch.float32, requires_grad=False, device='cuda:0')], dim=0).unsqueeze(0)
        shear_matrix_row_1 = torch.concat([shy, torch.tensor([1, 0], dtype=torch.float32, requires_grad=False, device='cuda:0')], dim=0).unsqueeze(0)
        shear_matrix = torch.concat([shear_matrix_row_0, shear_matrix_row_1, torch.tensor([0, 0, 1], dtype=torch.float32, requires_grad=False, device='cuda:0').unsqueeze(0)], dim=0)

        transform_matrix = torch.matmul(translate_matrix, torch.matmul(rotate_matrix, torch.matmul(shear_matrix, scale_matrix)))
        transformed_points = torch.matmul(transform_matrix, source_points.transpose(0, 1)).transpose(0, 1)

        distance = transformed_points[:, :2] - target_points[:, :2]
        error_distance = torch.mean(torch.norm(distance) * weight_tensor)
        error_rotate = torch.square(theta / configs.theta_error_threshold) * configs.theta_error_ratio
        error_scale = (torch.square((sx - 1) / configs.scale_error_threshold) + torch.square((sy - 1) / configs.scale_error_threshold)) * configs.scale_ratio
        error_shear = (torch.square(shx / configs.shear_error_threshold) + torch.square(shy / configs.shear_error_threshold)) * configs.shear_ratio
        # error = error_distance + error_rotate + error_scale + error_shear
        error = error_distance

        error_value = error.cpu().detach().numpy().item()
        error.backward()
        optimizer.step()
        if grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_([theta, tx, ty, sx, sy, shx, shy], grad_clip_value)

        grad_norm = torch.abs(theta.grad) + torch.abs(tx.grad) + torch.abs(ty.grad) + torch.abs(sx.grad) + torch.abs(sy.grad) + torch.abs(shx.grad) + torch.abs(shy.grad)

        # print(f"iteration: {iteration_index}, error: {error}, grad_norm: {torch.norm(transform_matrix_raw.grad)}")
        last_iteration_num = iteration_index
        # last_error = error_value
        if len(grad_norm_list) > 0:
            grad_norm_derivative_list.append(grad_norm.cpu().detach() - grad_norm_list[-1])
        grad_norm_list.append(grad_norm.cpu().detach())

        if (final_transform_matrix_raw is None
                or (abs(error_value) < 1000 and error_value * grad_norm < least_error * least_grad_norm)):
            least_error = error_value
            final_transform_matrix_raw = transform_matrix.clone()
            least_grad_norm = grad_norm
            least_index = iteration_index

        bool_stop = True
        # # 检查过去50项，保证他们都小于stop_grad_norm。
        # for grad_norm in grad_norm_list[-50:]:
        #     if grad_norm > stop_grad_norm:
        #         bool_stop = False
        #         break

        if grad_clip_value > 0 and grad_norm > grad_clip_value * 0.95:
            bool_stop = False
        else:
            for grad_norm_derivative in grad_norm_derivative_list[-50:]:
                if abs(grad_norm_derivative) > stop_grad_norm:
                    bool_stop = False
                    break

        if last_iteration_num > configs.gradient_descent_iteration_threshold and bool_stop:
            last_error = error_value
            least_index = max_iterations
            break
        else:
            last_error = error_value

    if least_index == max_iterations:
        transform_matrix = transform_matrix.cpu().detach().numpy()
    elif least_index == 0:
        # 如果没有做任何迭代，那么返回一个单位矩阵。
        transform_matrix = np.eye(3)
        grad_norm = 0
        last_iteration_num = least_index
    else:
        transform_matrix = final_transform_matrix_raw.cpu().detach().numpy()
        grad_norm = least_grad_norm
        last_iteration_num = least_index
    return transform_matrix, last_error, last_iteration_num, grad_norm
