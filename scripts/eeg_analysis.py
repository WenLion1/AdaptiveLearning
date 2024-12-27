import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


def compute_and_plot_dissimilarity_matrices(eeg_data,
                                            save_path,
                                            trial_range=None,
                                            time_range=None):
    """
    计算 EEG 数据中每个时间点的 trial 之间的不相似性矩阵，并保存图像。

    参数:
    - eeg_data: ndarray, EEG 数据，形状为 (channels, time_points, trials)
    - save_path: str, 保存不相似性矩阵图像的目录
    - trial_range: tuple, 指定要处理的 trial 范围 (start, end)，例如 (0, 100)。默认为 None，表示所有 trial
    - time_range: tuple, 指定要处理的时间点范围 (start, end)。默认为 None，表示所有时间点
    """
    # 检查保存路径是否存在，如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 数据维度 (channels, time_points, trials)
    channels, time_points, trials = eeg_data.shape

    # 提取指定 trial 范围的数据
    if trial_range is not None:
        start_trial, end_trial = trial_range
        eeg_data = eeg_data[:, :, start_trial:end_trial]
        trials = end_trial - start_trial  # 更新 trial 数量

    # 提取指定的时间点范围
    if time_range is not None:
        start_time, end_time = time_range
        eeg_data = eeg_data[:, start_time:end_time, :]
        time_points = end_time - start_time  # 更新时间点数量

    # 遍历每个时间点
    for t in range(time_points):
        # 提取当前时间点的所有 trial 数据，形状为 (channels, trials)
        data_at_time_t = eeg_data[:, t, :]  # shape: (channels, trials)

        # 转置数据，使得每一列表示一个 trial，形状变为 (trials, channels)
        data_at_time_t = data_at_time_t.T

        # 计算成对欧几里得距离
        condensed_dist_matrix = pdist(data_at_time_t, metric='euclidean')
        dissimilarity_matrix = squareform(condensed_dist_matrix)

        # 绘制不相似性矩阵的热图
        plt.figure(figsize=(8, 6))
        plt.imshow(dissimilarity_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(label='Dissimilarity')

        # 设置坐标轴标签为从底部0到顶部（trial数量-1）
        plt.xticks(ticks=np.arange(trials), labels=np.arange(trials))
        plt.yticks(ticks=np.arange(trials), labels=np.arange(trials))  # 正常排列标签
        plt.gca().invert_yaxis()  # 确保热图内容和坐标标签一致

        plt.title(f"Dissimilarity Matrix at Time Point {t}")
        plt.xlabel("Trial")
        plt.ylabel("Trial")

        # 保存图像
        save_file = os.path.join(save_path, f"dissimilarity_matrix_time_{t:03d}.png")
        plt.savefig(save_file, dpi=300)
        plt.close()  # 关闭图像，释放内存

    print(f"All dissimilarity matrices have been saved to {save_path}")


def compute_eeg_model_rdm_correlation(eeg_data,
                                      model_rdm,
                                      save_path,
                                      trial_range=None,
                                      time_range=None):
    """
    对 EEG 数据的每个时间点生成不相似性矩阵（RDM），
    并与给定模型的隐藏层 RDM 进行相关分析，保存相关矩阵。

    参数:
    - eeg_data: ndarray, EEG 数据，形状为 (channels, time_points, trials)
    - model_rdm: ndarray, 模型隐藏层 RDM，形状为 (trials, trials)
    - save_path: str, 保存相关矩阵的目录
    - trial_range: tuple, 指定要处理的 trial 范围 (start, end)，例如 (0, 100)。默认为 None，表示所有 trial
    - time_range: tuple, 指定要处理的时间点范围 (start, end)。默认为 None，表示所有时间点
    """
    # 检查保存路径是否存在，如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 数据维度 (channels, time_points, trials)
    channels, time_points, trials = eeg_data.shape

    # 提取指定 trial 范围的数据
    if trial_range is not None:
        start_trial, end_trial = trial_range
        eeg_data = eeg_data[:, :, start_trial:end_trial]
        # model_rdm = model_rdm[start_trial:end_trial, start_trial:end_trial]  # 裁剪模型 RDM
        trials = end_trial - start_trial  # 更新 trial 数量

    # 提取指定的时间点范围
    if time_range is not None:
        start_time, end_time = time_range
        eeg_data = eeg_data[:, start_time:end_time, :]
        time_points = end_time - start_time  # 更新时间点数量

    print(f"Shape of model_rdm: {model_rdm.shape}")
    print(f"Number of trials: {trials}")
    # 验证模型 RDM 的尺寸是否匹配
    if model_rdm.shape != (trials * trials / 2 - trials / 2,):
        raise ValueError("The model RDM must have the shape (trials, trials) matching the trial range.")

    # 创建存储相关值的数组
    correlations = np.zeros(time_points)

    # 遍历每个时间点
    for t in range(time_points):
        # 提取当前时间点的所有 trial 数据，形状为 (channels, trials)
        data_at_time_t = eeg_data[:, t, :]  # shape: (channels, trials)

        # 转置数据，使得每一列表示一个 trial，形状变为 (trials, channels)
        data_at_time_t = data_at_time_t.T
        # 计算 EEG 的 RDM（不相似性矩阵），并转换为向量形式
        condensed_dist_matrix = pdist(data_at_time_t, metric='euclidean')
        eeg_rdm_vector = condensed_dist_matrix
        condensed_dist_matrix = squareform(condensed_dist_matrix)

        corr, _ = pearsonr(eeg_rdm_vector, model_rdm)
        correlations[t] = corr

    # 保存相关矩阵为图片
    plt.figure(figsize=(10, 6))
    plt.plot(correlations, label='Correlation with Model RDM', color='b')
    plt.xlabel('Time Points')
    plt.ylabel('Correlation (r)')
    plt.title('EEG RDM vs Model RDM Correlation Over Time')
    plt.legend()
    plt.grid(True)
    save_file = os.path.join(save_path, 'eeg_model_rdm_correlation.png')
    plt.savefig(save_file, dpi=300)
    plt.close()

    # 保存相关值矩阵为文件
    np.save(os.path.join(save_path, 'eeg_model_rdm_correlation.npy'), correlations)

    print(f"Correlation matrix saved to {save_path}")


def generate_model_hidden_by_eeg(hidden_path,
                                 epoch_numbers,
                                 save_path=None,
                                 type="CP"):
    """
    根据EEG数据去除的trials，在模型的hidden中去除相应的trials

    :param hidden_path: 隐藏层矩阵文件路径 (假设为.npy文件)
    :param epoch_numbers: 保留的trial编号列表
    :param save_path: 处理后的hidden保存路径
    :param type: 当前epoch_number所属类型 ("CP" 或 "OB")
    """

    # 加载hidden矩阵
    hidden = torch.load(hidden_path)
    if type == "CP":

        # 调整 epoch_numbers：减去 241，去除小于241的值
        adjusted_epoch_numbers = (epoch_numbers[epoch_numbers >= 241] - 241).tolist()

        # 将hidden矩阵裁剪到只保留adjusted_epoch_numbers中对应的行
        adjusted_hidden = hidden[adjusted_epoch_numbers, :]

    elif type == "OB":
        pass

    # 检查是否有数据被保留
    if adjusted_hidden.size == 0:
        raise ValueError("调整后的hidden矩阵为空，请检查输入的 epoch_numbers 和类型是否正确")

    # 保存结果
    if save_path:
        torch.save(adjusted_hidden, save_path)
        print(f"处理后的hidden矩阵已保存到: {save_path}")
    else:
        print("未提供保存路径，结果未保存")

    return adjusted_hidden


def batch_generate_model_hidden_by_eeg(hidden_folder_path,
                                       save_root_path,
                                       extracted_eeg_data,
                                       type="CP"):
    """
    批量调用 generate_model_hidden_by_eeg 方法，处理 hidden_path 文件夹中的所有子文件夹。

    参数:
    - hidden_folder_path: str, hidden 文件的根目录。
    - save_root_path: str, 保存生成结果的根目录。
    - extracted_eeg_data: dict, 提取的 EEG 数据，key 为文件名，value 为对应的 epochNumbers。
    - type: str, 默认为 "CP"，生成模型时的固定参数。
    """

    # 遍历 hidden_folder_path 下的所有子文件夹
    for subdir, _, files in os.walk(hidden_folder_path):
        # 筛选出文件名包含 "CP" 的 .pt 文件
        pt_files = [f for f in files if f.endswith('.pt') and "CP" in f]

        if not pt_files:
            print(f"在文件夹 {subdir} 中未找到包含 'CP' 的 .pt 文件，跳过该文件夹。")
            continue

        # 获取当前子文件夹的名称
        subfolder_name = os.path.basename(subdir)
        # 找到 EEG 文件夹中对应的 epoch_numbers
        matched_eeg_file = next((key for key in extracted_eeg_data.keys() if subfolder_name in key), None)
        if not matched_eeg_file:
            print(f"未找到与子文件夹 {subfolder_name} 匹配的 EEG 文件，跳过该文件夹。")
            continue

        epoch_numbers = extracted_eeg_data[matched_eeg_file]['fields']['epochNumbers']

        # 创建保存路径
        save_folder_path = os.path.join(save_root_path, subfolder_name, "remove")
        os.makedirs(save_folder_path, exist_ok=True)

        # 遍历当前子文件夹中的每个符合条件的 .pt 文件
        for pt_file in pt_files:
            hidden_path = os.path.join(subdir, pt_file)
            save_path = os.path.join(save_folder_path, pt_file)

            print(f"正在处理文件: {hidden_path}")
            print(f"保存路径: {save_path}")

            # 调用生成隐藏状态的方法
            generate_model_hidden_by_eeg(hidden_path=hidden_path,
                                         epoch_numbers=epoch_numbers,
                                         save_path=save_path,
                                         type=type)

    print("批量处理完成。")


def read_mat_files_from_folder(folder_path,
                               fields_to_extract):
    """
    从指定文件夹内读取所有 .mat 文件，并提取 EEG 数据和指定字段。

    :param folder_path: str, 文件夹路径
    :param fields_to_extract: list, 需要提取的字段列表，内容为字符串
    :return: dict, 以文件名为键，提取结果为值的字典
    """
    # 初始化结果字典
    results = {}

    # 获取文件夹内所有 .mat 文件
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    for mat_file in mat_files:
        file_path = os.path.join(folder_path, mat_file)
        try:
            # 加载 .mat 文件
            mat_contents = loadmat(file_path)

            # 提取 EEG 数据
            if 'EEG' in mat_contents and 'data' in mat_contents['EEG'].dtype.names:
                eeg_data = mat_contents['EEG']['data'][0, 0]  # 提取 EEG 数据
            else:
                eeg_data = None

            # 提取指定字段
            extracted_fields = {}
            for field in fields_to_extract:
                if field in mat_contents:
                    extracted_fields[field] = mat_contents[field]
                else:
                    extracted_fields[field] = None

            # 将文件名作为键，提取的内容作为值存储在结果字典中
            results[mat_file] = {
                'eeg_data': eeg_data,
                'fields': extracted_fields
            }

        except Exception as e:
            print(f"Error reading file {mat_file}: {e}")
            continue

    return results


def batch_compute_eeg_model_rdm_correlation(extracted_eeg_data,
                                            model_path,
                                            results_folder_path,
                                            time_range, ):
    """
    批量计算 EEG 数据与模型 RDM 的相关性，trial_range 自动根据 epochNumbers 调整。

    参数:
    - folder_path: str, EEG 数据所在的文件夹路径
    - results_folder_path: str, 结果保存的顶层文件夹路径
    - time_range: tuple, 处理的时间点范围
    - fields_to_extract: list, 提取的字段，默认为 ["epochNumbers"]
    """

    # 遍历文件夹内的所有数字子文件夹
    for subfolder in sorted(os.listdir(model_path)):
        subfolder_path = os.path.join(model_path, subfolder)
        if not os.path.isdir(subfolder_path) or not subfolder.isdigit():
            continue

        print(f"Processing folder: {subfolder}")

        # 获取 RDM 文件路径
        rdm_folder = os.path.join(subfolder_path, "rdm")
        if not os.path.exists(rdm_folder):
            print(f"No RDM folder in {subfolder}. Skipping...")
            continue

        # 找到所有 .npy 文件
        rdm_files = [f for f in os.listdir(rdm_folder) if f.endswith(".npy")]
        if not rdm_files:
            print(f"No RDM .npy files in {rdm_folder}. Skipping...")
            continue

        for rdm_file in rdm_files:
            model_rdm_path = os.path.join(rdm_folder, rdm_file)
            model_rdm = np.load(model_rdm_path)

            # 构造 EEG 数据文件名
            eeg_filename = f"lal-hc-{subfolder}-task.mat"
            if eeg_filename not in extracted_eeg_data:
                print(f"{eeg_filename} not found in extracted EEG data. Skipping...")
                continue

            eeg_data = extracted_eeg_data[eeg_filename]["eeg_data"]
            epoch_numbers = extracted_eeg_data[eeg_filename]["fields"]["epochNumbers"][0]

            # 自动确定 trial_range
            start_trial = next((i for i, x in enumerate(epoch_numbers) if x > 240), None)
            if start_trial is None:
                print(f"No valid trial_range found for {eeg_filename}. Skipping...")
                continue
            trial_range = (start_trial, len(epoch_numbers))

            # 构造结果保存路径
            save_folder = os.path.join(results_folder_path, subfolder)
            os.makedirs(save_folder, exist_ok=True)

            # 调用计算相关性的函数
            compute_eeg_model_rdm_correlation(
                eeg_data=eeg_data,
                model_rdm=model_rdm,
                save_path=save_folder,
                trial_range=trial_range,
                time_range=time_range
            )

            print(f"Processed: {model_rdm_path}, trial_range: {trial_range}")

    print("All folders processed.")


def plot_npy_from_subfolders(folder_path,
                             saving_path):
    """
    读取文件夹中所有子文件夹内的 .npy 文件，将它们的内容绘制到一张图中并保存。

    参数:
    - folder_path: str, 顶层文件夹路径，包含子文件夹
    - saving_path: str, 保存绘制结果的路径
    """
    plt.figure(figsize=(12, 8))  # 设置图像大小
    legend_labels = []  # 用于存储图例标签

    # 遍历文件夹内的所有子文件夹
    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # 跳过非文件夹

        # 获取子文件夹内的 .npy 文件
        npy_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]
        if not npy_files:
            print(f"No .npy files in {subfolder_path}. Skipping...")
            continue

        # 逐个处理 .npy 文件
        for npy_file in npy_files:
            file_path = os.path.join(subfolder_path, npy_file)
            try:
                # 加载 .npy 文件
                data = np.load(file_path)
                if not (isinstance(data, np.ndarray) and data.ndim == 1):
                    print(f"Invalid data format in {file_path}. Skipping...")
                    continue

                # 绘制曲线
                plt.plot(data, label=f"{subfolder}/{npy_file}")
                legend_labels.append(f"{subfolder}/{npy_file}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

    # 设置图例和标题
    # plt.legend(legend_labels, loc='upper right', fontsize='small', bbox_to_anchor=(1.1, 1))
    plt.title("Visualization of .npy Files from Subfolders")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)

    # 保存图像
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    plt.savefig(saving_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved at {saving_path}")


if __name__ == "__main__":
    # # 读取 .mat 文件
    # mat_file_path = '../data/eeg/hc/lal-hc-405-task.mat'  # 替换为你的 .mat 文件路径
    # mat_contents = loadmat(mat_file_path)
    #
    # # 查看文件的顶级结构
    # print(mat_contents.keys())
    #
    # # 提取 eeg 字段
    # if 'EEG' in mat_contents:
    #     eeg_data = mat_contents['EEG']['data'][0, 0]  # 提取 data 字段
    #     print("EEG Data Shape:", eeg_data.shape)  # 打印数据的形状
    # else:
    #     print("eeg 字段未找到")
    #
    # if 'epochNumbers' in mat_contents:
    #     epoch_number = mat_contents['epochNumbers']
    #     print("epochNumbers Shape: ", epoch_number.shape)
    # else:
    #     print("epochNumber 字段未找到")

    # extracted_eeg_data = read_mat_files_from_folder(folder_path="../data/eeg/hc",
    #                                                 fields_to_extract=["epochNumbers"])

    # compute_and_plot_dissimilarity_matrices(eeg_data=eeg_data,
    #                                         save_path="../results/eeg/403/rdm/CP",
    #                                         trial_range=(250, 350), )

    # generate_model_hidden_by_eeg(hidden_path="../hidden/sub/405/rnn_layers_1_hidden_16_input_489_CP.pt",
    #                              epoch_numbers=epoch_number,
    #                              save_path="../hidden/sub/405/remove/rnn_layers_1_hidden_16_input_489_CP.pt",
    #                              type="CP")

    # batch_generate_model_hidden_by_eeg(hidden_folder_path="../hidden/sub/hc",
    #                                    save_root_path="../hidden/sub/hc",
    #                                    extracted_eeg_data=extracted_eeg_data,
    #                                    type="CP", )

    # # 单独计算rdm
    # model_rdm = np.load("../results/numpy/model/sub/hc/405/rdm/rnn_layers_1_hidden_16_input_489_CP_228.npy")
    # # time_range = (0, 100)
    # time_range = (100, 351)
    #
    # compute_eeg_model_rdm_correlation(eeg_data=extracted_eeg_data['lal-hc-403-task.mat']['eeg_data'],
    #                                   model_rdm=model_rdm,
    #                                   save_path="../results/numpy/eeg_model/correlation/baseline",
    #                                   trial_range=(235, 463),
    #                                   time_range=time_range, )

    # # 批量计算rdm
    # batch_compute_eeg_model_rdm_correlation(extracted_eeg_data=extracted_eeg_data,
    #                                         model_path="../results/numpy/model/sub/hc",
    #                                         results_folder_path="../results/numpy/eeg_model/correlation",
    #                                         time_range=(100, 351))

    # 画所有的rdm相关图
    plot_npy_from_subfolders(folder_path="../results/numpy/eeg_model/correlation",
                             saving_path="../results/png/sub/hc/all/rdm_all_correlation.png")
