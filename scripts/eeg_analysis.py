import os
import re
from collections import defaultdict

import mne
import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from mne.channels import find_ch_adjacency
from mne.decoding import SlidingEstimator, cross_val_multiscore
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, ttest_1samp, zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
from sklearn.svm import LinearSVC
from twisted.python.util import println

from scripts.analysis import decode_hidden_eeg
from scripts.test import batch_evaluate
from scripts.train_valid import train_valid
from scripts.utils import angular_difference_deg, remove_rows_by_npy


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
    对 EEG 数据的每个时间点(减去均值后)生成不相似性矩阵（RDM），
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
        raise ValueError(
            "The model RDM must have the shape (trials * trials / 2 - trials / 2,) matching the trial range.")

    # 创建存储相关值的数组
    correlations = np.zeros(time_points)

    # 遍历每个时间点
    for t in range(time_points):
        # 提取当前时间点的所有 trial 数据，形状为 (channels, trials)
        data_at_time_t = eeg_data[:, t, :]  # shape: (channels, trials)

        # 转置数据，使得每一列表示一个 trial，形状变为 (trials, channels)
        data_at_time_t = data_at_time_t.T

        mean_per_trial = data_at_time_t.mean(axis=1, keepdims=True)
        std_per_trial = data_at_time_t.std(axis=1, keepdims=True)
        data_at_time_t = (data_at_time_t - mean_per_trial) / (std_per_trial + 1e-8)  # 加 1e-8 避免除以 0

        # 将数据向右平移，并将第一个元素设置为随机值
        data_at_time_t = np.roll(data_at_time_t, 1, axis=0)
        data_at_time_t[0, :] = np.random.rand(channels)

        # 计算 EEG 的 RDM（不相似性矩阵），并转换为向量形式
        condensed_dist_matrix = pdist(data_at_time_t, metric='euclidean')
        eeg_rdm_vector = condensed_dist_matrix

        # # 对 condensed_dist_matrix 进行标准化
        # scaler = StandardScaler()
        # eeg_rdm_vector = scaler.fit_transform(condensed_dist_matrix.reshape(-1, 1)).flatten()

        corr, _ = spearmanr(eeg_rdm_vector, model_rdm)
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
        raise NotImplementedError("OB 类型尚未实现")

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


# def batch_compute_eeg_model_rdm_correlation(extracted_eeg_data,
#                                             model_path,
#                                             results_folder_path,
#                                             time_range, ):
#     """
#     批量计算 EEG 数据与模型 RDM 的相关性，trial_range 自动根据 epochNumbers 调整。
#
#     参数:
#     - folder_path: str, EEG 数据所在的文件夹路径
#     - results_folder_path: str, 结果保存的顶层文件夹路径
#     - time_range: tuple, 处理的时间点范围
#     - fields_to_extract: list, 提取的字段，默认为 ["epochNumbers"]
#     """
#
#     # 遍历文件夹内的所有数字子文件夹
#     for subfolder in sorted(os.listdir(model_path)):
#         subfolder_path = os.path.join(model_path, subfolder)
#         if not os.path.isdir(subfolder_path) or not subfolder.isdigit():
#             continue
#
#         print(f"Processing folder: {subfolder}")
#
#         # 获取 RDM 文件路径
#         rdm_folder = os.path.join(subfolder_path, "rdm")
#         if not os.path.exists(rdm_folder):
#             print(f"No RDM folder in {subfolder}. Skipping...")
#             continue
#
#         # 找到所有 .npy 文件
#         rdm_files = [f for f in os.listdir(rdm_folder) if f.endswith(".npy")]
#         if not rdm_files:
#             print(f"No RDM .npy files in {rdm_folder}. Skipping...")
#             continue
#
#         for rdm_file in rdm_files:
#             model_rdm_path = os.path.join(rdm_folder, rdm_file)
#             model_rdm = np.load(model_rdm_path)
#
#             # 构造 EEG 数据文件名
#             eeg_filename = f"lal-hc-{subfolder}-task.mat"
#             if eeg_filename not in extracted_eeg_data:
#                 print(f"{eeg_filename} not found in extracted EEG data. Skipping...")
#                 continue
#
#             eeg_data = extracted_eeg_data[eeg_filename]["eeg_data"]  # (99, 600, 458)
#             epoch_numbers = extracted_eeg_data[eeg_filename]["fields"]["epochNumbers"][0]
#
#             # 自动确定 trial_range
#             start_trial = next((i for i, x in enumerate(epoch_numbers) if x > 240), None)
#             if start_trial is None:
#                 print(f"No valid trial_range found for {eeg_filename}. Skipping...")
#                 continue
#             trial_range = (start_trial, len(epoch_numbers))
#
#             # 构造结果保存路径
#             save_folder = os.path.join(results_folder_path, subfolder)
#             os.makedirs(save_folder, exist_ok=True)
#
#             # 调用计算相关性的函数
#             compute_eeg_model_rdm_correlation(
#                 eeg_data=eeg_data,
#                 model_rdm=model_rdm,
#                 save_path=save_folder,
#                 trial_range=trial_range,
#                 time_range=time_range
#             )
#
#             print(f"Processed: {model_rdm_path}, trial_range: {trial_range}")
#
#     print("All folders processed.")

def computer_eeg_rdm(eeg_data,
                     save_path,
                     fig_save_path=None,
                     metric='correlation',
                     save_time=200,
                     is_number_label=False, ):
    """
    计算eeg的rdm

    :param fig_save_path:
    :param metric:
    :param eeg_data: (sub, trial, channel, time)
    :param save_path:
    :return:
    """

    sub_num, trial_num, channel_num, time_num = eeg_data.shape
    rdm_results = np.zeros((sub_num, time_num, trial_num * (trial_num - 1) // 2))  # 存储结果

    for sub in range(sub_num):
        println("sub: ", sub)
        for t in range(time_num):
            # 取出当前被试、当前时间点的数据 (trial × channel)
            trial_features = eeg_data[sub, :, :, t]  # (trial, channel)

            # 计算 RDM
            rdm = pdist(trial_features, metric=metric)  # 计算 pairwise 距
            rdm_results[sub, t] = rdm  # 存储到结果矩阵

            if t == save_time and fig_save_path is not None:
                dissimilarity_matrix = squareform(rdm)
                # 绘制热图
                plt.figure(figsize=(10, 8))
                plt.imshow(dissimilarity_matrix, cmap="viridis", aspect="auto")
                plt.colorbar(label="Dissimilarity")

                if is_number_label:
                    # 如果需要在坐标轴显示坐标数字
                    plt.xticks(ticks=np.arange(dissimilarity_matrix.shape[1]),
                               labels=np.arange(dissimilarity_matrix.shape[1]) + 1)
                    plt.yticks(ticks=np.arange(dissimilarity_matrix.shape[0]),
                               labels=np.arange(dissimilarity_matrix.shape[0]) + 1)
                else:
                    # 如果不需要显示坐标数字
                    plt.xticks([])
                    plt.yticks([])

                plt.gca().invert_yaxis()  # 反转纵轴
                plt.title("Dissimilarity Matrix")
                plt.xlabel("Trial")
                plt.ylabel("Trial")

                fig_save = os.path.join(fig_save_path, f"{sub}_{save_time}.png")
                plt.savefig(fig_save, dpi=300)
    np.save(save_path, rdm_results)


def plot_npy_from_subfolders(folder_path,
                             saving_path,
                             threshold=0.008):
    """
    读取文件夹中所有子文件夹内的 .npy 文件，将它们的内容绘制到一张图中，
    并标记高相关性时间段（平均值超过阈值），结果保存。

    参数:
    - folder_path: str, 顶层文件夹路径，包含子文件夹
    - saving_path: str, 保存绘制结果的路径
    - threshold: float, 用于标记高相关性时间段的平均值阈值
    """
    plt.figure(figsize=(12, 8))  # 设置图像大小
    legend_labels = []  # 用于存储图例标签
    all_data = []  # 用于存储所有文件的数据

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

                all_data.append(data)  # 收集数据以计算平均值
                # 绘制曲线
                plt.plot(data, alpha=0.5, label=f"{subfolder}/{npy_file}")
                legend_labels.append(f"{subfolder}/{npy_file}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

    # 计算每个时间点的平均值
    if all_data:
        all_data = np.array(all_data)
        mean_values = all_data.mean(axis=0)

        # 打印每个时间点的平均值
        print("Time Point Average Values:")
        print(mean_values)

        # 绘制平均值曲线
        plt.plot(mean_values, color='black', linewidth=2, label='Mean')

        # 标记高相关性时间段
        high_corr_indices = np.where(mean_values > threshold)[0]
        if len(high_corr_indices) > 0:
            plt.fill_between(
                range(len(mean_values)),
                mean_values,
                where=mean_values > threshold,
                color='red',
                alpha=0.3,
                label=f"High Correlation (>{threshold})"
            )

    # 图例和标题
    plt.title("Visualization of .npy Files from Subfolders")
    plt.xlabel("Time Point")
    plt.ylabel("Value")
    plt.grid(True)

    # 保存图像
    plt.savefig(saving_path)
    plt.show()


def analyze_significant_time_points(data,
                                    popmean=0,
                                    threshold=None,
                                    alpha=0.05,
                                    n_permutations=1000):
    """
    分析多个序列在不同时间点上的显著性，并通过集群置换检验进行多重比较校正。

    参数:
    - data: ndarray, 输入数据，形状为 (样本数, 时间点数)
    - popmean: float, 零假设下的均值（默认值为 0）
    - threshold: float, 集群置换检验的阈值
    - alpha: float, 显著性水平（默认值为 0.05）
    - n_permutations: int, 集群置换检验的置换次数

    返回:
    - T_obs: ndarray, 每个时间点的统计值（t 值）
    - clusters: list of slices, 每个显著集群的时间范围
    - cluster_p_values: ndarray, 每个集群的 p 值
    """

    # Step 1: 单样本 t 检验
    t_values, _ = ttest_1samp(data, popmean=popmean, axis=0)

    # Step 2: 集群置换检验
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [data], n_permutations=n_permutations, threshold=threshold, tail=1
    )
    print("T_obs: ", T_obs)
    print("clusters: ", clusters)
    print("cluster_p_values: ", cluster_p_values)
    print("H0: ", H0)
    asdasdad

    # Step 3: 可视化结果
    times = np.arange(data.shape[1])  # 时间点
    plt.figure(figsize=(10, 6))
    plt.plot(times, T_obs, label="T-values", color="b")

    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] <= alpha:  # 显著集群
            plt.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3, label="Significant Cluster")

    plt.xlabel("Time Points")
    plt.ylabel("T-values")
    plt.title("Significant Time Points Across Sequences")
    plt.legend()
    plt.grid(True)
    plt.show()

    return T_obs, clusters, cluster_p_values


def load_and_concatenate_npy_files(folder_path):
    """
    读取指定文件夹及其子文件夹中的所有 .npy 文件，并将这些文件中的数组合并成一个二维数组。

    :param folder_path: 包含 .npy 文件的文件夹路径
    :return: 合并后的二维数组
    """
    # 初始化一个空列表来存储所有数组
    all_arrays = []

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                # 构建完整的文件路径
                file_path = os.path.join(root, file)
                # 加载 .npy 文件中的数组
                array = np.load(file_path)
                # 将数组添加到列表中
                all_arrays.append(array)

    # 将所有数组合并成一个二维数组，每个一维数组作为一行
    combined_data = np.vstack(all_arrays)

    return combined_data


# 检验函数
def find_significant_periods(data, alpha=0.05, threshold=0.8):
    """
    找出时间段里大部分被试值显著大于0。

    参数：
    - data: numpy.ndarray, 形状为 (n_subjects, n_timepoints)
    - alpha: 显著性水平，默认0.05
    - threshold: 被试比例阈值，默认80%

    返回：
    - significant_timepoints: numpy.ndarray, 形状为 (n_timepoints,)
    """
    n_subjects = data.shape[0]
    significant_timepoints = []

    for t in range(data.shape[1]):
        # 单样本t检验
        t_stat, p_value = ttest_1samp(data[:, t], popmean=0, alternative='greater')

        # 判断显著性（p值小于 alpha）并统计显著被试比例
        if p_value < alpha:
            proportion_significant = np.sum(data[:, t] > 0) / n_subjects
            if proportion_significant >= threshold:
                significant_timepoints.append(t)

    significant_points = np.array(significant_timepoints)

    # 可视化
    timepoints = np.arange(data.shape[1])  # 时间点
    mean_values = np.mean(data, axis=0)  # 每个时间点的平均值

    plt.figure(figsize=(12, 6))

    # 绘制均值曲线
    plt.plot(timepoints, mean_values, label='Mean values', color='blue')

    # 标记显著时间点
    plt.scatter(significant_points, mean_values[significant_points],
                color='red', label='Significant timepoints', zorder=5)

    # 添加横线标记 0 位置
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # 图例和标题
    plt.title('Significant Timepoints Visualization', fontsize=16)
    plt.xlabel('Timepoints', fontsize=14)
    plt.ylabel('Mean Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    plt.show()

    return significant_points


def batch_compute_eeg_model_rdm_correlation(eeg_rdm,
                                            model_rdm,
                                            time_range,
                                            time_min,
                                            time_max,
                                            re_threhold,
                                            is_every=0,
                                            rsa_save_path='../rsa_results', ):  # 新增参数 is_every
    """
    批量计算 EEG 数据与模型 RDM 的相关性，trial_range 自动根据 epochNumbers 调整。

    :param eeg_rdm: EEG 的 RDM 数据
    :param model_rdm: 模型的 RDM 数据
    :param time_range: 时间范围
    :param time_min: 最小时间点
    :param time_max: 最大时间点
    :param re_threhold: 显著性检验的阈值
    :param is_every: 是否绘制每个被试的曲线 (1: 是, 0: 否)
    :return:
    """

    sub_num = eeg_rdm.shape[0]
    rsa_results_all = []  # 用于存储所有被试的 RSA 结果

    for s in range(sub_num):
        rsa_results = np.zeros((sub_num, time_range[1] - time_range[0] + 1))
        print("leave-one-out sub number :", s)  # 修复 println 为 print
        submat = np.delete(np.arange(sub_num), s)

        for z in submat:
            sub_eeg_data = eeg_rdm[z, :, :]  # (851, 114960)
            sub_model_data = model_rdm[z, :]  # (114960,)
            sub_model_data_normalized = zscore(sub_model_data)
            print(sub_model_data)

            for t in range(time_range[0], time_range[1] + 1):
                sub_eeg_data_cat = sub_eeg_data[t, :]
                sub_eeg_data_normalized = zscore(sub_eeg_data_cat)

                spearman_corr, p_value = spearmanr(sub_model_data_normalized, sub_eeg_data_normalized)
                rsa_results[z, t - time_range[0]] = spearman_corr

        # 创建画布
        plt.figure(figsize=(6, 5))
        time_points = np.linspace(time_min, time_max, num=rsa_results.shape[1])

        # 如果 is_every 为 1，则绘制每个被试的相关性曲线
        if is_every == 1:
            for z in range(sub_num):
                plt.plot(time_points, rsa_results[z, :], color='gray', linewidth=0.7, alpha=0.5,
                         label='_nolegend_')

        # 将所有被试的 RSA 结果保存为一个 NumPy 文件
        rsa_results = rsa_results[~np.all(rsa_results == 0, axis=1)]  # 去掉0的那一行
        rsa_results_all.append(rsa_results)  # 记录每个被试的 RSA 结果

        # 计算平均相关系数曲线
        mean_corr = np.mean(rsa_results, axis=0)
        sem_corr = np.std(rsa_results, axis=0) / np.sqrt(rsa_results.shape[0])

        # 绘制主曲线（所有被试平均）
        plt.plot(time_points, mean_corr, color='red', linewidth=2.5, label='Mean Correlation')
        plt.fill_between(time_points, mean_corr - sem_corr, mean_corr + sem_corr, color='red', alpha=0.3)

        # 绘制零线和竖直虚线
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

        # 选择 tail 值
        tail = 1 if re_threhold > 0 else -1

        # 进行显著性检验
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(rsa_results,
                                                                         threshold=re_threhold,
                                                                         n_permutations=5000,
                                                                         tail=tail)

        significant_clusters = np.where(cluster_pv < 0.05)[0]

        for i_clu, p_val in enumerate(cluster_pv):
            time_indices = clusters[i_clu][0]
            sig_times = time_points[time_indices]

            # 判断显著性
            significance = "显著" if p_val < 0.05 else "不显著"

            print(f"Cluster {i_clu}: 时间范围 {sig_times.min():.3f} ms ~ {sig_times.max():.3f} ms, "
                  f"p = {p_val:.5f} ({significance})")

        # 画显著性点
        for i_clu in significant_clusters:
            time_indices = clusters[i_clu][0]
            sig_times = time_points[time_indices]
            print(
                f"Cluster {i_clu}: 时间范围 {sig_times.min():.3f} ms ~ {sig_times.max():.3f} ms, p = {cluster_pv[i_clu]:.5f}")
            plt.scatter(sig_times, np.full_like(sig_times, mean_corr.min()), color='red', s=5)

        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Coefficient', fontsize=12)
        plt.title('Time-resolved RSA Correlation with Model RDM', fontsize=14)

        # 自动调整坐标轴范围
        plt.xlim(time_points.min(), time_points.max())
        # plt.ylim(-0.02, 0.1)
        # 自动调整 Y 轴范围，留出一些边距
        y_min = np.min(rsa_results) - 0.02
        y_max = np.max(rsa_results) + 0.02
        plt.ylim(y_min, y_max)

        # 处理图例，避免重复标签
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.grid(alpha=0.3)
        plt.tight_layout()

        # 保存图像
        plt.savefig(f"rsa_correlation_{s}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # rsa_results_all = np.array(rsa_results_all)
    # np.save(rsa_save_path, rsa_results_all)


def batch_compute_eeg_model_rdm_correlation_remove(eeg_rdm_folder,
                                                   epoch_data_path,
                                                   model_rdm_folder,
                                                   time_min,
                                                   time_max,
                                                   re_threshold,
                                                   rsa_save_path,
                                                   time_range=None,
                                                   is_every=0,
                                                   png_save_path=None,
                                                   type="time",
                                                   one_time=527,
                                                   montage_type=98,
                                                   model_type="combine", ):
    """
    批量计算 EEG 数据与模型 RDM 的相关性，并绘制显著性结果。

    :param model_type:
    :param montage_type:
    :param epoch_data_path:
    :param one_time:
    :param type: eeg rdm是以什么维度计算出来的
    :param png_save_path:
    :param eeg_rdm_folder: str, EEG RDM 文件夹路径。
    :param model_rdm_folder: str, 模型 RDM 文件夹路径。
    :param time_min: float, 最小时间点。
    :param time_max: float, 最大时间点。
    :param re_threshold: float, 显著性检验阈值。
    :param time_range: tuple, 需要计算相关性的时间范围 (start_time, end_time)，单位与time_min、time_max一致。
    :param is_every: bool, 是否绘制每个被试的曲线。
    :param rsa_save_path: str, 保存 RSA 结果的路径。
    """

    # 获取所有 EEG 和模型的被试编号
    eeg_subjects = sorted(
        {int(re.match(r"(\d+)_", d).group(1)) for d in os.listdir(eeg_rdm_folder)
         if os.path.isdir(os.path.join(eeg_rdm_folder, d)) and re.match(r"(\d+)_", d)},
        key=int
    )
    eeg_subjects_yuan = sorted(
        [d for d in os.listdir(eeg_rdm_folder) if os.path.isdir(os.path.join(eeg_rdm_folder, d))])
    model_subjects = sorted(
        [int(d) for d in os.listdir(model_rdm_folder) if os.path.isdir(os.path.join(model_rdm_folder, d))])

    print(eeg_subjects_yuan)
    print(eeg_subjects)
    print(model_subjects)

    # 筛选出两者都存在的被试编号
    common_subjects = list(set(eeg_subjects) & set(model_subjects))
    print(common_subjects)
    if not common_subjects:
        print("没有找到匹配的被试编号，请检查文件夹内容。")
        return

    print(f"找到 {len(common_subjects)} 个匹配的被试编号。")
    #
    # # 初始化 RSA 结果矩阵
    # if type == "time":
    #     # 初始化 RSA 结果矩阵
    #     rsa_results_all = np.zeros((len(common_subjects), time_range[1] - time_range[0]))
    #     time_dur = time_range[1] - time_range[0]
    #
    #     # 遍历被试
    #     for i, sub in enumerate(common_subjects):
    #         print(f"正在处理被试 {sub} ({i + 1}/{len(common_subjects)})")
    #
    #         # 加载 EEG RDM 和模型 RDM
    #         eeg_rdm_path = os.path.join(eeg_rdm_folder, str(sub) + "_clean", 'rdm')
    #         model_rdm_path = os.path.join(model_rdm_folder, str(sub), 'rdm', 'remove')
    #
    #         eeg_rdm_file = [f for f in os.listdir(eeg_rdm_path) if f.endswith('_by_time_rdm.npy')][0]
    #         if model_type == "combine":
    #             model_rdm_file = [f for f in os.listdir(model_rdm_path) if f.endswith('combine_processed.npy')][0]
    #         elif model_type == "reverse":
    #             model_rdm_file = [f for f in os.listdir(model_rdm_path) if f.endswith('reverse_processed.npy')][0]
    #
    #         eeg_rdm = np.load(os.path.join(eeg_rdm_path, eeg_rdm_file))
    #         model_rdm = np.load(os.path.join(model_rdm_path, model_rdm_file))
    #
    #         # 检查数据维度
    #         if eeg_rdm.shape[1] != model_rdm.shape[0]:
    #             raise ValueError(f"维度不匹配: EEG RDM {eeg_rdm.shape}, 模型 RDM {model_rdm.shape}")
    #
    #         # 初始化当前被试的 RSA 结果
    #         rsa_results = np.zeros((time_range[1] - time_range[0],))
    #
    #         # 逐时间点计算相关性
    #         for t in range(time_range[0], time_range[1]):
    #             eeg_rdm_t = eeg_rdm[t, :]
    #             model_rdm_z = zscore(model_rdm)
    #
    #             eeg_rdm_z = zscore(eeg_rdm_t)
    #
    #             spearman_corr, _ = spearmanr(model_rdm_z, eeg_rdm_z)
    #             rsa_results[t - time_range[0]] = spearman_corr
    #
    #         rsa_results_all[i, :] = rsa_results

    if type == "time":
        time_dur = time_range[1] - time_range[0]

        for sub in common_subjects:
            print(f"正在处理被试 {sub}")
            eeg_rdm_path = os.path.join(eeg_rdm_folder, str(sub) + "_clean", 'rdm')
            model_rdm_path = os.path.join(model_rdm_folder, str(sub), 'rdm', 'remove')

            eeg_rdm_file = [f for f in os.listdir(eeg_rdm_path) if f.endswith('_clean_rdm.npy')][0]
            eeg_rdm = np.load(os.path.join(eeg_rdm_path, eeg_rdm_file))

            model_rdm_files = [f for f in os.listdir(model_rdm_path) if f.endswith('_processed.npy')]

            for model_file in model_rdm_files:
                save_dir = os.path.join(rsa_save_path, str(sub))
                os.makedirs(save_dir, exist_ok=True)

                save_file = os.path.join(save_dir, model_file.replace(".npy", "_by_time_rsa.npy"))

                # 如果 save_file 已经存在，则跳过
                if os.path.exists(save_file):
                    print(f"文件 {save_file} 已存在，跳过处理。")
                    continue

                model_rdm = np.load(os.path.join(model_rdm_path, model_file))

                if eeg_rdm.shape[1] != model_rdm.shape[0]:
                    raise ValueError(f"维度不匹配: EEG RDM {eeg_rdm.shape}, 模型 RDM {model_rdm.shape}")

                rsa_results = np.zeros((time_dur,))

                for t in range(time_range[0], time_range[1]):
                    eeg_rdm_z = zscore(eeg_rdm[t, :])
                    model_rdm_z = zscore(model_rdm)
                    spearman_corr, _ = spearmanr(model_rdm_z, eeg_rdm_z)
                    rsa_results[t - time_range[0]] = spearman_corr

                np.save(save_file, rsa_results)

        # 获取 rsa_save_path 下的所有子文件夹
        sub_folders = [d for d in os.listdir(rsa_save_path) if os.path.isdir(os.path.join(rsa_save_path, d))]

        # 遍历每个子文件夹
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(rsa_save_path, sub_folder)
            model_files = [f for f in os.listdir(sub_folder_path) if f.endswith("_by_time_rsa.npy")]

            # 提取所有文件的前缀
            model_base_names = set(f.replace("_by_time_rsa.npy", "") for f in model_files)

            # 遍历每个前缀
            for model_base_name in model_base_names:
                # 保存为新的文件
                all_save_file = os.path.join(rsa_save_path, model_base_name + "_all.npy")
                # 如果 save_file 已经存在，则跳过
                if os.path.exists(all_save_file):
                    print(f"文件 {all_save_file} 已存在，跳过处理。")
                    continue
                # 在所有子文件夹中找到具有相同前缀的文件
                all_files = []
                for sub_folder in sub_folders:
                    sub_folder_path = os.path.join(rsa_save_path, sub_folder)
                    all_files.extend([os.path.join(sub_folder_path, f) for f in os.listdir(sub_folder_path) if
                                      f.startswith(model_base_name) and f.endswith("_by_time_rsa.npy")])

                # 如果没有找到文件，跳过
                if not all_files:
                    continue

                # 将相同前缀的文件加载到一个数组中
                rsa_results_all = [np.load(f) for f in all_files]
                rsa_results_all = np.array(rsa_results_all)

                np.save(all_save_file, rsa_results_all)

                # 如果有 png_save_path，则调用 plot_permutation_test
                if png_save_path:
                    png_file = os.path.join(png_save_path, model_base_name + "_all_by_time_rsa.png")
                    print(png_file)
                    plot_permutation_test(
                        rsa_result_path=all_save_file,
                        time_min=time_min,
                        time_max=time_max,
                        time_dur=time_dur,
                        re_threshold=re_threshold,
                        png_save_path=png_file,
                        epoch_data_path=epoch_data_path,
                        is_every=is_every,
                        type="time",
                        montage_type=montage_type
                    )
    elif type == "channel" or type == "one_time":
        # 初始化 RSA 结果矩阵
        rsa_results_all = np.zeros((len(common_subjects), montage_type))

        # 遍历被试
        for i, sub in enumerate(common_subjects):
            print(f"正在处理被试 {sub} ({i + 1}/{len(common_subjects)})")

            # 加载 EEG RDM 和模型 RDM
            eeg_rdm_path = os.path.join(eeg_rdm_folder, str(sub) + "_clean", 'rdm')
            model_rdm_path = os.path.join(model_rdm_folder, str(sub), 'rdm', 'remove')

            if type == "channel":
                eeg_rdm_file = [f for f in os.listdir(eeg_rdm_path) if f.endswith('_by_channel_rdm.npy')][0]
            elif type == "one_time":
                eeg_rdm_file = [f for f in os.listdir(eeg_rdm_path) if f'_one_time_{one_time}' in f][0]
                print(eeg_rdm_file)

            if model_type == "combine":
                model_rdm_file = [f for f in os.listdir(model_rdm_path) if f.endswith('combine_processed.npy')][0]
            elif model_type == "reverse":
                model_rdm_file = [f for f in os.listdir(model_rdm_path) if f.endswith('reverse_processed.npy')][0]

            eeg_rdm = np.load(os.path.join(eeg_rdm_path, eeg_rdm_file))
            model_rdm = np.load(os.path.join(model_rdm_path, model_rdm_file))

            # 检查数据维度
            if eeg_rdm.shape[1] != model_rdm.shape[0]:
                raise ValueError(f"维度不匹配: EEG RDM {eeg_rdm.shape}, 模型 RDM {model_rdm.shape}")

            # 初始化当前被试的 RSA 结果
            rsa_results = np.zeros((montage_type,))

            # 逐时间点计算相关性
            for c in range(montage_type):
                eeg_rdm_t = eeg_rdm[c, :]
                model_rdm_z = zscore(model_rdm)
                eeg_rdm_z = zscore(eeg_rdm_t)

                spearman_corr, _ = spearmanr(model_rdm_z, eeg_rdm_z)
                rsa_results[c] = spearman_corr

            rsa_results_all[i, :] = rsa_results

    if type != "time":
        # 保存 RSA 结果
        np.save(rsa_save_path, rsa_results_all)

    if type == "time":
        # plot_permutation_test(rsa_result_path=rsa_save_path,
        #                       time_min=time_min,
        #                       time_max=time_max,
        #                       time_dur=time_dur,
        #                       re_threshold=re_threshold,
        #                       png_save_path=png_save_path,
        #                       epoch_data_path=epoch_data_path,
        #                       is_every=is_every,
        #                       type="time",
        #                       montage_type=montage_type)
        pass
    elif type == "channel":
        plot_permutation_test(rsa_result_path=rsa_save_path,
                              time_min=time_min,
                              time_max=time_max,
                              time_dur=None,
                              re_threshold=re_threshold,
                              png_save_path=png_save_path,
                              epoch_data_path=epoch_data_path,
                              is_every=is_every,
                              type="channel",
                              montage_type=montage_type)
    elif type == "one_time":
        plot_permutation_test(rsa_result_path=rsa_save_path,
                              time_min=time_min,
                              time_max=time_max,
                              time_dur=time_dur,
                              re_threshold=re_threshold,
                              png_save_path=png_save_path,
                              epoch_data_path=epoch_data_path,
                              is_every=is_every,
                              type="one_time",
                              montage_type=montage_type)


def plot_permutation_test(rsa_result_path,
                          time_min,
                          time_max,
                          time_dur,
                          re_threshold,
                          png_save_path=None,
                          epoch_data_path=None,
                          is_every=1,
                          rsa_result2_path=None,
                          type="time",
                          montage_type=64):
    rsa_result = np.load(rsa_result_path)
    if rsa_result2_path is not None:
        rsa_result2 = np.load(rsa_result2_path)

    # 时间点转换
    if type == "time":
        axis = np.linspace(time_min, time_max, num=time_dur)
        rsa_results_all = rsa_result
    elif type == "channel" or type == "one_time":
        axis = np.arange(1, montage_type + 1)
        rsa_results_all = rsa_result
    elif type == "combine":
        axis = np.linspace(time_min, time_max, num=time_dur)
        rsa_results_all = np.concatenate([rsa_result, rsa_result2], axis=0)

    # 绘图
    plt.figure(figsize=(8, 6))

    # 绘制所有被试曲线
    if is_every == 1:
        for sub_data in rsa_results_all:
            plt.plot(axis, sub_data, color='gray', alpha=0.3)

    # 计算均值和标准误
    mean_corr = np.mean(rsa_results_all, axis=0)
    sem_corr = np.std(rsa_results_all, axis=0) / np.sqrt(rsa_results_all.shape[0])

    plt.plot(axis, mean_corr, color='red', label='Mean Correlation')
    plt.fill_between(axis, mean_corr - sem_corr, mean_corr + sem_corr, color='red', alpha=0.3)

    if type == "time" or type == "combine":
        # 进行显著性检验
        tail = 1 if re_threshold > 0 else -1
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(rsa_results_all,
                                                                         threshold=re_threshold,
                                                                         n_permutations=5000,
                                                                         tail=tail, )
    elif type == "channel":
        if montage_type == 98:
            epoch = mne.read_epochs(epoch_data_path)
            sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(epoch.info, "eeg")
        elif montage_type == 64:
            # 加载 MNE 的标准 montage（脑电帽布局）
            montage = mne.channels.make_standard_montage('biosemi64')  # 或 'standard_1020'，视实际而定

            info = mne.create_info(ch_names=montage.ch_names, sfreq=500, ch_types='eeg')
            info.set_montage(montage)

            # 获取通道邻接矩阵（稀疏矩阵形式）和通道名
            sensor_adjacency, ch_names = find_ch_adjacency(info, ch_type='eeg')

        # 进行显著性检验
        tail = 1 if re_threshold > 0 else -1
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(rsa_results_all,
                                                                         threshold=re_threshold,
                                                                         adjacency=sensor_adjacency,
                                                                         n_permutations=5000,
                                                                         tail=tail, )

    significant_clusters = np.where(cluster_pv < 0.05)[0]

    for i_clu in significant_clusters:
        time_indices = clusters[i_clu][0]
        sig_times = axis[time_indices]
        plt.scatter(sig_times, np.full_like(sig_times, mean_corr.min()), color='blue', s=5)
        print(f"显著性区域 {i_clu}: {sig_times.min():.3f} ms - {sig_times.max():.3f} ms, p = {cluster_pv[i_clu]:.5f}")

    # 绘制辅助线
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

    if type == "time":
        plt.xlabel('Time (ms)')
    elif type == "channel" or type == "one_time":
        plt.xlabel('Channel')
    plt.ylabel('Correlation Coefficient')
    plt.title('Time-resolved RSA Correlation with Model RDM')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    # 保存图像
    if png_save_path is not None:
        plt.savefig(png_save_path, dpi=300)
        print("相关性图像已保存。")
    plt.close()


def computer_eeg_rdm_remove(folder_path,
                            save_base_path,
                            fig_save_path=None,
                            metric='correlation',
                            save_time=200,
                            is_number_label=False,
                            time_range=None,
                            type="time",
                            one_time=527, ):
    """
    计算一个文件夹下所有 EEG 数据的 RDM，并保存结果。

    :param time_range:
    :param one_time:
    :param type: 根据什么维度计算rdm
    :param folder_path: str, 包含 EEG 数据的 .npy 文件的文件夹路径。
    :param save_base_path: str, 保存 RDM 结果的基础路径。
    :param fig_save_path: str, 保存 RDM 图像的路径。若为 None，则不保存图像。
    :param metric: str, 距离度量方法，默认为欧几里得距离。
    :param save_time: int, 指定绘图的时间点。
    :param is_number_label: bool, 是否在图像中显示数字标签。
    """

    # 获取所有 .npy 文件
    npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    if not npy_files:
        print("未找到任何 .npy 文件，退出程序。")
        return

    # 遍历每个 .npy 文件
    for npy_file in npy_files:
        # 加载 EEG 数据
        eeg_data = np.load(os.path.join(folder_path, npy_file), mmap_mode="r")
        epoch_num, channel_num, time_num = eeg_data.shape
        print(f"处理 {npy_file}，形状: {eeg_data.shape}")

        # 创建存储路径
        sub_save_path = os.path.join(save_base_path, os.path.splitext(npy_file)[0], "rdm")
        os.makedirs(sub_save_path, exist_ok=True)
        if fig_save_path:
            sub_fig_path = os.path.join(fig_save_path, os.path.splitext(npy_file)[0], "rdm")
            os.makedirs(sub_fig_path, exist_ok=True)

        # 初始化 RDM 结果
        if type == "time":
            if time_range is None:
                rdm_results = np.zeros((time_num, epoch_num * (epoch_num - 1) // 2))
                for t in range(time_num):
                    print(f"  时间点: {t + 1}/{time_num}")
                    trial_features = eeg_data[:, :, t]  # (epoch, channel)
                    rdm = pdist(trial_features, metric=metric)
                    rdm_results[t] = rdm
            else:
                start_index, end_index = time_range  # time_range 是一个形如 (start, end) 的索引元组
                time_window_len = end_index - start_index
                rdm_results = np.zeros((time_window_len, epoch_num * (epoch_num - 1) // 2))
                for i, t in enumerate(range(start_index, end_index)):
                    print(f"  时间点: {t + 1}/{time_window_len}")
                    trial_features = eeg_data[:, :, t]  # (epoch, channel)
                    rdm = pdist(trial_features, metric=metric)
                    rdm_results[i] = rdm
        elif type == "channel":
            rdm_results = np.zeros((channel_num, epoch_num * (epoch_num - 1) // 2))

            for c in range(channel_num):
                print(f"  电极点: {c + 1}/{channel_num}")
                # 提取当前时间点的数据 (epoch × time)

                trial_features = eeg_data[:, c, :]  # (epoch, time)
                if time_range is not None:
                    trial_features = trial_features[:, time_range[0]:time_range[1]]

                # 计算 RDM
                rdm = pdist(trial_features, metric=metric)  # 计算所有 epoch 的 pairwise 距离
                rdm_results[c] = rdm  # 存储到结果矩阵
        elif type == "one_time":
            rdm_results = np.zeros((channel_num, epoch_num * (epoch_num - 1) // 2))

            print(f"  时间点: {one_time}")
            trial_features = eeg_data[:, :, one_time]

            for c in range(channel_num):
                print(f"  电极点: {c + 1}/{channel_num}")
                # 提取当前时间点的数据 (epoch × time)
                eeg_data_2 = trial_features[:, c]  # (epoch, time)
                eeg_data_2 = eeg_data_2[:, np.newaxis]
                # 计算 RDM
                rdm = pdist(eeg_data_2, metric=metric)  # 计算所有 epoch 的 pairwise 距离
                rdm_results[c] = rdm  # 存储到结果矩阵
        else:
            print("无此维度！")
            return

        # 保存 RDM 结果
        if type == "time":
            save_path = os.path.join(sub_save_path, f"{os.path.splitext(npy_file)[0]}_by_time_rdm.npy")
        elif type == "channel":
            save_path = os.path.join(sub_save_path, f"{os.path.splitext(npy_file)[0]}_by_channel_rdm.npy")
        elif type == "one_time":
            save_path = os.path.join(sub_save_path, f"{os.path.splitext(npy_file)[0]}_one_time_{one_time}_rdm.npy")
        np.save(save_path, rdm_results)
        print(f"已保存 RDM 到 {save_path}")

    print("所有文件处理完成。")


def plot_topomap_by_correlation(correlation_array_path,
                                save_path=None,
                                vmin=None,
                                vmax=None,
                                montage_type=98, ):
    """
    根据相关值画脑部图
    :param montage_type:
    :param correlation_array_path:
    :param save_path:
    :param vmin:
    :param vmax:
    :return:
    """
    if montage_type == 98:
        # 加载 epochs（为了获取通道空间信息）
        epochs = mne.read_epochs("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/ob/405_clean.fif")
        info = epochs.info
    elif montage_type == 64:
        # 加载 MNE 的标准 montage（脑电帽布局）
        montage = mne.channels.make_standard_montage('biosemi64')  # 或 'standard_1020'，视实际而定

        info = mne.create_info(ch_names=montage.ch_names, sfreq=500, ch_types='eeg')
        info.set_montage(montage)

    # 加载相关性数据并计算平均（被试平均）
    data = np.load(correlation_array_path)  # shape: (n_subjects, n_channels)
    mean_data = np.mean(data, axis=0)  # shape: (n_channels,)

    # 获取 EEG 通道名
    eeg_picks = mne.pick_types(info, eeg=True, eog=False, stim=False, misc=False)
    eeg_ch_names = [info.ch_names[i] for i in eeg_picks]

    # 打印每个通道的值，方便调试
    for i in range(len(mean_data)):
        print("channel: ", eeg_ch_names[i])
        print("mean_data: ", mean_data[i])

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    # 画 topomap 并获取 image 对象（用于 colorbar）
    im, _ = mne.viz.plot_topomap(mean_data, info, ch_type='eeg', cmap='RdBu_r',
                                 names=eeg_ch_names, show=False, contours=0, axes=ax)
    if vmin is None or vmax is None:
        im.set_clim()
    else:
        im.set_clim(vmin=vmin, vmax=vmax)

    # 添加颜色条（colorbar），用于指示数值范围
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Correlation value", fontsize=12)

    # 保存图像
    if save_path:
        fig.savefig(save_path, dpi=300)
        print("图像已保存到:", save_path)
    else:
        plt.show()


def ridge_regression_eeg(hidden_data,
                         eeg_data,
                         train_ratio=0.8,
                         n_splits=10,
                         alpha=1.0,
                         save_path=None,
                         time_range=(100, 850)):
    """
    使用 Ridge Regression 对 EEG 数据进行回归分析并预测。

    参数:
    - hidden_data: ndarray, (n_trials, n_hidden) 模型隐藏层数据
    - eeg_data: ndarray, (n_trials, n_channels, n_times) EEG 数据
    - train_ratio: float, 训练集比例
    - n_splits: int, 数据划分的次数
    - alpha: float, Ridge 正则化系数
    - save_path: str, 保存结果的路径

    返回:
    - predicted_eeg: ndarray, (n_splits, n_test_trials, n_channels, n_times) 预测的 EEG 数据
    - coefs: ndarray, (n_splits, n_channels, n_hidden) 回归系数
    - mse_scores: ndarray, (n_splits, n_channels, n_times) 均方误差
    - corr_scores: ndarray, (n_splits, n_channels, n_times) 相关系数
    """
    n_trials, n_channels, _ = eeg_data.shape
    n_times = time_range[1] - time_range[0]
    n_hidden = hidden_data.shape[1]

    all_predicted_eeg = []
    test_all = []
    coefs = np.zeros((n_splits, n_times, n_channels, n_hidden))

    # 评估指标
    mse_train_scores = np.zeros((n_splits, n_channels, n_times))
    mse_test_scores = np.zeros((n_splits, n_channels, n_times))
    corr_train_scores = np.zeros((n_splits, n_channels, n_times))
    corr_test_scores = np.zeros((n_splits, n_channels, n_times))

    for split in range(n_splits):
        print(f"第 {split + 1}/{n_splits} 次数据划分和训练")

        # 数据划分
        train_idx, test_idx = train_test_split(np.arange(n_trials), train_size=train_ratio, random_state=split)
        test_all.append(test_idx)
        n_test_trials = len(test_idx)
        predicted_eeg = np.zeros((n_test_trials, n_channels, n_times))

        model = Ridge(alpha=alpha)

        for t in range(time_range[0], time_range[1]):
            eeg_train = eeg_data[train_idx, :, t]
            eeg_test = eeg_data[test_idx, :, t]

            # Ridge 回归训练
            model.fit(hidden_data[train_idx], eeg_train)
            coefs[split, t - time_range[0], :, :] = model.coef_

            # EEG 预测
            predicted_eeg[:, :, t - time_range[0]] = model.predict(hidden_data[test_idx])

            # # 训练集评估
            # eeg_train_pred = model.predict(hidden_data[train_idx])  # 训练集预测
            # mse_train_scores[split, :, t] = np.mean((eeg_train - eeg_train_pred) ** 2, axis=0)  # 训练集 MSE
            # for ch in range(n_channels):
            #     corr_train_scores[split, ch, t], _ = pearsonr(eeg_train[:, ch], eeg_train_pred[:, ch])
            #
            # # 测试集评估
            # mse_test_scores[split, :, t] = np.mean((eeg_test - predicted_eeg[:, :, t]) ** 2, axis=0)  # 测试集 MSE
            # for ch in range(n_channels):
            #     corr_test_scores[split, ch, t], _ = spearmanr(eeg_test[:, ch], predicted_eeg[:, ch, t])

        all_predicted_eeg.append(predicted_eeg)

    print("回归和预测完成！")

    # # 相关性转换为百分比
    # corr_train_scores *= 100
    # corr_test_scores *= 100
    #
    # print("训练集上的 MSE:", mse_train_scores.mean(axis=(0, 1, 2)))
    # print("测试集上的 MSE:", mse_test_scores.mean(axis=(0, 1, 2)))
    # print("训练集上的相关性 (%):", corr_train_scores.mean(axis=(0, 1, 2)))  # 百分比格式
    # print("测试集上的相关性 (%):", corr_test_scores.mean(axis=(0, 1, 2)))  # 百分比格式

    # 保存结果
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "predicted_eeg.npy"), np.array(all_predicted_eeg))
        np.save(os.path.join(save_path, "ridge_coefs.npy"), coefs)
        print(f"结果已保存至 {save_path}")

    return np.array(all_predicted_eeg), coefs, np.array(test_all)


def ridge_regression_eeg_by_eeg_rsa(true_eeg_data,
                                    pre_eeg_data,
                                    test_all,
                                    metric="correlation",
                                    time_range=(100, 850), ):
    """
    计算预测rdm和实际rdm的rsa矩阵
    :param time_range:
    :param true_eeg_data: 实际eeg数据
    :param pre_eeg_data: 模型预测eeg数据
    :param test_all: 截取的测试集
    :param metric: pdist方法
    :return:
    """
    n_split, n_test_trials, n_channels, n_times = pre_eeg_data.shape

    rsa_results = np.zeros((n_split, n_times))
    for split in range(n_split):

        test_now = test_all[split]
        true_eeg_now = true_eeg_data[test_now]
        pre_eeg_now = pre_eeg_data[split]

        for t in range(n_times):
            # 提取当前时间点的数据 (epoch × channel)
            true_trial_features = true_eeg_now[:, :, t + time_range[0]]  # (epoch, channel)
            pre_trial_features = pre_eeg_now[:, :, t]

            # 计算 RDM
            true_rdm = pdist(true_trial_features, metric=metric)  # 计算所有 epoch 的 pairwise 距离
            # true_rdm_results[t] = true_rdm  # 存储到结果矩阵
            pre_rdm = pdist(pre_trial_features, metric=metric)  # 计算所有 epoch 的 pairwise 距离
            # pre_rdm_results[t] = pre_rdm  # 存储到结果矩阵

            true_rdm = zscore(true_rdm)
            pre_rdm = zscore(pre_rdm)

            spearman_corr, _ = spearmanr(true_rdm, pre_rdm)
            rsa_results[split, t] = spearman_corr

    return rsa_results


def r2_score_channel_time_encoding_base_rsa(true_eeg_data,
                                            pre_eeg_data,
                                            test_all,
                                            time_range=(100, 850), ):
    """
    根据true_eeg_data和pre_eeg_data计算r2_score(对单个被试)
    :param time_range:
    :param true_eeg_data:
    :param pre_eeg_data:
    :param test_all:
    :return:
    """

    n_split, n_test_trials, n_channels, n_times = pre_eeg_data.shape
    r2_all_score = np.zeros((n_split, n_channels, n_times))

    for split in range(n_split):

        test_now = test_all[split]
        true_eeg_now = true_eeg_data[test_now]
        pre_eeg_now = pre_eeg_data[split]

        for t in range(n_times):
            # 提取当前时间点的数据 (epoch × channel)
            true_trial_features = true_eeg_now[:, :, t + time_range[0]]  # (epoch, channel)
            pre_trial_features = pre_eeg_now[:, :, t]

            r2_pre_channel = r2_score(true_trial_features, pre_trial_features, multioutput="raw_values")  # (channel,)
            r2_all_score[split, :, t] = r2_pre_channel

    r2_all_score_mean = np.mean(r2_all_score, axis=0)
    return r2_all_score_mean


def permutation_cluster_r2_score(r2_sub_score,
                                 epoch_data_path,
                                 n_permutations=50, ):
    """
    根据r2_score来计算channel和time上的permutation_cluster

    :param n_permutations:
    :param r2_sub_score:
    :param epoch_data_path:
    :return:
    """
    n_sub, n_channel, n_time = r2_sub_score.shape
    r2_sub_score_mean = np.mean(r2_sub_score, axis=1)

    # 数据必须是 2D -> reshape 成 (n_observations, n_features)
    X = r2_sub_score_mean

    # cluster test：只有时间维度，不需要空间 adjacency
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X,
        n_permutations=n_permutations,
        threshold=2,
        tail=1,  # 正向单尾
        out_type="mask",
        verbose=True,
    )

    # 可视化 T 值和显著的 cluster
    significant_mask = np.zeros_like(T_obs, dtype=bool)
    for c, p_val in enumerate(cluster_p_values):
        if p_val < 0.05:
            significant_mask |= clusters[c]

    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(X, axis=0), label='Mean R² across subjects')
    plt.fill_between(np.arange(n_time), 0, np.max(X), where=significant_mask,
                     color='red', alpha=0.3, label='Significant cluster (p < 0.05)')
    plt.xlabel("Time")
    plt.ylabel("R² score")
    plt.title("Permutation cluster test across time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    epoch = mne.read_epochs(epoch_data_path)
    sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(epoch.info, "eeg")

    adjacency = mne.stats.combine_adjacency(
        sensor_adjacency, n_time
    )  # (time*channel, time*channel)

    tail = 1
    # degrees_of_freedom = n_sub - 1
    # t_thresh = scipy.stats.t.ppf(1-0.001, df=degrees_of_freedom)
    t_thresh = 2
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        r2_sub_score,
        n_permutations=n_permutations,
        threshold=t_thresh,
        tail=tail,
        adjacency=adjacency,
        out_type="mask",
        verbose=True,
    )

    significant_mask = np.zeros_like(T_obs, dtype=bool)

    # 假设我们设定 p < 0.05 为显著
    for c, p_val in enumerate(cluster_p_values):
        if p_val < 0.05:
            significant_mask |= clusters[c]
    plt.figure(figsize=(10, 5))
    plt.imshow(
        T_obs,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-np.max(np.abs(T_obs)),
        vmax=np.max(np.abs(T_obs)),
    )
    plt.colorbar(label="T-values")
    plt.contour(significant_mask, colors="black", linewidths=0.5)
    plt.title("T-statistics with significant clusters")
    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.show()


def plot_encoding_base_rsa(rsa_results_all,
                           time_min=-1,
                           time_max=0.5,
                           is_every=1,
                           re_threshold=2,
                           type="single",
                           time_range=(100, 850),
                           png_save_path=None,
                           png_save_name=None, ):
    n_sub, n_split, n_time = rsa_results_all.shape
    axis = np.linspace(time_min, time_max, num=time_range[1] - time_range[0])

    if type == "single":
        for i in range(n_sub):
            rsa_results = rsa_results_all[i]
            # 绘图
            plt.figure(figsize=(8, 6))

            # 绘制所有被试曲线
            if is_every == 1:
                for split in rsa_results:
                    plt.plot(axis, split, color='gray', alpha=0.3)

            # 计算均值和标准误
            mean_corr = np.mean(rsa_results, axis=0)
            sem_corr = np.std(rsa_results, axis=0) / np.sqrt(n_split)

            plt.plot(axis, mean_corr, color='red', label='Mean Correlation')
            plt.fill_between(axis, mean_corr - sem_corr, mean_corr + sem_corr, color='red', alpha=0.3)

            # 进行显著性检验
            tail = 1 if re_threshold > 0 else -1
            t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(rsa_results,
                                                                             threshold=re_threshold,
                                                                             n_permutations=5000,
                                                                             tail=tail)

            significant_clusters = np.where(cluster_pv < 0.05)[0]

            for i_clu in significant_clusters:
                time_indices = clusters[i_clu][0]
                sig_times = axis[time_indices]
                plt.scatter(sig_times, np.full_like(sig_times, mean_corr.min()), color='blue', s=5)
                print(
                    f"显著性区域 {i_clu}: {sig_times.min():.3f} ms - {sig_times.max():.3f} ms, p = {cluster_pv[i_clu]:.5f}")

            # 绘制辅助线
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

            plt.xlabel('Time (ms)')
            plt.ylabel('Correlation Coefficient')
            plt.title(f'encoding base rsa sub:{i}')
            plt.legend(loc='upper right')
            plt.grid(alpha=0.3)

            # 保存图像
            if png_save_path is not None:
                save_name = os.path.join(png_save_path, f"encoding_base_rsa_sub_{i}.png")
                plt.savefig(save_name, dpi=300)
                print("相关性图像已保存。")
            plt.show()
            plt.close()
    elif type == "all":
        rsa_results_sub = np.zeros((n_sub, n_time))
        for i in range(n_sub):
            rsa_results = rsa_results_all[i]
            mean_corr = np.mean(rsa_results, axis=0)

            rsa_results_sub[i] = mean_corr
        # 绘图
        plt.figure(figsize=(8, 6))

        # 绘制所有被试曲线
        if is_every == 1:
            for split in rsa_results_sub:
                plt.plot(axis, split, color='gray', alpha=0.3)

        # 计算均值和标准误
        mean_corr = np.mean(rsa_results_sub, axis=0)
        sem_corr = np.std(rsa_results_sub, axis=0) / np.sqrt(n_split)

        plt.plot(axis, mean_corr, color='red', label='Mean Correlation')
        plt.fill_between(axis, mean_corr - sem_corr, mean_corr + sem_corr, color='red', alpha=0.3)

        # 进行显著性检验
        tail = 1 if re_threshold > 0 else -1
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(rsa_results_sub,
                                                                         threshold=re_threshold,
                                                                         n_permutations=5000,
                                                                         tail=tail)

        significant_clusters = np.where(cluster_pv < 0.05)[0]

        for i_clu in significant_clusters:
            time_indices = clusters[i_clu][0]
            sig_times = axis[time_indices]
            plt.scatter(sig_times, np.full_like(sig_times, mean_corr.min()), color='blue', s=5)
            print(
                f"显著性区域 {i_clu}: {sig_times.min():.3f} ms - {sig_times.max():.3f} ms, p = {cluster_pv[i_clu]:.5f}")

        # 绘制辅助线
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

        plt.xlabel('Time (ms)')
        plt.ylabel('Correlation Coefficient')
        plt.title(f'encoding base rsa sub:{i}')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        # 保存图像
        if png_save_path is not None and png_save_name is not None:
            save_name = os.path.join(png_save_path, png_save_name)
            plt.savefig(save_name, dpi=300)
            print("相关性图像已保存。")
        plt.show()
        plt.close()


def encoding_base_rsa(eeg_data_folder,
                      hidden_data_folder,
                      epoch_data_path,
                      time_range=(100, 850),
                      metrix="correlation",
                      train_ratio=0.8,
                      n_splits=10,
                      n_permutation=1000,
                      alpha=1.0,
                      plot_type="all",
                      type="r2_score",
                      predicted_eeg_coef_save_path=None,
                      rsa_results_save_path=None,
                      png_save_path=None,
                      png_save_name=None, ):
    eeg_subjects = sorted(
        {int(re.match(r"(\d+)_", d).group(1)) for d in os.listdir(eeg_data_folder)
         if os.path.isfile(os.path.join(eeg_data_folder, d)) and re.match(r"(\d+)_", d)},
        key=int
    )
    hidden_subjects = sorted(
        [int(d) for d in os.listdir(hidden_data_folder) if os.path.isdir(os.path.join(hidden_data_folder, d))])

    common_subjects = list(set(eeg_subjects) & set(hidden_subjects))
    print(eeg_subjects)
    print(hidden_subjects)
    if not common_subjects:
        print("没有找到匹配的被试编号，请检查文件夹内容。")
        return

    print(f"找到 {len(common_subjects)} 个匹配的被试编号。")

    rsa_results_all = np.zeros((len(common_subjects), n_splits, time_range[1] - time_range[0]))
    r2_sub_score = []

    for i, sub in enumerate(common_subjects):
        print("正在处理被试:", sub)
        eeg_data = os.path.join(eeg_data_folder, str(sub) + "_clean.npy")
        hidden_data = os.path.join(hidden_data_folder, str(sub),
                                   "remove/rnn_layers_1_hidden_16_input_489_combine_processed.npy")

        eeg_data = np.load(eeg_data)
        hidden_data = np.load(hidden_data)

        all_predicted_eeg, coefs, test_all = ridge_regression_eeg(
            hidden_data=hidden_data,
            eeg_data=eeg_data,
            n_splits=n_splits,
            time_range=time_range,
            train_ratio=train_ratio,
            alpha=alpha,
            save_path=predicted_eeg_coef_save_path,
        )  # all_predicted_egg: (split, test_trial, channel, time)

        if type == "r2_score":
            r2_all_score_mean = r2_score_channel_time_encoding_base_rsa(true_eeg_data=eeg_data,
                                                                        pre_eeg_data=all_predicted_eeg,
                                                                        test_all=test_all,
                                                                        time_range=time_range, )
            r2_sub_score.append(r2_all_score_mean)
        elif type == "ridge_regression":

            rsa_results = ridge_regression_eeg_by_eeg_rsa(true_eeg_data=eeg_data,
                                                          pre_eeg_data=all_predicted_eeg,
                                                          test_all=test_all,
                                                          metric=metrix,
                                                          time_range=time_range, )
            rsa_results_all[i] = rsa_results
        else:
            print("此方法暂无此type")
            return

    if type == "ridge_regression":
        plot_encoding_base_rsa(rsa_results_all=rsa_results_all,
                               type=plot_type,
                               png_save_path=png_save_path,
                               png_save_name=png_save_name, )
    elif type == "r2_score":
        r2_sub_score = np.array(r2_sub_score)
        permutation_cluster_r2_score(r2_sub_score=r2_sub_score,
                                     epoch_data_path=epoch_data_path,
                                     n_permutations=n_permutation, )


def batch_decode_hidden_eeg(hidden_eeg_path_folder,
                            csv_path_folder,
                            output_csv_path="decode_results.csv",
                            key="outcome_label_remove",
                            type="sub",
                            time_range=(100, 852),
                            is_random=0, ):
    """
    批量运行 decode_hidden_eeg 并记录每个文件的准确率。

    :param is_random:
    :param hidden_eeg_path_folder: .npy 或 .pt 文件所在的文件夹
    :param csv_path_folder: 包含被试子文件夹，每个子文件夹包含多个 CSV 文件
    :param output_csv_path: 输出保存准确率结果的 CSV 文件路径
    :param key: 用于匹配 CSV 文件名的关键词（如 "trials"）
    :param type: "sub" 或 "model"
    :param time_range: 适用于 "sub" 类型的时间范围
    """
    results = []

    for file_name in os.listdir(hidden_eeg_path_folder):
        if not file_name.endswith(('.npy', '.pt')):
            continue

        # 提取 ID（假设文件名是 123_xxx.npy）
        match = re.match(r"(\d+)_", file_name)
        if not match:
            print(f"文件名不符合格式：{file_name}")
            continue
        subject_id = match.group(1)

        # 定位 CSV 文件
        subject_folder = os.path.join(csv_path_folder, subject_id)
        if not os.path.isdir(subject_folder):
            print(f"未找到对应的子文件夹：{subject_id}")
            continue

        csv_files = [f for f in os.listdir(subject_folder) if key in f and f.endswith('.csv')]
        if not csv_files:
            print(f"未找到包含关键词 '{key}' 的 CSV 文件：{subject_folder}")
            continue
        csv_path = os.path.join(subject_folder, csv_files[0])

        # 构建特征路径
        hidden_eeg_path = os.path.join(hidden_eeg_path_folder, file_name)

        # 捕获输出准确率
        try:
            # 重写 decode_hidden_eeg 函数来返回准确率
            acc = decode_hidden_eeg(
                hidden_eeg_path, csv_path, time_range=time_range, type=type, is_random=is_random,
            )
            results.append({
                "file": file_name,
                "subject_id": subject_id,
                "csv_file": os.path.basename(csv_path),
                "accuracy": acc
            })
        except Exception as e:
            print(f"处理文件出错：{file_name}, 错误：{e}")
            results.append({
                "file": file_name,
                "subject_id": subject_id,
                "csv_file": "N/A",
                "accuracy": "error"
            })

    # 保存结果
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_csv_path, index=False)
    print(f"批量解码完成，结果已保存至 {output_csv_path}")


def plot_accuracy_from_csvs(csv_paths, png_save_path, title="Accuracy Comparison"):
    """
    从多个 CSV 文件中读取 accuracy 和 subject_id 列，画出折线图并保存，横坐标显示实际 subject_id。

    :param csv_paths: 一个或多个 CSV 文件路径（列表或元组）
    :param png_save_path: 保存的 PNG 文件路径
    :param title: 图表标题
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    plt.figure(figsize=(12, 6))

    all_subject_ids = set()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        if 'subject_id' not in df.columns or 'accuracy' not in df.columns:
            print(f"跳过 {csv_path}，缺少 'subject_id' 或 'accuracy' 列")
            continue

        # 确保 accuracy 为 float，subject_id 为 str
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        df['subject_id'] = df['subject_id'].astype(str)
        df_sorted = df.sort_values(by='subject_id')

        all_subject_ids.update(df_sorted['subject_id'])

        label = csv_path.split("/")[-1].replace(".csv", "")
        plt.plot(df_sorted['subject_id'], df_sorted['accuracy'], marker='o', label=label)

    # 设置横坐标标签为所有 subject_id，按顺序排列
    subject_id_list = sorted(all_subject_ids, key=lambda x: int(x) if x.isdigit() else x)
    plt.xticks(ticks=range(len(subject_id_list)), labels=subject_id_list, rotation=45)

    plt.xlabel("Subject ID")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_save_path)
    plt.close()
    print(f"图已保存到 {png_save_path}")


def decode_eeg_by_time(eeg_hidden_path,
                       label_path,
                       save_path,
                       cv=5):
    """
    在单个的每个时间点上训练，然后在 test epoch 的对应时间点上测试

    :param save_path: 图像保存的路径
    :param eeg_hidden_path: EEG 数据的 .npy 路径
    :param label_path: 标签的 .csv 路径
    :param cv: 交叉验证的折数
    """
    eeg_data = np.load(eeg_hidden_path)
    X = eeg_data[1:, :]  # 去掉第一行
    print("EEG shape:", X.shape)

    df = pd.read_csv(label_path)
    if 'label' not in df.columns:
        raise ValueError("CSV 文件中必须包含 'label' 列")
    y = df['label'].to_numpy()[1:]  # 同步去掉第一行
    print("Label shape:", y.shape)

    # 使用 RandomForestClassifier 代替 LogisticRegression
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)  # 使用随机森林分类器
    )

    time_decod = SlidingEstimator(clf, n_jobs=None, scoring="accuracy", verbose=True)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_multiscore(time_decod, X, y, cv=skf, n_jobs=None)
    scores = np.mean(scores, axis=0)

    # 绘图
    fig, ax = plt.subplots()
    fixed_times = np.linspace(-1, 0.5, len(scores))  # 创建和 scores 长度匹配的时间轴
    ax.plot(fixed_times, scores, label="score")
    ax.axhline(0.25, color="k", linestyle="--", label="chance")  # 设置 chance line
    ax.set_xlabel("Times")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Sensor space decoding")

    # 保存图像
    plt.savefig(save_path)  # 保存图像到指定路径
    plt.close()  # 关闭图像，释放内存


def batch_decode_eeg_by_time(eeg_data_folder,
                             label_data_folder,
                             save_path_folder,
                             cv=5,
                             label_key="outcome_label_remove"):
    """
    批量解码 EEG 数据并保存图像。遍历 eeg_data_folder 下的所有 .npy 文件，
    对每个文件进行解码并将结果保存。

    :param eeg_data_folder: EEG 数据文件夹路径
    :param label_data_folder: 标签数据文件夹路径
    :param save_path_folder: 保存图像的文件夹路径
    :param cv: 交叉验证折数
    :param label_key: 标签对应的关键字（部分文件名）
    """
    # 获取所有符合条件的文件
    eeg_files = [f for f in os.listdir(eeg_data_folder) if re.match(r"^\d+_", f) and f.endswith(".npy")]

    for eeg_file in eeg_files:
        # 提取文件名中的数字部分
        subject_id = eeg_file.split("_")[0]

        # 构建 eeg_hidden_path
        eeg_hidden_path = os.path.join(eeg_data_folder, eeg_file)
        print(f"Processing EEG file: {eeg_hidden_path}")

        # 找到对应的标签文件夹
        label_folder = os.path.join(label_data_folder, subject_id)

        if not os.path.exists(label_folder):
            print(f"Label folder for subject {subject_id} not found, skipping.")
            continue

        # 查找标签文件，文件名中包含 label_key
        label_file = None
        for label_file_candidate in os.listdir(label_folder):
            if label_key in label_file_candidate and label_file_candidate.endswith(".csv"):
                label_file = label_file_candidate
                break

        if label_file is None:
            print(f"Label file with key '{label_key}' not found for subject {subject_id}, skipping.")
            continue

        # 构建 label_path
        label_path = os.path.join(label_folder, label_file)
        print(f"Processing label file: {label_path}")

        # 构建保存路径
        save_path = os.path.join(save_path_folder, subject_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # 创建子文件夹

        save_image_path = os.path.join(save_path, "decode_eeg_by_time.png")
        print(f"Saving plot to: {save_image_path}")

        # 调用 decode_eeg_by_time 方法处理并保存图像
        decode_eeg_by_time(eeg_hidden_path,
                           label_path,
                           save_image_path,
                           cv=cv)
        print(f"Completed processing for subject {subject_id}")


def compare_behavior(model_csv,
                     human_csv,
                     corr_func=pearsonr, ):
    """
    比较模型和人类行为误差之间的相关性。

    :param model_csv:
    :param human_csv:
    :param corr_func:
    :return:
    """
    # 读取CSV文件
    model_df = pd.read_csv(model_csv)
    human_df = pd.read_csv(human_csv)

    # 确保两者行数一致
    if len(model_df) != len(human_df):
        raise ValueError("模型数据和人类数据的行数不一致！")

    # 提取 outcome 和 pred
    model_outcome = model_df["outcome"].values
    model_pred = model_df["pred"].values
    human_outcome = human_df["outcome"].values
    human_pred = human_df["pred"].values

    # 计算角度误差（范围在 [0, 2]）
    model_error = angular_difference_deg(model_outcome, model_pred)
    human_error = angular_difference_deg(human_outcome, human_pred)

    # np.random.shuffle(human_error)

    # 计算相关系数
    correlation, p_value = corr_func(model_error, human_error)

    return correlation


def batch_compare_behavior(model_folder,
                           human_folder,
                           corr_func=pearsonr,
                           type="ob"):
    """
    批量比较人类相似度

    :param model_folder:
    :param human_folder:
    :param output_path:
    :param corr_func:
    :param type:
    :return:
    """
    results = []

    for subdir in os.listdir(model_folder):
        model_subdir = os.path.join(model_folder, subdir)
        if not os.path.isdir(model_subdir) or not subdir.isdigit():
            continue

        human_subdir = os.path.join(human_folder, subdir)
        if not os.path.isdir(human_subdir):
            print(f"跳过未匹配的人类子文件夹: {human_subdir}")
            continue

        # 遍历 model_subdir 下的所有子文件夹
        for inner_dir in os.listdir(model_subdir):
            inner_path = os.path.join(model_subdir, inner_dir)
            if not os.path.isdir(inner_path):
                continue

            model_file = None
            for fname in os.listdir(inner_path):
                if type == "ob" and "combine_combine" in fname and "remove" in fname:
                    model_file = os.path.join(inner_path, fname)
                    break
                elif type == "cp" and "combine_reverse" in fname and "remove" in fname:
                    model_file = os.path.join(inner_path, fname)
                    break

            if model_file:
                human_file = None
                for fname in os.listdir(human_subdir):
                    if type == "ob" and f"_{subdir}_remove" in fname:
                        human_file = os.path.join(human_subdir, fname)
                        break
                    elif type == "cp" and "reverse_remove" in fname:
                        human_file = os.path.join(human_subdir, fname)
                        break

                if human_file:
                    corr = compare_behavior(model_file, human_file, corr_func)
                    results.append((subdir, inner_dir, corr))
                    print(f"[{subdir}/{inner_dir}] 相关系数: {corr:.4f}")
                else:
                    print(f"[{subdir}/{inner_dir}] 缺少 human 文件")
            else:
                print(f"[{subdir}/{inner_dir}] 缺少 model 文件")
    return results


# def batch_compare_behavior(model_folder,
#                            human_folder,
#                            output_path,
#                            corr_func=pearsonr,
#                            type='ob'):
#     """
#     批量比较多个模型和人类行为文件的相关性，并保存结果。
#
#     参数：
#         model_folder (str): 模型行为数据的文件夹路径
#         human_folder (str): 人类行为数据的文件夹路径
#         output_path (str): 保存相关系数数组的路径（.npy 或 .csv）
#         corr_func (function): 相关函数，默认为 pearsonr
#         type (str): 'ob' 或 'cp'，决定读取哪种人类行为文件
#     """
#     correlation_list = []
#
#     # 遍历所有数字命名的子文件夹
#     for subfolder in sorted(os.listdir(model_folder)):
#         if not subfolder.isdigit():
#             continue
#
#         model_subdir = os.path.join(model_folder, subfolder)
#         human_subdir = os.path.join(human_folder, subfolder)
#
#         # 模型文件：子文件夹中包含 'remove' 的文件
#         model_files = [f for f in os.listdir(model_subdir) if "remove" in f and f.endswith(".csv")]
#         if not model_files:
#             print(f"没有找到模型文件：{model_subdir}")
#             continue
#         model_csv = os.path.join(model_subdir, model_files[0])
#
#         # 人类文件名根据 type 变化
#         if type == 'cp':
#             human_filename = f"combine_{subfolder}_reverse_remove.csv"
#         else:  # 默认 'ob'
#             human_filename = f"combine_{subfolder}_remove.csv"
#
#         human_csv = os.path.join(human_subdir, human_filename)
#         if not os.path.exists(human_csv):
#             print(f"没有找到人类行为文件：{human_csv}")
#             continue
#
#         # 调用 compare_behavior
#         try:
#             corr = compare_behavior(model_csv,
#                                     human_csv,
#                                     corr_func=corr_func)
#             correlation_list.append(corr)
#             print(f"{subfolder} -> 相关系数: {corr:.4f}")
#         except Exception as e:
#             print(f"处理 {subfolder} 时出错: {e}")
#             correlation_list.append(np.nan)
#
#     # 保存结果
#     correlation_array = np.array(correlation_list)
#     if output_path.endswith(".npy"):
#         np.save(output_path, correlation_array)
#     elif output_path.endswith(".csv"):
#         np.savetxt(output_path, correlation_array, delimiter=",")
#     else:
#         raise ValueError("output_path 必须以 .npy 或 .csv 结尾")
#     print(f"\n所有相关系数已保存至: {output_path}")


def sliding_correlation_analysis_permutation_v2(path_a,
                                                path_b,
                                                path_c,
                                                path_d,
                                                time_range=(100, 200),
                                                corr_method="pearson",
                                                n_permutations=10000,
                                                seed=42):
    np.random.seed(seed)

    # 1. 加载数据
    a = np.load(path_a)  # (13,)
    b = np.load(path_b)  # (13, 750)
    c = np.load(path_c)  # (12,)
    d = np.load(path_d)  # (12, 750)

    # 验证维度
    if a.shape != (13,) or b.shape[0] != 13:
        print("a/b 输入数据维度不匹配")
        return -2
    if c.shape != (12,) or d.shape[0] != 12:
        print("c/d 输入数据维度不匹配")
        return -2

    t_start, t_end = time_range
    if t_start < 0 or t_end > b.shape[1] or t_start >= t_end or t_end > d.shape[1]:
        print(f"非法时间范围: ({t_start}, {t_end})")
        return -2

    # 2. 截取时间窗口并平均
    b_sub = b[:, t_start:t_end]  # shape: (13, time_window)
    b_mean = b_sub.mean(axis=1)  # shape: (13,)

    d_sub = d[:, t_start:t_end]  # shape: (12, time_window)
    d_mean = d_sub.mean(axis=1)  # shape: (12,)

    # 3. 拼接成两个向量 (25,)
    ab_vec = np.concatenate([a, c])
    cd_vec = np.concatenate([b_mean, d_mean])
    print(ab_vec)
    print(cd_vec)

    # 4. 计算原始相关系数
    if corr_method == 'pearson':
        r_obs, _ = pearsonr(ab_vec, cd_vec)
    elif corr_method == 'spearman':
        r_obs, _ = spearmanr(ab_vec, cd_vec)
    else:
        raise ValueError("仅支持 pearson 或 spearman")

    # 5. permutation test
    perm_rs = []
    for _ in range(n_permutations):
        cd_perm = np.random.permutation(cd_vec)
        if corr_method == 'pearson':
            r, _ = pearsonr(ab_vec, cd_perm)
        else:
            r, _ = spearmanr(ab_vec, cd_perm)
        perm_rs.append(r)

    perm_rs = np.array(perm_rs)

    # 计算p值：右尾概率
    p_value = np.mean(perm_rs >= r_obs)

    # 6. 可视化
    plt.figure(figsize=(10, 5))
    plt.hist(perm_rs, bins=50, color='lightgray', edgecolor='black', label='Null distribution')
    plt.axvline(r_obs, color='red', linestyle='--', linewidth=2, label=f'Observed r = {r_obs:.4f}')
    plt.title(f'Permutation Test ({corr_method.title()} correlation)\n'
              f'p = {p_value:.4f}, time: {t_start}-{t_end}')
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Observed correlation r = {r_obs:.4f}")
    print(f"P-value (permutation test): {p_value:.4f}")

    return r_obs, p_value, perm_rs


def create_model_behave_hidden(model_list,
                               hidden_state_list,
                               num_layers_list,
                               target_dir,
                               model_path,
                               sub_behave_path,
                               behave_dir,
                               model_hidden_path,
                               source_npy_folder,
                               type, ):
    """
    遍历所有模型和参数，如果不存在数据，则训练并得到行为和隐藏层数据

    :param type: 当前类型
    :param model_list: 模型列表
    :param hidden_state_list: 隐藏层列表
    :param num_layers_list: 层数列表
    :param target_dir: 人类相似度保存地址
    :param model_path: 模型h5保存地址
    :param sub_behave_path: 人类行为保存地址
    :param behave_dir: 模型行为保存地址
    :param model_hidden_path: 模型隐藏层保存地址
    :param source_npy_folder: remove文件保存地址
    :return:
    """
    # 遍历所有模型、隐藏层、层数组合
    for model in model_list:
        for hs in hidden_state_list:
            for nl in num_layers_list:
                filename = f"{model}_{hs}_{nl}.npy"
                filepath = os.path.join(target_dir, filename)
                if os.path.exists(filepath):
                    print(f"Skipping existing: {filepath}")
                    continue

                # 构造行为文件匹配子串
                match_str = f"_{model}_layers_{nl}_hidden_{hs}_"

                model_str = f"{model}_layers_{nl}_hidden_{hs}_input_489.h5"
                model_target = os.path.join(model_path, model_str)

                if not os.path.exists(model_target):
                    print(f"当前正在处理: {match_str}")

                    # 训练模型
                    train_valid(model_name=model,
                                hidden_size=hs,
                                num_layers=nl, )

                if not os.path.exists(model_target):
                    raise FileNotFoundError(f"Model path does not exist: {model_target}")

                sub_behave = os.path.join(sub_behave_path, type)
                batch_evaluate(data_folder_path=sub_behave,
                               model_path=model_target,
                               results_folder_path=behave_dir,
                               hidden_state_save_dir=model_hidden_path,
                               is_save_hidden_state=1,
                               num_layers=nl,
                               model_type=model,
                               hidden_size=hs,
                               skip_keys=["label", "remove"],
                               )

                source_npy_path = os.path.join(source_npy_folder, type)
                if type == "ob":
                    remove_rows_key = f"combine_combine_{model}_layers_{nl}_hidden_{hs}_"
                    remove_rows_by_npy(source_npy_folder=source_npy_path,
                                       target_csv_folder=behave_dir,
                                       key=remove_rows_key)
                elif type == "cp":
                    remove_rows_key = f"combine_reverse_{model}_layers_{nl}_hidden_{hs}_"
                    remove_rows_by_npy(source_npy_folder=source_npy_path,
                                       target_csv_folder=behave_dir,
                                       key=remove_rows_key)


def combine_behave_eeg(behave_similarity,
                       time_range=(0, 750),
                       eeg_folder="../results/numpy/eeg_model/correlation/hc/ob", ):
    """
    将行为相似度和eeg相似度结合

    :param behave_similarity:
    :param eeg_folder:
    :return:
    """

    """
    处理行为数据 
    """
    # 用 defaultdict 存储每个模型的值
    model_scores = defaultdict(list)

    # 分组汇总
    for _, model_name, score in behave_similarity:
        model_scores[model_name].append(score)

    # 计算每个模型的平均值
    model_avg = [(model_name, np.mean(scores)) for model_name, scores in model_scores.items()]

    # 排序（可选）
    model_avg.sort(key=lambda x: x[0])

    # 输出成数组形式
    behave_result_array = np.array(model_avg, dtype=object)

    """
    处理eeg数据
    """
    model_to_eeg = {}

    for fname in os.listdir(eeg_folder):
        if not fname.endswith(".npy"):
            continue

        file_path = os.path.join(eeg_folder, fname)

        for model_name in model_scores.keys():
            if model_name in fname:
                eeg_data = np.load(file_path)  # shape: (subjects, time)
                t_start, t_end = time_range

                if eeg_data.ndim != 2:
                    print(f"跳过形状不对的文件: {fname}")
                    continue

                eeg_slice = eeg_data[:, t_start:t_end]
                eeg_mean = np.mean(eeg_slice)
                model_to_eeg[model_name] = eeg_mean

    """
    合并
    """
    final_results = []
    for model_name, behave_score in behave_result_array:
        eeg_score = model_to_eeg.get(model_name, np.nan)  # 若无对应 EEG 文件，则为 NaN
        final_results.append([model_name, behave_score, eeg_score])

    final_results_array = np.array(final_results, dtype=object)

    print("\n最终结果（行为 + EEG 相似度）:")
    for row in final_results_array:
        print(row)

    return final_results_array


def draw_human_similarity_final_results_array(final_results_array,
                                              png_save_path="../results/png/eeg_model/human_similarity/hc/ob"):
    """
    画出二维人类相似度散点图（EEG 相似度 vs 行为相似度）

    :param final_results_array: numpy array，形如 [['model_name', 行为相似度, EEG相似度], ...]
    :param png_save_path: 保存图片的文件夹路径，文件名强制为 human_similarity_timerange_0_750.png
    """
    # 提取行为和EEG相似度
    behave_sim = np.array([row[1] for row in final_results_array], dtype=float)
    eeg_sim = np.array([row[2] for row in final_results_array], dtype=float)

    # 替换 NaN 为 0
    behave_sim = np.nan_to_num(behave_sim, nan=0.0)
    eeg_sim = np.nan_to_num(eeg_sim, nan=0.0)

    # 创建画布
    plt.figure(figsize=(8, 6))

    # 设置文本偏移距离
    dx, dy = -0.0005, -0.0005

    # 绘制散点图和标签
    for i, row in enumerate(final_results_array):
        model_name = row[0]
        label = model_name.split("hidden_")[-1]  # 只取 hidden_ 后面的数字
        x, y = eeg_sim[i], behave_sim[i]
        plt.scatter(x, y, color='blue')
        plt.text(x + dx, y + dy, label, fontsize=8, ha='left', va='bottom')

    plt.xlabel("EEG similarity")
    plt.ylabel("Behavioral similarity")
    plt.title("Human Similarity")

    # 强制设定保存文件名
    save_path = os.path.join(png_save_path, "human_similarity.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_human_similarity_matrix(model_list=["rnn"],
                                hidden_state_list=[4, 8, 16, 32, 100, 200, 300],
                                num_layers_list=[1],
                                num_save_path="../results/numpy/eeg_model/human_similarity",
                                model_behave_path="../results/csv/sub/hc",
                                model_path="../models/240_rule",
                                model_hidden_path="../hidden/sub/hc",
                                sub_behave_path="../data/sub/hc",
                                source_npy_folder="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/bad_epochs",
                                eeg_rsa_path="../results/numpy/eeg_model/correlation/hc/",
                                human_similarity_png_save_path="../results/png/eeg_model/human_similarity/hc",
                                type="ob",
                                behave_corr_func=pearsonr,
                                time_range=(0, 750),
                                ):
    """
    计算所给模型和参数列表的人类相似度，并画成图像

    :param time_range:
    :param eps:
    :param eeg_rsa_path:
    :param human_similarity_png_save_path:
    :param behave_corr_func: 计算行为相似性的方法
    :param model_list: 模型列表
    :param hidden_state_list: 隐藏层列表
    :param num_layers_list: 层数列表
    :param num_save_path: 人类相似度保存列表
    :param model_behave_path: 模型行为列表
    :param model_path: 模型h5保存路径
    :param model_hidden_path: 模型隐藏层保存路径
    :param sub_behave_path: 人类行为保存路径
    :param source_npy_folder: remove文件保存路径
    :param type: 类型
    :return:
    """
    to_process = []  # 存储需要处理的组合

    # 构造路径前缀
    target_dir = os.path.join(num_save_path, type)
    os.makedirs(target_dir, exist_ok=True)  # 如果目录不存在则创建

    # 构造行为数据目录路径
    behave_dir = os.path.join(model_behave_path, type)

    model_behave_folder = os.path.join(model_behave_path, type)
    human_behave_folder = os.path.join(sub_behave_path, type)
    eeg_rsa_folder = os.path.join(eeg_rsa_path, type)
    human_similarity_png_save_folder = os.path.join(human_similarity_png_save_path, type)

    # 遍历所有模型和参数，如果不存在数据，则训练并得到行为和隐藏层数据
    create_model_behave_hidden(model_list=model_list,
                               hidden_state_list=hidden_state_list,
                               num_layers_list=num_layers_list,
                               target_dir=target_dir,
                               model_path=model_path,
                               sub_behave_path=sub_behave_path,
                               behave_dir=behave_dir,
                               model_hidden_path=model_hidden_path,
                               source_npy_folder=source_npy_folder,
                               type=type, )
    behave_similarity = batch_compare_behavior(model_folder=model_behave_folder,
                                               human_folder=human_behave_folder,
                                               corr_func=behave_corr_func,
                                               type=type, )
    final_results_array = combine_behave_eeg(behave_similarity,
                                             eeg_folder=eeg_rsa_folder,
                                             time_range=time_range, )
    draw_human_similarity_final_results_array(final_results_array=final_results_array,
                                              png_save_path=human_similarity_png_save_folder,)


if __name__ == "__main__":
    """
    计算eeg rdm
    """
    # data = np.load("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/eeg_preprocessing_458_470_473_478.npy", mmap_mode="r")
    # computer_eeg_rdm(eeg_data=data,
    #                  save_path="../results/numpy/eeg_rdm/hc/2base_-1_0.5_baseline(6)_0_0.2/eeg_rdm_458_470_473_478.npy",
    #                  fig_save_path=None)

    # computer_eeg_rdm_remove(folder_path="../data/eeg/yuanwen/npy",
    #                         save_base_path="../results/numpy/eeg_rdm/yuanwen",
    #                         fig_save_path=None,
    #                         time_range=(650, 750),  # channel (650, 750), time (250, 1000)
    #                         metric="correlation",
    #                         type="channel",
    #                         one_time=527)
    # data = np.load("../results/numpy/eeg_rdm/yuanwen_need/203_clean/rdm/203_clean_by_channel_rdm.npy")
    # print(data.shape)

    """
    比较eeg和model的rdm
    """
    # eeg_rdm = np.load(
    #     "../results/numpy/eeg_rdm/hc/2base_-1_0.5_baseline(6)_0_0.2_remove/456_clean/rdm/456_clean_rdm.npy",
    #     mmap_mode="r")
    # model_rdm = np.load(
    #     "../results/numpy/model/sub/hc/456/rdm/remove/rnn_layers_1_hidden_16_input_489_reverse_processed.npy")
    # print(eeg_rdm.shape)
    # print(model_rdm.shape)

    # batch_compute_eeg_model_rdm_correlation(eeg_rdm=eeg_rdm,
    #                                         model_rdm=model_rdm,
    #                                         time_range=(100, 850),
    #                                         time_min=-1,
    #                                         time_max=0.5,
    #                                         re_threhold=2,
    #                                         is_every=1,
    #                                         rsa_save_path="../results/numpy/eeg_model/correlation/OB_first_rsa_result.npy")
    # batch_compute_eeg_model_rdm_correlation_remove(
    #     eeg_rdm_folder="../results/numpy/eeg_rdm/hc/2base_-1_0.5_baseline(6)_0_0.2_remove/all_time_problem/ob",
    #     epoch_data_path=None,
    #     model_rdm_folder="../results/numpy/model/sub/hc/",
    #     time_range=(100, 850),
    #     time_min=-1,
    #     time_max=0.5,
    #     is_every=1,
    #     re_threshold=2,
    #     rsa_save_path="../results/numpy/eeg_model/correlation/hc/ob",
    #     png_save_path="../results/png/eeg_model/correlation/hc/ob",
    #     type="time",
    #     one_time=527,
    #     montage_type=64,
    #     model_type="combine",)
    # plot_permutation_test(rsa_result_path="../results/numpy/eeg_model/correlation/hc/rsa_results_ob_first.npy",
    #                       rsa_result2_path="../results/numpy/eeg_model/correlation/hc/rsa_results_cp_first.npy",
    #                       time_min=-1,
    #                       time_max=0.5,
    #                       is_every=1,
    #                       time_dur=750,
    #                       type="combine",
    #                       re_threshold=2,
    #                       png_save_path="../results/png/eeg_model/correlation/hc/rsa_results_combine.png")

    """
    根据相关矩阵画出电极点图
    """

    # plot_topomap_by_correlation(
    #     correlation_array_path="../results/numpy/eeg_model/correlation/yuanwen/rsa_results_by_channel_unknow_OB_first.npy",
    #     save_path="../results/png/eeg_model/correlation/yuanwen/rsa_results_unknow_OB_first_comtap_0.015.png",
    #     montage_type=64,
    #     vmin=-0.015,
    #     vmax=0.015,)

    """
    encoding base RSA
    """
    # encoding_base_rsa(eeg_data_folder="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/npy/",
    #                   hidden_data_folder="../hidden/sub/hc/",
    #                   n_splits=10,
    #                   n_permutation=50,
    #                   png_save_path="../results/png/eeg_model/correlation",
    #                   png_save_name="test.png",
    #                   type="ridge_regression",
    #                   epoch_data_path="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/408.fif")

    """
    根据eeg解码四个方向
    """
    # decode_hidden_eeg(
    #     hidden_eeg_path="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/clean_npy/ob/merged.npy",
    #     csv_path="../data/sub/hc/ob/merged.csv",
    #     type="sub",
    #     time_range=(100, 850),
    #     is_random=0,)
    # batch_decode_hidden_eeg(
    #     hidden_eeg_path_folder="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/clean_npy/ob/",
    #     csv_path_folder="../data/sub/hc/ob/",
    #     output_csv_path="../results/csv/decode/decode_result_ob_100_500.csv",
    #     key="outcome_label_remove",
    #     type="sub",
    #     time_range=(100, 500),
    #     is_random=0,
    # )
    # plot_accuracy_from_csvs(
    #     csv_paths=["../results/csv/decode/decode_result_ob.csv",
    #                "../results/csv/decode/decode_result_ob_random.csv",
    #                "../results/csv/decode/decode_result_ob_500_700.csv",],
    #     png_save_path="../results/png/decode/decode_acc_ob.png")

    # decode_eeg_by_time(
    #     eeg_hidden_path="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/clean_npy/ob/405_clean.npy",
    #     label_path="../data/sub/hc/ob/405/combine_405_outcome_label_remove.csv",
    #     save_path="../results/png/decode/sub/hc/405/decode_eeg_by_time.png")
    # batch_decode_eeg_by_time(eeg_data_folder="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/clean_npy/cp/",
    #                          label_data_folder="../data/sub/hc/cp/",
    #                          save_path_folder="../results/png/decode/sub/hc/cp",
    #                          cv=5,
    #                          label_key="outcome_label_remove",
    #                          )

    """
    人类相似度
    """
    # print(compare_behavior(
    #     model_csv="../results/csv/sub/hc/ob/407/combine_combine_rnn_layers_1_hidden_16_input_489_cos_remove.csv",
    #     human_csv="../data/sub/hc/ob/407/combine_407_remove.csv",
    #     corr_func=pearsonr, ))
    # batch_compare_behavior(model_folder="../results/csv/sub/hc/cp/",
    #                        human_folder="../data/sub/hc/cp/",
    #                        output_path="../results/numpy/eeg_model/human_similarity/cp/pre_dis_cor.npy",
    #                        corr_func=pearsonr,
    #                        type="cp",)
    # sliding_correlation_analysis_permutation_v2(path_a="../results/numpy/eeg_model/human_similarity/ob/pre_dis_cor.npy",
    #                                             path_b="../results/numpy/eeg_model/correlation/hc/rsa_results_ob_first.npy",
    #                                             path_c="../results/numpy/eeg_model/human_similarity/cp/pre_dis_cor.npy",
    #                                             path_d="../results/numpy/eeg_model/correlation/hc/rsa_results_cp_first.npy",
    #                                             time_range=(400, 500),
    #                                             corr_method='pearson',
    #                                             n_permutations=10000, )

    """
    从两个维度计算人类相似度
    """
    get_human_similarity_matrix(hidden_state_list=[1, 2, 4, 8, 16, 32, 100, 200, 300],
                                time_range=(400, 500),)
