import os
import re

import mne
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, ttest_1samp, zscore
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
from twisted.python.util import println


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
                                                   model_rdm_folder,
                                                   time_min,
                                                   time_max,
                                                   re_threshold,
                                                   time_range=None,
                                                   is_every=0,
                                                   rsa_save_path=None,
                                                   png_save_path=None,
                                                   type="time",
                                                   one_time=527, ):
    """
    批量计算 EEG 数据与模型 RDM 的相关性，并绘制显著性结果。

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
    if not common_subjects:
        print("没有找到匹配的被试编号，请检查文件夹内容。")
        return

    print(f"找到 {len(common_subjects)} 个匹配的被试编号。")

    # 初始化 RSA 结果矩阵
    if type == "time":
        # 初始化 RSA 结果矩阵
        rsa_results_all = np.zeros((len(common_subjects), time_range[1] - time_range[0]))
        time_dur = time_range[1] - time_range[0]

        # 遍历被试
        for i, sub in enumerate(common_subjects):
            print(f"正在处理被试 {sub} ({i + 1}/{len(common_subjects)})")

            # 加载 EEG RDM 和模型 RDM
            eeg_rdm_path = os.path.join(eeg_rdm_folder, str(sub) + "_clean", 'rdm')
            model_rdm_path = os.path.join(model_rdm_folder, str(sub), 'rdm', 'remove')

            eeg_rdm_file = [f for f in os.listdir(eeg_rdm_path) if f.endswith('_clean_rdm.npy')][0]
            model_rdm_file = [f for f in os.listdir(model_rdm_path) if f.endswith('.npy')][0]

            eeg_rdm = np.load(os.path.join(eeg_rdm_path, eeg_rdm_file))
            model_rdm = np.load(os.path.join(model_rdm_path, model_rdm_file))

            # 检查数据维度
            if eeg_rdm.shape[1] != model_rdm.shape[0]:
                raise ValueError(f"维度不匹配: EEG RDM {eeg_rdm.shape}, 模型 RDM {model_rdm.shape}")

            # 初始化当前被试的 RSA 结果
            rsa_results = np.zeros((time_range[1] - time_range[0],))

            # 逐时间点计算相关性
            for t in range(time_range[0], time_range[1]):
                eeg_rdm_t = eeg_rdm[t, :]
                model_rdm_z = zscore(model_rdm)
                eeg_rdm_z = zscore(eeg_rdm_t)

                spearman_corr, _ = spearmanr(model_rdm_z, eeg_rdm_z)
                rsa_results[t - time_range[0]] = spearman_corr

            rsa_results_all[i, :] = rsa_results
    elif type == "channel" or type == "one_time":
        # 初始化 RSA 结果矩阵
        rsa_results_all = np.zeros((len(common_subjects), 98))

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
            model_rdm_file = [f for f in os.listdir(model_rdm_path) if f.endswith('.npy')][0]

            eeg_rdm = np.load(os.path.join(eeg_rdm_path, eeg_rdm_file))
            model_rdm = np.load(os.path.join(model_rdm_path, model_rdm_file))

            # 检查数据维度
            if eeg_rdm.shape[1] != model_rdm.shape[0]:
                raise ValueError(f"维度不匹配: EEG RDM {eeg_rdm.shape}, 模型 RDM {model_rdm.shape}")

            # 初始化当前被试的 RSA 结果
            rsa_results = np.zeros((98,))

            # 逐时间点计算相关性
            for c in range(98):
                eeg_rdm_t = eeg_rdm[c, :]
                model_rdm_z = zscore(model_rdm)
                eeg_rdm_z = zscore(eeg_rdm_t)

                spearman_corr, _ = spearmanr(model_rdm_z, eeg_rdm_z)
                rsa_results[c] = spearman_corr

            rsa_results_all[i, :] = rsa_results

    # 保存 RSA 结果
    if rsa_save_path is not None:
        np.save(rsa_save_path, rsa_results_all)
        print("RSA 结果已保存。")

    # 时间点转换
    if type == "time":
        axis = np.linspace(time_min, time_max, num=time_dur)
    elif type == "channel" or type == "one_time":
        axis = np.arange(1, 99)

    # 绘图
    plt.figure(figsize=(8, 6))

    # 绘制所有被试曲线
    if is_every == 1:
        for sub_data in rsa_results_all:
            plt.plot(axis, sub_data, color='gray', alpha=0.3)

    # 计算均值和标准误
    mean_corr = np.mean(rsa_results_all, axis=0)
    sem_corr = np.std(rsa_results_all, axis=0) / np.sqrt(len(common_subjects))

    plt.plot(axis, mean_corr, color='red', label='Mean Correlation')
    plt.fill_between(axis, mean_corr - sem_corr, mean_corr + sem_corr, color='red', alpha=0.3)

    # 进行显著性检验
    tail = 1 if re_threshold > 0 else -1
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(rsa_results_all,
                                                                     threshold=re_threshold,
                                                                     n_permutations=5000,
                                                                     tail=tail)

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
    plt.show()
    plt.close()


def computer_eeg_rdm_remove(folder_path,
                            save_base_path,
                            fig_save_path=None,
                            metric='correlation',
                            save_time=200,
                            is_number_label=False,
                            type="time",
                            one_time=527, ):
    """
    计算一个文件夹下所有 EEG 数据的 RDM，并保存结果。

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
            rdm_results = np.zeros((time_num, epoch_num * (epoch_num - 1) // 2))

            for t in range(time_num):
                print(f"  时间点: {t + 1}/{time_num}")
                # 提取当前时间点的数据 (epoch × channel)
                trial_features = eeg_data[:, :, t]  # (epoch, channel)

                # 计算 RDM
                rdm = pdist(trial_features, metric=metric)  # 计算所有 epoch 的 pairwise 距离
                rdm_results[t] = rdm  # 存储到结果矩阵
        elif type == "channel":
            rdm_results = np.zeros((channel_num, epoch_num * (epoch_num - 1) // 2))

            for c in range(channel_num):
                print(f"  电极点: {c + 1}/{channel_num}")
                # 提取当前时间点的数据 (epoch × time)
                trial_features = eeg_data[:, c, :]  # (epoch, time)

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
                                save_path=None, ):
    epochs = mne.read_epochs("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/405_clean.fif")
    data = np.load(correlation_array_path)
    mean_data = np.mean(data, axis=0)

    info = epochs.info
    eeg_picks = mne.pick_types(info, eeg=True, eog=False, stim=False, misc=False)
    eeg_ch_names = [info.ch_names[i] for i in eeg_picks]

    for i in range(len(mean_data)):
        print("channel: ", eeg_ch_names[i])
        print("mean_data: ", mean_data[i])

    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(layout="constrained")

    mne.viz.plot_topomap(mean_data, info, ch_type='eeg', cmap='RdBu_r', names=eeg_ch_names, show=True, contours=0,
                         axes=ax)

    # sm = plt.cm.ScalarMappable(cmap="RdBu_r")
    # sm.set_array(mean_data)
    # plt.colorbar(sm, ax=ax, label='Correlation')

    # fig.title("The distribution of average correlation values on the EEG scalp")
    # fig.show()

    fig.savefig(save_path, dpi=300)
    print("图像已保存")


def ridge_regression_eeg(hidden_data, eeg_data, train_ratio=0.8, n_splits=10, alpha=1.0, save_path=None):
    """
    使用 Ridge Regression 对 EEG 数据进行回归分析并预测，同时评估预测准确性。

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
    eeg_data = np.load(eeg_data)
    hidden_data = np.load(hidden_data)
    n_trials, n_channels, n_times = eeg_data.shape
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

        for t in range(n_times):
            eeg_train = eeg_data[train_idx, :, t]
            eeg_test = eeg_data[test_idx, :, t]

            # Ridge 回归训练
            model.fit(hidden_data[train_idx], eeg_train)
            coefs[split, t, :, :] = model.coef_

            # EEG 预测
            predicted_eeg[:, :, t] = model.predict(hidden_data[test_idx])

            # 训练集评估
            eeg_train_pred = model.predict(hidden_data[train_idx])  # 训练集预测
            mse_train_scores[split, :, t] = np.mean((eeg_train - eeg_train_pred) ** 2, axis=0)  # 训练集 MSE
            for ch in range(n_channels):
                corr_train_scores[split, ch, t], _ = pearsonr(eeg_train[:, ch], eeg_train_pred[:, ch])

            # 测试集评估
            mse_test_scores[split, :, t] = np.mean((eeg_test - predicted_eeg[:, :, t]) ** 2, axis=0)  # 测试集 MSE
            for ch in range(n_channels):
                corr_test_scores[split, ch, t], _ = spearmanr(eeg_test[:, ch], predicted_eeg[:, ch, t])

        all_predicted_eeg.append(predicted_eeg)

    print("回归和预测完成！")

    # 相关性转换为百分比
    corr_train_scores *= 100
    corr_test_scores *= 100

    print("训练集上的 MSE:", mse_train_scores.mean(axis=(0, 1, 2)))
    print("测试集上的 MSE:", mse_test_scores.mean(axis=(0, 1, 2)))
    print("训练集上的相关性 (%):", corr_train_scores.mean(axis=(0, 1, 2)))  # 百分比格式
    print("测试集上的相关性 (%):", corr_test_scores.mean(axis=(0, 1, 2)))  # 百分比格式

    # 保存结果
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "predicted_eeg.npy"), np.array(all_predicted_eeg))
        np.save(os.path.join(save_path, "ridge_coefs.npy"), coefs)
        print(f"结果已保存至 {save_path}")

    return np.array(all_predicted_eeg), coefs, np.array(test_all)


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

    # # 提取eeg数据
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
    #
    # # 画所有的rdm相关图
    # plot_npy_from_subfolders(folder_path="../results/numpy/eeg_model/correlation",
    #                          saving_path="../results/png/sub/hc/all/rdm_all_correlation_roll.png")

    # data = load_and_concatenate_npy_files(folder_path="../results/numpy/eeg_model/correlation")
    # # analyze_significant_time_points(data=data,
    # #                                 threshold=None)
    #
    # significant_points = find_significant_periods(data,
    #                                               threshold=0.7)
    # print(significant_points)
    #
    """
    计算eeg rdm
    """
    # data = np.load("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/eeg_preprocessing_458_470_473_478.npy", mmap_mode="r")
    # computer_eeg_rdm(eeg_data=data,
    #                  save_path="../results/numpy/eeg_rdm/hc/2base_-1_0.5_baseline(6)_0_0.2/eeg_rdm_458_470_473_478.npy",
    #                  fig_save_path=None)

    # computer_eeg_rdm_remove(folder_path="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/npy",
    #                         save_base_path="../results/numpy/eeg_rdm/hc/2base_-1_0.5_baseline(6)_0_0.2_remove/",
    #                         fig_save_path=None,
    #                         metric="euclidean",
    #                         type="one_time",
    #                         one_time=527)

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
    #     eeg_rdm_folder="../results/numpy/eeg_rdm/hc/2base_-1_0.5_baseline(6)_0_0.2_remove",
    #     model_rdm_folder="../results/numpy/model/sub/hc/",
    #     time_range=(100, 850),
    #     time_min=-1,
    #     time_max=0.5,
    #     is_every=1,
    #     re_threshold=2,
    #     rsa_save_path="../results/numpy/eeg_model/correlation/rsa_results_one_time_527_cp_first.npy",
    #     png_save_path="../results/png/eeg_model/correlation/rsa_results_one_time_527_cp_first.png",
    #     type="one_time",
    #     one_time=527, )

    """
    根据相关矩阵画出电极点图
    """

    # plot_topomap_by_correlation(
    #     correlation_array_path="../results/numpy/eeg_model/correlation/rsa_results_by_channel_cp_first.npy",
    #     save_path="../results/png/eeg_model/correlation/rsa_cp_first_topomap.png")

    """
    encoding base RSA
    """
    all_predicted_eeg, coefs, test_all = ridge_regression_eeg(
        hidden_data="../hidden/sub/hc/405/remove/rnn_layers_1_hidden_16_input_489_combine_processed.npy",
        eeg_data="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/npy/405_clean.npy",
        n_splits=2)
    print(all_predicted_eeg.shape)
    print(coefs.shape)
    print(test_all.shape)
    print(test_all)
