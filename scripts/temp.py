import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from mne.stats import permutation_cluster_1samp_test
from scipy.spatial.distance import squareform
from scipy.stats import vonmises, zscore, spearmanr

import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def merge_csv_in_subfolders(parent_folder, output_folder):
    """
    遍历父文件夹内的所有子文件夹，在每个子文件夹内合并两个 CSV 文件，并将结果保存到 output_folder。

    :param parent_folder: 父文件夹路径，包含多个子文件夹
    :param output_folder: 保存合并结果的文件夹
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有子文件夹
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)

        if os.path.isdir(subfolder_path):  # 确保是文件夹
            # 获取子文件夹内的所有 CSV 文件
            csv_files = glob.glob(os.path.join(subfolder_path, "*.csv"))

            if len(csv_files) !=2:
                print(f"跳过 {subfolder}：该子文件夹内的 CSV 文件数不是 2 个")
                continue

            # 读取并合并两个 CSV 文件
            df1 = pd.read_csv(csv_files[0])
            df2 = pd.read_csv(csv_files[1])
            merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)

            # 生成输出文件路径（保持原子文件夹名称）
            output_file = os.path.join(subfolder_path, f"combine_{subfolder}.csv")

            # 保存合并后的 CSV 文件
            merged_df.to_csv(output_file, index=False)
            print(f"合并完成：{output_file}")

def extract_outcome_from_csv(root_dir):
    """
    遍历指定文件夹及其子文件夹，查找包含 'combine' 的 csv 文件并提取 'outcome' 字段。
    最后将所有提取的数组合并成一个二维数组。

    :param root_dir: 根文件夹路径
    :return: 合并后的二维数组
    """
    outcome_list = []  # 用来存储所有 'outcome' 字段数据

    # 遍历文件夹及子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 如果文件名包含 "combine" 且是 CSV 文件
            if "combine" in filename and filename.endswith(".csv"):
                file_path = os.path.join(dirpath, filename)
                print(f"读取文件: {file_path}")

                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 确保文件中有 "outcome" 字段
                if 'outcome' in df.columns:
                    outcome_data = df['outcome'].to_numpy()  # 提取 outcome 字段
                    outcome_list.append(outcome_data)  # 将数据添加到 outcome_list 中

    # 将 outcome_list 合并为一个二维数组
    combined_outcome_array = np.array(outcome_list)

    return combined_outcome_array

def compute_rdm(data):
    """
    计算给定数据的RDM（Representational Dissimilarity Matrix），
    使用欧几里得距离计算每一对观测之间的差异。

    :param data: 输入数据矩阵，形状为 (30, 480)
    :return: 计算出的 RDM，形状为 (30, 30)
    """
    # # 标准化数据
    # data_normalized = zscore(data, axis=1, nan_policy='omit')

    # 计算行之间的欧几里得距离，pdist 返回一个扁平化的矩阵
    rdm = pdist(data, metric='correlation')

    # 将结果转换为方阵形式，得到 30x30 的 RDM
    rdm_square = squareform(rdm)

    return rdm_square


def compare_rdm_matrices(eeg_rdm, model_rdm):
    """
    比较两个 RDM 矩阵之间的相似性，并返回一个 sub x sub 的相似性矩阵。

    :param eeg_rdm: EEG RDM 矩阵，形状为 (sub, rdm)
    :param model_rdm: 模型 RDM 矩阵，形状为 (sub, rdm)
    :return: sub x sub 的相似性矩阵
    """
    sub_num = eeg_rdm.shape[0]
    similarity_matrix = np.zeros((sub_num, sub_num))  # 初始化相似性矩阵

    for i in range(sub_num):
        for j in range(sub_num):
            # 直接计算两个矩阵的 RDM 相关性
            corr, _ = spearmanr(eeg_rdm[i, :], model_rdm[j, :])
            print(corr)
            # 将计算结果填充到相似性矩阵
            similarity_matrix[i, j] = corr

    return similarity_matrix


def calculate_and_plot_correlation(rdm1, rdm2, time_min=-1, time_max=0.5, threshold=-2.0):
    # 确保数据维度一致
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 和 rdm2 形状不一致，无法计算相关性")

    n_times = rdm1.shape[1]
    correlations = np.zeros(n_times)
    p_values = np.zeros(n_times)

    # 逐时间点计算 Spearman 相关性
    for t in range(n_times):
        corr, p_val = spearmanr(rdm1[0, t, :], rdm2[0, t, :])
        correlations[t] = corr
        p_values[t] = p_val

    # 绘制相关性曲线
    time_points = np.linspace(time_min, time_max, n_times)
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, correlations, color='red', label='Spearman Correlation')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Correlation')
    plt.title('Time-resolved RDM Correlation')

    # 显著性检验
    rsa_results = correlations.reshape(1, -1)
    tail = 1
    if threshold < 0:
        tail = -1
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        rsa_results,
        threshold=threshold,
        n_permutations=5000,
        tail=tail
    )

    # 标出显著相关区段
    significant_clusters = np.where(cluster_pv < 0.05)[0]
    for i_clu in significant_clusters:
        time_indices = clusters[i_clu][0]
        sig_times = time_points[time_indices]
        plt.fill_between(sig_times, correlations.min(), correlations.max(), color='red', alpha=0.3,
                         label='p < 0.05' if i_clu == 0 else "")

    # 图例和显示
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def convert_fif_to_npy(folder_path, baseline=1, baseline_range=(0, 0.2)):
    """
    将指定文件夹中的所有 .fif 文件转换为 .npy 文件，并保存到原文件夹中。

    参数:
        folder_path (str): 包含 .fif 文件的文件夹路径。
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为 .fif
        if filename.endswith(".fif"):
            # 构造完整的文件路径
            fif_file_path = os.path.join(folder_path, filename)
            # 构造对应的 .npy 文件路径
            npy_file_path = os.path.join(folder_path, filename.replace(".fif", ".npy"))

            epochs = mne.read_epochs(fif_file_path, preload=True)

            if baseline:
                # 应用基线校正
                epochs.apply_baseline(baseline_range)

            data = epochs.get_data(picks=['eeg'])

            # 保存为 .npy 文件
            np.save(npy_file_path, data)

            print(f"已将 {fif_file_path} 转换为 {npy_file_path}")


def calculate_correlation_significance(data1, data2, time_points, n_permutations=1000):
    """
    计算每个时间点的两个文件之间的相关值，并标出显著相关的区间。

    参数:
        data1 (np.ndarray): 第一个数据文件，形状为 (subjects, time_points, rdm_size)。
        data2 (np.ndarray): 第二个数据文件，形状为 (subjects, time_points, rdm_size)。
        time_points (int): 时间点的数量。
        n_permutations (int): 置换测试的次数。
    """
    correlations = []
    p_values = []

    for t in range(time_points):
        print("t: ", t)
        # 提取当前时间点的数据
        rdm1 = data1[:, t, :]
        rdm2 = data2[:, t, :]

        # 计算每个被试的相关性
        subject_correlations = []
        for subject in range(rdm1.shape[0]):
            print("subject: ", subject)
            corr_coef, _ = spearmanr(rdm1[subject], rdm2[subject])
            subject_correlations.append(corr_coef)

        # 计算平均相关性
        avg_corr_coef = np.mean(subject_correlations)
        correlations.append(avg_corr_coef)

        # # 置换测试
        # permuted_corrs = []
        # for _ in range(n_permutations):
        #     # 随机打乱rdm2的行
        #     permuted_rdm2 = rdm2[np.random.permutation(rdm2.shape[0]), :]
        #     permuted_subject_correlations = []
        #     for subject in range(rdm1.shape[0]):
        #         permuted_corr_coef, _ = spearmanr(rdm1[subject], permuted_rdm2[subject])
        #         permuted_subject_correlations.append(permuted_corr_coef)
        #     permuted_corrs.append(np.mean(permuted_subject_correlations))
        #
        # # 计算p值
        # p_value = np.mean(np.abs(permuted_corrs) >= np.abs(avg_corr_coef))
        # p_values.append(p_value)

    # 绘制相关值和显著性
    plt.figure(figsize=(10, 5))
    plt.plot(correlations, label='Average Correlation Coefficient')
    plt.plot(p_values, label='p-value')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (0.05)')
    plt.xlabel('Time Points')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Average Correlation and Significance over Time')
    plt.show()


if __name__ == "__main__":
    cp1 = np.load("../results/numpy/model/sub/hc/456/rdm/not_remove/rnn_layers_1_hidden_16_input_489_CP.npy")
    cp2 = np.load("../results/numpy/model/sub/hc/457/rdm/not_remove/rnn_layers_1_hidden_16_input_489_CP.npy")
    cp3 = np.load("../results/numpy/model/sub/hc/458/rdm/not_remove/rnn_layers_1_hidden_16_input_489_CP.npy")

    ob1 = np.load("../results/numpy/model/sub/ob/405/rdm/not_remove/rnn_layers_1_hidden_16_input_489_CP.npy")
    ob2 = np.load("../results/numpy/model/sub/ob/407/rdm/not_remove/rnn_layers_1_hidden_16_input_489_CP.npy")
    ob3 = np.load("../results/numpy/model/sub/ob/408/rdm/not_remove/rnn_layers_1_hidden_16_input_489_CP.npy")
    print(cp1.shape)

    # 计算 Spearman 相关性
    spearman_corr, spearman_p = spearmanr(ob3.flatten(), ob1.flatten())
    print(f"Spearman相关系数: {spearman_corr}, p值: {spearman_p}")