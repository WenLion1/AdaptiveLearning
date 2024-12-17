import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform


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
        sns.heatmap(
            dissimilarity_matrix,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Dissimilarity'}
        )
        plt.title(f"Dissimilarity Matrix at Time Point {t}")
        plt.xlabel("Trial")
        plt.ylabel("Trial")

        # 保存图像
        save_file = os.path.join(save_path, f"dissimilarity_matrix_time_{t:03d}.png")
        plt.savefig(save_file, dpi=300)
        plt.close()  # 关闭图像，释放内存

    print(f"All dissimilarity matrices have been saved to {save_path}")


if __name__ == "__main__":
    # 读取 .mat 文件
    mat_file_path = '../data/eeg/hc/lal-hc-403-task.mat'  # 替换为你的 .mat 文件路径
    mat_contents = loadmat(mat_file_path)

    # 查看文件的顶级结构
    print(mat_contents.keys())

    # 提取 eeg 字段
    if 'EEG' in mat_contents:
        eeg_data = mat_contents['EEG']['icaact'][0, 0]  # 提取 data 字段
        print("EEG Data Shape:", eeg_data.shape)  # 打印数据的形状
    else:
        print("eeg 字段未找到")

    compute_and_plot_dissimilarity_matrices(eeg_data=eeg_data,
                                            save_path="../results/eeg/403/rdm",
                                            trial_range=(0, 223),)
