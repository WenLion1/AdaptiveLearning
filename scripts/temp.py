import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
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


if __name__ == "__main__":
    # merge_csv_in_subfolders(csv1_path="../data/sub/hc/403/ADL_B_403_DataCP_403.csv",
    #                csv2_path="../data/sub/hc/403/ADL_B_403_DataOddball_403.csv",
    #                output_path="../data/sub/hc/403/combine_403.csv", )

    # merge_csv_in_subfolders(parent_folder="../data/sub/hc",
    #                         output_folder="../data/sub/hc")

    # 读取数据
    data = np.load("../results/numpy/model/sub/hc/not_remove_model_rdm.npy")  # 形状 (27, 114960)

    # **Step 1: 计算原始 RDM**
    original_rdm = compute_rdm(data)

    # **Step 2-3: 生成 3000 个随机置换 RDM**
    num_permutations = 3000
    permuted_rdms = np.zeros((num_permutations, 27, 27))

    for i in range(num_permutations):
        print("循环: ", i)
        # 沿着第二维打乱数据（每行独立打乱）
        shuffled_data = np.apply_along_axis(np.random.permutation, axis=1, arr=data)

        # 计算打乱后的 RDM
        permuted_rdms[i] = compute_rdm(shuffled_data)

    # **Step 4: 计算原始 RDM 在置换分布中的位置**
    # 这里可以选取某些统计量，比如均值或特定位置的值进行比较
    original_mean = np.mean(original_rdm)
    permuted_means = np.mean(permuted_rdms, axis=(1, 2))  # 对每个置换 RDM 求均值

    # 计算原始 RDM 均值在置换分布中的百分位
    p_value = np.mean(permuted_means >= original_mean)

    # **Step 5: 绘制分布直方图**
    plt.hist(permuted_means, bins=50, color='gray', alpha=0.7, label="Permutation Distribution")
    plt.axvline(original_mean, color='red', linestyle='dashed', linewidth=2, label="Original RDM Mean")
    plt.xlabel("Mean RDM Distance")
    plt.ylabel("Frequency")
    # plt.title(f"Permutation Test (p={p_value:.5f})")
    plt.legend()
    plt.show()

    # **Step 6: 输出结果**
    print(f"原始 RDM 的均值: {original_mean:.5f}")
    print(f"随机置换 RDM 分布的均值: {np.mean(permuted_means):.5f} ± {np.std(permuted_means):.5f}")
    print(f"原始 RDM 在置换分布中的 p 值: {p_value:.5f}")

    # raw = mne.io.read_epochs_eeglab('C:/Learn/Project/bylw/eeg/2 remove channels + waterprint/lal-hc-453-task.set')
    #
    # print(type(raw))

    # model1 = np.load("../results/numpy/model/sub/hc/not_remove_model_rdm.npy")
    # model2 = np.load("../results/numpy/model/sub/hc/not_remove_model_rdm_copy.npy")
    # model3 = np.load("../results/numpy/model/sub/hc/not_remove_model_rdm_reverse.npy")
    # print(model1.shape)
    # print(model2.shape)
    #
    # array = compare_rdm_matrices(model1, model3)
    # # 可视化 RDM
    # plt.figure(figsize=(8, 6))
    # plt.imshow(array, cmap='viridis', interpolation='none')
    # plt.colorbar(label='Dissimilarity')
    # plt.title('Representational Dissimilarity Matrix (RDM)', fontsize=16)
    # plt.xlabel('Sub', fontsize=12)
    # plt.ylabel('Sub', fontsize=12)
    # plt.tight_layout()
    # plt.show()
