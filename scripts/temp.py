import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from mne.stats import permutation_cluster_1samp_test
from scipy.spatial.distance import squareform
from scipy.stats import vonmises, zscore, spearmanr
from autoreject import AutoReject
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import scipy.io
import h5py


def plot_average_topomap_from_folder(folder_path, time_point, ch_type='eeg', show=True, verbose=True):
    """
    从指定文件夹中加载所有 .fif 文件，提取某一时间点的电极值，进行平均后绘制 topomap。

    :param folder_path: 包含 .fif 文件的文件夹路径
    :param time_point: 要提取的时间点（单位：秒）
    :param ch_type: 通道类型，默认 'eeg'
    :param show: 是否显示图像
    :param verbose: 是否打印加载信息
    """
    all_data_at_time = []
    template_evoked = None

    fif_files = [f for f in os.listdir(folder_path) if f.endswith(".fif")]
    if len(fif_files) == 0:
        raise ValueError("指定文件夹中没有 .fif 文件！")

    for filename in fif_files:
        filepath = os.path.join(folder_path, filename)
        if verbose:
            print(f"加载: {filename}")
        epochs = mne.read_epochs(filepath, preload=True)
        evoked = epochs.average()
        if template_evoked is None:
            template_evoked = evoked.copy()

        try:
            time_idx = evoked.time_as_index(time_point)[0]
        except Exception as e:
            raise ValueError(f"时间点 {time_point}s 超出了数据范围！") from e

        data_at_time = evoked.data[:, time_idx]
        all_data_at_time.append(data_at_time)

    # 平均所有数据
    average_data = np.mean(all_data_at_time, axis=0)

    # 用 template_evoked 做模板，只保留 time_point 的数据
    template_evoked.data[:, :] = 0
    template_evoked.data[:, template_evoked.time_as_index(time_point)[0]] = average_data

    # 绘图
    template_evoked.plot_topomap(times=[time_point], ch_type=ch_type, show=show)


def read_eeg_data_from_mat(mat_file_path):
    """
    从指定的 .MAT 文件（MATLAB v7.3 格式）中读取 EEG 子文件的 data 字段。

    参数:
        mat_file_path (str): .MAT 文件的路径

    返回:
        numpy.ndarray: EEG 的 data 字段数据，如果不存在则返回 None
    """
    try:
        # 打开 .MAT 文件
        with h5py.File(mat_file_path, 'r') as file:
            # 检查是否存在 'EEG' 子文件
            if 'EEG' in file:
                eeg_group = file['EEG']

                # 检查是否存在 'data' 字段
                if 'data' in eeg_group:
                    eeg_data_field = np.array(eeg_group['data'])
                    print(f"EEG data field extracted successfully from {mat_file_path}!")
                    return eeg_data_field
                else:
                    print(f"No 'data' field found in 'EEG' in {mat_file_path}.")
            else:
                print(f"No 'EEG' subfile found in {mat_file_path}.")
    except Exception as e:
        print(f"Error loading .MAT file {mat_file_path}: {e}")

    return None


def plot_erp_from_folder(folder_path, sampling_rate=500, time_window=None, baseline_window=(0, 100), is_baseline=0):
    """
    从指定文件夹中的所有 .MAT 文件读取 EEG 数据，并绘制合并的 ERP 图。

    参数:
        folder_path (str): 包含 .MAT 文件的文件夹路径
        sampling_rate (int): 采样率（Hz），默认为 500 Hz
        time_window (tuple): 要显示的时间范围（起始时间，结束时间），单位为秒。默认为 None，表示显示全部时间范围
        baseline_window (tuple): 基线校正的时间窗口（起始时间点，结束时间点），默认为 (0, 100)
        is_baseline (int): 是否进行基线校正，0 表示不校正，1 表示校正
    """
    # 获取文件夹中的所有 .MAT 文件
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    # 初始化一个列表来存储每个文件的 ERP 数据
    all_erps = []
    all_time_axes = []

    # 遍历每个 .MAT 文件
    for mat_file in mat_files:
        mat_file_path = os.path.join(folder_path, mat_file)
        eeg_data = read_eeg_data_from_mat(mat_file_path)
        if eeg_data is not None:
            # 获取数据的维度
            trials, time_points, channels = eeg_data.shape

            # 计算时间轴
            time_axis = np.arange(time_points) / sampling_rate

            # 基线校正
            if is_baseline == 1:
                baseline_mean = np.mean(eeg_data[:, baseline_window[0]:baseline_window[1], :], axis=1, keepdims=True)
                eeg_data = eeg_data - baseline_mean

            # 计算 ERP（对所有试验进行平均）
            erp = np.mean(eeg_data, axis=0)

            # 如果指定了时间范围，则筛选数据
            if time_window is not None:
                start_time, end_time = time_window
                start_index = int(start_time * sampling_rate)
                end_index = int(end_time * sampling_rate)
                time_axis = time_axis[start_index:end_index]
                erp = erp[start_index:end_index, :]

            # 将当前文件的 ERP 数据和时间轴添加到列表中
            all_erps.append(erp)
            all_time_axes.append(time_axis)

    # 检查是否有有效的 ERP 数据
    if not all_erps:
        print("No valid ERP data found in the folder.")
        return

    # 绘制合并的 ERP 图
    plt.figure(figsize=(12, 8))
    for i, erp in enumerate(all_erps):
        time_axis = all_time_axes[i]
        for channel in range(erp.shape[1]):
            plt.plot(time_axis, erp[:, channel], label=f'Channel {channel + 1} (File {i + 1})')

    plt.title('Event-Related Potentials (ERP) from Multiple Files')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.show()


def save_false_npy(file_path):
    """
    保存一个 shape 为 (480,) 且所有值为 False 的 .npy 文件。

    参数：
    - file_path (str): 保存的完整路径（包含文件名和 .npy 后缀）
    """
    data = np.zeros(480, dtype=bool)  # 生成全部为 False 的布尔数组
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 自动创建目录（如不存在）
    np.save(file_path, data)
    print(f"已保存 False 数组到：{file_path}")


def merge_npy_files_in_folder(folder_path):
    """
    合并文件夹中所有 .npy 文件（每个文件去掉第一个样本），按 axis=0 合并，
    并保存为 merged.npy 到该文件夹下。

    :param folder_path: 包含多个 .npy 文件的文件夹路径
    """
    npy_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith('.npy')
    ])

    if not npy_files:
        raise ValueError("该文件夹中没有 .npy 文件")

    # 加载所有数据并合并（每个文件去掉第一个样本）
    data_list = []
    for fname in npy_files:
        full_path = os.path.join(folder_path, fname)
        arr = np.load(full_path)

        if arr.shape[0] < 2:
            raise ValueError(f"{fname} 样本数量不足，无法去掉第一个样本")

        data_list.append(arr[1:])  # 去掉第一个样本

    merged = np.concatenate(data_list, axis=0)
    save_path = os.path.join(folder_path, "merged.npy")
    np.save(save_path, merged)
    print(f"合并完成，保存路径: {save_path}")


def merge_csvs_from_subfolders(folder_path, key, save_csv_path):
    """
    遍历主文件夹下的所有子文件夹，读取其中包含指定关键词 key 的 CSV 文件，
    去掉每个 CSV 的前两行，按顺序合并并保存为一个新 CSV 文件。

    :param folder_path: 主文件夹路径，里面包含多个子文件夹
    :param key: 筛选 CSV 文件名中包含的关键词
    :param save_csv_path: 合并后的 CSV 要保存的完整路径（包含文件名）
    """
    merged_dfs = []

    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('.csv') and key in file:
                    csv_path = os.path.join(subfolder_path, file)
                    df = pd.read_csv(csv_path, skiprows=2, header=None)  # 跳过前两行
                    merged_dfs.append(df)

    if not merged_dfs:
        raise ValueError(f"在 {folder_path} 的子文件夹中找不到包含关键字 '{key}' 的 CSV 文件")

    merged_df = pd.concat(merged_dfs, ignore_index=True)
    merged_df.to_csv(save_csv_path, index=False, header=False)
    print(f"CSV 合并完成，保存至: {save_csv_path}")


def add_label_header(csv_path):
    """
    给一个只有一列的 CSV 文件添加表头 'label'。

    :param csv_path: 目标 CSV 文件路径
    """
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['label']
    df.to_csv(csv_path, index=False)
    print(f"已为 {csv_path} 添加表头 'label'")


if __name__ == "__main__":
    # eeg_data = read_eeg_data_from_mat(
    #     "C:/Learn/Project/bylw/数据/eeg-elife/203_Cannon_FILT_altLow_STIM.mat/203_Cannon_FILT_altLow_STIM.mat")
    # print(eeg_data.shape)

    # plot_erp_from_folder(folder_path="C:/Learn/Project/bylw/数据/eeg-elife",
    #                      is_baseline=1,
    #                      baseline_window=(0, 100),
    #                      time_window=(0.5, 2))

    # save_false_npy("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/bad_epochs/cp/470_bad_epochs.npy")

    # merge_npy_files_in_folder("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/clean_npy/ob")
    # data = np.load("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/clean_npy/ob/merged.npy")
    # print(data.shape)
    # merge_csvs_from_subfolders(folder_path="../data/sub/hc/ob",
    #                            key="outcome_label_remove",
    #                            save_csv_path="../data/sub/hc/ob/merged.csv")
    add_label_header("../data/sub/hc/ob/merged.csv")
