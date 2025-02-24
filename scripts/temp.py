import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import vonmises

import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt


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

            if len(csv_files) != 2:
                print(f"跳过 {subfolder}：该子文件夹内的 CSV 文件数不是 2 个")
                continue

            # 读取并合并两个 CSV 文件
            df1 = pd.read_csv(csv_files[1])
            df2 = pd.read_csv(csv_files[0])
            merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)

            # 生成输出文件路径（保持原子文件夹名称）
            output_file = os.path.join(subfolder_path, f"combine_{subfolder}.csv")

            # 保存合并后的 CSV 文件
            merged_df.to_csv(output_file, index=False)
            print(f"合并完成：{output_file}")

if __name__ == "__main__":
    # merge_csv_in_subfolders(csv1_path="../data/sub/hc/403/ADL_B_403_DataCP_403.csv",
    #                csv2_path="../data/sub/hc/403/ADL_B_403_DataOddball_403.csv",
    #                output_path="../data/sub/hc/403/combine_403.csv", )

    merge_csv_in_subfolders(parent_folder="../data/sub/hc",
                            output_folder="../data/sub/hc")

    # raw = mne.io.read_epochs_eeglab('C:/Learn/Project/bylw/eeg/2 remove channels + waterprint/lal-hc-453-task.set')
    #
    # print(type(raw))
