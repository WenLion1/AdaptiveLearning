import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import vonmises

import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

def merge_csv_rows(csv1_path, csv2_path, output_path):
    """
    将两个 CSV 文件按行合并并保存到一个新文件。

    :param csv1_path: 第一个 CSV 文件路径
    :param csv2_path: 第二个 CSV 文件路径
    :param output_path: 合并后保存的文件路径
    """
    # 读取两个 CSV 文件
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # 合并两个 DataFrame，按行连接
    merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)

    # 保存合并结果到新的 CSV 文件
    merged_df.to_csv(output_path, index=False)
    print(f"合并完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    # merge_csv_rows(csv1_path="../data/sub/hc/403/ADL_B_403_DataCP_403.csv",
    #                csv2_path="../data/sub/hc/403/ADL_B_403_DataOddball_403.csv",
    #                output_path="../data/sub/hc/403/combine_403.csv",)

    raw = mne.io.read_epochs_eeglab('C:/Learn/Project/bylw/eeg/2 remove channels + waterprint/lal-hc-453-task.set')

    print(type(raw))