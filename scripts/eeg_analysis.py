import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat

if __name__ == "__main__":
    # 读取 .mat 文件
    mat_file_path = '../data/eeg/hc/lal-hc-403-task.mat'  # 替换为你的 .mat 文件路径
    mat_contents = loadmat(mat_file_path)

    # 查看文件的顶级结构
    print(mat_contents.keys())

    # 提取 eeg 字段
    if 'EEG' in mat_contents:
        eeg_data = mat_contents['EEG']['data'][0, 0]  # 提取 data 字段
        print("EEG Data Shape:", eeg_data.shape)  # 打印数据的形状
    else:
        print("eeg 字段未找到")

    print(mat_contents['EEG'].dtype.names)