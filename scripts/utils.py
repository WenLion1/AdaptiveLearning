# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:42:26 2023

@author: ning
"""

import os, torch
import re
import shutil

import h5py
import mne
import numpy as np
import pandas as pd
import scipy.io

torch.manual_seed(12345)
np.random.seed(12345)

import torch.nn as nn
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score

from typing import List, Callable, Union, Any, TypeVar, Tuple

###############################################################################
Tensor = TypeVar('torch.tensor')
###############################################################################
invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1., 1., 1.]),
                               ])
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


###############################################################################
def noise_func(x, noise_level):
    """
    add guassian noise to the images during agumentation procedures
    Inputs
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, level of noise, between 0 and 1
    """

    generator = torch.distributions.Normal(x.mean(), x.std())
    noise = generator.sample(x.shape)
    new_x = x * (1 - noise_level) + noise * noise_level
    new_x = torch.clamp(new_x, x.min(), x.max(), )
    return new_x


def concatenate_transform_steps(image_resize: int = 128,
                                num_output_channels: int = 3,
                                noise_level: float = 0.,
                                flip: bool = False,
                                rotate: float = 0.,
                                fill_empty_space: int = 255,
                                grayscale: bool = True,
                                center_crop: bool = False,
                                center_crop_size: tuple = (1200, 1200),
                                ):
    """
    from image to tensors

    Parameters
    ----------
    image_resize : int, optional
        DESCRIPTION. The default is 128.
    num_output_channels : int, optional
        DESCRIPTION. The default is 3.
    noise_level : float, optional
        DESCRIPTION. The default is 0..
    flip : bool, optional
        DESCRIPTION. The default is False.
    rotate : float, optional
        DESCRIPTION. The default is 0.,
    fill_empty_space : int, optional
        DESCRIPTION. The defaultis 130.
    grayscale: bool, optional
        DESCRIPTION. The default is True.
    center_crop : bool, optional
        DESCRIPTION. The default is False.
    center_crop_size : Tuple, optional
        DESCRIPTION. The default is (1200, 1200)

    Returns
    -------
    transformer_steps : TYPE
        DESCRIPTION.

    """
    transformer_steps = []
    # crop the image - for grid like layout
    if center_crop:
        transformer_steps.append(transforms.CenterCrop(center_crop_size))
    # resize
    transformer_steps.append(transforms.Resize((image_resize, image_resize)))
    # flip
    if flip:
        transformer_steps.append(transforms.RandomHorizontalFlip(p=.5))
        transformer_steps.append(transforms.RandomVerticalFlip(p=.5))
    # rotation
    if rotate > 0.:
        transformer_steps.append(transforms.RandomRotation(degrees=rotate,
                                                           fill=fill_empty_space,
                                                           ))
    # grayscale
    if grayscale:
        transformer_steps.append(  # it needs to be 3 if we want to use pretrained CV models
            transforms.Grayscale(num_output_channels=num_output_channels)
        )
    # rescale to [0,1] from int8
    transformer_steps.append(transforms.ToTensor())
    # add noise
    if noise_level > 0:
        transformer_steps.append(transforms.Lambda(lambda x: noise_func(x, noise_level)))
    # normalization
    if num_output_channels == 3:
        transformer_steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                                 )
    elif num_output_channels == 1:
        transformer_steps.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        transformer_steps = transforms.Compose(transformer_steps)
    return transformer_steps


invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1., 1., 1.]),
                               ])


def generator(image_size=224, lamda=4, thetaRad_base=45, ):
    """
    Inputs
    -------------
    image_size: int, image size
    lamda: float, better be in range between 4 and 32
    thetaRad_base:float, base value of theta, in degrees
    """

    # convert degree to pi-based
    thetaRad = thetaRad_base * np.pi / 180
    # Sanjeev's algorithm
    X = np.arange(image_size)
    X0 = (X / image_size) - .5
    freq = image_size / lamda
    Xf = X0 * freq * 2 * np.pi
    sinX = np.sin(Xf)
    Xm, Ym = np.meshgrid(X0, X0)
    Xt = Xm * np.cos(thetaRad)
    Yt = Ym * np.sin(thetaRad)
    XYt = Xt + Yt
    XYf = XYt * freq * 2 * np.pi

    grating = np.sin(XYf)

    s = 0.075
    w = np.exp(-(0.3 * ((Xm ** 2) + (Ym ** 2)) / (2 * s ** 2))) * 2
    w[w > 1] = 1
    gabor = ((grating - 0.5) * w) + 0.5

    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gabor, cmap=plt.cm.gray)
    ax.axis('off')
    return fig2img(fig)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def determine_training_stops(nets: List[nn.Module],
                             idx_epoch: int,
                             warmup_epochs: int,
                             valid_loss: Tensor,
                             counts: int = 0,
                             best_valid_loss=np.inf,
                             tol: float = 1e-4,
                             saving_names: dict = {'model_saving_name': True, },
                             ) -> Tuple[Tensor, int]:
    """
   

    Parameters
    ----------
    nets : List[nn.Module]
        DESCRIPTION. 
    idx_epoch : int
        DESCRIPTION.
    warmup_epochs : int
        DESCRIPTION.
    valid_loss : Tensor
        DESCRIPTION.
    counts : int, optional
        DESCRIPTION. The default is 0.
    best_valid_loss : TYPE, optional
        DESCRIPTION. The default is np.inf.
    tol : float, optional
        DESCRIPTION. The default is 1e-4.
    saving_names : dict, optional
        DESCRIPTION. The default is {'model_saving_name':True,}.

    Returns
    -------
    Tuple[Tensor,int]
        DESCRIPTION.

    """
    if idx_epoch >= warmup_epochs:  # warming up
        temp = valid_loss
        if np.logical_and(temp < best_valid_loss, np.abs(best_valid_loss - temp) >= tol):
            best_valid_loss = valid_loss
            for net, (saving_name, save_model) in zip(nets, saving_names.items()):
                if save_model:
                    torch.save(net.state_dict(), saving_name)  # why do i need state_dict()?
                    print(f'save {saving_name}')
            counts = 0
        else:
            counts += 1

    return best_valid_loss, counts


def add_noise_instance_for_training(batch_features: Tensor,
                                    n_noise: int = 1,
                                    clip_output: bool = False,
                                    ) -> Tensor:
    """
    

    Parameters
    ----------
    batch_features : Tensor
        DESCRIPTION.
    n_noise : int, optional
        DESCRIPTION. The default is 1.
    clip_output : bool, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    batch_features : Tensor
        DESCRIPTION.

    """
    if n_noise > 0:
        noise_generator = torch.distributions.normal.Normal(batch_features.mean(),
                                                            batch_features.std(), )
        noise_features = noise_generator.sample(batch_features.shape)[:n_noise]
        if clip_output:
            temp = invTrans(batch_features[:n_noise])
            idx_pixels = torch.where(temp == 1)
            temp = invTrans(noise_features)
            temp[idx_pixels] = 1
            noise_features = normalizer(temp)
        batch_features = torch.cat([batch_features, noise_features])
    else:
        pass
    return batch_features


def classify_file_by_endwith_num(folder_path, ):
    """
    根据文件末尾数字将其分类到对应数字命名的文件夹内（不存在该文件夹则创建）

    :param folder_path: 文件夹地址
    :return:
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        raise ValueError(f"文件夹 '{folder_path}' 不存在。")

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 确保处理的是文件而不是子文件夹
        if os.path.isfile(file_path):
            # 使用正则表达式匹配文件名结尾的数字
            match = re.search(r'(\d+)(?=\.\w+$)', filename)

            if match:
                # 获取文件名中的数字
                folder_num = match.group(1)

                # 创建对应的子文件夹路径
                target_folder = os.path.join(folder_path, folder_num)

                # 如果子文件夹不存在，则创建
                os.makedirs(target_folder, exist_ok=True)

                # 移动文件到对应的子文件夹
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"已移动文件 {filename} 到文件夹 {target_folder}")


def get_mat_csv_batch(file_path,
                      key_names,
                      column_names,
                      csv_save_path, ):
    """
    批量将mat文件转换为csv文件

    :param file_path: mat文件夹路径
    :param key_name: 需要转换的key list, 若为空则转换该mat的所有key_name
    :param column_names: 需要转换的列名 list
    :param csv_save_path: csv保存路径（文件夹）
    :return:
    """

    os.makedirs(csv_save_path, exist_ok=True)

    for file_name in os.listdir(file_path):
        if file_name.endswith(".mat"):
            mat_file_path = os.path.join(file_path, file_name)
            base_name = os.path.splitext(file_name)[0]
            mat_data = scipy.io.loadmat(mat_file_path)
            current_key_names = key_names if key_names else [key for key in mat_data.keys() if not key.startswith("__")]

            for key_name in current_key_names:
                csv_file_name = f"{base_name}_{key_name}.csv"
                save_path = os.path.join(csv_save_path, csv_file_name)

                try:
                    get_mat_to_csv(mat_file_path=mat_file_path,
                                   key_name=key_name,
                                   column_names=column_names,
                                   csv_save_path=save_path,
                                   change_num=-1, )
                except ValueError as e:
                    print(f"跳过文件{file_name}, 错误信息：{e}")
    print("批量处理完毕")


def get_mat_to_csv(mat_file_path,
                   key_name,
                   column_names,
                   csv_save_path,
                   change_num=-1, ):
    """
    提取一个mat文件数据到csv中

    :param change_num: 如果遇到Nan，则替换成对应数字
    :param mat_file_path: mat文件路径
    :param key_name: 需要提取的文件key
    :param column_names: 需要提取的列 list
    :param csv_save_path: csv文件保存路径
    :return:
    """

    # 加载mat文件
    mat_data = scipy.io.loadmat(mat_file_path)

    # 检查键名
    if key_name not in mat_data:
        raise ValueError(f"MATLAB文件中找不到键‘{key_name}’，请确认数据的键名。")

    # 提取数据
    data = mat_data[key_name]

    # 创建一个空字典以存储提取的数据
    extracted_data = {}

    # 遍历每个列名并提取对应的数据
    for column in column_names:
        if data[f"{column}"] is not None:
            # 获取并展平数据
            column_data = data[column][0][0].flatten()

            # 检查该列数据是否全为 NaN
            if np.isnan(column_data).all():
                # 如果该列全为 NaN，则用 -1 替换
                column_data = np.full_like(column_data, -1, dtype=float)

            extracted_data[column] = column_data  # 添加到提取的数据字典中
        else:
            raise ValueError(f"指定的列名‘{column}’在数据中不存在。")

    # 将提取的数据转换为DataFrame
    df = pd.DataFrame(extracted_data)

    # 保存为CSV文件
    df.to_csv(csv_save_path, index=False)
    print(f"数据已保存到 {csv_save_path}。")


def add_rule(folder_path):
    """
    遍历指定文件夹及其所有子文件夹的CSV文件，按文件名的规则添加"rule"列

    :param folder_path: 需要处理的文件夹路径
    :return:
    """

    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(folder_path):
        for filename in files:
            # 只处理CSV文件
            if filename.endswith('.csv'):
                # 生成文件的完整路径
                file_path = os.path.join(root, filename)

                # 判断文件名是否包含 "Oddball" 或 "CP"
                if "Oddball" in filename:
                    rule_value = -1
                elif "CP" in filename:
                    rule_value = 1
                else:
                    continue  # 如果文件名不包含指定关键词，跳过

                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 添加"rule"列，并设置相应的值
                df['rule'] = rule_value

                # 保存更新后的文件
                df.to_csv(file_path, index=False)
                print(f"已更新文件：{file_path}，添加 rule 列，值为 {rule_value}")


def change_oddball_to_isoddball(folder_name):
    """
    遍历指定文件夹及其子文件夹中的所有CSV文件，
    将其中的 'oddball' 列名修改为 'is_oddball'。

    参数：
    - folder_name: str，目标文件夹的路径。
    """
    # 遍历文件夹及子文件夹
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # 读取CSV文件
                    df = pd.read_csv(file_path)

                    # 检查是否存在 'oddball' 列
                    if 'oddBall' in df.columns:
                        # 修改列名
                        df.rename(columns={'oddBall': 'is_oddball'}, inplace=True)

                        # 将修改后的数据保存回CSV文件
                        df.to_csv(file_path, index=False)
                        print(f"已修改文件: {file_path} 中的 'oddBall' 列为 'is_oddball'")
                    else:
                        print(f"文件: {file_path} 中未找到 'oddBalcl' 列")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")


def merge_csv_rows_in_folder(folder_path,
                             output_path):
    """
    将指定文件夹及其子文件夹内所有以 ADL 开头的 CSV 文件按行合并并保存到一个新文件中。

    :param folder_path: 根文件夹路径
    :param output_path: 合并后保存的文件路径
    """
    # 用于存储所有符合条件的 DataFrame
    all_dfs = []

    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 判断文件是否以 ADL 开头且是 CSV 文件
            if file.startswith("ADL") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"正在处理文件: {file_path}")
                # 读取 CSV 文件并存入列表
                df = pd.read_csv(file_path)
                all_dfs.append(df)

    # 合并所有 DataFrame
    if all_dfs:
        merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        # 保存到指定的输出路径
        merged_df.to_csv(output_path, index=False)
        print(f"合并完成，结果已保存到 {output_path}")
    else:
        print("未找到符合条件的文件！")


def update_is_changepoint_in_place(csv_file):
    """
    处理 CSV 文件中的数据：
    1. 如果不存在 `is_changepoint` 列，则添加此列。
    2. 当 `is_oddball = -1` 时，根据 `distMean` 列的变化情况更新 `is_changepoint` 列：
       - 如果 `distMean` 发生变化，则将 `is_changepoint` 设置为 1。
       - 否则设置为 0。
    3. 当 `is_oddball != -1` 时，将 `is_changepoint` 设置为 -1。

    :param csv_file: 输入的 CSV 文件路径，处理结果将直接覆盖原文件。
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 确保必要的列存在
    if not {'is_oddball', 'distMean'}.issubset(df.columns):
        raise ValueError("CSV 文件必须包含 'is_oddball' 和 'distMean' 列")

    # 如果 `is_changepoint` 列不存在，则添加此列，默认值为 0
    if 'is_changepoint' not in df.columns:
        df['is_changepoint'] = 0

    # 处理 `is_oddball = -1` 的行
    oddball_mask = df['is_oddball'] == -1
    distMean_values = df.loc[oddball_mask, 'distMean']

    # 找到 `distMean` 列变化的行
    distMean_changes = distMean_values != distMean_values.shift()

    # 更新 `is_changepoint` 列
    df.loc[oddball_mask, 'is_changepoint'] = distMean_changes.astype(int)

    # 将 `is_oddball != -1` 的行的 `is_changepoint` 设置为 -1
    df.loc[~oddball_mask, 'is_changepoint'] = -1

    # 将结果保存回原文件
    df.to_csv(csv_file, index=False)
    print(f"文件 {csv_file} 处理完成。")


def merge_data_files_per_subfolder(root_folder,
                                   first_file_type='DataCP'):
    """
    遍历 root_folder 下的所有子文件夹，查找 DataCP 和 DataOddball 文件，
    合并其内容（去掉第二个文件的第一行），并保存到当前子文件夹中。

    参数：
        root_folder (str): 根目录路径。
        first_file_type (str): 'DataCP' 或 'DataOddball'，决定合并顺序。
    """
    assert first_file_type in ['DataCP', 'DataOddball'], "first_file_type must be 'DataCP' or 'DataOddball'"

    for dirpath, dirnames, filenames in os.walk(root_folder):
        cp_file = None
        oddball_file = None

        for fname in filenames:
            if 'DataCP' in fname:
                cp_file = os.path.join(dirpath, fname)
            elif 'DataOddball' in fname:
                oddball_file = os.path.join(dirpath, fname)

        if cp_file and oddball_file:
            with open(cp_file, 'r', encoding='utf-8') as f_cp, open(oddball_file, 'r', encoding='utf-8') as f_odd:
                cp_lines = f_cp.readlines()
                odd_lines = f_odd.readlines()

                if first_file_type == 'DataCP':
                    merged_lines = cp_lines + odd_lines[1:]  # 去掉 oddball 表头
                    suffix = '_reverse'
                else:
                    merged_lines = odd_lines + cp_lines[1:]  # 去掉 cp 表头
                    suffix = ''

                subfolder_name = os.path.basename(dirpath.rstrip('/\\'))
                save_filename = f"combine_{subfolder_name}{suffix}.csv"
                save_path = os.path.join(dirpath, save_filename)

                with open(save_path, 'w', encoding='utf-8') as f_out:
                    f_out.writelines(merged_lines)

                print(f"[已保存] {save_path}")


def extract_bad_epochs(mat_folder,
                       save_folder,
                       total_trials=480):
    """
    在原文数据中提取bad_epochs数组

    :param mat_folder:
    :param save_folder:
    :param total_trials:
    :return:
    """
    os.makedirs(save_folder, exist_ok=True)

    for filename in os.listdir(mat_folder):
        if filename.endswith(".mat"):
            mat_path = os.path.join(mat_folder, filename)

            try:
                with h5py.File(mat_path, 'r') as f:
                    # epochNumbers 是个引用对象，需要转置并转换成1D numpy数组
                    if 'epochNumbers' not in f:
                        print(f"Warning: 'epochNumbers' not found in {filename}, skipping.")
                        continue

                    epoch_numbers = f['epochNumbers'][()]
                    epoch_numbers = np.array(epoch_numbers).flatten().astype(int)
                    epoch_numbers = epoch_numbers - 1  # MATLAB 索引从 1 开始，Python 从 0 开始

                    bad_epochs = np.ones(total_trials, dtype=bool)
                    bad_epochs[epoch_numbers] = False

                    prefix = filename.split('_')[0]
                    save_name = f"{prefix}_bad_epochs.npy"
                    save_path = os.path.join(save_folder, save_name)

                    np.save(save_path, bad_epochs)
                    print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")


def convert_mat_to_clean_npy(mat_folder, save_folder):
    """
    将MAT文件中的EEG数据读取并转换为形状为 (epoch, channel, time) 的 numpy 数组，并保存为 .npy 文件。

    参数：
        mat_folder: str，包含 .mat 文件的文件夹路径
        save_folder: str，输出 .npy 文件保存路径
    """
    os.makedirs(save_folder, exist_ok=True)

    for filename in os.listdir(mat_folder):
        if filename.endswith(".mat"):
            mat_path = os.path.join(mat_folder, filename)
            try:
                with h5py.File(mat_path, 'r') as f:
                    # 读取 EEG.data 字段
                    eeg_group = f['EEG']
                    data_ref = eeg_group['data']
                    data = np.array(data_ref)  # 原始形状: (epoch, time, channel)

                    # 转为 (epoch, channel, time)
                    data = np.transpose(data, (0, 2, 1))

                    # 提取前缀数字作为保存文件名
                    match = re.match(r"(\d+)_", filename)
                    if match:
                        file_id = match.group(1)
                    else:
                        file_id = os.path.splitext(filename)[0]

                    save_path = os.path.join(save_folder, f"{file_id}_clean.npy")
                    np.save(save_path, data)
                    print(f"保存成功: {save_path}")

            except Exception as e:
                print(f"跳过文件 {filename}，因为读取失败: {e}")


if __name__ == "__main__":
    """
    将被试行为mat文件转化为csv文件供模型测试
    """
    # get_mat_csv_batch(file_path="C:/Learn/Project/bylw/cannonBehavData_forDryad",
    #                   key_names=[],
    #                   column_names=["distMean", "outcome", "pred", "oddBall"],
    #                   csv_save_path="../data/sub/yuanwen", )
    #
    # add_rule(folder_path="../data/sub/yuanwen")
    #
    # change_oddball_to_isoddball(folder_name="../data/sub/yuanwen")
    #
    # classify_file_by_endwith_num(folder_path="../data/sub/yuanwen")
    #
    # merge_csv_rows_in_folder("../data/sub/hc",
    #                          "../data/sub/hc/all_combine_sub.csv")

    # merge_data_files_per_subfolder(root_folder="../data/sub/yuanwen",
    #                                first_file_type="DataOddball", )

    # update_is_changepoint_in_place("../data/sub/hc/404/ADL_B_404_DataCP_404.csv")

    """
    在原文数据中提取bad_epochs数组
    """
    # extract_bad_epochs(mat_folder="C:/Learn/Project/bylw/数据/eeg-elife",
    #                    save_folder="../data/eeg/yuanwen_need/bad_epochs",
    #                    )

    """
    将eeg数据从mat中提取出来并保存
    """
    convert_mat_to_clean_npy(mat_folder="C:/Learn/Project/bylw/数据/eeg-elife",
                             save_folder="../data/eeg/yuanwen_need/npy")
