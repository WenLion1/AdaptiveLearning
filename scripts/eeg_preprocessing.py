import glob
import os
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from autoreject import AutoReject, Ransac
from autoreject.utils import interpolate_bads
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, find_bad_channels_maxwell
from IPython.display import display, HTML


def load_data(file_path):
    """
    设置脑电帽，加载eeg数据

    :param file_path: eeg数据路径
    :return:
    """
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

    raw = mne.io.read_raw_egi(file_path, preload=True)
    events, event_dict = mne.events_from_annotations(raw)
    raw.set_montage(montage, on_missing='warn')

    return raw, events, event_dict


def change_one_mark(events,
                    min=1,
                    max=17, ):
    """
    将所有1号mark的id改为1，方便后续处理

    :param events:
    :param min: 1号mark最小id
    :param max: 最大id
    :return:
    """
    target_event_ids = list(range(min, max))
    # 遍历修改
    for i in range(len(events)):
        if events[i, 2] in target_event_ids:
            events[i, 2] = 1  # 修改事件 ID 为 1
    # 显示修改后的结果
    print("修改后的 events：", events)

    return events


def del_edge_channels(raw,
                      edge_list,
                      eogs_list, ):
    """
    删除边缘channel，并且添加eogs channel
    :param raw:
    :param edge_list:
    :param eogs_list:
    :return:
    """

    # 去除channel
    raw.drop_channels(edge_list)
    # raw_copy.ch_names
    eogs = {item: 'eog' for item in eogs_list}
    raw.set_channel_types({'VREF': 'misc'})
    raw.set_channel_types(eogs)

    return raw


def eeg_reference(raw,
                  method='average',
                  type='eeg', ):
    """
    重参考
    :param raw:
    :param method: 重参考的方法
    :param type: 重参考的电极
    :return:
    """

    raw.set_eeg_reference(method,
                          ch_type=type)

    return raw


def eeg_filter(raw,
               picks,
               l_freq=0.1,
               h_freq=30,
               fir_design='firwin',
               phase='zero-double', ):
    """
    滤波
    :param raw:
    :param picks:
    :param l_freq:
    :param h_freq:
    :param fir_design:
    :param phase:
    :return:
    """

    raw.filter(l_freq=l_freq,
               h_freq=h_freq,
               fir_design=fir_design,
               phase=phase,
               picks=picks, )
    return raw


def cut_epoch(raw,
              events,
              tmin=-0.7,
              tmax=1,
              event_id=19, ):
    """
    截取需要的epoch
    :param raw:
    :param events:
    :param tmin:
    :param tmax:
    :param event_id:
    :return:
    """

    epochs = mne.Epochs(raw,
                        events,
                        event_id=event_id,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=None,
                        preload=True, )

    return epochs


def eeg_ica(epochs,
            n_components=15,
            baseline_range=(-3, -2.8)):
    """
    ica
    
    :param baseline_range:
    :param epochs:
    :param n_components: 
    :return: 
    """
    epochs.apply_baseline(baseline_range)

    # ica
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, method="infomax", max_iter=3000)
    ica.fit(epochs)

    eog_inds, scores = ica.find_bads_eog(epochs, measure='correlation', threshold=0.4)
    ica.exclude = eog_inds
    print("去除的成分有：", eog_inds)

    epochs = ica.apply(epochs.copy())

    return epochs


def save_eeg(epochs,
             save_path, ):
    """
    保存处理好的eeg数据

    :param epochs:
    :param save_path:
    :return:
    """

    epochs.save(save_path,
                overwrite=True, )
    print("文件已保存至：", save_path)


def batch_eeg_preprocessing(eeg_folder,
                            edge_list,
                            eogs_list,
                            t_min=-0.7,
                            t_max=0.5,
                            event_id="2   ",
                            save_path="../data/eeg/hc/",
                            baseline_range=(-3, -2.8)):
    """
    批量预处理一个文件夹内的eeg

    :param t_min:
    :param t_max:
    :param event_id:
    :param save_path:
    :param eogs_list:
    :param edge_list:
    :param eeg_folder: 装存eeg的文件夹名
    :return:
    """

    for filename in os.listdir(eeg_folder):
        if filename.endswith('.mff'):
            match = re.search(r"lal-hc-(\d+)", filename)
            if match:
                sub_name = match.group(1)
            else:
                print("以下文件未找到被试编号：", filename)
                continue

            eeg_path = os.path.join(eeg_folder, filename)

            # 加载数据
            raw, events, events_id = load_data(eeg_path)

            event_id_2 = events_id.get(event_id)

            events = change_one_mark(events=events,
                                     min=1,
                                     max=event_id_2, )

            # 删除边缘电极以及设置eog电极
            raw = del_edge_channels(raw,
                                    edge_list=edge_list,
                                    eogs_list=eogs_list, )

            # 重定位
            raw = eeg_reference(raw)

            # 滤波
            picks = mne.pick_types(raw.info, eeg=True, eog=True)
            raw = eeg_filter(raw,
                             picks=picks)

            # 切分需要的epoch
            epochs = cut_epoch(raw,
                               events,
                               tmin=t_min,
                               tmax=t_max,
                               event_id=event_id_2, )

            # ica
            epochs = eeg_ica(epochs,
                             baseline_range=baseline_range)

            # 保存文件
            save_file_name = sub_name + ".fif"
            save_path_last = os.path.join(save_path, save_file_name)
            save_eeg(epochs,
                     save_path=save_path_last, )


def get_numpy_from_fif(folder_path,
                       save_folder_path,
                       is_baseline=False,
                       baseline_range=(-0.2, 0),):
    """
    将eeg数据提取为numpy矩阵, 要跑一点时间

    :param baseline_range:
    :param is_baseline:
    :param save_folder_path:
    :param folder_path:
    :return:
    """

    fif_files = [f for f in os.listdir(folder_path) if f.endswith(".fif")]
    eeg_data_list = []

    for fif_file in fif_files:
        fif_path = os.path.join(folder_path, fif_file)

        epochs = mne.read_epochs(fif_path, preload=True)

        if is_baseline:
            # 应用基线校正
            epochs.apply_baseline(baseline_range)

        data = epochs.get_data(picks=['eeg'])

        eeg_data_list.append(data)

        print(f"Loaded {fif_file}: shape {data.shape}")

    eeg_data_list = np.array(eeg_data_list)
    if is_baseline:
        save_path = os.path.join(save_folder_path, "eeg_preprocessing_data_baseline.npy")
    else:
        save_path = os.path.join(save_folder_path, "eeg_preprocessing_data.npy")
    np.save(save_path, eeg_data_list)


if __name__ == "__main__":

    # # 设置边缘电极和eog电极
    # edge_list = ['E1', 'E2', 'E9', 'E14', 'E15', 'E17', 'E21', 'E22', 'E26', 'E32', 'E38', 'E39', 'E43', 'E44', 'E45',
    #              'E48', 'E49', 'E108', 'E113', 'E114', 'E115', 'E119', 'E120', 'E121', 'E125', 'E128']
    # eogs_list = ['E8', 'E25', 'E126', 'E127']
    #
    # save_path = "../data/eeg/hc/2base_-3_0_baseline_-3_-2.8"
    # # 批量预处理
    # batch_eeg_preprocessing(r"C:\Learn\Project\bylw\eeg\1",
    #                         t_min=-3,
    #                         t_max=0,
    #                         event_id="2   ",
    #                         edge_list=edge_list,
    #                         eogs_list=eogs_list,
    #                         save_path=save_path,
    #                         baseline_range=(-3, -2.8))

    # # 设置数据文件夹路径
    # data_folder = "../data/eeg/hc"  # 修改为你的实际路径
    #
    # # 获取所有 .fif 文件
    # file_list = glob.glob(os.path.join(data_folder, "*.fif"))
    #
    # # 存储所有 evoke 对象的列表
    # evokes = []
    #
    # # 遍历所有 .fif 文件
    # for file_path in file_list:
    #     print(f"Processing file: {file_path}")
    #
    #     # 读取 .fif 文件
    #     epochs = mne.read_epochs(file_path, preload=True)
    #
    #     # 应用基线校正
    #     epochs.apply_baseline((-0.2, 0))
    #
    #     # 计算 ERP（evoked response）
    #     evoke = epochs.average()
    #
    #     # 选择需要的通道（这里是所有通道，你也可以指定某些通道）
    #     evokes.append(evoke)
    #
    # # 确保至少有一个 evoke 计算完成
    # if len(evokes) == 0:
    #     print("No .fif files found or processed!")
    #     exit()
    #
    # # 对所有 evoke 进行 **通道平均**
    # evoke_mean = mne.grand_average(evokes)
    #
    # # 选择特定通道（E6, E62）
    # evoke_mean.pick(["E6", "E62"])
    #
    # # 绘制最终平均 ERP 曲线
    # evoke_mean.plot_joint(title="Grand Average ERP")

    # epochs = mne.read_epochs("../data/eeg/hc/462.fif", preload=True)
    # epochs.apply_baseline((-0.2, 0))
    #
    # evoke = epochs.average()
    # # evoke.pick(["E6", "E62"])
    # evoke.plot_joint()

    # 将eeg的fif转化为numpy
    get_numpy_from_fif(folder_path="../data/eeg/hc/2base_-3_0_baseline_-3_-2.8",
                       save_folder_path="../data/eeg/hc/2base_-3_0_baseline_-3_-2.8",
                       is_baseline=False,
                       baseline_range=(-1.5, -1.3))

    data = np.load("../data/eeg/hc/2base_-3_0_baseline_-3_-2.8/eeg_preprocessing_data.npy")
    print(data.shape)


