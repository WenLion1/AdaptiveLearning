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


def cut_epoch_two_part(raw,
                       events,
                       tmin=-0.7,
                       tmax=1,
                       event_id=19,
                       is_pre=1):
    """
    将要分析的 EEG 片段和 6 号 mark 之后的 0.2s 片段拼接在一起，并根据 is_pre 参数选择前或后 240 个 epochs。

    :param raw: 原始 EEG 数据 (mne.io.Raw)
    :param events: 事件信息 (array, shape: n_events x 3)
    :param tmin: 事件后的最小时间（默认 -0.7s）
    :param tmax: 事件后的最大时间（默认 1s）
    :param event_id: 要提取的事件 ID（默认 19）
    :param is_pre: 1 表示取前 240 个，0 表示取后 240 个
    :return: 拼接后的 epochs (mne.Epochs)
    """

    print("event_id: ", event_id)
    # 1. 提取目标事件（event_id=19）的 epochs
    epochs_main = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                             baseline=None, preload=True)

    baseline_tmin = 0  # 取 event_id=6 之后0.2s
    baseline_tmax = 0.2

    # 2. 提取基线 epochs
    epochs_baseline = mne.Epochs(raw, events, event_id=event_id + 5, tmin=baseline_tmin, tmax=baseline_tmax,
                                 baseline=None, preload=True)

    # 确保两组 epochs trial 数量匹配
    print("len(epochs_main): ", len(epochs_main))
    print("len(epochs_baseline): ", len(epochs_baseline))

    if len(epochs_main) != len(epochs_baseline):
        raise ValueError("epochs_main 或 epochs_baseline长度不同，无法合并")

    # 获取数据
    main_data = epochs_main.get_data()
    baseline_data = epochs_baseline.get_data()

    # 复制数据并进行移动
    shifted_data = np.copy(baseline_data)
    shifted_data[1:] = baseline_data[:-1]

    # 3. 逐个 trial 拼接（concatenate 方式）
    combined_data = np.concatenate([shifted_data, main_data], axis=-1)

    # 4. 根据 is_pre 参数选择前 240 或后 240 个
    if len(combined_data) < 240:
        raise ValueError("数据量不足 240 个，无法按需求切分")

    if is_pre == 1:
        print("提取前 240 个 epochs")
        combined_data = combined_data[:240]
        # 5. 生成新的 epochs
        new_info = epochs_main.info
        combined_epochs = mne.EpochsArray(combined_data, new_info, tmin=baseline_tmin, events=epochs_main.events[:240])
        print("len(combined_epochs): ", len(combined_epochs))
    elif is_pre == 0:
        print("提取后 240 个 epochs")
        combined_data = combined_data[-240:]
        # 5. 生成新的 epochs
        new_info = epochs_main.info
        combined_epochs = mne.EpochsArray(combined_data, new_info, tmin=baseline_tmin, events=epochs_main.events[-240:])
        print("len(combined_epochs): ", len(combined_epochs))
    new_info = epochs_main.info
    combined_epochs = mne.EpochsArray(combined_data, new_info, tmin=baseline_tmin, events=epochs_main.events)
    print("len(combined_epochs): ", len(combined_epochs))

    return combined_epochs


def cut_epoch_by_od_cp(raw,
                       events,
                       tmin=-0.7,
                       tmax=1,
                       event_id=19):
    """
    只切分od或cp试次

    :param raw:
    :param events:
    :param tmin:
    :param tmax:
    :param event_id:
    :return:
    """


def eeg_ica(epochs,
            n_components=15,
            is_baseline=True,
            baseline_range=(-3, -2.8)):
    """
    ica
    
    :param is_baseline:
    :param baseline_range:
    :param epochs:
    :param n_components: 
    :return: 
    """
    if is_baseline:
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


def extract_events(events,
                   target_event_ids=[23, 24],
                   max_time_diff=10000,
                   end_id=22, ):
    """
    从事件数组中提取符合条件的事件：
    - 选择第一个 `target_event_ids` 之前最近的 `ID=1` 作为起点
    - `ID=1` 必须在目标事件前 `max_time_diff` 以内，否则跳过当前目标事件，继续查找
    - 选择最后一个 `target_event_ids` 之后最近的 `ID=22` 作为终点
    - 如果找不到符合条件的 `ID=1`，返回 None

    :param end_id:
    :param events: 事件数组，形状为 (N, 3)，第一列是时间点，第三列是事件 ID
    :param target_event_ids: 目标事件 ID 列表，例如 [23, 24]
    :param max_time_diff: `ID=1` 到目标事件的最大 **时间差**，超过则跳过该目标事件
    :return: 截取的 `events`（如果找到），否则返回 None
    """
    # 找到所有目标事件（target_event_ids）的索引
    target_idx = np.where(np.isin(events[:, 2], target_event_ids))[0]

    # 记录最终的起点和终点
    start_idx, end_idx = None, None

    for idx in target_idx:
        # 目标事件的时间
        target_time = events[idx, 0]

        # 在当前目标事件之前，寻找最近的 ID=1
        prev_1_idx = np.where(events[:idx, 2] == 1)[0]

        if len(prev_1_idx) > 0:
            last_1_idx = prev_1_idx[-1]  # 取最近的 1 事件
            last_1_time = events[last_1_idx, 0]  # 该 ID=1 的时间

            # 确保 ID=1 到目标事件的时间差不超过 max_time_diff
            if target_time - last_1_time <= max_time_diff:
                start_idx = last_1_idx
                end_idx = idx  # 暂时设置终点
                break  # 找到符合条件的目标事件就退出循环

    if start_idx is not None:
        # 从找到的起点开始，寻找最后一个目标事件
        last_target_idx = target_idx[target_idx >= end_idx][-1]

        # 在最后一个目标事件后，寻找最近的 ID=end_id
        next_22_idx = np.where(events[last_target_idx:, 2] == end_id)[0]

        if len(next_22_idx) > 0:
            end_idx = last_target_idx + next_22_idx[0]  # 取最近的 end_id 事件

        # 截取符合范围的 `events`
        selected_events = events[start_idx:end_idx + 1]
        print("起点索引: ", start_idx, "时间: ", events[start_idx, 0])
        print("终点索引: ", end_idx, "时间: ", events[end_idx, 0])

        return selected_events  # 返回截取的 `events`
    else:
        print("未找到符合条件的事件")
        return None


def batch_eeg_preprocessing(eeg_folder,
                            edge_list,
                            eogs_list,
                            is_two_part=False,
                            is_baseline=True,
                            t_min=-0.7,
                            t_max=0.5,
                            event_id="2   ",
                            save_path="../data/eeg/hc/",
                            baseline_range=(-3, -2.8),
                            is_pre=1, ):
    """
    批量预处理一个文件夹内的eeg

    :param is_pre:
    :param baseline_range:
    :param is_two_part:
    :param is_baseline:
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

            # events = change_one_mark(events=events,
            #                          min=1,
            #                          max=event_id_2, )

            # events = extract_events(events,
            #                         target_event_ids=[event_id_2+8, event_id_2+9],
            #                         end_id=event_id_2+5,)
            # print(events)

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

            # # 切分需要的epoch
            if is_two_part:
                epochs = cut_epoch_two_part(raw,
                                            events,
                                            tmin=t_min,
                                            tmax=t_max,
                                            event_id=event_id_2,
                                            is_pre=is_pre, )
            else:
                epochs = cut_epoch(raw,
                                   events,
                                   tmin=t_min,
                                   tmax=t_max,
                                   event_id=event_id_2, )
            print(epochs)

            # ica
            epochs = eeg_ica(epochs,
                             is_baseline=is_baseline,
                             baseline_range=baseline_range)

            # 保存文件
            save_file_name = sub_name + ".fif"
            save_path_last = os.path.join(save_path, save_file_name)
            save_eeg(epochs,
                     save_path=save_path_last, )


def get_numpy_from_fif(folder_path, is_baseline=False, baseline_range=(-0.2, 0)):
    """
    将 EEG 数据提取为 numpy 矩阵，每个 .fif 文件保存为对应的 .npy 文件，存储在原文件夹的 npy 子文件夹中。

    :param folder_path: .fif 文件所在的文件夹路径
    :param is_baseline: 是否进行基线校正
    :param baseline_range: 基线校正范围，默认(-0.2, 0)
    """
    # 获取所有 .fif 文件
    fif_files = [f for f in os.listdir(folder_path) if f.endswith(".fif")]

    if not fif_files:
        print("没有找到 .fif 文件。")
        return

    # 创建 npy 文件夹
    npy_folder_path = os.path.join(folder_path, 'npy')
    os.makedirs(npy_folder_path, exist_ok=True)
    print(f"所有 .npy 文件将保存至：{npy_folder_path}\n")

    for fif_file in fif_files:
        fif_path = os.path.join(folder_path, fif_file)

        try:
            # 读取 EEG 数据
            epochs = mne.read_epochs(fif_path, preload=True)

            # 是否进行基线校正
            if is_baseline:
                epochs.apply_baseline(baseline_range)

            # 获取 EEG 数据并保存
            data = epochs.get_data(picks=['eeg'])
            npy_file_path = os.path.join(npy_folder_path, os.path.splitext(fif_file)[0] + '.npy')
            np.save(npy_file_path, data)

            print(f"处理完成：{fif_file}, 数据 shape: {data.shape}, 保存至: {npy_file_path}")

        except Exception as e:
            print(f"处理 {fif_file} 时出错: {e}")

    print("\n 所有文件处理完成！")


def autoreject_fif_files(input_folder):
    # 定义输出文件夹路径
    autoreject_folder = os.path.join(input_folder, "autoreject")
    os.makedirs(autoreject_folder, exist_ok=True)
    print(f"清洗后的数据和信息将保存至：{autoreject_folder}")

    # 遍历所有 .fif 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.fif'):
            file_path = os.path.join(input_folder, file_name)
            print(f"\n正在处理文件: {file_name}")

            try:
                # 读取数据
                data = mne.read_epochs(file_path, preload=True)

                # 应用 AutoReject
                ar = AutoReject(consensus=[0.9], random_state=11, verbose=True)
                data_clean, reject_log = ar.fit_transform(data, return_log=True)

                # 保存清洗后的数据
                clean_output_path = os.path.join(autoreject_folder, f"{file_name.replace('.fif', '_clean.fif')}")
                data_clean.save(clean_output_path, overwrite=True)
                print(f"清洗后的数据已保存至：{clean_output_path}")

                # 处理坏 Epoch 信息
                bad_epochs_sum = sum(reject_log.bad_epochs)
                if bad_epochs_sum > 0:
                    print(f"检测到 {bad_epochs_sum} 个坏 Epoch")

                    # # 绘制坏 Epochs
                    # data[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

                    # 保存坏 Epoch 信息
                    bad_epochs_folder = os.path.join(autoreject_folder, "bad_epochs")
                    os.makedirs(bad_epochs_folder, exist_ok=True)
                    bad_epochs_path = os.path.join(bad_epochs_folder, f"{file_name.replace('.fif', '_bad_epochs.npy')}")
                    np.save(bad_epochs_path, reject_log.bad_epochs)
                    print(f"坏 Epoch 信息已保存至：{bad_epochs_path}")
                else:
                    print("没有检测到坏的 Epoch。")

                # # 绘制拒绝日志
                # reject_log.plot('horizontal')

                # # 绘制平均 ERP 图
                # data_clean.average().plot()

                # 保存插值过的通道信息
                interpolated_folder = os.path.join(autoreject_folder, "interpolated_channels")
                os.makedirs(interpolated_folder, exist_ok=True)
                interpolated_channels_path = os.path.join(interpolated_folder,
                                                          f"{file_name.replace('.fif', '_interpolated_channels.npy')}")
                np.save(interpolated_channels_path, reject_log.labels)
                print(f"插值过的通道信息已保存至：{interpolated_channels_path}")

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")


if __name__ == "__main__":
    """
    批量预处理
    # """
    # # 设置边缘电极和eog电极
    # edge_list = ['E1', 'E2', 'E9', 'E14', 'E15', 'E17', 'E21', 'E22', 'E26', 'E32', 'E38', 'E39', 'E43', 'E44', 'E45',
    #              'E48', 'E49', 'E108', 'E113', 'E114', 'E115', 'E119', 'E120', 'E121', 'E125', 'E128']
    # eogs_list = ['E8', 'E25', 'E126', 'E127']
    #
    # save_path = "../data/eeg/hc/test"
    # # 批量预处理
    # batch_eeg_preprocessing(r"C:\Learn\Project\bylw\eeg\1",
    #                         t_min=-1,
    #                         t_max=0.5,
    #                         event_id="2   ",
    #                         is_two_part=True,
    #                         is_baseline=True,
    #                         edge_list=edge_list,
    #                         eogs_list=eogs_list,
    #                         save_path=save_path,
    #                         baseline_range=(0, 0.2),
    #                         is_pre=1,)

    # autoreject_fif_files("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2")

    """
    做ERP
    """
    # # 设置数据文件夹路径
    # data_folder = "../data/eeg/hc/2base_0_1.5_baseline(1)_-0.5_0_ob"  # 修改为你的实际路径
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
    #     epochs.apply_baseline((-0.5, 0))
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

    """
    单个被试ERP
    """
    # # 2base_0_1.5_baseline(1)_-0.5_0_ob
    # epochs = mne.read_epochs("../data/eeg/hc/2base_0_1.5_baseline(1)_-0.5_0_ob/405.fif", preload=True)
    # epochs.apply_baseline((-0.5, 0))
    #
    # evoke = epochs.average()
    # evoke.pick(["E6", "E62"])
    # evoke.plot_joint()

    """
    将eeg的fif转化为numpy
    """
    get_numpy_from_fif(folder_path="../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject",
                       is_baseline=True,
                       baseline_range=(0, 0.2))

    # data = np.load("../data/eeg/hc/2base_-1.5_0.5_nobaseline/eeg_preprocessing_data.npy")
    # print(data.shape)
