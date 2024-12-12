import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from util.dataloader_util import draw_image
from util.utils_deep import concatenate_transform_steps


def get_right_angle(angle):
    """
    返回正确的角度值

    :param angle: 角度
    :return: 正确的角度
    """
    return angle % 360


def generate_dist_mean(min_n_trail=5,
                       max_n_trail=15,
                       all_n_trail=3000,
                       cp_min_angle=100,
                       cp_max_angle=180,
                       ob_min_angle=-10,
                       ob_max_angle=10,
                       combine_trail_num=240,
                       trail_type="CP"):
    """
    生成每回合的落点均值（大炮角度）

    :param ob_min_angle: OB情况下的最小角度
    :param ob_max_angle: OB情况下的最大角度
    :param cp_min_angle: 改变点的最小角度
    :param cp_max_angle: 改变点的最大角度
    :param trail_type: 实验类型：CP-changepoint、OB-oddball、combine-混合
    :param min_n_trail: 每个角度持续的最小trail数
    :param max_n_trail: 每个角度持续的最大trail数
    :param all_n_trail: 总共生成多少trail
    :return:
    """
    angles = []  # 用于存储角度
    changepoint_flags = []  # 用于存储changepoint标记
    rules = []  # 用于存储规则标记

    if trail_type == "CP":
        cp_num = 0
        now_trails = 0  # 初始化为0，确保能正确计算
        angle = 0

        while now_trails < all_n_trail:
            if now_trails == 0:
                angle = np.random.randint(0, 360)
                changepoint_flags.append(1)  # 第一个点标记为changepoint
                rules.append(1)  # 第一个点规则标记为1
            else:
                min_value = min(get_right_angle(angle - cp_min_angle), get_right_angle(angle - cp_max_angle))
                max_value = max(get_right_angle(angle - cp_min_angle), get_right_angle(angle - cp_max_angle))
                angle_left = np.random.randint(min_value, max_value)

                min_value = min(get_right_angle(angle + cp_min_angle), get_right_angle(angle + cp_max_angle))
                max_value = max(get_right_angle(angle + cp_min_angle), get_right_angle(angle + cp_max_angle))
                angle_right = np.random.randint(min_value, max_value)

                angle = np.random.choice([angle_left, angle_right])
                changepoint_flags.append(1)  # 选择新角度时标记为changepoint
                cp_num += 1
                # ！！！
                rules.append(1)  # 变更角度后规则标记为0

            n_current_trails = np.random.randint(min_n_trail, max_n_trail + 1)
            if now_trails + n_current_trails > all_n_trail:
                n_current_trails = all_n_trail - now_trails

            angles.extend([angle] * n_current_trails)
            changepoint_flags.extend([0] * (n_current_trails - 1))  # 除第一个点外，剩下的标记为0
            # ！！！
            rules.extend([1] * (n_current_trails - 1))  # 规则标记为0
            now_trails += n_current_trails  # 更新当前的轨迹数
            print("changepoint_num: ", cp_num)

        # 确保最终结果长度为 all_n_trail
        angles = angles[:all_n_trail]
        changepoint_flags = changepoint_flags[:all_n_trail]  # 确保 changepoint 标记的长度也一致
        rules = rules[:all_n_trail]  # 确保规则标记的长度也一致

    elif trail_type == "OB":
        angles.append(random.randint(0, 360))
        rules.append(-1)  # 第一行为-1
        changepoint_flags = [-1] * all_n_trail  # OB情况下，changepoint_flags均为-1

        for _ in range(all_n_trail - 1):
            diff_angle = random.randint(ob_min_angle, ob_max_angle)
            new_angle = get_right_angle(angles[-1] + diff_angle)
            angles.append(new_angle)
            # ！！！
            rules.append(-1)  # 其他为0

    elif trail_type == "constant":
        constant_angle = random.randint(50, 200)
        angles = [constant_angle] * all_n_trail
        rules = [1] + [0] * (all_n_trail - 1)  # 第一行为1，其他为0

    elif trail_type == "combine":
        if all_n_trail % combine_trail_num != 0:
            print("trail总数无法整除每个epoch的个数")
        else:
            flag = 1
            for i in range(int(all_n_trail / combine_trail_num)):
                if flag == 1:
                    now_trails = 0  # 初始化为0，确保能正确计算
                    angle = 0
                    while now_trails < combine_trail_num:
                        if now_trails == 0:
                            angle = np.random.randint(0, 360)
                            changepoint_flags.append(1)  # 第一个点标记为changepoint
                            rules.append(1)  # 第一个点规则标记为1
                        else:
                            min_value = min(get_right_angle(angle - cp_min_angle),
                                            get_right_angle(angle - cp_max_angle))
                            max_value = max(get_right_angle(angle - cp_min_angle),
                                            get_right_angle(angle - cp_max_angle))
                            angle_left = np.random.randint(min_value, max_value)
                            min_value = min(get_right_angle(angle + cp_min_angle),
                                            get_right_angle(angle + cp_max_angle))
                            max_value = max(get_right_angle(angle + cp_min_angle),
                                            get_right_angle(angle + cp_max_angle))
                            angle_right = np.random.randint(min_value, max_value)
                            angle = np.random.choice([angle_left, angle_right])
                            changepoint_flags.append(1)  # 选择新角度时标记为changepoint
                            # !!!
                            rules.append(1)  # 变更角度后规则标记为0
                        n_current_trails = np.random.randint(min_n_trail, max_n_trail + 1)
                        if now_trails + n_current_trails > combine_trail_num:
                            n_current_trails = combine_trail_num - now_trails
                        angles.extend([angle] * n_current_trails)
                        changepoint_flags.extend([0] * (n_current_trails - 1))  # 除第一个点外，剩下的标记为0
                        # !!!
                        rules.extend([1] * (n_current_trails - 1))  # 规则标记为1
                        now_trails += n_current_trails  # 更新当前的轨迹数

                    flag = -1

                elif flag == -1:

                    angles.append(random.randint(0, 360))
                    rules.append(-1)  # 第一行为-1
                    changepoint_flags.extend([-1] * combine_trail_num)  # OB情况下，changepoint_flags均为-1

                    for _ in range(combine_trail_num - 1):
                        diff_angle = random.randint(ob_min_angle, ob_max_angle)
                        new_angle = get_right_angle(angles[-1] + diff_angle)
                        angles.append(new_angle)
                        # !!!
                        rules.append(-1)  # 其他为0
                    flag = 1

    else:
        print("无此实验类型！")

    print(len(angles))
    print(len(changepoint_flags))
    print(len(rules))
    df = pd.DataFrame({'distMean': angles,
                       'is_changepoint': changepoint_flags,
                       'rule': rules})

    return df


def generate_outcome_angle(df,
                           sigma=10,
                           oddball_min=5,
                           oddball_max=15,
                           oddball_angle_range_min=100,
                           oddball_angle_range_max=180,
                           combine_trail_num=240,
                           trail_type="CP"):
    """
    根据distMean生成实际落点

    :param oddball_angle_range_max: 奇异点的最大角度差
    :param oddball_angle_range_min: 奇异点最小角度差
    :param oddball_max: 奇异点最大间隔trail数
    :param oddball_min: 奇异点最小间隔trail数
    :param df: distMean
    :param sigma: 标准差
    :param trail_type: 实验类型：CP-changepoint、OB-oddball、combine-混合
    :return:
    """
    if trail_type == "CP":
        df['outcome'] = df['distMean'].apply(lambda x: int(np.random.normal(loc=x, scale=sigma)))
        df['is_oddball'] = -1
    elif trail_type == "OB":
        ob_num = 0
        df['outcome'] = df['distMean'].apply(lambda x: int(np.random.normal(loc=x, scale=sigma)))
        df['is_oddball'] = 0  # 初始化is_oddball列，非奇异点的trial为0

        # 设定初始位置为一个随机数以间隔产生奇异点
        next_oddball = random.randint(oddball_min, oddball_max)

        for i in range(len(df)):
            if i == next_oddball:
                ob_num += 1
                # 生成奇异点，且该试次为奇异点
                sign = random.choice([-1, 1])
                oddball_angle = df.loc[i, 'distMean'] + sign * random.randint(oddball_angle_range_min,
                                                                              oddball_angle_range_max)
                df.at[i, 'outcome'] = oddball_angle
                df.at[i, 'is_oddball'] = 1  # 标记为奇异点

                # 计算下一个奇异点的试次
                next_oddball += random.randint(oddball_min, oddball_max)
        print("ob_num: ", ob_num)
    elif trail_type == "combine":
        total_trials = len(df)
        num_epochs = total_trials // combine_trail_num
        current_trail_type = "CP"  # 初始化为CP

        for epoch in range(num_epochs):
            start_index = epoch * combine_trail_num

            if current_trail_type == "CP":
                # 处理CP部分
                cp_indices = range(start_index, start_index + combine_trail_num)  # 当前240个试次全部为CP
                df.loc[cp_indices, 'outcome'] = df.loc[cp_indices, 'distMean'].apply(
                    lambda x: int(np.random.normal(loc=x, scale=sigma)))
                df.loc[cp_indices, 'is_oddball'] = -1  # 标记为非奇异点
            else:
                # 处理OB部分
                ob_indices = range(start_index, start_index + combine_trail_num)  # 当前240个试次全部为OB
                df.loc[ob_indices, 'outcome'] = df.loc[ob_indices, 'distMean'].apply(
                    lambda x: int(np.random.normal(loc=x, scale=sigma)))
                df.loc[ob_indices, 'is_oddball'] = 0  # 初始化is_oddball列，非奇异点的trial为0

                # 设定初始位置为一个随机数以间隔产生奇异点
                next_oddball = random.randint(oddball_min, oddball_max) + start_index

                for i in ob_indices:
                    if i == next_oddball:
                        # 生成奇异点，且该试次为奇异点
                        sign = random.choice([-1, 1])
                        oddball_angle = df.loc[i, 'distMean'] + sign * random.randint(oddball_angle_range_min,
                                                                                      oddball_angle_range_max)
                        df.at[i, 'outcome'] = oddball_angle
                        df.at[i, 'is_oddball'] = 1  # 标记为奇异点

                        # 计算下一个奇异点的试次
                        next_oddball += random.randint(oddball_min, oddball_max)

            # 切换下一个试次类型
            current_trail_type = "OB" if current_trail_type == "CP" else "CP"

    return df


# class generate_dataset(Dataset):
#     def __init__(self,
#                  df,
#                  sequence_length=100):
#         self.df = df
#         self.dist_mean = self.df['distMean'].values.astype(float)
#         self.outcome = self.df['outcome'].values.astype(float)
#         self.is_oddball = self.df['is_oddball'].values.astype(float)
#         self.rule = self.df['rule'].values.astype(float)  # 添加对rule列的处理
#         self.sequence_length = sequence_length
#
#     def __len__(self):
#         return len(self.df) - self.sequence_length + 1
#
#     def __getitem__(self, index):
#         if index + self.sequence_length > len(self.df):
#             return None
#         else:
#             distMean = self.dist_mean[index: index + self.sequence_length]
#             outcome = self.outcome[index: index + self.sequence_length]
#             is_oddball = self.is_oddball[index: index + self.sequence_length]
#             rule = self.rule[index: index + self.sequence_length]  # 提取rule列
#
#             return distMean, outcome, is_oddball, rule  # 返回rule

class generate_dataset(Dataset):
    def __init__(self,
                 df,
                 transform_steps=None,
                 image_size=128,
                 fill_empty_space=255,
                 sequence_length=240):
        # if transform_steps is None:
        #     self.transform_steps = concatenate_transform_steps(image_resize=image_size,
        #                                                        fill_empty_space=fill_empty_space,
        #                                                        grayscale=False,
        #                                                        )
        # else:
        #     self.transform_steps = transform_steps
        self.df = df
        self.dist_mean = self.df['distMean'].values.astype(float)
        self.outcome = self.df['outcome'].values.astype(float)
        self.is_oddball = self.df['is_oddball'].values.astype(float)
        self.rule = self.df['rule'].values.astype(float)  # 添加对rule列的处理
        self.sequence_length = sequence_length

    def __len__(self):
        # 返回非重叠的数据块数量
        return (len(self.df) // self.sequence_length)

    def __getitem__(self, index):
        # 无重复的后移，因此索引步长是 sequence_length
        start_index = index * self.sequence_length
        end_index = start_index + self.sequence_length

        if end_index > len(self.df):
            return None  # 防止越界
        else:
            distMean = self.dist_mean[start_index:end_index]
            outcome = self.outcome[start_index:end_index]
            is_oddball = self.is_oddball[start_index:end_index]
            rule = self.rule[start_index:end_index]  # 提取rule列

            # # 将outcome列的值转换为图片
            # images = []
            # for angle in outcome:
            #     image = draw_image(angle)
            #     image = self.transform_steps(image)
            #     # 将图片添加到列表中
            #     images.append(image)
            #
            # images = torch.stack(images, dim=0)

            # 返回图片列表和其他数据
            return distMean, outcome, is_oddball, rule


if __name__ == "__main__":
    # df_train_CP = generate_dist_mean(all_n_trail=240,
    #                                  trail_type="CP")
    # df_train_CP = generate_outcome_angle(df_train_CP,
    #                                      trail_type="CP")
    # df_valid_CP = generate_dist_mean(all_n_trail=240,
    #                                  trail_type="CP")
    # df_valid_CP = generate_outcome_angle(df_valid_CP,
    #                                      trail_type="CP")
    # df_test_CP = generate_dist_mean(all_n_trail=240,
    #                                 trail_type="CP")
    # df_test_CP = generate_outcome_angle(df_test_CP,
    #                                     trail_type="CP")

    # df_train_OB = generate_dist_mean(all_n_trail=240,
    #                                  trail_type="OB")
    # df_train_OB = generate_outcome_angle(df_train_OB,
    #                                      trail_type="OB", )
    # df_valid_OB = generate_dist_mean(all_n_trail=240,
    #                                  trail_type="OB")
    # df_valid_OB = generate_outcome_angle(df_valid_OB,
    #                                      trail_type="OB", )
    # df_test_OB = generate_dist_mean(all_n_trail=240,
    #                                 trail_type="OB")
    # df_test_OB = generate_outcome_angle(df_test_OB,
    #                                     trail_type="OB", )
    #
    # df_train_combine = generate_dist_mean(all_n_trail=240000,
    #                                       trail_type="combine")
    # df_train_combine = generate_outcome_angle(df_train_combine,
    #                                           trail_type="combine", )
    # df_valid_combine = generate_dist_mean(all_n_trail=240000,
    #                                       trail_type="combine")
    # df_valid_combine = generate_outcome_angle(df_valid_combine,
    #                                           trail_type="combine", )
    # df_test_combine = generate_dist_mean(all_n_trail=24000,
    #                                      trail_type="combine")
    # df_test_combine = generate_outcome_angle(df_test_combine,
    #                                          trail_type="combine", )

    # csv_path = "../data/df_test_CP_100.csv"
    # df_OB_in_CP = pd.read_csv(csv_path)
    # df_OB_in_CP = generate_dist_mean(trail_type="CP")
    # df_OB_in_CP = generate_outcome_angle(df_OB_in_CP,
    #                                      trail_type="OB")
    #
    data_dir = '../data/240_rule'

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # df_OB_in_CP.to_csv(os.path.join(data_dir, 'df_valid_OB_in_CP.csv'), index=False)
    # df_OB.to_csv(os.path.join(data_dir, 'df_OB_100.csv'), index=False)
    # df_train_CP.to_csv(os.path.join(data_dir, 'df_train_CP.csv'), index=False)
    # df_valid_CP.to_csv(os.path.join(data_dir, 'df_valid_CP.csv'), index=False)
    # df_test_CP.to_csv(os.path.join(data_dir, 'df_test_CP_temp.csv'), index=False)

    # df_train_OB.to_csv(os.path.join(data_dir, 'df_train_OB.csv'), index=False)
    # df_valid_OB.to_csv(os.path.join(data_dir, 'df_valid_OB.csv'), index=False)
    # df_test_OB.to_csv(os.path.join(data_dir, 'df_test_OB_temp.csv'), index=False)
    #
    # df_train_combine.to_csv(os.path.join(data_dir, 'df_train_combine.csv'), index=False)
    # df_valid_combine.to_csv(os.path.join(data_dir, 'df_valid_combine.csv'), index=False)
    df_test_combine.to_csv(os.path.join(data_dir, 'df_test_combine_100.csv'), index=False)

    # df_train = pd.read_csv(os.path.join(data_dir, "df_train_combine.csv"))
    # #
    # dataset = generate_dataset(df_train,
    #                            transform_steps=None)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # count = 0
    # for i, data in enumerate(dataset):
    #     images, distMean, outcome, is_oddball, rule = data
    #     if i == 0:  # 只显示第一个数据块的图片
    #         for j, img in enumerate(images):
    #             img = np.transpose(img, (1, 2, 0))
    #
    #             # 现在你可以使用imshow来显示图像
    #             plt.imshow(img)
    #             plt.show()
    #
    #             # print(img.shape)
    #         break  # 只显示第一个数据块，然后退出循环
