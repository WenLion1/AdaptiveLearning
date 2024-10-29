import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def signed_angle_diff(angle1,
                      angle2):
    """
    在圆周上计算两个角度的最短角度差

    """
    # 计算顺时针和逆时针的角度差
    diff = (angle2 - angle1) % 360  # 计算模360后的差值
    # 如果差值大于180，使用逆时针的差值
    if diff > 180:
        diff -= 360
    return diff


def paint_table(csv_path,
                save_path="../results/png",
                save_name="CP_100",
                trail_type="CP", ):
    # 读取 CSV 文件
    data = pd.read_csv(csv_path)

    # 设置 trial 数量
    trials = range(len(data))
    data_len = len(data)

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制 distMean_true 和 outcome_pre 作为线
    plt.plot(trials, data['distMean_true'], label='Cannon', color='blue', linewidth=2)
    plt.plot(trials, data['outcome_pre'], label='prediction_position', color='orange', linewidth=2)

    # 绘制 outcome_true 作为点
    if trail_type == "CP":
        plt.scatter(trials, data['outcome_true'], label='Cannonball', color='red', marker='o', s=50)
    elif trail_type == "OB":
        oddball_label_added = False  # 用于控制只设置一次标签
        normal_label_added = False
        for i in trials:
            if data['is_oddball'].iloc[i] == 1:
                if not oddball_label_added:
                    plt.scatter(i, data['outcome_true'].iloc[i], color='green', marker='o', s=50,
                                label='Cannonball (oddball)')
                    oddball_label_added = True  # 标记已添加过标签
                else:
                    plt.scatter(i, data['outcome_true'].iloc[i], color='green', marker='o', s=50)
            else:
                if not normal_label_added:
                    plt.scatter(i, data['outcome_true'].iloc[i], color='red', marker='o', s=50,
                                label='Cannonball (normal)')
                    normal_label_added = True  # 标记已添加过标签
                else:
                    plt.scatter(i, data['outcome_true'].iloc[i], color='red', marker='o', s=50)

    # 添加标签和标题
    plt.xlabel(f'Trial (0 to {data_len})')
    plt.ylabel('Angle (0 to 400)')
    plt.title(f'{save_name}')
    plt.xlim(0, data_len)  # 设置 x 轴范围
    plt.ylim(0, 400)  # 设置 y 轴范围
    plt.legend(loc='lower right')  # 显示图例并设置位置
    plt.grid()

    # 保存图像
    image_file_path = os.path.join(save_path, f"{save_name}.png")  # 设置图像文件名
    plt.savefig(image_file_path)

    plt.show()


def paint_2C_figure(file_path,
                    csv_save_path="../results/csv/2C",
                    csv_save_name="difference_output",
                    png_save_path="../results/png/2C",
                    png_save_name="difference_output",
                    ignore_number=20,
                    is_nihe=0, ):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 提取 outcome_true 和 outcome_pre 列
    outcome_true = df['outcome_true']
    outcome_pre = df['outcome_pre']
    is_oddball = df['is_oddball']
    # is_changepoint = df['is_changepoint']  # 提取 changepoint 列

    # 将 outcome_pre 列向前移动一位
    outcome_pre_shifted = outcome_pre.shift(-1)

    # 计算有符号角度差
    signed_diffs_true = []
    signed_diffs_pre = []

    # 计算与上一行的有符号角度差
    for i in range(1, len(df)):
        signed_diff_true = signed_angle_diff(outcome_true[i - 1], outcome_true[i])
        signed_diff_pre = signed_angle_diff(outcome_pre_shifted[i - 1], outcome_pre_shifted[i])  # 使用向前移动的列

        signed_diffs_true.append(signed_diff_true)
        signed_diffs_pre.append(signed_diff_pre)

    # 在结果列表中插入 NaN 以保持与原 DataFrame 的长度一致
    signed_diffs_true.insert(0, None)  # 第一行没有上一行，插入 None
    signed_diffs_pre.insert(0, None)  # 第一行没有上一行，插入 None

    # 将结果保存到新的 DataFrame
    result_df = pd.DataFrame({
        'outcome_true': outcome_true,
        'outcome_pre': outcome_pre,
        'outcome_pre_shifted': outcome_pre_shifted,  # 加入向前移动的列
        'signed_diff_true': signed_diffs_true,
        'signed_diff_pre': signed_diffs_pre,
        'is_oddball': is_oddball,
        # 'is_changepoint': is_changepoint,  # 添加 changepoint 列
    })

    # 保存结果到新的 CSV 文件
    os.makedirs(csv_save_path, exist_ok=True)  # 创建保存路径
    output_file_path = os.path.join(csv_save_path, f"{csv_save_name}.csv")
    result_df.to_csv(output_file_path, index=False)

    # 创建散点图，忽略前20行的数据
    plt.figure(figsize=(10, 6))  # 设置图形大小
    # 过滤数据，只保留需要绘制的部分数据
    signed_diffs_true_filtered = signed_diffs_true[ignore_number:]
    signed_diffs_pre_filtered = signed_diffs_pre[ignore_number:]
    is_oddball_filtered = is_oddball[ignore_number:]
    # is_changepoint_filtered = is_changepoint[ignore_number:]  # 过滤 changepoint 列

    # 将is_oddball和is_changepoint转换为列表，避免索引错误
    is_oddball_filtered = is_oddball_filtered.tolist()
    # is_changepoint_filtered = is_changepoint_filtered.tolist()

    # 绘制非oddball点
    normal_mask = [i != 1 for i in is_oddball_filtered]
    oddball_mask = [i == 1 for i in is_oddball_filtered]

    # 绘制正常点
    plt.scatter([sd for i, sd in enumerate(signed_diffs_true_filtered) if normal_mask[i]],
                [sd for i, sd in enumerate(signed_diffs_pre_filtered) if normal_mask[i]],
                alpha=0.7, color='blue', label='Normal Points')

    # 绘制oddball点
    plt.scatter([sd for i, sd in enumerate(signed_diffs_true_filtered) if oddball_mask[i]],
                [sd for i, sd in enumerate(signed_diffs_pre_filtered) if oddball_mask[i]],
                alpha=0.7, color='red', label='Oddball Points')

    # # 绘制changepoint点
    # changepoint_mask = [i == 1 for i in is_changepoint_filtered]
    # plt.scatter([sd for i, sd in enumerate(signed_diffs_true_filtered) if changepoint_mask[i]],
    #             [sd for i, sd in enumerate(signed_diffs_pre_filtered) if changepoint_mask[i]],
    #             alpha=0.7, color='orange', label='Changepoint Points')  # 采用橘色

    if is_nihe == 1:
        # 拟合曲线
        # 对正常点进行线性回归拟合
        x_normal = np.array(signed_diffs_true_filtered)[normal_mask]
        y_normal = np.array(signed_diffs_pre_filtered)[normal_mask]

        # 进一步过滤 NaN 值
        mask_valid = ~np.isnan(y_normal)  # 只选择 y_normal 的有效值
        x_normal = x_normal[mask_valid]
        y_normal = y_normal[mask_valid]

        if len(x_normal) > 0 and len(y_normal) > 0:
            slope_normal, intercept_normal, _, _, _ = stats.linregress(x_normal, y_normal)
            x_fit_normal = np.linspace(-180, 180, 100)
            y_fit_normal = slope_normal * x_fit_normal + intercept_normal
            if len(x_normal) > 0 and len(y_normal) > 0:
                print(f'Slope: {slope_normal}, Intercept: {intercept_normal}')

            plt.plot(x_fit_normal, y_fit_normal, color='green', label='Fit Line Normal')

        # 对oddball点进行线性回归拟合
        x_oddball = np.array(signed_diffs_true_filtered)[oddball_mask]
        y_oddball = np.array(signed_diffs_pre_filtered)[oddball_mask]
        if len(x_oddball) > 0 and len(y_oddball) > 0:
            slope_oddball, intercept_oddball, _, _, _ = stats.linregress(x_oddball, y_oddball)
            x_fit_oddball = np.linspace(-180, 180, 100)
            y_fit_oddball = slope_oddball * x_fit_oddball + intercept_oddball
            plt.plot(x_fit_oddball, y_fit_oddball, color='orange', label='Fit Line Oddball')

    # 设置横坐标和纵坐标的范围
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)

    # 添加标题和标签
    plt.title('Scatter Plot of Signed Angle Differences (Ignoring First 20 Rows)')
    plt.xlabel('Prediction error')
    plt.ylabel('Update')

    # 添加水平黑色虚线
    plt.axhline(0, color='black', linestyle='--')
    # 添加 y=x 的黑色虚线
    plt.axline((0, 0), slope=1, color='black', linestyle='--')

    os.makedirs(png_save_path, exist_ok=True)  # 创建保存路径
    image_file_path = os.path.join(png_save_path, f"{png_save_name}.png")  # 设置图像文件名
    # 保存图形到指定路径
    plt.savefig(image_file_path)

    # 显示图形
    plt.show()


def paint_2C_combined_figure(file_path_1,
                             file_path_2,
                             csv_save_path="../results/csv/2C",
                             csv_save_name="combined_difference_output",
                             png_save_path="../results/png/2C",
                             png_save_name="combined_difference_output",
                             is_nihe=0, ):
    # 读取第一个 CSV 文件
    df1 = pd.read_csv(file_path_1)
    # 读取第二个 CSV 文件
    df2 = pd.read_csv(file_path_2)

    # 提取 outcome_true 和 outcome_pre 列
    distMean1 = df1['distMean_true']
    outcome_true_1 = df1['outcome_true']
    outcome_pre_1 = df1['outcome_pre']
    outcome_pre_shifted_1 = outcome_pre_1.shift(-1)

    distMean2 = df2['distMean_true']
    outcome_true_2 = df2['outcome_true']
    outcome_pre_2 = df2['outcome_pre']
    outcome_pre_shifted_2 = outcome_pre_2.shift(-1)

    # 计算 signed_diff
    signed_diffs_true_1 = [signed_angle_diff(outcome_true_1[i - 1], outcome_true_1[i]) for i in
                           range(1, len(outcome_true_1))]
    signed_diffs_true_1.insert(0, None)  # 插入 None

    signed_diffs_pre_1 = [signed_angle_diff(outcome_pre_shifted_1[i - 1], outcome_pre_shifted_1[i]) for i in
                          range(1, len(outcome_pre_shifted_1))]
    signed_diffs_pre_1.insert(0, None)  # 插入 None

    signed_diffs_true_2 = [signed_angle_diff(outcome_true_2[i - 1], outcome_true_2[i]) for i in
                           range(1, len(outcome_true_2))]
    signed_diffs_true_2.insert(0, None)  # 插入 None

    signed_diffs_pre_2 = [signed_angle_diff(outcome_pre_shifted_2[i - 1], outcome_pre_shifted_2[i]) for i in
                          range(1, len(outcome_pre_shifted_2))]
    signed_diffs_pre_2.insert(0, None)  # 插入 None

    # 调整 signed_diffs_pre_1 的符号，使其与 signed_diffs_true_1 符号一致
    for i in range(1, len(signed_diffs_true_1)):
        if signed_diffs_true_1[i] is not None and signed_diffs_pre_1[i] is not None:
            # 检查符号是否一致，如果不一致则调整
            if (signed_diffs_true_1[i] >= 0 and signed_diffs_pre_1[i] < 0) or (
                    signed_diffs_true_1[i] < 0 and signed_diffs_pre_1[i] >= 0):
                signed_diffs_pre_1[i] *= -1  # 反转符号

    for i in range(1, len(signed_diffs_true_2)):
        if signed_diffs_true_2[i] is not None and signed_diffs_pre_2[i] is not None:
            if (signed_diffs_true_2[i] >= 0 and signed_diffs_pre_2[i] < 0) or (
                    signed_diffs_true_2[i] < 0 and signed_diffs_pre_2[i] >= 0):
                signed_diffs_pre_2[i] *= -1  # 反转符号

    # 创建散点图
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 调整 size_factor 的值并更改公式，使中心点的点更小
    size_factor = 30  # 调整大小因子，较之前小一些
    sizes1 = size_factor * (1 + (np.abs(np.array(signed_diffs_true_1[20:])) / 180)) ** 2  # 用平方放大远离中心的点
    sizes2 = size_factor * (1 + (np.abs(np.array(signed_diffs_true_2[20:])) / 180)) ** 2  # 用平方放大远离中心的点

    # 绘制第一个图的散点图
    plt.scatter(signed_diffs_true_1[20:], signed_diffs_pre_1[20:], alpha=0.7, color='blue',
                label='CP', s=sizes1)
    # 绘制第二个图的散点图
    plt.scatter(signed_diffs_true_2[20:], signed_diffs_pre_2[20:], alpha=0.7, color='red',
                label='OB', s=sizes2)

    if is_nihe == 1:
        x1 = np.array(signed_diffs_true_1[20:]).astype(float)  # 确保为浮点数
        y1 = np.array(signed_diffs_pre_1[20:]).astype(float)
        mask1 = ~np.isnan(x1) & ~np.isnan(y1)  # 过滤 NaN 值
        if np.sum(mask1) > 0:  # 确保有有效的数据点
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1[mask1], y1[mask1])
            x_fit1 = np.linspace(-180, 180, 100)
            y_fit1 = slope1 * x_fit1 + intercept1
            plt.plot(x_fit1, y_fit1, color='green', label='Fit Line OB')

        # 数据集2
        x2 = np.array(signed_diffs_true_2[20:]).astype(float)
        y2 = np.array(signed_diffs_pre_2[20:]).astype(float)
        mask2 = ~np.isnan(x2) & ~np.isnan(y2)  # 过滤 NaN 值
        if np.sum(mask2) > 0:  # 确保有有效的数据点
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2[mask2], y2[mask2])
            x_fit2 = np.linspace(-180, 180, 100)
            y_fit2 = slope2 * x_fit2 + intercept2
            plt.plot(x_fit2, y_fit2, color='orange', label='Fit Line CP')

    # 设置横坐标和纵坐标的范围
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)

    # 添加水平黑色虚线
    plt.axhline(0, color='black', linestyle='--')
    # 添加 y=x 的黑色虚线
    plt.axline((0, 0), slope=1, color='black', linestyle='--')

    # 添加标题和标签
    plt.title('Combined Scatter Plot of Signed Angle Differences')
    plt.xlabel('Prediction error')
    plt.ylabel('Update')

    # 添加图例
    plt.legend()

    os.makedirs(png_save_path, exist_ok=True)  # 创建保存路径
    image_file_path = os.path.join(png_save_path, f"{png_save_name}.png")  # 设置图像文件名
    # 保存图形到指定路径
    plt.savefig(image_file_path)

    # 显示图形
    plt.show()


if __name__ == "__main__":
    # csv_path = "../results/csv/240_rule/combine_CP_GRU_240.csv"
    # saving_name = "combine_CP_GRU_240"
    # saving_path = "../results/png/240_rule"
    # paint_table(csv_path=csv_path,
    #             save_name=saving_name,
    #             save_path=saving_path,
    #             trail_type="CP", )

    # file_path = "../results/csv/240_rule/combine_OB_RNN_240.csv"
    # paint_2C_figure(file_path=file_path,
    #                 csv_save_name="combine_OB_RNN_240",
    #                 png_save_name="combine_OB_RNN_240",
    #                 png_save_path="../results/png/240_rule/combine",
    #                 ignore_number=20, )
    paint_2C_combined_figure(file_path_1="../results/csv/240_rule/combine_CP_LSTM_l_3_h_256_240.csv",
                             file_path_2="../results/csv/240_rule/combine_OB_LSTM_l_3_h_256_240.csv",
                             png_save_path="../results/png/240_rule/combine",
                             png_save_name="combineCP_combineOB_LSTM_l_3_h_256_240",
                             is_nihe=1)
