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
                trail_type="CP",
                is_show=0, ):
    # 读取 CSV 文件
    data = pd.read_csv(csv_path)

    # 设置 trial 数量
    trials = range(len(data))
    data_len = len(data)

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制 distMean_true 和 outcome_pre 作为线
    plt.plot(trials, data['distMean'], label='Cannon', color='blue', linewidth=2)
    plt.plot(trials, data['pred'], label='prediction_position', color='orange', linewidth=2)

    # 绘制 outcome_true 作为点
    if trail_type == "CP":
        plt.scatter(trials, data['outcome'], label='Cannonball', color='red', marker='o', s=50)
    elif trail_type == "OB":
        oddball_label_added = False  # 用于控制只设置一次标签
        normal_label_added = False
        for i in trials:
            if data['is_oddball'].iloc[i] == 1:
                if not oddball_label_added:
                    plt.scatter(i, data['outcome'].iloc[i], color='green', marker='o', s=50,
                                label='Cannonball (oddball)')
                    oddball_label_added = True  # 标记已添加过标签
                else:
                    plt.scatter(i, data['outcome'].iloc[i], color='green', marker='o', s=50)
            else:
                if not normal_label_added:
                    plt.scatter(i, data['outcome'].iloc[i], color='red', marker='o', s=50,
                                label='Cannonball (normal)')
                    normal_label_added = True  # 标记已添加过标签
                else:
                    plt.scatter(i, data['outcome'].iloc[i], color='red', marker='o', s=50)

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

    if is_show == 1:
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
                             is_nihe=0,
                             is_show=0, ):
    # 设置字体，以支持中文字符
    plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 读取第一个 CSV 文件
    df1 = pd.read_csv(file_path_1)
    # 读取第二个 CSV 文件
    df2 = pd.read_csv(file_path_2)

    # 提取 outcome_true 和 outcome_pre 列
    distMean1 = df1['distMean']
    outcome_true_1 = df1['outcome']
    outcome_pre_1 = df1['pred']
    # 将数据都向上移动一位
    outcome_pre_shifted_1 = outcome_pre_1.shift(-1)

    distMean2 = df2['distMean']
    outcome_true_2 = df2['outcome']
    outcome_pre_2 = df2['pred']
    outcome_pre_shifted_2 = outcome_pre_2.shift(-1)

    outcome_pre_shifted_1_rad = np.deg2rad(outcome_pre_shifted_1)
    outcome_true_1_rad = np.deg2rad(outcome_true_1)
    outcome_pre_shifted_2_rad = np.deg2rad(outcome_pre_shifted_2)
    outcome_true_2_rad = np.deg2rad(outcome_true_2)

    # 计算 signed_diff
    signed_diffs_true_1 = [circ_dist(outcome_pre_shifted_1_rad[i - 1], outcome_true_1_rad[i]) for i in
                           range(1, len(outcome_true_1))]
    signed_diffs_true_1.insert(0, None)  # 插入 None

    signed_diffs_pre_1 = [circ_dist(outcome_pre_shifted_1_rad[i - 1], outcome_pre_shifted_1_rad[i]) for i in
                          range(1, len(outcome_pre_shifted_1))]
    signed_diffs_pre_1.insert(0, None)  # 插入 None

    signed_diffs_true_2 = [circ_dist(outcome_pre_shifted_2_rad[i - 1], outcome_true_2_rad[i]) for i in
                           range(1, len(outcome_true_2))]
    signed_diffs_true_2.insert(0, None)  # 插入 None

    signed_diffs_pre_2 = [circ_dist(outcome_pre_shifted_2_rad[i - 1], outcome_pre_shifted_2_rad[i]) for i in
                          range(1, len(outcome_pre_shifted_2))]
    signed_diffs_pre_2.insert(0, None)  # 插入 None

    # # 调整 signed_diffs_pre_1 的符号，使其与 signed_diffs_true_1 符号一致
    # for i in range(1, len(signed_diffs_true_1)):
    #     if signed_diffs_true_1[i] is not None and signed_diffs_pre_1[i] is not None:
    #         # 检查符号是否一致，如果不一致则调整
    #         if (signed_diffs_true_1[i] >= 0 and signed_diffs_pre_1[i] < 0) or (
    #                 signed_diffs_true_1[i] < 0 and signed_diffs_pre_1[i] >= 0):
    #             signed_diffs_pre_1[i] *= -1  # 反转符号
    #
    # for i in range(1, len(signed_diffs_true_2)):
    #     if signed_diffs_true_2[i] is not None and signed_diffs_pre_2[i] is not None:
    #         if (signed_diffs_true_2[i] >= 0 and signed_diffs_pre_2[i] < 0) or (
    #                 signed_diffs_true_2[i] < 0 and signed_diffs_pre_2[i] >= 0):
    #             signed_diffs_pre_2[i] *= -1  # 反转符号

    # 创建散点图
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 调整 size_factor 的值并更改公式，使中心点的点更小
    size_factor = 30  # 调整大小因子，较之前小一些
    sizes1 = size_factor * (1 + (np.abs(np.array(signed_diffs_true_1[20:])) / 180)) ** 5  # 用平方放大远离中心的点
    sizes2 = size_factor * (1 + (np.abs(np.array(signed_diffs_true_2[20:])) / 180)) ** 5  # 用平方放大远离中心的点

    # 绘制第一个图的散点图
    plt.scatter(signed_diffs_true_1[20:], signed_diffs_pre_1[20:], alpha=0.7, color='orange',
                label='改变点条件', s=sizes1)
    # 绘制第二个图的散点图
    plt.scatter(signed_diffs_true_2[20:], signed_diffs_pre_2[20:], alpha=0.7, color='deepskyblue',
                label='奇异点条件', s=sizes2)

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
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)

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

    if is_show == 1:
        # 显示图形
        plt.show()


def batch_generate_figure(folder_path,
                          figure_type,
                          save_path,
                          figure_mode="combine",
                          is_show=0, ):
    """
    遍历文件夹中的每个子文件夹，调用 paint_2C_combined_figure 函数进行处理

    :param is_show: 生成图像后是否显示，0-不显示，1-显示
    :param figure_mode: 选择图片模式，"single"-CP和OB分别的折线图，"combine"-合并的UP-PE图
    :param folder_path: 文件夹路径
    :param figure_type: str，决定PNG文件命名的类型，可以是 "model" 或 "sub"
    :param save_path: str，PNG文件保存的主路径
    :return:
    """
    # 遍历文件夹中的每个子文件夹
    for subdir, _, files in os.walk(folder_path):
        # 筛选出两个 CSV 文件
        csv_files = [f for f in files if f.endswith('.csv')]

        if len(csv_files) < 2:
            print(f"在文件夹 {subdir} 中找不到两个 CSV 文件，跳过该文件夹。")
            continue  # 如果不是两个文件，跳过此子文件
        elif len(csv_files) > 2:
            print(f"在文件夹 {subdir} 中超过两个 CSV 文件，请重新确认需要画图的两个文件。")
            continue

        # 获取文件路径
        file_path_1 = os.path.join(subdir, csv_files[0])
        file_path_2 = os.path.join(subdir, csv_files[1])

        # 生成 PNG 保存路径和名称
        subfolder_name = os.path.basename(subdir)  # 获取子文件夹名称
        png_save_path = os.path.join(save_path, subfolder_name)  # 拼接新的保存路径

        # 确保子文件夹存在，若不存在则创建
        os.makedirs(png_save_path, exist_ok=True)  # 创建目录

        if figure_mode == "combine":
            if figure_type == "model":
                png_save_name = f"model_{subfolder_name}_{figure_mode}"  # 使用 "model_" 前缀
            elif figure_type == "sub":
                png_save_name = f"sub_{subfolder_name}_{figure_mode}"  # 使用 "sub_" 前缀
            else:
                raise ValueError("figure_type 必须是 'model' 或 'sub'")
            # 调用绘图函数
            paint_2C_combined_figure(file_path_1, file_path_2, png_save_path=png_save_path, png_save_name=png_save_name,
                                     is_show=is_show)
        elif figure_mode == "single":
            if figure_type == "model":
                png_save_name1 = f"model_{subfolder_name}_{figure_mode}_CP"  # 使用 "model_" 前缀
                png_save_name2 = f"model_{subfolder_name}_{figure_mode}_OB"  # 使用 "model_" 前缀
            elif figure_type == "sub":
                png_save_name1 = f"sub_{subfolder_name}_{figure_mode}_CP"  # 使用 "sub_" 前缀
                png_save_name2 = f"sub_{subfolder_name}_{figure_mode}_OB"  # 使用 "sub_" 前缀
            else:
                raise ValueError("figure_type 必须是 'model' 或 'sub'")

            paint_table(csv_path=file_path_1,
                        save_name=png_save_name1,
                        save_path=png_save_path,
                        trail_type="CP",
                        is_show=is_show, )

            paint_table(csv_path=file_path_2,
                        save_name=png_save_name2,
                        save_path=png_save_path,
                        trail_type="OB",
                        is_show=is_show)


def circ_dist(a, b):
    """Compute circular distance between two angles in radians."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def delete_file(folder_path,
                key="", ):
    """
    删除指定文件夹及其子文件夹内所有文件名包含key的文件。

    :param folder_path: 文件夹路径
    :param key: 删除关键字，如果位空则不删除
    :return:
    """

    if not os.path.exists(folder_path):
        print(f"文件夹路径'{folder_path}'不存在！")

    if key == "":
        print("未指定关键字，未删除任何文件。")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if key in file:
                file_path = os.path.join(root, file)

                try:
                    os.remove(file_path)
                    print(f"已删除文件：{file_path}")
                except Exception as e:
                    print(f"删除文件{file}时出错：{e}")


if __name__ == "__main__":
    csv_path = "../results/combine_UNKNOWN_19_16_39_lstm_layers_3_hidden_1024_input_489_cos.csv"
    saving_name = "combine_480_403"
    saving_path = "../results/png/sub/hc/403"
    paint_table(csv_path=csv_path,
                save_name=saving_name,
                save_path=saving_path,
                trail_type="CP", )
    #
    # csv_path = "../results/csv/sub/403/OB_sub.csv"
    # saving_name = "OB_sub"
    # saving_path = "../results/png/sub/403"
    # paint_table(csv_path=csv_path,
    #             save_name=saving_name,
    #             save_path=saving_path,
    #             trail_type="OB", )

    # file_path = "../results/csv/sub/401/combine_CP_LSTM_l_3_h_1024_i_1024.csv"
    # paint_2C_figure(file_path=file_path,
    #                 csv_save_name="combine_CP_LSTM_l_3_h_1024_i_1024",
    #                 png_save_name="combine_CP_LSTM_l_3_h_1024_i_1024",
    #                 png_save_path="../results/png/sub/401",
    #                 ignore_number=20, )
    # paint_2C_combined_figure(
    #     file_path_1="../results/csv/sub/403/combine_CP_4_17_20_lstm_layers_3_hidden_1024_input_10_cos.csv",
    #     file_path_2="../results/csv/sub/403/combine_OB_4_17_20_lstm_layers_3_hidden_1024_input_10_cos.csv",
    #     png_save_path="../results/png/sub/403",
    #     png_save_name="CP_OB_403",
    #     is_nihe=0)
    # delete_file(folder_path="../results/csv/sub/hc",
    #             key="19_16_39")
    #
    # batch_generate_figure(folder_path="../results/csv/sub/hc",
    #                       figure_type="model",
    #                       save_path="../results/png/sub/hc",
    #                       figure_mode="combine",
    #                       is_show=1, )
