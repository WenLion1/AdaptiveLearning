import random

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from scipy.stats import ttest_ind, norm, stats
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from scripts.test import evaluate_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torch
import numpy as np

import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def svm_hidden_states(hidden_states_dir, test_size=0.2):
    """
    用svm解码rnn隐藏层，区分OB和CP两种情况，并可视化决策边界

    :param test_size: 测试集占总体集合比例
    :param hidden_states_dir: 隐藏层的存储目录
    :return:
    """
    # 加载隐藏层
    hidden_states = torch.load(hidden_states_dir)

    # 生成标签
    length_per_label = 240
    num_labels = hidden_states.shape[0] // (2 * length_per_label)
    labels = np.tile([0] * length_per_label + [1] * length_per_label, num_labels)

    # 区分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(hidden_states, labels, test_size=test_size, random_state=2001)

    # 训练SVM模型
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train, Y_train)

    # 预测
    Y_pred = svm_model.predict(X_test)

    # 计算并输出更多性能指标
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    print(f"SVM模型在测试集上的准确率: {accuracy:.4f}")
    # print(f"精确率: {precision:.4f}")
    # print(f"召回率: {recall:.4f}")
    # print(f"F1得分: {f1:.4f}")


def svm_hidden_states_singlepoint(hidden_states_dir,
                                  label_csv,
                                  type="CP", ):
    """
    用svm解码rnn隐藏层，区分当前trial是OB\CP，使用交叉验证

    :param type: 当前是区分CP还是OB
    :param label_csv: svm标签所存放的csv
    :param hidden_states_dir: 隐藏层的存储目录
    :return:
    """
    # 加载隐藏层
    hidden_states = torch.load(hidden_states_dir)

    # 生成标签
    # 使用pandas读取CSV文件
    df = pd.read_csv(label_csv)

    if type == "CP":
        # 检查是否存在is_changepoint列
        if 'is_changepoint' in df.columns:
            # 提取is_changepoint列，并转换成list（数组）
            labels = df['is_changepoint'].tolist()
        else:
            print("CSV文件中不存在名为'is_changepoint'的列。")
            return
    if type == "OB":
        if 'is_oddball' in df.columns:

            labels = df['is_oddball'].tolist()
        else:
            print("CSV文件中不存在名为'is_oddball'的列。")
            return
    # labels = [random.choice([0, 1]) for _ in range(240)]

    # 训练SVM模型并进行交叉验证
    svm_model = SVC(kernel="linear")
    # 使用交叉验证，这里设置为5折交叉验证
    scores = cross_val_score(svm_model, hidden_states, labels, cv=5, scoring='accuracy')

    # 计算平均准确率
    accuracy = scores.mean()
    print(f"SVM模型在交叉验证上的平均准确率: {accuracy:.4f}")

    # 如果需要计算其他性能指标，可以在训练集上训练模型，然后在测试集上计算
    # 但由于交叉验证不分割训练集和测试集，所以这里不适用
    # 可以选择在交叉验证的每一折中分别计算，但这需要手动实现交叉验证的逻辑


def sub_csv_distance(csv_path, type):
    """
    计算特殊点（is_changepoint 或 is_oddball）与前一个点的 outcome 差值，
    以及前一个点与下一个点的 outcome 差值，并与普通点之间的差值进行比较。

    :param csv_path: 输入的 CSV 文件路径
    :param type: 指定处理类型 ("CP" 或 "OB")
    :return: 包含结果统计信息的字典
    """
    # 加载数据
    data = pd.read_csv(csv_path)

    # 检查 type 参数合法性
    if type not in ["CP", "OB"]:
        raise ValueError("type 参数必须为 'CP' 或 'OB'")

    # 根据 type 确定列名和筛选条件
    type_col = "is_changepoint" if type == "CP" else "is_oddball"

    # 筛选有效数据（不等于 -1 的点）
    filtered_data = data[data[type_col] != -1].reset_index(drop=True)

    # 获取特殊点的索引
    special_indices = filtered_data.index[filtered_data[type_col] == 1]
    all_indices = filtered_data.index

    # 特殊点到前一个点的差值，以及前一个点与下一个点之间的差值
    special_to_previous_distances = []
    previous_to_next_distances = []

    for idx in special_indices:
        if idx - 1 >= 0:
            diff_previous = abs(filtered_data.loc[idx, "outcome"] - filtered_data.loc[idx - 1, "outcome"])
            special_to_previous_distances.append(diff_previous)

            if idx + 1 < len(filtered_data):
                diff_previous_next = abs(filtered_data.loc[idx + 1, "outcome"] - filtered_data.loc[idx - 1, "outcome"])
                previous_to_next_distances.append(diff_previous_next)

    # 普通点之间的差值
    normal_indices = np.setdiff1d(all_indices, special_indices)
    normal_distances = []

    for i in range(len(normal_indices) - 1):
        idx1, idx2 = normal_indices[i], normal_indices[i + 1]
        diff = abs(filtered_data.loc[idx2, "outcome"] - filtered_data.loc[idx1, "outcome"])
        normal_distances.append(diff)

    # 转换为 numpy 数组
    special_to_previous_distances = np.array(special_to_previous_distances)
    previous_to_next_distances = np.array(previous_to_next_distances)
    normal_distances = np.array(normal_distances)

    # 统计信息
    special_to_previous_mean, special_to_previous_std = np.mean(special_to_previous_distances), np.std(
        special_to_previous_distances)
    previous_to_next_mean, previous_to_next_std = np.mean(previous_to_next_distances), np.std(previous_to_next_distances)
    normal_mean, normal_std = np.mean(normal_distances), np.std(normal_distances)

    # t 检验
    t_stat_to_previous, p_value_to_previous = ttest_ind(special_to_previous_distances, normal_distances,
                                                        equal_var=False)
    t_stat_previous_next, p_value_previous_next = ttest_ind(previous_to_next_distances, normal_distances,
                                                            equal_var=False)

    # 返回结果
    return {
        "special_to_previous_distances": {
            "mean": special_to_previous_mean,
            "std": special_to_previous_std,
            "count": len(special_to_previous_distances),
        },
        "previous_to_next_distances": {
            "mean": previous_to_next_mean,
            "std": previous_to_next_std,
            "count": len(previous_to_next_distances),
        },
        "normal_distances": {
            "mean": normal_mean,
            "std": normal_std,
            "count": len(normal_distances),
        },
        "t_test_to_previous": {
            "t_statistic": t_stat_to_previous,
            "p_value": p_value_to_previous,
            "significant": p_value_to_previous < 0.05
        },
        "t_test_previous_next": {
            "t_statistic": t_stat_previous_next,
            "p_value": p_value_previous_next,
            "significant": p_value_previous_next < 0.05
        }
    }


def compare_distances(hidden_states_dir, test_csv, type="OB", oddball_col="is_oddball",
                      changepoint_col="is_changepoint"):
    """
    比较特殊点（oddball 或 changepoint）与其后一个点的距离，以及特殊点和特殊点后第二个点两者之间的距离。

    :param hidden_states_dir: 隐藏层数据的地址（通过 torch.load 加载）
    :param test_csv: 包含测试数据的 CSV 文件路径
    :param type: "OB"（oddball）或 "CP"（changepoint），决定使用哪种方式处理
    :param oddball_col: CSV 中标记 oddball 的列名，默认 "is_oddball"
    :param changepoint_col: CSV 中标记 changepoint 的列名，默认 "is_changepoint"
    :return: 包含两类距离均值、标准差和 t 检验结果的字典
    """
    # 加载隐藏层数据
    hidden_states = torch.load(hidden_states_dir).numpy()

    # 读取 CSV 文件
    test_data = pd.read_csv(test_csv)

    # 根据类型选择列，并过滤数据
    if type == "OB":
        special_col = oddball_col
        test_data = test_data[test_data[special_col] != -1]  # 只保留 is_oddball != -1 的行
    elif type == "CP":
        special_col = changepoint_col
        test_data = test_data[test_data[special_col] != -1]  # 只保留 is_changepoint != -1 的行
    else:
        raise ValueError("type 参数只能是 'OB' 或 'CP'")

    # 提取特殊点的索引
    special_indices = test_data.index[test_data[special_col] == 1].to_numpy()

    # 确保索引有效，过滤掉超出范围的点
    special_indices = special_indices[(special_indices > 0) & (special_indices + 2 < hidden_states.shape[0])]

    # 计算特殊点到其下一个点的距离
    special_to_next_distances = np.linalg.norm(hidden_states[special_indices + 1] - hidden_states[special_indices],
                                               axis=1)

    # 计算特殊点和特殊点后第二个点两者之间的距离
    special_to_second_next_distances = np.linalg.norm(
        hidden_states[special_indices + 2] - hidden_states[special_indices], axis=1)

    # 计算普通点之间的距离
    all_indices = test_data.index.to_numpy()
    normal_indices = np.setdiff1d(all_indices, special_indices)  # 普通点索引
    if len(normal_indices) < 2:
        raise ValueError("普通点数量不足，无法计算距离。")

    normal_distances = []
    for i in range(len(normal_indices) - 1):
        dist = np.linalg.norm(hidden_states[normal_indices[i + 1]] - hidden_states[normal_indices[i]])
        normal_distances.append(dist)

    normal_distances = np.array(normal_distances)

    # 统计信息
    special_to_next_mean, special_to_next_std = np.mean(special_to_next_distances), np.std(special_to_next_distances)
    special_to_second_next_mean, special_to_second_next_std = np.mean(special_to_second_next_distances), np.std(
        special_to_second_next_distances)
    normal_mean, normal_std = np.mean(normal_distances), np.std(normal_distances)

    # 进行 t 检验
    t_stat_to_next, p_value_to_next = stats.ttest_ind(special_to_next_distances, normal_distances, equal_var=False)
    t_stat_to_second_next, p_value_to_second_next = stats.ttest_ind(special_to_second_next_distances, normal_distances,
                                                                    equal_var=False)

    # 返回结果
    return {
        "special_to_next_distances": {
            "mean": special_to_next_mean,
            "std": special_to_next_std,
            "count": len(special_to_next_distances),
        },
        "special_to_second_next_distances": {
            "mean": special_to_second_next_mean,
            "std": special_to_second_next_std,
            "count": len(special_to_second_next_distances),
        },
        "normal_distances": {
            "mean": normal_mean,
            "std": normal_std,
            "count": len(normal_distances),
        },
        "t_test_to_next": {
            "t_statistic": t_stat_to_next,
            "p_value": p_value_to_next,
            "significant": p_value_to_next < 0.05
        },
        "t_test_to_second_next": {
            "t_statistic": t_stat_to_second_next,
            "p_value": p_value_to_second_next,
            "significant": p_value_to_second_next < 0.05
        }
    }




def plot_distance_sub_csv_comparison(oddball_data, changepoint_data, save_path=None):
    """
    绘制 Oddball 和 Changepoint 的距离比较图，并标注 t 值和 p 值。

    :param oddball_data: 包含 Oddball 数据的字典
    :param changepoint_data: 包含 Changepoint 数据的字典
    :param save_path: 图表保存路径，如果为 None 则直接显示图表
    """
    # 提取数据
    labels = ["Oddball", "Changepoint"]
    special_to_previous_means = [
        oddball_data["special_to_previous_distances"]["mean"],
        changepoint_data["special_to_previous_distances"]["mean"]
    ]
    special_to_previous_stds = [
        oddball_data["special_to_previous_distances"]["std"],
        changepoint_data["special_to_previous_distances"]["std"]
    ]
    previous_to_next_means = [
        oddball_data["previous_to_next_distances"]["mean"],
        changepoint_data["previous_to_next_distances"]["mean"]
    ]
    previous_to_next_stds = [
        oddball_data["previous_to_next_distances"]["std"],
        changepoint_data["previous_to_next_distances"]["std"]
    ]
    normal_means = [
        oddball_data["normal_distances"]["mean"],
        changepoint_data["normal_distances"]["mean"]
    ]
    normal_stds = [
        oddball_data["normal_distances"]["std"],
        changepoint_data["normal_distances"]["std"]
    ]
    t_values_to_previous = [
        oddball_data["t_test_to_previous"]["t_statistic"],
        changepoint_data["t_test_to_previous"]["t_statistic"]
    ]
    p_values_to_previous = [
        oddball_data["t_test_to_previous"]["p_value"],
        changepoint_data["t_test_to_previous"]["p_value"]
    ]
    t_values_previous_next = [
        oddball_data["t_test_previous_next"]["t_statistic"],
        changepoint_data["t_test_previous_next"]["t_statistic"]
    ]
    p_values_previous_next = [
        oddball_data["t_test_previous_next"]["p_value"],
        changepoint_data["t_test_previous_next"]["p_value"]
    ]

    # 设置图形
    x = np.arange(len(labels))  # x轴的位置
    width = 0.2  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    ax.bar(x - width, special_to_previous_means, width, yerr=special_to_previous_stds, label="Special to Previous",
           color="orange", capsize=5)
    ax.bar(x, normal_means, width, yerr=normal_stds, label="Normal", color="blue", capsize=5)
    ax.bar(x + width, previous_to_next_means, width, yerr=previous_to_next_stds, label="Previous to Next", color="green",
           capsize=5)

    # 添加 t 值和 p 值
    for i, label in enumerate(labels):
        ax.text(
            x[i] - width, special_to_previous_means[i] + special_to_previous_stds[i] + 10,
            f"t={t_values_to_previous[i]:.2f}\np={p_values_to_previous[i]:.2e}",
            ha="center", va="bottom", fontsize=10, color="black"
        )
        ax.text(
            x[i] + width, previous_to_next_means[i] + previous_to_next_stds[i] + 10,
            f"t={t_values_previous_next[i]:.2f}\np={p_values_previous_next[i]:.2e}",
            ha="center", va="bottom", fontsize=10, color="black"
        )

    # 图表设置
    ax.set_xlabel("Distance Type", fontsize=12)
    ax.set_ylabel("Mean Distance", fontsize=12)
    ax.set_title("Comparison of Special and Normal Distances", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 坐标轴范围固定为 0 到 300
    ax.set_ylim(0, 300)

    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_distance_comparison(oddball_data, changepoint_data, save_path=None):
    """
    绘制 Oddball 和 Changepoint 的距离比较图，并标注 t 值和 p 值。

    :param oddball_data: 包含 Oddball 数据的字典
    :param changepoint_data: 包含 Changepoint 数据的字典
    :param save_path: 图表保存路径，如果为 None 则直接显示图表
    """
    # 提取数据
    labels = ["Oddball", "Changepoint"]
    special_to_next_means = [
        oddball_data["special_to_next_distances"]["mean"],
        changepoint_data["special_to_next_distances"]["mean"]
    ]
    special_to_next_stds = [
        oddball_data["special_to_next_distances"]["std"],
        changepoint_data["special_to_next_distances"]["std"]
    ]
    special_to_second_next_means = [
        oddball_data["special_to_second_next_distances"]["mean"],
        changepoint_data["special_to_second_next_distances"]["mean"]
    ]
    special_to_second_next_stds = [
        oddball_data["special_to_second_next_distances"]["std"],
        changepoint_data["special_to_second_next_distances"]["std"]
    ]
    normal_means = [
        oddball_data["normal_distances"]["mean"],
        changepoint_data["normal_distances"]["mean"]
    ]
    normal_stds = [
        oddball_data["normal_distances"]["std"],
        changepoint_data["normal_distances"]["std"]
    ]
    t_values_to_next = [
        oddball_data["t_test_to_next"]["t_statistic"],
        changepoint_data["t_test_to_next"]["t_statistic"]
    ]
    p_values_to_next = [
        oddball_data["t_test_to_next"]["p_value"],
        changepoint_data["t_test_to_next"]["p_value"]
    ]
    t_values_to_second_next = [
        oddball_data["t_test_to_second_next"]["t_statistic"],
        changepoint_data["t_test_to_second_next"]["t_statistic"]
    ]
    p_values_to_second_next = [
        oddball_data["t_test_to_second_next"]["p_value"],
        changepoint_data["t_test_to_second_next"]["p_value"]
    ]

    # 设置图形
    x = np.arange(len(labels))  # x轴的位置
    width = 0.2  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    ax.bar(x - width, special_to_next_means, width, yerr=special_to_next_stds, label="Special to Next", color="orange",
           capsize=5)
    ax.bar(x, normal_means, width, yerr=normal_stds, label="Normal", color="blue", capsize=5)
    ax.bar(x + width, special_to_second_next_means, width, yerr=special_to_second_next_stds,
           label="Special to Second Next", color="green", capsize=5)

    # 添加 t 值和 p 值
    for i, label in enumerate(labels):
        ax.text(
            x[i] - width, max(special_to_next_means[i] + special_to_next_stds[i], normal_means[i] + normal_stds[i],
                              special_to_second_next_means[i] + special_to_second_next_stds[i]) + 0.1,
            f"t={t_values_to_next[i]:.2f}\np={p_values_to_next[i]:.2e}",
            ha="center", va="bottom", fontsize=10, color="black"
        )
        ax.text(
            x[i] + width, max(special_to_next_means[i] + special_to_next_stds[i], normal_means[i] + normal_stds[i],
                              special_to_second_next_means[i] + special_to_second_next_stds[i]) + 0.1,
            f"t={t_values_to_second_next[i]:.2f}\np={p_values_to_second_next[i]:.2e}",
            ha="center", va="bottom", fontsize=10, color="black"
        )

    # 图表设置
    ax.set_xlabel("Distance Type", fontsize=12)
    ax.set_ylabel("Mean Distance", fontsize=12)
    ax.set_title("Comparison of Special and Normal Distances", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 坐标轴范围设置一致
    max_y = max(
        max(special_to_next_means[i] + special_to_next_stds[i], normal_means[i] + normal_stds[i],
            special_to_second_next_means[i] + special_to_second_next_stds[i]) for i in range(len(labels))
    )
    ax.set_ylim(0, max_y + 0.5)

    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_hidden_state_trajectories(hidden_states_dir,
                                   test_csv="",
                                   reduction_type="PCA",
                                   dimensions=3,
                                   plot_type="all",  # 'all', 'normal', 'special'
                                   save_path=None):
    """
    绘制隐藏层状态在 2D 或 3D 空间的运动轨迹，并染色和添加旋转动画。

    :param hidden_states_dir: 隐藏层存储地址
    :param test_csv: RNN 测试时使用的 CSV 数据集
    :param reduction_type: 降维方法 ("PCA" 或 "MDS")
    :param dimensions: 降维到的维度 (2 或 3)
    :param plot_type: 绘制点类型 ('all', 'normal', 'special')
    :param save_path: 图像或动画保存路径 (.gif 或 .mp4)
    :return: None
    """

    # 加载隐藏层
    hidden_states = torch.load(hidden_states_dir).numpy()

    # 加载测试数据集 CSV
    if test_csv:
        test_data = pd.read_csv(test_csv)
        is_cp = test_data["is_changepoint"]
        is_ob = test_data["is_oddball"]

        # 根据 plot_type 筛选点
        if plot_type == "normal":
            selected_indices = test_data.index[(is_cp == 0) | (is_ob == 0)].to_numpy()
        elif plot_type == "special":
            selected_indices = test_data.index[(is_cp == 1) | (is_ob == 1)].to_numpy()
        elif plot_type == "all":
            selected_indices = np.arange(len(hidden_states))
        else:
            raise ValueError("plot_type 参数必须是 'all', 'normal', 或 'special'")
    else:
        selected_indices = np.arange(len(hidden_states))

    # 筛选数据
    hidden_states = hidden_states[selected_indices]

    # 降维
    if reduction_type == "PCA":
        reducer = PCA(n_components=dimensions)
    elif reduction_type == "MDS":
        reducer = MDS(n_components=dimensions, dissimilarity='euclidean', random_state=42, normalized_stress='auto')
    else:
        raise ValueError("reduction_type 参数只能是 'PCA' 或 'MDS'")
    hidden_states_reduced = reducer.fit_transform(hidden_states)

    # 获取坐标
    coords = [hidden_states_reduced[:, i] for i in range(dimensions)]

    # 初始化图形
    fig = plt.figure(figsize=(10, 7))
    if dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(coords[0].min(), coords[0].max())
        ax.set_ylim(coords[1].min(), coords[1].max())
        ax.set_zlim(coords[2].min(), coords[2].max())
    elif dimensions == 2:
        ax = fig.add_subplot(111)
        ax.set_xlim(coords[0].min(), coords[0].max())
        ax.set_ylim(coords[1].min(), coords[1].max())
    else:
        raise ValueError("dimensions 参数只能是 2 或 3")

    # 设置颜色
    point_colors = []
    for idx in selected_indices:
        if plot_type == "all":
            # 'all' 情况下，所有 is_changepoint 染红，所有 is_oddball 染蓝
            if idx in test_data.index[(is_cp == 1) | (is_cp == 0)]:
                point_colors.append('#FF0000')  # CP (红色)
            elif idx in test_data.index[(is_ob == 1) | (is_ob == 0)]:
                point_colors.append('#0000FF')  # OB (蓝色)
            else:
                point_colors.append('#808080')  # 普通点 (灰色)
        elif plot_type == "normal":
            # 'normal' 情况下，is_changepoint = 0 染红，is_oddball = 0 染蓝
            if idx in test_data.index[(is_cp == 0)]:
                point_colors.append('#FF0000')  # Normal - CP (红色)
            elif idx in test_data.index[(is_ob == 0)]:
                point_colors.append('#0000FF')  # Normal - OB (蓝色)
            else:
                point_colors.append('#808080')  # 普通点 (灰色)
        else:
            # 'special' 情况下，is_changepoint = 1 染红，is_oddball = 1 染蓝
            if idx in test_data.index[(is_cp == 1)]:
                point_colors.append('#FF0000')  # CP (红色)
            elif idx in test_data.index[(is_ob == 1)]:
                point_colors.append('#0000FF')  # OB (蓝色)
            else:
                point_colors.append('#808080')  # 普通点 (灰色)

    scatter = None
    if dimensions == 3:
        scatter = ax.scatter(coords[0], coords[1], coords[2], c=point_colors, s=50, edgecolors='k')
    else:
        ax.scatter(coords[0], coords[1], c=point_colors, s=50, edgecolors='k')

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=10,
                   label='CP (Change Point)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0000FF', markersize=10, label='OB (Oddball)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#808080', markersize=10, label='Normal'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 创建动画或保存图片
    if dimensions == 3 and save_path:
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return scatter,

        anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=20)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=20)
        else:
            raise ValueError("保存路径必须以 '.gif' 或 '.mp4' 结尾")
    elif save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def get_trial_vars_from_pes_cannon(noise,
                                   PE,
                                   modHaz,
                                   newBlock,
                                   allHeliVis,
                                   initRU,
                                   heliVisVar,
                                   lw,
                                   ud,
                                   driftRate,
                                   isOdd,
                                   outcomeSpace):
    """
    用于根据预测误差和一组模型参数计算主观的变化点概率（CPP）和相对不确定性（RU）的函数。

    参数:
    - noise: 高斯噪声分布的标准差
    - PE: 每个试次的预测误差
    - modHaz: 危险率
    - newBlock: 一个逻辑数组，表示新块的起始位置
    - allHeliVis: 数组，表示直升机是否可见（1表示可见）
    - initRU: 初始化的相对不确定性
    - heliVisVar: 可见直升机预测线索的方差
    - lw: 对惊讶敏感性的似然权重
    - ud: 不确定性耗竭因子
    - driftRate: 用于增加不确定性的漂移率
    - isOdd: 一个逻辑数组，表示该试次是否属于奇异条件
    - outcomeSpace: 结果空间的大小

    返回值:
    - errBased_pCha: 基于误差的变化点概率
    - errBased_RU: 基于误差的相对不确定性
    - errBased_LR: 基于误差的学习率
    - errBased_UP: 基于学习率和预测误差的模型更新
    """

    # 初始化输出数组
    errBased_RU = np.full_like(PE, np.nan, dtype=float)
    errBased_pCha = np.full_like(PE, np.nan, dtype=float)
    errBased_LR = np.full_like(PE, np.nan, dtype=float)
    errBased_UP = np.full_like(PE, np.nan, dtype=float)
    H = modHaz  # 危险率

    for i in range(1, len(noise)):
        # 首先计算相对不确定性（RU）
        if newBlock[i]:
            errBased_RU[i] = initRU
        else:
            nVar = noise[i - 1] ** 2
            cp = errBased_pCha[i - 1]
            tPE = PE[i - 1]
            inRU = errBased_RU[i - 1]

            # 计算运行长度
            runLength = (1 - inRU) / inRU

            if not isOdd[i]:
                # 对于变化点条件，更新不确定性
                numerator = (cp * nVar) + ((1 - cp) * inRU * nVar) + cp * (1 - cp) * (tPE * (1 - inRU)) ** 2
            else:
                # 对于奇异条件，更新不确定性
                numerator = (cp * nVar / runLength) + ((1 - cp) * nVar / (runLength + 1)) + cp * (1 - cp) * (
                        tPE * inRU) ** 2

            # 如果是漂移条件，根据漂移率增加不确定性
            if driftRate[i] > 0:
                numerator += driftRate[i] ** 2

            numerator /= ud  # 用常数除以不确定性
            denominator = numerator + nVar  # 分母是分子加噪声方差
            errBased_RU[i] = numerator / denominator  # RU是分数

            # 如果有直升机可见，调整不确定性
            if allHeliVis[i] == 1:
                inRU = ((errBased_RU[i] * nVar * heliVisVar) / (errBased_RU[i] * nVar + heliVisVar)) / nVar
                if np.isnan(inRU):
                    inRU = 0
                errBased_RU[i] = inRU

        if not np.isfinite(errBased_RU[i]):
            raise ValueError("非有限的错误：errBased_RU")

        # 计算基于误差的CPP（变化点概率）
        totUnc = (noise[i] ** 2) / (1 - errBased_RU[i])

        pSame = (1 - H) * norm.pdf(PE[i], 0, np.sqrt(totUnc)) ** lw
        pNew = H * (1 / outcomeSpace) ** lw
        errBased_pCha[i] = pNew / (pSame + pNew)

        # 计算学习率
        if not isOdd[i]:
            errBased_LR[i] = errBased_RU[i] + errBased_pCha[i] - errBased_RU[i] * errBased_pCha[i]
        else:
            errBased_LR[i] = errBased_RU[i] - errBased_RU[i] * errBased_pCha[i]

    # 计算模型更新
    errBased_UP = errBased_LR * PE

    return errBased_pCha, errBased_RU, errBased_LR, errBased_UP


if __name__ == "__main__":
    # plot_hidden_state_trajectories(hidden_states_dir="../hidden/3_10_42_rnn_layers_1_hidden_16_input_489_combine.pt",
    #                                test_csv="../data/sub/hc/all_combine_sub.csv",
    #                                reduction_type="MDS",
    #                                dimensions=3,
    #                                plot_type="all",
    #                                save_path="../results/png/sub/hc/hidden_trajectories/mds_3_lstm_layers_1_hidden_16_input_489_combine.gif", )

    # result_cp = compare_distances(hidden_states_dir="../hidden/3_10_42_rnn_layers_1_hidden_16_input_489_combine.pt",
    #                               test_csv="../data/sub/hc/all_combine_sub.csv",
    #                               type="CP")
    # result_ob = compare_distances(hidden_states_dir="../hidden/3_10_42_rnn_layers_1_hidden_16_input_489_combine.pt",
    #                               test_csv="../data/sub/hc/all_combine_sub.csv",
    #                               type="OB")
    #
    # plot_distance_comparison(oddball_data=result_ob, changepoint_data=result_cp,
    #                          save_path="../results/png/all_sub_distance.png")

    result_cp = sub_csv_distance(csv_path="../data/sub/hc/all_combine_sub.csv",
                                 type="CP")
    result_ob = sub_csv_distance(csv_path="../data/sub/hc/all_combine_sub.csv",
                                 type="OB")
    plot_distance_sub_csv_comparison(oddball_data=result_ob,
                                     changepoint_data=result_cp,
                                     save_path="../results/png/sub/hc/true_sub_distance.png")
