import glob
import os
import random

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind, norm, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

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


def perform_kmeans_clustering(data, k=3, plot_elbow=False, max_k=10):
    """
    对给定数据执行 K-Means 聚类，并选择性绘制肘部法则图。

    :param data: numpy.ndarray，形状为 (n_samples, n_features) 的数据矩阵
    :param k: int，聚类数（默认为 3）
    :param plot_elbow: bool，是否绘制肘部法则图以选择最佳聚类数
    :param max_k: int，用于肘部法则的最大聚类数（仅在 plot_elbow=True 时有效）
    :return: tuple (cluster_labels, cluster_centers, sse)，分别是样本的簇标签，聚类中心，以及 SSE（总误差平方和）
    """
    if plot_elbow:
        # 计算不同 k 值的 SSE
        sse = []
        k_values = range(1, max_k + 1)
        for i in k_values:
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)

        # 绘制肘部法则图
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, sse, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Sum of squared distances (SSE)')
        plt.title('Elbow Method for Optimal k')
        plt.show()

    # 执行 K-Means 聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    cluster_centers = kmeans.cluster_centers_
    sse = kmeans.inertia_

    return cluster_labels, cluster_centers, sse


def plot_clusters(reduced_states, labels, centers, dimensions=2, save_path=None):
    """
    绘制聚类结果，支持 2D 和 3D，可保存为图片或 GIF 动画。

    :param reduced_states: 降维后的数据点 (n_samples, n_features)
    :param labels: 聚类标签 (n_samples,)
    :param centers: 聚类中心 (n_clusters, n_features)
    :param dimensions: 绘图维度 (2 或 3)
    :param save_path: 保存路径（2D 图保存为图片，3D 图保存为 .gif 文件）
    """
    unique_labels = np.unique(labels)

    if dimensions == 2:
        # 2D 图
        plt.figure(figsize=(8, 6))
        for label in unique_labels:
            cluster_points = reduced_states[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
        plt.scatter(centers[:, 0], centers[:, 1], color="black", marker="x", s=100, label="Centroids")
        plt.legend()
        plt.title("K-Means Clustering Results (2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)

        # 保存或显示 2D 图
        if save_path:
            print(f"Saving 2D plot to {save_path}...")
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    elif dimensions == 3:
        # 3D 图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for label in unique_labels:
            cluster_points = reduced_states[labels == label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {label}")

        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="black", marker="x", s=100, label="Centroids")
        ax.set_title("K-Means Clustering Results (3D)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.legend()

        # 保存为 GIF 动画
        if save_path and save_path.endswith(".gif"):
            def update(frame):
                ax.view_init(elev=30, azim=frame)
                return fig,

            anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
            print(f"Saving 3D animation to {save_path}...")
            anim.save(save_path, writer="pillow", fps=20)
        else:
            plt.show()

    else:
        raise ValueError("Only 2D or 3D plotting is supported.")


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
    previous_to_next_mean, previous_to_next_std = np.mean(previous_to_next_distances), np.std(
        previous_to_next_distances)
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
    ax.bar(x + width, previous_to_next_means, width, yerr=previous_to_next_stds, label="Previous to Next",
           color="green",
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
                                   type="combine",
                                   save_path=None,
                                   is_line=False):
    """
    绘制隐藏层状态在 2D 或 3D 空间的运动轨迹，并染色和添加旋转动画。
    当 is_line 为 True 时，绘制特殊点和下一个普通点之间的连线。

    :param hidden_states_dir: 隐藏层存储地址
    :param test_csv: RNN 测试时使用的 CSV 数据集
    :param reduction_type: 降维方法 ("PCA" 或 "MDS")
    :param dimensions: 降维到的维度 (2 或 3)
    :param plot_type: 绘制点类型 ('all', 'normal', 'special')
    :param type: 绘制点的类型 ('combine', 'CP', 'OB')
    :param save_path: 图像或动画保存路径 (.gif 或 .mp4)
    :param is_line: 是否绘制特殊点和下一个普通点之间的连线
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
            if type == "combine":
                if idx in test_data.index[(is_cp == 1) | (is_cp == 0)]:
                    point_colors.append('#FF0000')  # CP (红色)
                elif idx in test_data.index[(is_ob == 1) | (is_ob == 0)]:
                    point_colors.append('#0000FF')  # OB (蓝色)
                else:
                    point_colors.append('#808080')  # 普通点 (灰色)
            elif type == "CP":
                if idx in test_data.index[(is_cp == 1)]:
                    point_colors.append('#FF0000')  # CP (红色)
                else:
                    point_colors.append('#808080')  # 普通点 (灰色)
            elif type == "OB":
                if idx in test_data.index[(is_ob == 1)]:
                    point_colors.append('#0000FF')  # OB (蓝色)
                else:
                    point_colors.append('#808080')  # 普通点 (灰色)
        elif plot_type == "normal":
            if idx in test_data.index[(is_cp == 0)]:
                point_colors.append('#FF0000')  # Normal - CP (红色)
            elif idx in test_data.index[(is_ob == 0)]:
                point_colors.append('#0000FF')  # Normal - OB (蓝色)
            else:
                point_colors.append('#808080')  # 普通点 (灰色)
        else:
            if idx in test_data.index[(is_cp == 1)]:
                point_colors.append('#FF0000')  # CP (红色)
            elif idx in test_data.index[(is_ob == 1)]:
                point_colors.append('#0000FF')  # OB (蓝色)
            else:
                point_colors.append('#808080')  # 普通点 (灰色)

    # 绘制点
    scatter = None
    if dimensions == 3:
        scatter = ax.scatter(coords[0], coords[1], coords[2], c=point_colors, s=50, edgecolors='k')
    else:
        scatter = ax.scatter(coords[0], coords[1], c=point_colors, s=50, edgecolors='k')

    # 如果 is_line 为 True，绘制特殊点和下一个普通点的连线
    if is_line and test_csv:
        for idx in test_data.index[(is_cp == 1) | (is_ob == 1)]:  # 筛选特殊点
            next_idx = idx + 1
            if next_idx < len(hidden_states_reduced):  # 确保索引不越界
                x_values = [hidden_states_reduced[idx, 0], hidden_states_reduced[next_idx, 0]]
                y_values = [hidden_states_reduced[idx, 1], hidden_states_reduced[next_idx, 1]]
                if dimensions == 3:
                    z_values = [hidden_states_reduced[idx, 2], hidden_states_reduced[next_idx, 2]]
                    ax.plot(x_values, y_values, z_values, color='red', linestyle='-', linewidth=2)
                else:
                    ax.plot(x_values, y_values, color='red', linestyle='-', linewidth=2)

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


def plot_average_hidden_state_trajectories_from_files(folder_path,
                                                      dimensions=3,
                                                      test_csv=None,
                                                      save_path=None):
    """
    从文件夹中加载所有文件的数据，分别进行 PCA 降维后取平均，
    根据测试集标注 changepoint 和 oddball 点，绘制轨迹。
    2D 保存最终静态图像，3D 动态显示或保存动画。

    :param folder_path: 包含数据文件的文件夹路径
    :param dimensions: 降维后的维度，支持 2 或 3
    :param test_csv: 测试集 CSV 文件路径，包含 is_changepoint 和 is_oddball 列
    :param save_path: 图像或动画保存路径 (.gif 或 .mp4)，为 None 则直接显示图像
    :return: None
    """
    if dimensions not in [2, 3]:
        raise ValueError("dimensions 参数只能是 2 或 3")

    # 获取文件夹中的所有文件
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        raise ValueError("文件夹中没有找到任何文件")

    # 存储所有数据
    hidden_states_list = []

    # 加载每个文件中的数据
    for file_path in files:
        try:
            data = torch.load(file_path).numpy()  # 加载为 NumPy 数组
            hidden_states_list.append(data)
        except Exception as e:
            print(f"文件 {file_path} 无法加载为 NumPy 数组，错误：{e}")
            continue

    if not hidden_states_list:
        raise ValueError("所有文件均无法加载为有效数据")

    # 检查数据维度一致性
    sample_shape = hidden_states_list[0].shape
    for hs in hidden_states_list:
        if hs.shape != sample_shape:
            raise ValueError(f"所有数据形状必须一致，{hs.shape} 与 {sample_shape} 不匹配")

    # 对每个隐藏状态进行 PCA 降维
    reduced_states_list = []
    for hidden_states in hidden_states_list:
        pca = PCA(n_components=dimensions)
        reduced_states = pca.fit_transform(hidden_states)
        reduced_states_list.append(reduced_states)

    # 计算所有降维结果的平均值
    reduced_states_average = np.mean(reduced_states_list, axis=0)

    # 分离坐标
    coords = [reduced_states_average[:, i] for i in range(dimensions)]

    # 如果提供了测试集 CSV，加载并读取标记
    is_changepoint = None
    is_oddball = None
    if test_csv:
        test_data = pd.read_csv(test_csv)
        is_changepoint = test_data["is_changepoint"].values
        is_oddball = test_data["is_oddball"].values

    # 固定坐标轴范围
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    if dimensions == 3:
        z_min, z_max = coords[2].min(), coords[2].max()

    if dimensions == 2:
        # 绘制 2D 图像
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title("2D Average Hidden State Trajectory")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 绘制所有点和特殊点连线
        for i in range(len(coords[0])):
            color = "gray"  # 默认普通点为灰色
            if test_csv and is_changepoint[i] == 1:
                color = "red"
            elif test_csv and is_oddball[i] == 1:
                color = "blue"

            ax.scatter(coords[0][i], coords[1][i], color=color, s=50)

            if i < len(coords[0]) - 1 and (is_changepoint[i] == 1 or is_oddball[i] == 1):
                line_color = "red" if is_changepoint[i] == 1 else "blue"
                ax.plot([coords[0][i], coords[0][i + 1]],
                        [coords[1][i], coords[1][i + 1]],
                        color=line_color)

        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"2D 图像已保存到 {save_path}")
        else:
            plt.show()

    elif dimensions == 3:
        # 初始化 3D 图像
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Average Hidden State Trajectory")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # 动态绘制函数
        def update(frame):
            ax.cla()
            ax.set_title("3D Average Hidden State Trajectory")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.view_init(elev=30, azim=frame)

            for i in range(frame):
                color = "gray"
                if test_csv and is_changepoint[i] == 1:
                    color = "red"
                elif test_csv and is_oddball[i] == 1:
                    color = "blue"

                ax.scatter(coords[0][i], coords[1][i], coords[2][i], color=color, s=50)

                if i < frame - 1 and (is_changepoint[i] == 1 or is_oddball[i] == 1):
                    line_color = "red" if is_changepoint[i] == 1 else "blue"
                    ax.plot([coords[0][i], coords[0][i + 1]],
                            [coords[1][i], coords[1][i + 1]],
                            [coords[2][i], coords[2][i + 1]],
                            color=line_color)

        # 创建动画
        anim = FuncAnimation(fig, update, frames=len(coords[0]), interval=300, repeat=False)

        # 保存或显示动画
        if save_path:
            print(f"Saving 3D animation to {save_path}...")
            if save_path.endswith(".gif"):
                anim.save(save_path, writer="pillow", fps=5)
            elif save_path.endswith(".mp4"):
                anim.save(save_path, writer="ffmpeg", fps=5)
            else:
                raise ValueError("保存路径必须以 '.gif' 或 '.mp4' 结尾")
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

    errBased_UP = errBased_LR * PE

    return errBased_pCha, errBased_RU, errBased_LR, errBased_UP


def pairwise_distance_matrix(x,
                             saving_path,
                             is_number_label=False,
                             save_matrix_path=None, ):
    """
    绘制数据点之间的成对距离矩阵的热图。

    :param is_number_label: 是否在坐标轴上显示坐标数字
    :param x: 输入的二维数据，形状为 (n_samples, n_features)
    :param saving_path: 保存热图的路径，如果为 None，则不保存
    """
    # 计算成对欧几里得距离并转化为方阵形式
    condensed_dist_matrix = pdist(x, metric='euclidean')
    dissimilarity_matrix = squareform(condensed_dist_matrix)

    if save_matrix_path:
        np.save(save_matrix_path, condensed_dist_matrix)
        print(f"Dissimilarity matrix saved to {save_matrix_path}")

    # 绘制热图
    plt.figure(figsize=(10, 8))
    plt.imshow(dissimilarity_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Dissimilarity")

    if is_number_label:
        # 如果需要在坐标轴显示坐标数字
        plt.xticks(ticks=np.arange(dissimilarity_matrix.shape[1]),
                   labels=np.arange(dissimilarity_matrix.shape[1]) + 1)
        plt.yticks(ticks=np.arange(dissimilarity_matrix.shape[0]),
                   labels=np.arange(dissimilarity_matrix.shape[0]) + 1)
    else:
        # 如果不需要显示坐标数字
        plt.xticks([])
        plt.yticks([])

    plt.gca().invert_yaxis()  # 反转纵轴
    plt.title("Dissimilarity Matrix")
    plt.xlabel("Trial")
    plt.ylabel("Trial")

    # 保存图像
    if saving_path:
        plt.savefig(saving_path, dpi=300)


def batch_pairwise_distance_matrix(hidden_path,
                                   saving_base_path,
                                   matrix_base_path,
                                   is_number_label):
    """
    先将模型按trials减去均值，再批量计算和保存 RDM。

    :param hidden_path: str, 文件夹路径，包含子文件夹，子文件夹中有 `not_remove` 文件夹，存储 pt 文件。
    :param saving_base_path: str, 保存图像的基础路径。
    :param matrix_base_path: str, 保存矩阵的基础路径。
    :param is_number_label: bool, 是否在图像中显示数字标签。
    """
    # 遍历主文件夹下的所有子文件夹
    for subdir, _, files in os.walk(hidden_path):
        if "not_remove" in subdir:  # 筛选出包含 "remove" 的文件夹
            pt_files = [f for f in files if f.endswith('.pt')]  # 筛选出 .pt 文件
            if not pt_files:
                print(f"在文件夹 {subdir} 中未找到任何 .pt 文件，跳过该文件夹。")
                continue

            # 获取当前子文件夹的数字名
            subfolder_name = os.path.basename(os.path.dirname(subdir))

            # 遍历每个 .pt 文件
            for pt_file in pt_files:
                # 加载 .pt 文件的路径
                pt_file_path = os.path.join(subdir, pt_file)
                # 加载 hidden_states
                try:
                    hidden_states = torch.load(pt_file_path).numpy()
                    print(hidden_states.shape)
                    mean_value = hidden_states[0].mean()

                    # mean_per_trial = hidden_states.mean(axis=1, keepdims=True)
                    # std_per_trial = hidden_states.std(axis=1, keepdims=True)
                    # # hidden_states = (hidden_states - mean_per_trial)
                    # hidden_states = (hidden_states - mean_per_trial) / (std_per_trial + 1e-8) # 加 1e-8 避免除以 0

                except Exception as e:
                    print(f"加载 {pt_file_path} 时出错: {e}")
                    continue

                # 构建保存路径
                saving_path = os.path.join(
                    saving_base_path,
                    subfolder_name,
                    f"model_dm_CP.png"
                )
                matrix_save_path = os.path.join(
                    matrix_base_path,
                    subfolder_name,
                    "rdm",
                    "not_remove",
                    f"{pt_file.split('.pt')[0]}.npy"
                )

                # 确保保存目录存在
                os.makedirs(os.path.dirname(saving_path), exist_ok=True)
                os.makedirs(os.path.dirname(matrix_save_path), exist_ok=True)

                # 调用 pairwise_distance_matrix 函数
                print(f"正在处理 {pt_file_path}，保存到 {saving_path} 和 {matrix_save_path}...")
                pairwise_distance_matrix(
                    hidden_states,
                    saving_path=None,
                    is_number_label=is_number_label,
                    save_matrix_path=matrix_save_path
                )

    print("所有文件处理完成。")


def find_npy_files(root_dir, keyword="combine"):
    """
    递归查找 root_dir 及所有子文件夹中包含特定关键字的 .npy 文件

    :param root_dir: 需要遍历的根目录
    :param keyword: 需要匹配的文件名关键字
    :return: 符合条件的 .npy 文件列表
    """
    npy_files = []
    for dirpath, _, filenames in os.walk(root_dir):  # 递归遍历所有子目录
        for file in filenames:
            if keyword in file and file.endswith(".npy"):  # 过滤包含关键字的 .npy 文件
                npy_files.append(os.path.join(dirpath, file))
    return npy_files


def merge_model_rdm_files(root_dir, save_path, keyword="combine"):
    """
    遍历所有子文件夹，读取名字包含关键字的 .npy 文件，并合并它们为一个 2D 数组后保存。

    :param root_dir: 需要遍历的根目录
    :param save_path: 合并后的 .npy 文件保存路径
    :param keyword: 需要匹配的文件名关键字
    """
    npy_files = find_npy_files(root_dir, keyword)
    all_arrays = []

    if not npy_files:
        print("未找到任何符合条件的 .npy 文件，未执行合并操作。")
        return

    for npy_file in npy_files:
        print(f"加载文件: {npy_file}")
        data = np.load(npy_file)

        # 只支持 1D 数组，转换为 (1, N) 以便后续合并
        if data.ndim == 1:
            data = data[np.newaxis, :]  # 变成 (1, 20)
        else:
            print(f"警告: {npy_file} 不是 1D 数组，跳过。")
            continue

        all_arrays.append(data)

    # 确保所有数组的形状兼容
    try:
        merged_array = np.concatenate(all_arrays, axis=0)  # 按行合并成 (N, 20)
        np.save(save_path, merged_array)
        print(f"合并完成，数据已保存到 {save_path}")
    except ValueError as e:
        print(f"数据形状不兼容，无法合并: {e}")


if __name__ == "__main__":
    # # 隐藏层轨迹
    # plot_hidden_state_trajectories(hidden_states_dir="../hidden/rnn_layers_1_hidden_16_input_489_CP_100_120.pt",
    #                                test_csv="../data/240_rule/df_100_120_CP.csv",
    #                                reduction_type="PCA",
    #                                dimensions=3,
    #                                plot_type="all",
    #                                type="CP",
    #                                is_line=True,
    #                                save_path="../results/png/240_rule/df_100_120_PCA_3.gif", )

    # # 距离图
    # result_cp = compare_distances(hidden_states_dir="../hidden/3_10_42_rnn_layers_1_hidden_16_input_489_combine.pt",
    #                               test_csv="../data/sub/hc/all_combine_sub.csv",
    #                               type="CP")
    # result_ob = compare_distances(hidden_states_dir="../hidden/3_10_42_rnn_layers_1_hidden_16_input_489_combine.pt",
    #                               test_csv="../data/sub/hc/all_combine_sub.csv",
    #                               type="OB")
    #
    # plot_distance_comparison(oddball_data=result_ob, changepoint_data=result_cp,
    #                          save_path="../results/png/all_sub_distance.png")

    # # 被试距离图
    # result_cp = sub_csv_distance(csv_path="../data/sub/hc/all_combine_sub.csv",
    #                              type="CP")
    # result_ob = sub_csv_distance(csv_path="../data/sub/hc/all_combine_sub.csv",
    #                              type="OB")
    # plot_distance_sub_csv_comparison(oddball_data=result_ob,
    #                                  changepoint_data=result_cp,
    #                                  save_path="../results/png/sub/hc/true_sub_distance.png")

    # # 平均隐藏层轨迹
    # plot_average_hidden_state_trajectories_from_files(folder_path="../hidden/10/404/CP",
    #                                                   dimensions=2,
    #                                                   test_csv="../data/sub/hc/404/ADL_B_404_DataCP_404.csv",
    #                                                   save_path="../results/png/10/404/average_hidden_states_2.png")

    # # 聚族
    # data = torch.load("../hidden/rnn_layers_1_hidden_16_input_489_CP_100_120.pt").numpy()
    # pca = PCA(n_components=3)
    # reduced_states = pca.fit_transform(data)
    #
    # labels, centers, sse = perform_kmeans_clustering(reduced_states,
    #                                                  k=3,
    #                                                  plot_elbow=True,
    #                                                  max_k=10, )
    # plot_clusters(reduced_states,
    #               labels,
    #               centers,
    #               dimensions=3,
    #               save_path="../results/png/240_rule/df_100_120_PCA_3.gif")
    #
    # print("每个样本的簇标签:")
    # print(labels)
    # print("聚类中心:")
    # print(centers)
    # print(f"误差平方和 (SSE): {sse}")

    # # 不相似性矩阵
    # hidden_states = torch.load("../hidden/sub/405/remove/rnn_layers_1_hidden_16_input_489_CP.pt").numpy()
    # pairwise_distance_matrix(hidden_states,
    #                          saving_path="../results/png/sub/hc/405/model_dm_CP_228.png",
    #                          is_number_label=False,
    #                          save_matrix_path="../results/numpy/model/sub/hc/405/rdm/rnn_layers_1_hidden_16_input_489_CP_228.npy")

    # # 批量产生不相似性矩阵
    # batch_pairwise_distance_matrix(hidden_path="../hidden/sub/hc",
    #                                saving_base_path="../results/png/sub/hc",
    #                                matrix_base_path="../results/numpy/model/sub/hc",
    #                                is_number_label=False)

    merge_model_rdm_files(root_dir="../results/numpy/model/sub/hc",
                          save_path="../results/numpy/model/sub/hc/not_remove_model_rdm_CP_frist_OB.npy",
                          keyword="reverse")

    # data = np.load("../results/numpy/model/sub/hc/not_remove_model_rdm.npy")
    # print(data.shape)
