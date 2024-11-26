import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import ttest_ind
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


def plot_hidden_state_trajectories(hidden_states_dir,
                                   test_csv="",
                                   reduction_type="PCA",
                                   dimensions=3,
                                   test_type="CP",
                                   highlight_points=False,
                                   save_path=None):
    """
    绘制隐藏层状态在 2D 或 3D 空间的运动轨迹，并保存动画。

    :param hidden_states_dir: 隐藏层存储地址
    :param test_csv: RNN 测试时使用的 CSV 数据集
    :param reduction_type: 降维方法 ("PCA" 或 "MDS")
    :param dimensions: 降维到的维度 (2 或 3)
    :param test_type: 测试类型 ("CP" 或 "OB")
    :param highlight_points: 是否标记 changepoint 或 oddball 点
    :param save_path: 动画保存路径 (.gif 或 .mp4)
    :return: None
    """

    # 加载隐藏层
    hidden_states = torch.load(hidden_states_dir).numpy()

    # 加载测试数据集 CSV
    if test_csv:
        test_data = pd.read_csv(test_csv)
        if test_type == "CP":
            highlight_col = "is_changepoint"
        elif test_type == "OB":
            highlight_col = "is_oddball"
        else:
            raise ValueError("test_type 参数只能是 'CP' 或 'OB'")
        highlight_indices = test_data.index[test_data[highlight_col] == 1].to_numpy()
    else:
        highlight_indices = []

    # 降维
    if reduction_type == "PCA":
        reducer = PCA(n_components=dimensions)
    elif reduction_type == "MDS":
        reducer = MDS(n_components=dimensions, dissimilarity='euclidean', random_state=42)
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

    scatter = ax.scatter([], [], c=[], cmap='viridis', s=50, edgecolors='k')

    if highlight_points:
        highlight_scatter = ax.scatter([], [], color='orange', s=100, edgecolors='k')
    else:
        highlight_scatter = None

    # 动态连线列表
    lines = []

    # 动画更新函数
    def update(frame):
        indices = np.arange(frame + 1)
        normal_indices = np.setdiff1d(indices, highlight_indices)

        # 颜色随时间变化
        colors = np.linspace(0, 1, len(normal_indices)) ** (frame / len(coords[0]))  # 颜色随帧数加深

        if dimensions == 3:
            scatter._offsets3d = (coords[0][normal_indices], coords[1][normal_indices], coords[2][normal_indices])
            scatter.set_array(colors)  # 设置颜色数组

            if highlight_points:
                highlight_scatter._offsets3d = (
                    coords[0][highlight_indices[highlight_indices <= frame]],
                    coords[1][highlight_indices[highlight_indices <= frame]],
                    coords[2][highlight_indices[highlight_indices <= frame]]
                )

                # 绘制连线
                for line in lines:
                    line.remove()
                lines.clear()
                for idx in highlight_indices[highlight_indices <= frame]:
                    if idx + 1 < len(coords[0]):
                        line, = ax.plot(
                            [coords[0][idx], coords[0][idx + 1]],
                            [coords[1][idx], coords[1][idx + 1]],
                            [coords[2][idx], coords[2][idx + 1]],
                            color='red', linestyle='--', linewidth=2
                        )
                        lines.append(line)
            ax.view_init(elev=30, azim=frame % 360)
        else:  # 2D
            scatter.set_offsets(np.c_[coords[0][normal_indices], coords[1][normal_indices]])
            scatter.set_array(colors)  # 设置颜色数组

            if highlight_points:
                highlight_scatter.set_offsets(
                    np.c_[coords[0][highlight_indices[highlight_indices <= frame]],
                    coords[1][highlight_indices[highlight_indices <= frame]]]
                )

                # 绘制连线
                for line in lines:
                    line.remove()
                lines.clear()
                for idx in highlight_indices[highlight_indices <= frame]:
                    if idx + 1 < len(coords[0]):
                        line, = ax.plot(
                            [coords[0][idx], coords[0][idx + 1]],
                            [coords[1][idx], coords[1][idx + 1]],
                            color='red', linestyle='--', linewidth=2
                        )
                        lines.append(line)
        return (scatter, highlight_scatter) if highlight_points else (scatter,)

    def init():
        if dimensions == 3:
            scatter._offsets3d = ([], [], [])
            if highlight_points:
                highlight_scatter._offsets3d = ([], [], [])
        else:
            scatter.set_offsets(np.empty((0, 2)))
            if highlight_points:
                highlight_scatter.set_offsets(np.empty((0, 2)))
        return (scatter, highlight_scatter) if highlight_points else (scatter,)

    anim = FuncAnimation(fig, update, frames=len(coords[0]), init_func=init, blit=False, interval=100, repeat=True)

    # 保存动画
    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=10)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=10)
        else:
            raise ValueError("保存路径必须以 '.gif' 或 '.mp4' 结尾")

    # 显示动画
    plt.show()

def compare_distances(hidden_states_dir, test_csv, type="OB", oddball_col="is_oddball",
                      changepoint_col="is_changepoint"):
    """
    比较特殊点（oddball 或 changepoint）与其后一个点的距离 和 普通点之间的距离。

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

    # 根据类型选择列
    if type == "OB":
        special_col = oddball_col
    elif type == "CP":
        special_col = changepoint_col
    else:
        raise ValueError("type 参数只能是 'OB' 或 'CP'")

    # 提取特殊点的索引
    special_indices = test_data.index[test_data[special_col] == 1].to_numpy()

    # 确保索引有效，过滤掉超出范围的点
    special_indices = special_indices[special_indices + 1 < hidden_states.shape[0]]

    # 计算特殊点到其下一个点的距离
    special_distances = np.linalg.norm(hidden_states[special_indices + 1] - hidden_states[special_indices], axis=1)

    # 计算普通点之间的距离
    all_indices = np.arange(len(hidden_states))
    normal_indices = np.setdiff1d(all_indices, special_indices)  # 普通点索引
    if len(normal_indices) < 2:
        raise ValueError("普通点数量不足，无法计算距离。")

    normal_distances = []
    for i in range(len(normal_indices) - 1):
        dist = np.linalg.norm(hidden_states[normal_indices[i + 1]] - hidden_states[normal_indices[i]])
        normal_distances.append(dist)

    normal_distances = np.array(normal_distances)

    # 统计信息
    special_mean, special_std = np.mean(special_distances), np.std(special_distances)
    normal_mean, normal_std = np.mean(normal_distances), np.std(normal_distances)

    # 进行 t 检验
    t_stat, p_value = ttest_ind(special_distances, normal_distances, equal_var=False)

    # 返回结果
    return {
        "special_distances": {
            "mean": special_mean,
            "std": special_std,
            "count": len(special_distances),
        },
        "normal_distances": {
            "mean": normal_mean,
            "std": normal_std,
            "count": len(normal_distances),
        },
        "t_test": {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    }


def plot_distance_comparison(oddball_data, changepoint_data, save_path=None):
    """
    绘制 Oddball 和 Changepoint 的距离比较图，并标注 t 值和 p 值。

    :param oddball_data: 包含 Oddball 数据的字典
    :param changepoint_data: 包含 Changepoint 数据的字典
    :param save_path: 图表保存路径，如果为 None 则直接显示图表
    """
    # 提取数据
    labels = ["Oddball", "Changepoint"]
    special_means = [
        oddball_data["special_distances"]["mean"],
        changepoint_data["special_distances"]["mean"]
    ]
    special_stds = [
        oddball_data["special_distances"]["std"],
        changepoint_data["special_distances"]["std"]
    ]
    normal_means = [
        oddball_data["normal_distances"]["mean"],
        changepoint_data["normal_distances"]["mean"]
    ]
    normal_stds = [
        oddball_data["normal_distances"]["std"],
        changepoint_data["normal_distances"]["std"]
    ]
    t_values = [
        oddball_data["t_test"]["t_statistic"],
        changepoint_data["t_test"]["t_statistic"]
    ]
    p_values = [
        oddball_data["t_test"]["p_value"],
        changepoint_data["t_test"]["p_value"]
    ]

    # 设置图形
    x = np.arange(len(labels))  # x轴的位置
    width = 0.35  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制柱状图
    ax.bar(x - width / 2, special_means, width, yerr=special_stds, label="Special", color="orange", capsize=5)
    ax.bar(x + width / 2, normal_means, width, yerr=normal_stds, label="Normal", color="blue", capsize=5)

    # 添加 t 值和 p 值
    for i, label in enumerate(labels):
        ax.text(
            x[i], max(special_means[i] + special_stds[i], normal_means[i] + normal_stds[i]) + 0.1,
            f"t={t_values[i]:.2f}\np={p_values[i]:.2e}",
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
        max(special_means[i] + special_stds[i], normal_means[i] + normal_stds[i]) for i in range(len(labels))
    )
    ax.set_ylim(0, max_y + 0.5)

    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # svm_hidden_states(hidden_states_dir="../hidden/6_19_43_layers_3_hidden_1024_input_10.pt")

    # svm_hidden_states_singlepoint(hidden_states_dir="../hidden/21_19_56_layers_3_hidden_1024_input_489.pt",
    #                               label_csv="../data/240_rule/df_test_OB.csv",
    #                               type="OB", )

    # hidden_state_trajectories_pca_2d(hidden_states_dir="../hidden/23_11_0_layers_3_hidden_1024_input_489_sub_OB.pt",
    #                                  test_csv="../data/sub/hc/403/ADL_B_403_DataOddball_403.csv",
    #                                  type="OB", )

    # hidden_state_trajectories_mds(hidden_states_dir="../hidden/23_11_6_layers_3_hidden_1024_input_489_sub_OB.pt",
    #                               test_csv="../data/sub/hc/404/ADL_B_404_DataOddball_404.csv",
    #                               type="OB", )
    # plot_hidden_state_trajectories(hidden_states_dir="../hidden/23_10_36_layers_3_hidden_1024_input_489_CP.pt",
    #                                test_csv="../data/240_rule/df_test_CP.csv",
    #                                test_type="CP",
    #                                reduction_type="MDS",
    #                                dimensions=2,
    #                                highlight_points=True,
    #                                save_path="../results/png/240_rule/df_test_CP/hidden_trajectories/mds_2_model.gif", )

    result_OB = compare_distances(hidden_states_dir="../hidden/23_11_0_layers_3_hidden_1024_input_489_sub_OB.pt",
                                  test_csv="../data/sub/hc/403/ADL_B_403_DataOddball_403.csv",
                                  type="OB", )
    result_CP = compare_distances(hidden_states_dir="../hidden/23_10_36_layers_3_hidden_1024_input_489_CP.pt",
                                  test_csv="../data/240_rule/df_test_CP.csv",
                                  type="CP", )
    print("Oddball Distances:", result_OB)
    print("Changepoint Distances:", result_CP)

    plot_distance_comparison(result_OB, 
                             result_CP,
                             save_path="../results/png/distance.png")
