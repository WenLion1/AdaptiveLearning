import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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


import torch
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def hidden_state_trajectories_pca(hidden_states_dir,
                                  test_csv="",
                                  type="CP",
                                  save_path=None):
    """
    画hidden_state在3维的运动轨迹，并保存动画

    :param hidden_states_dir: 隐藏层存储地址
    :param test_csv: rnn再哪个csv数据集下跑的
    :param type: 此次test的类型，CP or OB
    :param save_path: 动画保存路径（.gif 或 .mp4）
    :return:
    """
    import pandas as pd

    # 加载隐藏层
    hidden_states = torch.load(hidden_states_dir).numpy()

    # 加载测试数据集 CSV
    if test_csv:
        test_data = pd.read_csv(test_csv)
        if type == "CP":
            highlight_col = "is_changepoint"
        elif type == "OB":
            highlight_col = "is_oddball"
        else:
            raise ValueError("type 参数只能是 'CP' 或 'OB'")

        # 找到需要高亮的时间点索引
        highlight_indices = test_data.index[test_data[highlight_col] == 1].to_numpy()
    else:
        highlight_indices = []

    # 使用 PCA 降维到 3D
    pca = PCA(n_components=3)
    hidden_states_3d = pca.fit_transform(hidden_states)  # [240, 3]

    # 获取 3D 坐标
    x = hidden_states_3d[:, 0]
    y = hidden_states_3d[:, 1]
    z = hidden_states_3d[:, 2]

    # 初始化 3D 图形
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 设置固定坐标轴范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)

    # 初始状态
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=50, edgecolors='k')  # 仅绘制普通点
    highlight_scatter = ax.scatter([], [], [], color='orange', s=100, edgecolors='k')  # 橙色点用于高亮

    # 动画更新函数
    def update(frame):
        # 获取当前帧普通点索引，排除高亮点
        normal_indices = np.setdiff1d(range(frame + 1), highlight_indices)

        # 当前普通点
        current_x = x[normal_indices]
        current_y = y[normal_indices]
        current_z = z[normal_indices]

        # 更新普通点的位置和颜色
        scatter._offsets3d = (current_x, current_y, current_z)
        scatter.set_array(np.linspace(0, 1, len(normal_indices)))  # 动态更新颜色渐变

        # 更新高亮点
        highlight_x = x[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_y = y[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_z = z[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_scatter._offsets3d = (highlight_x, highlight_y, highlight_z)

        return scatter, highlight_scatter

    # 动画初始化函数
    def init():
        scatter._offsets3d = ([], [], [])
        scatter.set_array([])
        highlight_scatter._offsets3d = ([], [], [])
        return scatter, highlight_scatter

    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=False, interval=100, repeat=False)

    # 在动画完成后显示最后一帧
    def on_animation_complete():
        update(len(x) - 1)

    anim._stop = on_animation_complete

    # 如果指定了保存路径，则保存动画
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


def hidden_state_trajectories_pca_2d(hidden_states_dir,
                                     test_csv="",
                                     type="CP",
                                     save_path=None):
    """
    画hidden_state在2维的运动轨迹，并保存动画

    :param hidden_states_dir: 隐藏层存储地址
    :param test_csv: rnn在哪个csv数据集下跑的
    :param type: 此次test的类型，CP or OB
    :param save_path: 动画保存路径（.gif 或 .mp4）
    :return:
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    from sklearn.decomposition import PCA
    import torch

    # 加载隐藏层
    hidden_states = torch.load(hidden_states_dir).numpy()

    # 加载测试数据集 CSV
    if test_csv:
        test_data = pd.read_csv(test_csv)
        if type == "CP":
            highlight_col = "is_changepoint"
        elif type == "OB":
            highlight_col = "is_oddball"
        else:
            raise ValueError("type 参数只能是 'CP' 或 'OB'")

        # 找到需要高亮的时间点索引
        highlight_indices = test_data.index[test_data[highlight_col] == 1].to_numpy()
    else:
        highlight_indices = []

    # 使用 PCA 降维到 2D
    pca = PCA(n_components=2)
    hidden_states_2d = pca.fit_transform(hidden_states)  # [240, 2]

    # 获取 2D 坐标
    x = hidden_states_2d[:, 0]
    y = hidden_states_2d[:, 1]

    # 初始化 2D 图形
    fig, ax = plt.subplots(figsize=(8, 8))

    # 设置固定坐标轴范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Hidden State Trajectories (2D)", fontsize=16)
    ax.set_xlabel("PCA Dim 1")
    ax.set_ylabel("PCA Dim 2")

    # 初始状态
    scatter = ax.scatter([], [], c=[], cmap='viridis', s=50, edgecolors='k')  # 普通点
    highlight_scatter = ax.scatter([], [], color='orange', s=100, edgecolors='k')  # 高亮点

    # 动画更新函数
    def update(frame):
        # 获取当前帧普通点索引，排除高亮点
        normal_indices = np.setdiff1d(range(frame + 1), highlight_indices)

        # 当前普通点
        current_x = x[normal_indices]
        current_y = y[normal_indices]

        # 更新普通点的位置和颜色
        scatter.set_offsets(np.c_[current_x, current_y])
        scatter.set_array(np.linspace(0, 1, len(normal_indices)))  # 动态更新颜色渐变

        # 更新高亮点
        highlight_x = x[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_y = y[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_scatter.set_offsets(np.c_[highlight_x, highlight_y])

        return scatter, highlight_scatter

    # 动画初始化函数
    def init():
        scatter.set_offsets(np.empty((0, 2)))  # 修复错误：提供一个形状为 (0, 2) 的空数组
        scatter.set_array([])
        highlight_scatter.set_offsets(np.empty((0, 2)))  # 同样提供空数组
        return scatter, highlight_scatter

    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=False, interval=100, repeat=False)

    # 如果指定了保存路径，则保存动画
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


def hidden_state_trajectories_mds(hidden_states_dir,
                                  test_csv="",
                                  save_path=None,
                                  type="CP"):
    """
    画hidden_state在3维的运动轨迹，并保存动画（使用MDS方法降维）

    :param hidden_states_dir: 隐藏层存储地址
    :param test_csv: rnn在哪个csv数据集下跑的
    :param type: 此次test的类型，CP or OB
    :param save_path: 动画保存路径（.gif 或 .mp4）
    :return:
    """
    import pandas as pd
    from sklearn.manifold import MDS
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import torch
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # 加载隐藏层
    hidden_states = torch.load(hidden_states_dir).numpy()

    # 加载测试数据集 CSV
    if test_csv:
        test_data = pd.read_csv(test_csv)
        if type == "CP":
            highlight_col = "is_changepoint"
            highlight_indices = test_data.index[test_data[highlight_col] == 1].to_numpy()
            blue_indices = []  # CP 情况下没有蓝色点
        elif type == "OB":
            highlight_col = "is_oddball"
            oddball_indices = test_data.index[test_data[highlight_col] == 1].to_numpy()
            highlight_indices = oddball_indices

            # 蓝色点：is_oddball 为 1 的下一个时间点
            blue_indices = (oddball_indices + 1)[oddball_indices + 1 < len(test_data)]
        else:
            raise ValueError("type 参数只能是 'CP' 或 'OB'")
    else:
        highlight_indices = []
        blue_indices = []

    # 使用 MDS 降维到 3D
    mds = MDS(n_components=3, dissimilarity='euclidean', random_state=42)
    hidden_states_3d = mds.fit_transform(hidden_states)

    # 获取 3D 坐标
    x = hidden_states_3d[:, 0]
    y = hidden_states_3d[:, 1]
    z = hidden_states_3d[:, 2]

    # 初始化 3D 图形
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 设置固定坐标轴范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)

    # 初始状态
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=50, edgecolors='k')  # 普通点
    highlight_scatter = ax.scatter([], [], [], color='orange', s=100, edgecolors='k')  # 橙色点
    blue_scatter = ax.scatter([], [], [], color='blue', s=100, edgecolors='k')  # 蓝色点

    # 动态连线
    lines = Line3DCollection([], colors='blue', linestyles='--', lw=2)
    ax.add_collection3d(lines)

    # 动画更新函数
    def update(frame):
        # 获取当前帧蓝色点及橙色点索引
        if frame in blue_indices:
            scatter._offsets3d = ([], [], [])  # 隐藏普通点
        else:
            normal_indices = np.setdiff1d(range(frame + 1), highlight_indices)
            scatter._offsets3d = (x[normal_indices], y[normal_indices], z[normal_indices])
            scatter.set_array(np.linspace(0, 1, len(normal_indices)))

        # 更新高亮点
        highlight_x = x[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_y = y[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_z = z[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_scatter._offsets3d = (highlight_x, highlight_y, highlight_z)

        # 更新蓝色点
        blue_x = x[np.intersect1d(blue_indices, range(frame + 1))]
        blue_y = y[np.intersect1d(blue_indices, range(frame + 1))]
        blue_z = z[np.intersect1d(blue_indices, range(frame + 1))]
        blue_scatter._offsets3d = (blue_x, blue_y, blue_z)

        # 更新连线
        if len(highlight_x) > 0 and len(blue_x) > 0:
            segments = [[(highlight_x[-1], highlight_y[-1], highlight_z[-1]),
                         (blue_x[-1], blue_y[-1], blue_z[-1])]]
            lines.set_segments(segments)
        else:
            lines.set_segments([])

        return scatter, highlight_scatter, blue_scatter, lines

    # 动画初始化函数
    def init():
        scatter._offsets3d = ([], [], [])
        scatter.set_array([])
        highlight_scatter._offsets3d = ([], [], [])
        blue_scatter._offsets3d = ([], [], [])
        lines.set_segments([])
        return scatter, highlight_scatter, blue_scatter, lines

    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=False, interval=100, repeat=False)

    # 如果指定了保存路径，则保存动画
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


def hidden_state_trajectories_mds_2d(hidden_states_dir,
                                     test_csv="",
                                     save_path=None,
                                     type="CP"):
    """
    画hidden_state在2维的运动轨迹，并保存动画（使用MDS方法降维）

    :param hidden_states_dir: 隐藏层存储地址
    :param test_csv: rnn在哪个csv数据集下跑的
    :param type: 此次test的类型，CP or OB
    :param save_path: 动画保存路径（.gif 或 .mp4）
    :return:
    """
    import pandas as pd
    from sklearn.manifold import MDS
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import torch
    from matplotlib.lines import Line2D

    # 加载隐藏层
    hidden_states = torch.load(hidden_states_dir).numpy()

    # 加载测试数据集 CSV
    if test_csv:
        test_data = pd.read_csv(test_csv)
        if type == "CP":
            highlight_col = "is_changepoint"
            highlight_indices = test_data.index[test_data[highlight_col] == 1].to_numpy()
            blue_indices = []  # CP 情况下没有蓝色点
        elif type == "OB":
            highlight_col = "is_oddball"
            oddball_indices = test_data.index[test_data[highlight_col] == 1].to_numpy()
            highlight_indices = oddball_indices

            # 蓝色点：is_oddball 为 1 的下一个时间点
            blue_indices = (oddball_indices + 1)[oddball_indices + 1 < len(test_data)]
        else:
            raise ValueError("type 参数只能是 'CP' 或 'OB'")
    else:
        highlight_indices = []
        blue_indices = []

    # 使用 MDS 降维到 2D
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress="auto")
    hidden_states_2d = mds.fit_transform(hidden_states)

    # 获取 2D 坐标
    x = hidden_states_2d[:, 0]
    y = hidden_states_2d[:, 1]

    # 初始化 2D 图形
    fig, ax = plt.subplots(figsize=(10, 7))

    # 设置固定坐标轴范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 初始状态
    scatter = ax.scatter([], [], c=[], cmap='viridis', s=50, edgecolors='k')  # 普通点
    highlight_scatter = ax.scatter([], [], color='orange', s=100, edgecolors='k')  # 高亮点
    blue_scatter = ax.scatter([], [], color='blue', s=100, edgecolors='k')  # 蓝色点

    # 动态连线
    line = Line2D([], [], color='blue', linestyle='--', lw=2)
    ax.add_line(line)

    # 动画更新函数
    def update(frame):
        # 获取当前帧普通点索引，排除高亮点
        normal_indices = np.setdiff1d(range(frame + 1), highlight_indices)

        # 当前普通点
        current_x = x[normal_indices]
        current_y = y[normal_indices]

        # 更新普通点的位置和颜色
        scatter.set_offsets(np.c_[current_x, current_y])
        scatter.set_array(np.linspace(0, 1, len(normal_indices)))  # 动态更新颜色渐变

        # 更新高亮点
        highlight_x = x[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_y = y[np.intersect1d(highlight_indices, range(frame + 1))]
        highlight_scatter.set_offsets(np.c_[highlight_x, highlight_y])

        # 更新蓝色点
        blue_x = x[np.intersect1d(blue_indices, range(frame + 1))]
        blue_y = y[np.intersect1d(blue_indices, range(frame + 1))]
        blue_scatter.set_offsets(np.c_[blue_x, blue_y])

        # 更新连线（连接橙色点与其蓝色点）
        if len(highlight_x) > 0 and len(blue_x) > 0:
            line.set_data([highlight_x[-1], blue_x[-1]], [highlight_y[-1], blue_y[-1]])
        else:
            line.set_data([], [])  # 如果没有点，则不绘制连线

        return scatter, highlight_scatter, blue_scatter, line

    # 动画初始化函数
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array([])
        highlight_scatter.set_offsets(np.empty((0, 2)))
        blue_scatter.set_offsets(np.empty((0, 2)))
        line.set_data([], [])
        return scatter, highlight_scatter, blue_scatter, line

    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=False, interval=100, repeat=False)

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

    result_OB = compare_distances(hidden_states_dir="../hidden/23_11_0_layers_3_hidden_1024_input_489_sub_OB.pt",
                                  test_csv="../data/sub/hc/403/ADL_B_403_DataOddball_403.csv",
                                  type="OB", )
    result_CP = compare_distances(hidden_states_dir="../hidden/23_10_36_layers_3_hidden_1024_input_489_CP.pt",
                                  test_csv="../data/240_rule/df_test_CP.csv",
                                  type="CP", )
    print("Oddball Distances:", result_OB)
    print("Changepoint Distances:", result_CP)
