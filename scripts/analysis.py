import numpy as np
import torch
from matplotlib import pyplot as plt
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
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1得分: {f1:.4f}")


if __name__ == "__main__":
    # svm_hidden_states(hidden_states_dir="../hidden/6_19_43_layers_3_hidden_1024_input_10.pt")

    # 投影后的数据点
    A_proj = [3, 5, 7]
    B_proj = [3, 5, 7]

    # 绘制数据点
    plt.scatter(A_proj, [0] * len(A_proj), color='blue', label='类别A')
    plt.scatter(B_proj, [1] * len(B_proj), color='red', label='类别B', alpha=0.5)  # 使用不同的y值以区分

    plt.xlabel('投影维度')
    plt.title('一维空间中的投影数据')
    plt.legend()
    plt.show()