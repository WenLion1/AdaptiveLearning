import random

import numpy as np
import pandas as pd
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
                                  type="CP",):
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


if __name__ == "__main__":
    # svm_hidden_states(hidden_states_dir="../hidden/6_19_43_layers_3_hidden_1024_input_10.pt")

    svm_hidden_states_singlepoint(hidden_states_dir="../hidden/21_19_56_layers_3_hidden_1024_input_489.pt",
                                  label_csv="../data/240_rule/df_test_OB.csv",
                                  type="OB",)
