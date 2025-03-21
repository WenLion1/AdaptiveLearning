import math
import os
import time
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataloader import generate_dataset
from scripts.model import perceptual_network, perceptual_network_vm
from scripts.utils import determine_training_stops

# 打印运行机器类型：cpu、gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'working on {device}')


def evaluate_model(data_dir,
                   model_path,
                   results_dir,
                   test_type="combine",
                   hidden_states_save_dir="../hidden",
                   model_type="lstm",
                   sequence_length=240,
                   input_size=489,
                   hidden_size=1024,
                   num_layers=3,
                   output_size=2,
                   batch_size=1,
                   is_save_hidden_state=0, ):
    """
    评估预训练模型在提供的数据集上的表现。

    参数：
    - is_save_hidden_state: int，是否保存隐藏层，1-保存，0-不保存
    - data_dir: str，包含CSV数据的目录路径。
    - model_path: str，预训练模型文件的路径。
    - results_dir: str，结果将保存到的目录路径
    - hidden_states_save_dir: str，隐藏层将保存到的目录路径。
    - sequence_length: int，输入序列的长度。
    - input_size: int，输入特征的大小。
    - hidden_size: int，模型中隐藏单元的数量。
    - num_layers: int，模型中LSTM层的数量。
    - output_size: int，输出特征的数量。
    - batch_size: int，评估时的批次大小。

    返回：
    - csv_path: str，保存结果的CSV文件路径。
    - hidden_states: list 每个时间点的隐藏层
    """

    # 实验参数
    hidden_states = []
    right_now = time.localtime()
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据框
    df_test = pd.read_csv(data_dir)

    # 构建数据加载器
    dataset_CP = generate_dataset(df_test, sequence_length=sequence_length)
    dataloader_CP = DataLoader(dataset_CP, batch_size=batch_size, shuffle=False)

    # 构建模型
    network = perceptual_network_vm(device=device,
                                    model_name=model_type,
                                    input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=output_size,
                                    num_layers=num_layers).to(device)
    network.load_state_dict(torch.load(model_path))
    for p in network.parameters():
        p.requires_grad = False
    network.eval()

    # 初始化变量
    loss = 0
    columns = ['distMean', 'outcome', 'pred', 'is_oddball']
    results = pd.DataFrame(columns=columns)

    for idx_batch, (distMean_true, outcome_true, is_oddball, rule) in tqdm(enumerate(dataloader_CP),
                                                                           total=len(dataloader_CP)):
        # 根据is_oddball设置trail_type
        trail_type = 0 if is_oddball[0, 0].item() == -1 else 1

        distMean_true = distMean_true.unsqueeze(-1)  # 添加一个维度
        outcome_true = outcome_true.unsqueeze(-1)  # 添加一个维度
        rule = rule.unsqueeze(-1)

        new_outcome = torch.zeros_like(outcome_true)
        for i in range(outcome_true.shape[0]):
            new_outcome[i, 0, :] = torch.tensor(0.)
            new_outcome[i, 1:, :] = outcome_true[i, :-1, :]

        # 前向传播
        outcome_pre, hidden_layer = network(new_outcome.to(device).float(), rule.to(device).float())

        # 保存隐藏层
        for time_step in range(hidden_layer.shape[1]):
            hidden_state_at_time = hidden_layer[:, time_step, :].cpu().detach().numpy()
            hidden_states.append(hidden_state_at_time)

        angles_rad = np.radians(outcome_true.cpu().numpy())
        sin = torch.tensor(np.sin(angles_rad)).to(device)
        cos = torch.tensor(np.cos(angles_rad)).to(device)
        sin_cos = torch.cat((sin, cos), dim=2).to(device)

        # 计算损失
        cosine_loss = 1 - F.cosine_similarity(sin_cos.float(), outcome_pre.float(), dim=2)
        outcome_loss = torch.mean(cosine_loss)

        batch_all_loss = outcome_loss
        loss += batch_all_loss.item()

        # 初始化列表以存储角度
        angles = []
        for i in range(outcome_pre.shape[1]):
            sin_value = outcome_pre[0, i, 0]
            cos_value = outcome_pre[0, i, 1]
            angle = math.atan2(sin_value, cos_value)  # 计算角度
            angle_degrees = math.degrees(angle) % 360  # 转换为度
            angles.append(angle_degrees)

        # 转换数据为NumPy数组
        distMean_true_np = np.squeeze(distMean_true.cpu().numpy())
        outcome_true_np = np.squeeze(outcome_true.cpu().numpy())
        is_oddball_np = np.squeeze(is_oddball.cpu().numpy())

        # 将结果组合成DataFrame
        batch_data = pd.DataFrame({
            'distMean': distMean_true_np,
            'outcome': outcome_true_np,
            'pred': angles,
            'is_oddball': is_oddball_np,
        })
        batch_data.dropna(how='all', axis=1, inplace=True)  # 删除全是NaN的列
        results = pd.concat([results, batch_data], ignore_index=True)  # 追加到结果中

    if is_save_hidden_state == 1:
        hidden_states = np.vstack(hidden_states)
        hidden_states_save_dir += "/not_remove"
        os.makedirs(hidden_states_save_dir, exist_ok=True)
        path = os.path.join(hidden_states_save_dir, f"{model_type}_layers_{num_layers}_hidden_{hidden_size}_input_{input_size}_{test_type}.pt")
        print("隐藏层正保存至: ", path)
        # 如果文件已存在，删除它（可选）
        if os.path.exists(path):
            os.remove(path)
        torch.save(torch.tensor(hidden_states), path)

    # 计算平均损失并打印
    average_loss = loss / len(dataloader_CP)
    print(f"最终平均损失: {average_loss:.4f}")

    # 根据data_dir中的文件名确定结果文件名
    base_filename = os.path.basename(data_dir)
    if "CP" in base_filename:
        ob_type = "CP"
    elif "Oddball" in base_filename:
        ob_type = "OB"
    elif "reverse" in base_filename:
        ob_type = "reverse"
    elif "combine" in base_filename:
        ob_type = "combine"
    else:
        ob_type = "UNKNOWN"  # 如果都没有则标记为UNKNOWN

    # 将结果保存到CSV
    csv_path = os.path.join(results_dir, f'combine_{ob_type}_{os.path.basename(model_path).split(".")[0]}_cos.csv')
    results.to_csv(csv_path, index=False)

    return csv_path, hidden_states


def batch_evaluate(data_folder_path,
                   model_path,
                   results_folder_path,
                   hidden_state_save_dir="../hidden",
                   model_type="lstm",
                   sequence_length=240,
                   input_size=489,
                   hidden_size=1024,
                   num_layers=3,
                   output_size=2,
                   batch_size=1,
                   is_save_hidden_state=0, ):
    """
    遍历 data_folder_path 下的所有子文件夹，评估每个 CSV 文件并保存结果到 results_folder_path。
    """

    # 遍历 data_folder_path 的所有子文件夹
    for subdir, _, files in os.walk(data_folder_path):
        # 筛选出 CSV 文件
        csv_files = [f for f in files if f.endswith('.csv')]

        if not csv_files:
            print(f"在文件夹 {subdir} 中未找到任何 CSV 文件，跳过该文件夹。")
            continue  # 如果没有找到 CSV 文件，则跳过

        # 为当前子文件夹创建相应的结果保存路径
        subfolder_name = os.path.basename(subdir)
        current_results_dir = os.path.join(results_folder_path, subfolder_name)
        current_hidden_dir = os.path.join(hidden_state_save_dir, subfolder_name)
        os.makedirs(current_results_dir, exist_ok=True)  # 创建子文件夹（如果尚不存在）


        # 遍历当前子文件夹中的每个 CSV 文件
        for csv_file in csv_files:
            data_dir = os.path.join(subdir, csv_file)  # 完整的 CSV 文件路径
            print(f"正在评估文件: {data_dir}")

            if "DataCP" in csv_file:
                test_type = "CP"
            elif "DataOddball" in csv_file:
                test_type = "OB"
            elif "reverse" in csv_file:
                test_type = "reverse"
            elif "combine" in csv_file:
                test_type = "combine"
            else:
                test_type = "unknown"

            # 调用评估模型函数
            evaluate_model(data_dir,
                           model_path,
                           current_results_dir,
                           test_type=test_type,
                           hidden_states_save_dir=current_hidden_dir,
                           model_type=model_type,
                           sequence_length=sequence_length,
                           input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           output_size=output_size,
                           batch_size=batch_size,
                           is_save_hidden_state=is_save_hidden_state)

    print("所有文件评估完成。")


if __name__ == "__main__":
    batch_evaluate(data_folder_path="../data/sub/hc",
                   model_path="../models/240_rule/test_OB_first.h5",
                   results_folder_path="../results/csv/sub/test_OB_first",
                   hidden_state_save_dir="../hidden/sub/test_OB_first",
                   is_save_hidden_state=1,
                   num_layers=1,
                   model_type="rnn",
                   hidden_size=16,)

    # evaluate_model(data_dir="../data/sub/hc/405/ADL_B_405_DataCP_405.csv",
    #                model_path="../models/10/rnn_layers_1_hidden_16_input_489_10.h5",
    #                results_dir="../results",
    #                hidden_states_save_dir="../hidden",
    #                is_save_hidden_state=1,
    #                test_type="combine",
    #                model_type="rnn",
    #                num_layers=1,
    #                hidden_size=16,)
