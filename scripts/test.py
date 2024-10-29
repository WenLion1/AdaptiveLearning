import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataloader import generate_dataset
from scripts.model import perceptual_network
from scripts.utils import determine_training_stops

torch.manual_seed(20010509)
np.random.seed(20010509)

# 打印运行机器类型：cpu、gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'working on {device}')


if __name__ == "__main__":
    # 实验参数
    right_now = time.localtime()
    n_epoch = 1000

    # 模型参数
    model_name = "lstm"
    input_size = 2048
    hidden_size = 256
    num_layers = 3
    output_size = 1
    batch_size = 1
    sequence_length = 240
    model_dir = "../models"
    results_dir = "../results/csv/240_rule"
    results_dir_sub = "../results/csv/sub"
    os.makedirs(results_dir, exist_ok=True)

    learning_rate = 1e-4
    l2_decay = 1e-8
    n_epochs = int(1e3)
    patience = 5
    warmup_epochs = 2
    tol = 1e-4

    # load dataframes
    data_dir = '../data/240_rule'
    df_test = pd.read_csv(os.path.join(data_dir, 'df_test_CP_temp.csv'))
    # df_OB = pd.read_csv(os.path.join(data_dir, 'df_test_OB_300.csv'))
    # df_OB_in_CP = pd.read_csv(os.path.join(data_dir, 'df_test_OBinCP_300.csv'))

    # data_dir_sub = '../data/sub'
    # df_sub = pd.read_csv(os.path.join(data_dir_sub, 'CP_400.csv'))

    # build dataloaders
    dataset_CP = generate_dataset(df_test,
                                  sequence_length=sequence_length, )
    dataloader_CP = DataLoader(dataset_CP,
                               batch_size=batch_size,
                               shuffle=False, )
    #
    # dataset_OB = generate_dataset(df_OB,
    #                               sequence_length=sequence_length, )
    # dataloader_OB = DataLoader(dataset_OB,
    #                            batch_size=batch_size,
    #                            shuffle=False, )

    # dataset_OB_in_CP = generate_dataset(df_OB_in_CP,
    #                                     sequence_length=sequence_length, )
    # dataloader_OB_in_CP = DataLoader(dataset_OB_in_CP,
    #                                  batch_size=batch_size,
    #                                  shuffle=False, )

    # dataset_sub = generate_dataset(df_sub,
    #                                sequence_length=sequence_length, )
    # dataloader_sub = DataLoader(dataset_sub,
    #                             batch_size=batch_size,
    #                             shuffle=False, )

    # build models
    network = perceptual_network(device=device,
                                 input_size=input_size,
                                 hidden_size=hidden_size,
                                 output_size=output_size,
                                 num_layers=num_layers,
                                 model_name=model_name, ).to(device)
    network.load_state_dict(torch.load("../models/240_rule/28_18_36_lstm_layers_3_hidden_256_combine.h5"))
    for p in network.parameters(): p.requires_grad = False
    network.eval()

    # !!!!!!!!!!!!!!!!
    dataloader = dataloader_CP
    iterator = tqdm(enumerate(dataloader))
    loss = 0

    columns = ['distMean_true', 'outcome_true', 'outcome_pre', 'is_oddball']
    results = pd.DataFrame(columns=columns)

    trail_type = 0  # 0代表CP，1代表OB
    for idx_batch, (distMean_true, outcome_true, is_oddball, rule) in iterator:

        # 根据 is_oddball 的第一个值设置 trail_type
        trail_type = 0 if is_oddball[0, 0].item() == -1 else 1

        distMean_true = distMean_true.unsqueeze(-1)  # 在最后添加一个维度
        outcome_true = outcome_true.unsqueeze(-1)  # 在最后添加一个维度
        rule = rule.unsqueeze(-1)

        new_outcome = torch.zeros_like(outcome_true)
        for i in range(outcome_true.shape[0]):
            new_outcome[i, 0, :] = torch.tensor(0.)
            new_outcome[i, 1:, :] = outcome_true[i, :-1, :]

        loss_func = nn.MSELoss()

        # 模型向前传播
        outcome_pre, out = network(new_outcome.to(device).float(), rule.to(device).float())

        # 计算损失
        outcome_loss = loss_func(outcome_true.float().to(device),
                                 outcome_pre.float().to(device), )

        batch_all_loss = outcome_loss
        loss += batch_all_loss.item()

        # 将数据转换为 NumPy 数组
        distMean_true_np = distMean_true.cpu().numpy()
        outcome_true_np = outcome_true.cpu().numpy()
        outcome_pre_np = outcome_pre.detach().cpu().numpy()

        distMean_true_np = np.squeeze(distMean_true_np)
        outcome_true_np = np.squeeze(outcome_true_np)
        outcome_pre_np = np.squeeze(outcome_pre_np)

        # is_changepoint_np = is_changepoint.cpu().numpy()
        # is_changepoint_np = np.squeeze(is_changepoint_np)
        is_oddball_np = is_oddball.cpu().numpy()
        is_oddball_np = np.squeeze(is_oddball_np)

        # 将每个样本的值与批次中的其他样本组合在一起
        batch_data = pd.DataFrame({
            'distMean_true': distMean_true_np,
            'outcome_true': outcome_true_np,
            'outcome_pre': outcome_pre_np,
            'is_oddball': is_oddball_np,
            # 'is_changepoint': is_changepoint_np
        })

        # 去除全是NA的列
        batch_data.dropna(how='all', axis=1, inplace=True)

        # 将批次数据追加到总数据
        results = pd.concat([results, batch_data], ignore_index=True)

        message = f"""{len(dataloader):4.0f}/{100 * (idx_batch + 1) / len(dataloader):2.3f}%, loss = {loss / (idx_batch + 1):2.4f}""".replace(
            '\n', '')
        iterator.set_description(message)

    # 将最终数据写入 CSV 文件，保存在 results_dir 中
    csv_path = os.path.join(results_dir, 'combine_CP_LSTM_l_3_h_256_240.csv')
    results.to_csv(csv_path, index=False)
