import os
import time
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataloader import generate_dataset
from scripts.model import perceptual_network, CNN, perceptual_network_vm
from scripts.utils import determine_training_stops

torch.manual_seed(20010509)
np.random.seed(20010509)

# 打印运行机器类型：cpu、gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'working on {device}')


def fit_one_cycle(network,
                  dataloader,
                  optimizer,
                  loss_func,
                  device=torch.device("cpu"),
                  idx_epoch=0,
                  train=True,
                  sequence_length=100, ):
    """
    一次训练或者验证过程

    :param network: 模型网络
    :param dataloader: 数据集合
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param device: cpu、gpu
    :param idx_epoch: 当前第几轮
    :param train: 是否是训练过程
    :param verbose: 输出信息的详细程度：大于0更详细
    :return:
    """

    # 设置模型模式
    if train:
        network.train(True).to(device)
    else:
        network.eval().to(device)

    loss = 0

    iterator = tqdm(enumerate(dataloader))
    for idx_batch, (distMean_true, outcome_true, is_oddball, rule) in iterator:

        distMean_true = distMean_true.unsqueeze(-1)  # 在最后添加一个维度
        outcome_true = outcome_true.unsqueeze(-1)  # 在最后添加一个维度
        rule = rule.unsqueeze(-1)

        new_outcome = torch.zeros_like(outcome_true)
        for i in range(outcome_true.shape[0]):
            new_outcome[i, 0, :] = torch.tensor(0.)
            new_outcome[i, 1:, :] = outcome_true[i, :-1, :]

        # 加入图像

        # # 创建一个与 x 形状相同的张量，用于存储结果
        # shifted_image = torch.zeros_like(image)
        # # 获取 x 的维度
        # batch_size, sequence_length, channels, height, width = image.shape
        # # 使用 x 的维度来创建随机张量，除了时间维度（sequence_length）我们只需要1
        # random_tensor = torch.randn(batch_size, 1, channels, height, width)
        # # 将 x 向后顺移
        # shifted_image[:, 1:, :, :, :] = image[:, :-1, :, :, :]
        # # 将随机数填充到第一个位置
        # shifted_image[:, 0, :, :, :] = random_tensor
        # shifted_image = shifted_image.squeeze(0)

        # 画图

        # # 假设 shifted_image 的形状为 (sequence_length, channels, height, width)
        # sequence_length = shifted_image.shape[0]  # 序列长度
        # channels = shifted_image.shape[1]  # 通道数
        #
        # # 检查 channels 是否为 1 或 3 以适配灰度图或 RGB 图
        # if channels == 1:
        #     shifted_image = shifted_image.squeeze(1)  # 去除单通道维度
        # elif channels == 3:
        #     shifted_image = shifted_image.permute(0, 2, 3, 1)  # 调整通道顺序为 (sequence_length, height, width, channels)
        #
        # # 逐帧显示每个图像
        # for i in range(sequence_length):
        #     plt.imshow(shifted_image[i].cpu().numpy(), cmap='gray' if channels == 1 else None)
        #     plt.title(f"Frame {i + 1}")
        #     plt.axis('off')  # 隐藏坐标轴
        #     plt.show()
        #
        #     time.sleep(1)  # 每张图显示 1 秒，可以根据需要调整时间
        #     plt.close()  # 关闭当前图，以显示下一张图
        # fcsdfc

        # 重置优化器
        optimizer.zero_grad()

        # cnn = CNN().to(device)
        # x = cnn(shifted_image.to(device))

        # 模型向前传播
        outcome_pre, out = network(new_outcome.to(device).float(), rule.to(device).float())

        angles_rad = np.radians(outcome_true.cpu().numpy())
        sin = torch.tensor(np.sin(angles_rad)).to(device)
        cos = torch.tensor(np.cos(angles_rad)).to(device)
        sin_cos = torch.cat((sin, cos), dim=2).to(device)

        # 计算损失
        cosine_loss = 1 - F.cosine_similarity(sin_cos.float(),
                                              outcome_pre.float(),
                                              dim=2)
        outcome_loss = torch.mean(cosine_loss)

        batch_all_loss = outcome_loss
        loss += batch_all_loss.item()

        if train:
            # 反向传播
            batch_all_loss.backward()
            # 修改权重
            optimizer.step()

        message = f"""epoch {idx_epoch + 1}-{idx_batch + 1:3.0f}-{len(dataloader):4.0f}/{100 * (idx_batch + 1) / len(dataloader):2.3f}%, loss = {loss / (idx_batch + 1):2.4f}""".replace(
            '\n', '')
        iterator.set_description(message)

    return network, loss / (idx_batch + 1)


def train_valid_loop(network,
                     dataloader_train,
                     dataloader_valid,
                     optimizer,
                     loss_func,
                     device=torch.device('cpu'),
                     n_epochs=1000,
                     verbose=0,
                     patience=5,
                     warmup_epochs=5,
                     tol=1e-4,
                     saving_name="perceptual_network.h5",
                     sequence_length=100, ):
    """
    总体训练以及验证循环函数

    :param network: 需要训练的模型
    :param dataloader_train: 训练集
    :param dataloader_valid: 验证集
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param device: cpu、gpu
    :param n_epochs: 循环次数
    :param verbose: 输出信息的详细程度：大于0更详细
    :param patience: patience轮次内效果不改善则早停
    :param warmup_epochs: 在训练初期的warmup_epochs轮次内逐步提高学习率
    :param tol: 改善值低于tol值则认定模型未改善
    :param saving_name: 模型参数保存名
    :return:
    """

    best_valid_loss = np.inf  # 记录最佳验证损失
    losses = []
    counts = 0
    for idx_epoch in range(n_epochs):
        print("Training ...")
        network, loss_train = fit_one_cycle(network,
                                            dataloader=dataloader_train,
                                            optimizer=optimizer,
                                            loss_func=loss_func,
                                            device=device,
                                            idx_epoch=idx_epoch,
                                            train=True,
                                            sequence_length=sequence_length, )
        print("Validating ...")
        with torch.no_grad():
            _, loss_valid = fit_one_cycle(network,
                                          dataloader=dataloader_valid,
                                          optimizer=optimizer,
                                          loss_func=loss_func,
                                          device=device,
                                          idx_epoch=idx_epoch,
                                          train=False,
                                          sequence_length=sequence_length, )
        losses.append([loss_train, loss_valid])

        best_valid_loss, counts = determine_training_stops(nets=[network, ],
                                                           idx_epoch=idx_epoch,
                                                           warmup_epochs=warmup_epochs,
                                                           valid_loss=loss_valid,
                                                           counts=counts,
                                                           best_valid_loss=best_valid_loss,
                                                           tol=tol,
                                                           saving_names={saving_name: True, },
                                                           )
        if counts > patience:
            break
        else:
            message = f"""epoch {idx_epoch + 1}, best validation loss = {best_valid_loss:.4f}, count = {counts}""".replace(
                '\n', '')
            print(message)
    return network, losses


if __name__ == "__main__":
    # 实验参数
    right_now = time.localtime()
    n_epoch = 1000

    # 模型参数
    model_name = "rnn"
    input_size = 489
    hidden_size = 16
    num_layers = 1
    output_size = 2
    batch_size = 1
    sequence_length = 240
    model_dir = "../models/240_rule"

    learning_rate = 1e-4
    l2_decay = 1e-8
    n_epochs = int(1e5)
    patience = 2
    warmup_epochs = 2
    tol = 1e-4
    saving_name = os.path.join(model_dir,
                               f'test_OB_first.h5')

    # load dataframes
    data_dir = '../data/240_rule'
    df_train = pd.read_csv(os.path.join(data_dir, 'df_train_combine_OB_first.csv'))
    df_valid = pd.read_csv(os.path.join(data_dir, 'df_valid_combine_OB_first.csv'))

    # build dataloaders
    dataset_train = generate_dataset(df_train,
                                     transform_steps=None,
                                     sequence_length=sequence_length, )
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=False, )
    dataset_valid = generate_dataset(df_valid,
                                     transform_steps=None,
                                     sequence_length=sequence_length, )
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=batch_size,
                                  shuffle=False, )

    # build models
    network = perceptual_network_vm(device=device,
                                    input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=output_size,
                                    num_layers=num_layers,
                                    model_name=model_name, ).to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(params=network.parameters(),
                                 lr=learning_rate,
                                 weight_decay=l2_decay, )
    loss_func = nn.BCELoss()

    # 训练
    network, losses = train_valid_loop(network=network,
                                       dataloader_train=dataloader_train,
                                       dataloader_valid=dataloader_valid,
                                       optimizer=optimizer,
                                       loss_func=loss_func,
                                       device=device,
                                       n_epochs=n_epochs,
                                       patience=patience,
                                       warmup_epochs=warmup_epochs,
                                       tol=tol,
                                       saving_name=saving_name,
                                       sequence_length=sequence_length, )
    del network
