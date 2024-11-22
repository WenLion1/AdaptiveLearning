import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from scipy.stats import vonmises
from util.dataloader_util import draw_image
from util.utils_deep import concatenate_transform_steps


def model_type(model_name: str, input_size: int, hidden_size: int, num_layers: int, batch_first: bool):
    """
    选择模型类型，rnn、lstm还是gru

    :param batch_first: 批次数是否在第一个位置
    :param num_layers: 一次输出中间的层数
    :param hidden_size: 隐藏层维度大小
    :param input_size: 输入维度大小
    :param model_name: rnn、lstm、gru
    :return: 具体的模型
    """

    model = None
    if model_name == "rnn":
        model = nn.RNN(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=batch_first, )
    elif model_name == "lstm":
        model = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=batch_first, )
    elif model_name == "gru":
        model = nn.GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=batch_first, )
    else:
        print("模型加载出现错误!")

    return model


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 第一层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 第一层池化
        )
        # 计算经过卷积和池化后的特征图尺寸
        # 输入尺寸为128x128，经过一次池化后变为64x64
        # 因此，全连接层的输入特征数为32*64*64
        self.fc = nn.Linear(32 * 64 * 64, 1)  # 调整线性层的输入特征数

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc(x)
        return x


class perceptual_network_with_image(nn.Module):
    def __init__(self,
                 device='cpu',
                 input_size=2,
                 hidden_size=128,
                 num_layers=1,
                 batch_first=True,
                 output_size=1,
                 model_name="rnn", ):
        super(perceptual_network_with_image, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.model_name = model_name
        self.device = device

        self.network = model_type(model_name=self.model_name,
                                  input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_first=self.batch_first, ).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.fc_rule = nn.Linear(1, 1023).to(self.device)
        self.cnn = CNN()

    def forward(self, image, rule):
        rule_input = self.fc_rule(rule)

        # 将 image 张量的形状从 [1, 1024] 调整为 [1, 1, 1024]
        image_features = image_features.unsqueeze(0)  # 现在 image 的形状是 [1, 1, 1024]

        # 沿着第二个维度（dim=2）合并这两个张量
        combined = torch.cat((image_features, rule_input), dim=2)  # 结果形状为 [1, 1, 2048]

        out, _ = self.network(combined)
        # out = self.fc(out[:, -1, :])

        outcome_pre = self.fc(out)
        return outcome_pre, out


class perceptual_network(nn.Module):
    def __init__(self,
                 device='cpu',
                 input_size=2,
                 hidden_size=128,
                 num_layers=1,
                 batch_first=True,
                 output_size=1,
                 model_name="rnn", ):
        super(perceptual_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.model_name = model_name
        self.device = device

        self.cnn = CNN().to(self.device)

        self.network = model_type(model_name=self.model_name,
                                  input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_first=self.batch_first, ).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.fc_rule = nn.Linear(1, 8).to(self.device)

    def forward(self, angle, rule):
        angles_rad = np.radians(angle.cpu().numpy())
        sin = torch.tensor(np.sin(angles_rad)).to(self.device)
        cos = torch.tensor(np.cos(angles_rad)).to(self.device)
        sin_cos = torch.cat((sin, cos), dim=2).to(self.device)

        rule_input = self.fc_rule(rule)

        combined = torch.cat((sin_cos, rule_input), dim=2)

        angles = []
        # 遍历张量中的每个元素
        for i in range(sin_cos.shape[1]):
            # 提取第三维度的值
            sin_value = sin_cos[0, i, 0]
            cos_value = sin_cos[0, i, 1]

            # 计算角度
            angle = math.atan2(sin_value, cos_value)  # 使用atan2来计算角度

            # 将弧度转换为角度
            angle_degrees = math.degrees(angle)

            # 将角度添加到列表中
            angles.append(angle_degrees)

        out, hn = self.network(combined)
        # out = self.fc(out[:, -1, :])

        outcome_pre = self.fc(out)
        return outcome_pre, out


class perceptual_network_vm(nn.Module):
    def __init__(self,
                 device='cpu',
                 input_size=489,
                 hidden_size=361,
                 num_layers=1,
                 batch_first=True,
                 output_size=1,
                 model_name="rnn",
                 sigma=0.1745, ):
        super(perceptual_network_vm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.model_name = model_name
        self.device = device
        self.sigma = sigma
        self.kappa = 1 / self.sigma**2

        self.network = model_type(model_name=self.model_name,
                                  input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_first=self.batch_first, ).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.fc_rule = nn.Linear(1, 128).to(self.device)

    def forward(self, angle, rule):
        angle_radians = np.deg2rad(angle.cpu())
        initial_input = np.arange(0, 361)
        initial_input = initial_input * (np.pi / 180)
        angle_input = torch.zeros((angle_radians.shape[1], 361))
        for i in range(angle_radians.shape[1]):
            mu = angle_radians[0][i][0]
            vonmises_pdf = torch.from_numpy(vonmises.pdf(initial_input, self.kappa, loc=mu)).float()

            angle_input[i] = vonmises_pdf

        rule_input = self.fc_rule(rule)
        angle_input = angle_input.unsqueeze(0)
        combined = torch.cat((rule_input.to(self.device), angle_input.to(self.device)), dim=2)

        out, hn = self.network(combined)
        # out = self.fc(out[:, -1, :])

        outcome_pre = self.fc(out)
        return outcome_pre, out


if __name__ == "__main__":
    # transform_steps = concatenate_transform_steps(image_resize=256,
    #                                               fill_empty_space=255,
    #                                               grayscale=False,
    #                                               )
    # angle = torch.tensor([[[32.]]])
    # image = draw_image(angle)
    # image = transform_steps(image)
    # image = image.unsqueeze(0)
    rule = torch.tensor([[[1.], [0.], [1], [0]]])
    angle = torch.tensor([[[30.], [45.], [60.], [120.]]])

    model = perceptual_network_vm(input_size=360)
    out, _ = model(angle, rule)
