import torch
from torch import nn


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

        self.network = model_type(model_name=self.model_name,
                                  input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_first=self.batch_first, ).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.fc_rule = nn.Linear(1, 2047).to(self.device)

    def forward(self, angle, rule):
        rule_input = self.fc_rule(rule)
        combined = torch.cat((angle, rule_input), dim=2)

        out, _ = self.network(combined)
        # out = self.fc(out[:, -1, :])

        outcome_pre = self.fc(out)
        return outcome_pre, out


if __name__ == "__main__":
    angle = torch.tensor([[[32.]]])
    rule = torch.tensor([[[1.]]])

    model = perceptual_network()
    out, _ = model(angle, rule)
    print(out)
