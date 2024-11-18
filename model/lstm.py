import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3801, hidden_size=256, num_layers=2, num_classes=8):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出进行分类

        return out

if __name__ == "__main__":
    # 定义模型参数
    input_size = 3801
    hidden_size = 128
    num_layers = 2
    num_classes = 8

    # 初始化模型
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)

    # 输出模型结构
    print(model)