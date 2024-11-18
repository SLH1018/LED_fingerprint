
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvTransformerModel(nn.Module):
    def __init__(self, input_size=3801, num_classes=8):
        super(ConvTransformerModel, self).__init__()
        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Transformer 注意力机制
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        # 线性层分类器
        self.fc1 = nn.Linear(121536, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # 在通道维度上增加一个维度
        # 卷积层
        x = self.conv1d(x)
        x = F.relu(x)
        x = self.pool(x)
        # 调整特征维度
        x = x.permute(2, 0, 1).contiguous()
        # Transformer 注意力机制
        x = self.transformer_encoder(x)
        # 将特征维度转换为分类维度
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(batch_size, -1)

        # 线性层分类器
        x = self.fc2(self.fc1(x))

        return x

if __name__ == "__main__":
    # 创建模型实例
    model = ConvTransformerModel(input_size=3801, num_classes=8)
    # 打印模型概述
    summary(model.cuda(), (3801,))