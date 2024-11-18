import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 中间部分
        self.middle = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 编码器
        x1 = self.encoder(x)
        # 中间部分
        x2 = self.middle(x1)
        # 解码器
        x3 = self.decoder(x2)

        return x3

# 输入数据形状为(batch_size, 1, 301)，输出为(batch_size, 3, 301)
model = UNet1D(in_channels=1, out_channels=3)

summary(model.cuda(), input_size=(2, 301))
