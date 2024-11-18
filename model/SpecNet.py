import torch
import torch.nn as nn
from torchsummary import summary


class WaveletTransform(nn.Module):
    """
    基于小波变换的特征变换模块
    """
    def __init__(self, input_size):
        super(WaveletTransform, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 计算小波变换后的数据维度，用于后续的拼接操作
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, input_size)
            self.feature_dim = self.transform(dummy_x).view(1, -1).shape[1]

    def forward(self, x):
        return self.transform(x).view(x.size(0), -1)


class SpecNet(nn.Module):
    """
    双流特征输入分类网络
    """
    def __init__(self, input_size, num_classes):
        super(SpecNet, self).__init__()

        # 原始数据分支
        self.raw_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 变换数据分支
        self.wavelet_transform = WaveletTransform(input_size)

        # 融合层
        self.fcn = nn.Sequential(
            nn.Linear(64 * 947 + self.wavelet_transform.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_raw, x_wavelet):
        x_raw = self.raw_cnn(x_raw)
        x_wavelet = self.wavelet_transform(x_wavelet)
        x = torch.cat([x_raw, x_wavelet], dim=1)
        x = self.fcn(x)
        return x




if __name__=="__main__":
    # 定义一些超参数
    input_size = 3801
    num_classes = 8

    # 创建模型实例
    model = SpecNet(input_size=input_size, num_classes=num_classes)
    summary(model.cuda(), input_size=(3801,))