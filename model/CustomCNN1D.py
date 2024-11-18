import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class CustomCNN1D(nn.Module):
    def __init__(self,num_classes):
        super(CustomCNN1D, self).__init__()

        # 第一层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, padding=1)
        # 第二层
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, padding=1)
        # 第三层
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, padding=1)
        # 第四层
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, padding=1)
        # 全局平均池化
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        # 全连接层1
        self.fc1 = nn.Linear(64, 256)
        # 全连接层2
        self.fc2 = nn.Linear(256, 64)
        # 输出层
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    model = CustomCNN1D(12)
    summary(model.cuda(), input_size=(3801,))
