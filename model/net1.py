
import torch
import torch.nn as nn
# 定义分类网络
from torchsummary import summary


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(3801, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 12)  # 3个类别

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 示例创建网络
if __name__=="__main__":
    model = Net1()
    summary(model.cuda(), input_size=(3800,))