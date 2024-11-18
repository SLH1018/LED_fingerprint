import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.dismax import DisMaxLossFirstPart



# 自定义注意力机制模块
class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class SpectralNet_DisMaxLoss(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SpectralNet_DisMaxLoss, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.attention1 = Attention(in_dim=256)
        self.fc2 = nn.Linear(256, 128)
        self.attention2 = Attention(in_dim=128)
        # self.fc3 = nn.Linear(128, num_classes)
        self.classifier = DisMaxLossFirstPart(128, num_classes)

    def forward(self, x):

        x1 = F.relu(self.fc1(x))
        x1 = x1.unsqueeze(1)  # 增加一个维度以适应注意力机制的输入
        x1 = self.attention1(x1)
        x1 = x1.squeeze(1)  # 去掉多余的维度
        x1 = F.relu(self.fc2(x1))
        # x1 = x1.unsqueeze(1)  # 增加一个维度以适应注意力机制的输入
        # x1 = self.attention2(x1)
        # x1 = x1.squeeze(1)
        outputs = self.classifier(x1)
        return outputs

if __name__ == "__main__":
    # 打印模型的参数量和输出形状
    from torchsummary import summary
    # Instantiate the model

    model = SpectralNet_DisMaxLoss(3801,12)

    # Print the model summary
    summary(model.cuda(), (3801,))
