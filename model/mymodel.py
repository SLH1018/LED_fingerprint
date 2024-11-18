import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#
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

class SpectralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SpectralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.attention = Attention(in_dim=256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 前向传播
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)
        features = F.relu(self.fc2(x))
        outputs = self.fc3(features)
        # 在推理阶段，仅返回特征
        return outputs,features



if __name__ == "__main__":


    from torchsummary import summary
    # Instantiate the model
    model = SpectralNet(3801,8)
    # Print the model summary
    summary(model.cuda(), (3801,))


