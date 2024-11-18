import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.W_q = nn.Linear(input_dim, attention_dim * num_heads)
        self.W_k = nn.Linear(input_dim, attention_dim * num_heads)
        self.W_v = nn.Linear(input_dim, attention_dim * num_heads)
        self.W_o = nn.Linear(attention_dim * num_heads, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        A = F.softmax(attention_scores, dim=-1)

        O = torch.matmul(A, V)
        O = O.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.attention_dim)
        O = self.W_o(O)
        return O


class SpectralAttentionWithGlobal(nn.Module):
    def __init__(self, input_dim, attention_dim, num_heads):
        super(SpectralAttentionWithGlobal, self).__init__()
        self.multi_head_attention = MultiHeadAttention(input_dim, attention_dim, num_heads)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(input_dim, input_dim)
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        O = self.multi_head_attention(x)
        G = self.global_pool(O.transpose(1, 2)).squeeze(-1)
        G = self.linear(G)
        O = O + G.unsqueeze(1)  # 结合全局特征
        O = O.transpose(1, 2)  # 转置以适应卷积操作
        Y = self.conv(O)
        Y = Y.transpose(1, 2)  # 转置回原来的形状
        Y = self.norm(Y)  # 层归一化
        Y = self.dropout(Y)  # Dropout
        return Y


# 轻量化光谱分类网络
class SpNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SpNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.attention = SpectralAttentionWithGlobal(input_dim=256, attention_dim=64, num_heads=8)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm(x)  # 层归一化
        x = self.attention(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout
        x = self.fc3(x)
        x=x.squeeze(1)
        return x



if __name__ == "__main__":
    # 打印模型的参数量和输出形状
    from torchsummary import summary
    # Instantiate the model

    model = SpNet(3801,12)

    # Print the model summary
    summary(model.cuda(), (3801,))


