import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Simple CNN module
class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten_dim = 32 * (input_dim // 4)  # Assuming input_dim is divisible by 4

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(x.size(0), -1)


# Simple Transformer module
class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleTransformer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = F.relu

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        linear_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + linear_output
        x = self.norm2(x)
        return x


# Main model with CNN and Transformer
class Transformer_cnn(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Transformer_cnn, self).__init__()
        self.cnn_branch = SimpleCNN(input_dim)
        embed_dim = 128  # Assume a suitable embedding dimension
        self.trans_branch = SimpleTransformer(embed_dim=embed_dim, num_heads=4)
        self.fc1 = nn.Linear(self.cnn_branch.flatten_dim, embed_dim)  # Project CNN output to Transformer input size
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # B, 1, L for CNN input
        cnn_features = self.cnn_branch(x)

        # Prepare input for transformer
        transformer_input = self.fc1(cnn_features).unsqueeze(0)

        transformer_output = self.trans_branch(transformer_input)
        transformer_output = transformer_output.squeeze(0)  # Remove batch dimension

        output = self.fc_out(transformer_output)
        return output,transformer_output


if __name__ == "__main__":
    # define the model
    # 实例化模型
    model = Transformer_cnn(3801,12)
    # print model summary
    from torchsummary import summary

    summary(model.cuda(), (3801,))