import torch
import torch.nn as nn

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=7, num_layers=3, hidden_dim=21):
        super(TransformerClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x=x.reshape(x.shape[0],-1,21)

        # x = self.embedding(x)  # (batch_size, seq_length, hidden_dim)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # global average pooling
        x = self.fc(x)
        return x

# # 示例：创建一个 TransformerModel 实例
# num_classes = 12
# seq_length = 3801
# input_dim = 3801
#
# model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes)
#
# from torchsummary import summary
#
# summary(model.cuda(), (3801,))