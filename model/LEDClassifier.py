import torch
import torch.nn as nn
import torch.nn.functional as F

# 网络架构定义
class SpectralClassifier(nn.Module):
    def __init__(self):
        super(SpectralClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64*475, 512)  # (760/2^3) * 64 = 1890
        self.fc2 = nn.Linear(512, 8)  # 12 classes

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64*475)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Initialize the model
    model = SpectralClassifier()

    # Move the model to GPU
    model = model.cuda()

    # Print the model summary
    from torchsummary import summary
    summary(model, (3801,))
