import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_2(nn.Module):
    def __init__(self, num_classes=10, input_size=28):
        super(Conv_2, self).__init__()
        self.feat_size = 12544 if input_size == 32 else 12544 if input_size == 28 else -1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.feat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x, mask=None):
        x1 = self.conv1(x)
        if mask: x1 = x1 * mask[0]
        x2 = F.relu(F.max_pool2d(x1, 1))
        x3 = self.conv2(x2)
        if mask: x3 = x3 * mask[1]
        x4 = F.relu(F.max_pool2d(x3, 2))
        x4 = x4.view(-1, self.feat_size)
        x5 = F.relu(self.fc1(x4))
        if mask: x5 = x5 * mask[2]
        x6 = F.relu(self.fc2(x5))
        if mask: x6 = x6 * mask[3]
        x7 = F.log_softmax(self.fc3(x6), dim=1)
        return x7

    def forward_features(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 1))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x2 = x2.view(-1, self.feat_size)
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = F.log_softmax(self.fc3(x4), dim=1)
        return [x2, x3, x4, x5]

    def forward_param_features(self, x):
        x1 = self.conv1(x)
        x2 = F.relu(F.max_pool2d(x1, 1))
        x3 = self.conv2(x2)
        x4 = F.relu(F.max_pool2d(x3, 2))
        x4 = x4.view(-1, self.feat_size)
        x5 = F.relu(self.fc1(x4))
        x6 = F.relu(self.fc2(x5))
        x7 = F.log_softmax(self.fc3(x6), dim=1)
        return [x1, x3, x5, x6, x7]