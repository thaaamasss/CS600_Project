import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):

    def __init__(self, input_channels, num_classes, input_size):

        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.feature_size = self._get_feature_size(input_channels, input_size)

        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def _get_feature_size(self, input_channels, input_size):

        with torch.no_grad():

            x = torch.zeros(1, input_channels, input_size, input_size)

            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))

            x = torch.flatten(x, 1)

        return x.shape[1]


    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x