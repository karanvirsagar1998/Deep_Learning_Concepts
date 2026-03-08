import torch
import torch.nn as nn

class HousePriceModel(nn.Module):

    def __init__(self):
        super(HousePriceModel, self).__init__()

        self.layer1 = nn.Linear(3, 10)
        self.layer2 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.output(x)

        return x

