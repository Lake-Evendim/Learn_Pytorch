import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1], [0, 1, 2, 3, 1], [1, 2, 1, 0, 0], [5, 2, 3, 1, 1], [2, 1, 0, 1, 1]], dtype=torch.float)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        x = self.maxpool(input)
        return x


tudui = Tudui()
output = tudui(input)
print(output)
print(output.shape)