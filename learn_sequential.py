import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)

        )

    def forward(self, x):
        x = self.model(x)
        return x


tudui = Tudui()
print(tudui)
input = torch.ones((64,3,32,32))
output = tudui(input)
print(output.shape)

writer = SummaryWriter('logs')
writer.add_graph(tudui, input)

writer.close()