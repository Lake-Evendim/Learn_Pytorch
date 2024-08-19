import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.pool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.pool(x)
        return x


tudui = Tudui()
# print(tudui)

writer = SummaryWriter('logs')

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images('input', imgs, step)
    writer.add_images('output', output, step)
    step += 1
writer.close()