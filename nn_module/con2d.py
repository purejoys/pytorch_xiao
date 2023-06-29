import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

net = Net()

writer = SummaryWriter("logs")

step = 0

for data in dataloader:
    imgs, targets = data
    output = net(imgs)
    output = torch.reshape(output, (-1, 3, 30, 30))
    print(imgs.shape)
    print(output.shape)
    writer.add_images("inpurt", imgs, step)
    writer.add_images("output", output, step)

    step = step + 1

writer.close()