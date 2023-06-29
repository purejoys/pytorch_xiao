import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
#
# input = torch.reshape(input, (-1, 1, 2, 2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.relu1 = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

net = Net()
#output = net(input)
#print(output)

writer = SummaryWriter("logs_relu")

step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = net(imgs)
    writer.add_images("output", output, global_step=step)

    step += 1

writer.close()