from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

# python 的用法 => tensor 数据类型

image_path = "/home/b612/Downloads/FFCTL-main/pytorch_xiao/dataset/13_23.jpg"
img = Image.open(image_path)

writer = SummaryWriter("logs")

# 1、transforms 该如何使用（python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

writer.close()