from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

writer = SummaryWriter("logs")
image = Image.open("/home/b612/Downloads/FFCTL-main/pytorch_xiao/dataset/13_23.jpg")
print(image)

# ToTensor
tran_tensor = transforms.ToTensor()
img_tensor = tran_tensor(image)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [2, 0.5, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Noramlize", img_norm, 2)

# Resize
print(image.size)
tran_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = tran_resize(image)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = tran_tensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
tran_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
tran_compose = transforms.Compose([tran_resize_2, tran_tensor])
img_resize_2 = tran_compose(image)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
tran_random = transforms.RandomCrop((500, 1000))
tran_compose_2 = transforms.Compose([tran_random, tran_tensor])
for i in range(10):
    img_crop = tran_compose_2(image)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()