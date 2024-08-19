from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = 'data/train/ants_image/0013035.jpg'
img = Image.open(img_path)
# print(img)

# ToTensor
# transforms.ToTensor是一个Class而不是一个函数，因此在使用时要先创建一个ToTensor类的实例
transform_totensor = transforms.ToTensor()
tensor_img = transform_totensor(img)
print(tensor_img)
writer = SummaryWriter('logs')
writer.add_image('tensor_img',tensor_img)

# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
norm_img = trans_norm(tensor_img)
print(norm_img[0][0][0])
writer.add_image('norm_img',norm_img)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
resize_img = trans_resize(img)
resize_img = transform_totensor(resize_img)
writer.add_image('resize',resize_img)
print(resize_img)

# Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, transform_totensor])
compose_img = trans_compose(img)
writer.add_image('resize',compose_img,1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,transform_totensor])
for i in range(10):
    random_img = trans_compose_2(img)
    writer.add_image('RandomCrop',random_img,i)


writer.close()