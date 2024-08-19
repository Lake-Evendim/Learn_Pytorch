import torch
import torchvision

# 加载方式1
model = torch.load('vgg16_save_1.pth')
# print(model)

# 加载方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_save_2.pth'))
print(vgg16)