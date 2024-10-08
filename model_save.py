import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1  结构+参数
torch.save(vgg16,'vgg16_save_1.pth')

# 保存方式2  参数
torch.save(vgg16.state_dict(),'vgg16_save_2.pth')