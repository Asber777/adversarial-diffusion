# import torch as t
# import numpy as np
# from torchvision import transforms
# a = np.load('/root/hhtpro/123/result/default_descirbe/adv-2022-07-22-13-30-14-680579/samples_5x256x256x3.npz')
# x = t.from_numpy(a['arr_0']).float()
# x = t.cat([pic for pic in x], 2)
# unloader = transforms.ToPILImage()
# unloader(x).save("result/256_guide_pic/result.jpg")






# from robustbench.data import load_imagenet
# # from robustbench.utils import clean_accuracy, load_model
# from robustbench.data import PREPROCESSINGS
# # model = load_model('Standard_R50', dataset='imagenet', threat_model='Linf').cuda()
# import torchvision.transforms as transforms
# tfs = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(256),
#         transforms.ToTensor()
#     ])
# x, y = load_imagenet(n_examples=5, data_dir='/root/hhtpro/123/imagenet', transforms_test=tfs)
# path = "result/256_guide_pic/samples_5x256x256x3.npz"

# import numpy as np
# all_images = [[xx.numpy()] for xx in x]
# all_labels = [[yy.numpy()] for yy in y]
# arr = np.concatenate(all_images, axis=0)
# label_arr = np.concatenate(all_labels, axis=0)
# np.savez(path, arr, label_arr)


# acc = clean_accuracy(model, x.cuda(), y.cuda(), device=torch.device('cuda'))
# x_in is 64x64 in [-1, 1]
# x = x.cuda()
# y = y.cuda()
# print(model(x))


# import os
 
# # __file__ 为当前执行的文件

# #当前文件路径
# print()
# #当前文件所在的目录，即父路径
# print(os.path.split(os.path.realpath(__file__))[0])
# #找到父路径下的其他文件，即同级的其他文件
# # print(os.path.join(proDir,"config.ini"))
# import torch 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/root/hhtpro/123/guided-diffusion/scripts/testwhy')
for t in range(100):
    writer.add_scalar('y=2x_origi', t * 2, t)
def cond_fn(t):
    writer.add_scalar('y=2x', t * 2, t)
from test2 import fun_out
fun_out(cond_fn)


# def func(i):
#     time = i.detach().cpu().float().item()
#     writer.add_scalar('loss', i*2, time)

# for i in torch.range(1, 100):
#     func(i)


# writer.close()

# from robustbench.data import load_imagenet
# load_imagenet(n_examples=50, data_dir='/root/hhtpro/123/imagenet')
# # x_test, y_test = load_cifar10(n_examples=1000, corruptions=corruptions, severity=5)
# print(__name__)