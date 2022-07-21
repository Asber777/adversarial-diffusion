
from robustbench.data import load_imagenet
from robustbench.utils import clean_accuracy, load_model

model = load_model('Standard_R50', dataset='imagenet', threat_model='Linf').cuda()
x, y = load_imagenet(n_examples=1000, data_dir='/root/hhtpro/123/imagenet', )
print(len(x))
acc = clean_accuracy(model, x.cuda(), y.cuda(), device=torch.device('cuda'))


# import os
 
# # __file__ 为当前执行的文件
 
# #当前文件路径
# print()
# #当前文件所在的目录，即父路径
# print(os.path.split(os.path.realpath(__file__))[0])
# #找到父路径下的其他文件，即同级的其他文件
# # print(os.path.join(proDir,"config.ini"))
import torch 
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('/testwhy')

# x = range(100)
# for i in x:
#     writer.add_scalar('y=2x', i * 2, i)


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