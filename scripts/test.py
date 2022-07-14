# import os
 
# # __file__ 为当前执行的文件
 
# #当前文件路径
# print()
# #当前文件所在的目录，即父路径
# print(os.path.split(os.path.realpath(__file__))[0])
# #找到父路径下的其他文件，即同级的其他文件
# # print(os.path.join(proDir,"config.ini"))
from tensorboardX import SummaryWriter
writer = SummaryWriter()
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
writer.close()