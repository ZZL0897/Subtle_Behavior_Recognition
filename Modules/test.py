import time

from colour import Color
import colour
import cv2
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator  # 导入tensorboard的事件解析器

# green = Color("Lime")
# print(Color.get_green(green))
# colors = list(green.range_to(Color("red"), 256))
# print(colors)
#
#
# def f(gray):
#     return Color.get_red(colors[int(gray)]), Color.get_green(colors[int(gray)])
#
#
# map_list = tuple(map(f, range(0, 256)))
# print(map_list[16])
#
#
# def f2(gray):
#     return map_list[gray][0], map_list[gray][1]
#
#
# # for i, c in enumerate(colors):
# #     a = colour.Color.get_rgb(c)
# #     b = colour.Color.get_red(c)
# #     print(a)
# #     print(b)
#
# x = np.random.randint(0, 255, (500, 500))
#
# f_n = np.frompyfunc(f2, 1, 2)
#
# s = time.time()
# for i in range(0, 200):
#     y = f_n(x)
# print(time.time() - s)

log_dir = '../TrainModel/log/two_resnet50_bs16_trans/'
writer = SummaryWriter(log_dir=log_dir)

# for i in range(1, 100):
#     print(i)
#     writer.add_scalar('train/train_acc', 0.5/i, i)
#     writer.add_scalar('train/train_loss', 0.1/i, i)
#     writer.add_scalar('test/test_acc', 0.5/i*i, i)
#     writer.add_scalar('test/test_loss', 0.2/i**2, i)
#     writer.add_scalar('learning_rate', 1/i, i)

ea = event_accumulator.EventAccumulator(log_dir)  # 初始化EventAccumulator对象
ea.Reload()  # 这一步是必须的，将事件的内容都导进去
print(ea.scalars.Keys())  # 我们知道tensorboard可以保存Image scalars等对象，我们主要关注scalars
train_loss = ea.scalars.Items("val/acc_per_epoch")  # 读取train_loss
print(train_loss)



