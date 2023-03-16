import time

import numpy as np
import torch
import cv2
from torchvision import transforms
from torchvision.transforms import functional as f
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.io import read_image, ImageReadMode

img_path = r'G:\0018.png'

# image = plt.imread(img_path)
i=0
b = time.time()
while 1:
    image = cv2.imread(img_path)
    # image = Image.open(img_path).convert('RGB')
    # image = read_image(img_path, mode=ImageReadMode.RGB).cuda()
    # print(image.shape)
    # image = Image.fromarray(np.uint8(image))
    image = torch.from_numpy(image).cuda()
    # print(image.size())
    # image = torch.unsqueeze(image, 0)
    # imgs = torch.vstack((image, image, image, image, image))
    image = torch.permute(image, (2, 0, 1))
    print(image.size())
    image = torch.unsqueeze(image, 0)
    imgs = torch.cat((image, image, image))
    print(imgs.size())

    image_r = f.rotate(imgs, 45, center=[521, 205], fill=0)
    # image_r = f.perspective(imgs, [[510, 50], [665, 246], [356, 182], [511, 335]],
    #                         [[346, 44], [662, 38], [346, 348], [662, 346]])

    # image_r.show()
    # img = image_r.numpy()
    # # print(type(img))
    # print(img.shape)
    #
    # img = np.squeeze()
    #
    image_save = f.to_pil_image(image_r[i])
    image_save.show(image_save)

    # plt.imshow(img)
    # plt.show()
    i+=1
    print(i)
    if i==10:
        break

print(time.time()-b)
