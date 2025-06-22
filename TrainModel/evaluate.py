import os

import torchvision.io.image
import torch.functional
from utils import evaluate
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


if __name__ == '__main__':

    test_data = MyDataset(r'G:\val_jxsy_two\test_info.json')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=8, drop_last=False,
                             pin_memory=True)

    print('Number of steps per epoch: {}'.format(len(test_loader)))

    net = torch.load('model/jxsy_two_resnet50_bs16_trans.pth').to(device)

    criterion = nn.CrossEntropyLoss().to(device)  # use a Classification Cross-Entropy loss

    evaluate(net, device, test_loader, criterion)

    # path = r'D:\Users\ZZL\Desktop\hzz'
    # img_list = os.listdir(path)
    # x = 0
    # with torch.no_grad():
    #     for i in img_list:
    #         img = torchvision.io.image.read_image(os.path.join(path, i))
    #         img = img / 255
    #         img = torchvision.transforms.Resize((224, 224))(img).to(device)
    #         output = net(torch.unsqueeze(img, 0))
    #         _, pred_label = torch.max(output.data, 1)
    #         if pred_label.item() == 4:
    #             x += 1
    #         else:
    #             print(i)
    #     print(x)




