import os.path
import torch.nn as nn
import torch.optim as optim
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import torch
from utils import train, val
from torchvision import transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model_in_2020paper import Paper2020


def net_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def set_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.01


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(20),
                                    transforms.RandomPerspective(distortion_scale=0.1, p=0.6),
                                    transforms.RandomAffine(degrees=0, shear=10)])
    """
    训练时要注意的地方：
    如果采用两块ROI，那么训练的时候验证集中的0行为会被丢弃掉一半，以保证训练集与验证集类别比例均匀，已在MyDataset中设置了随机种子
    需要将MyDataset中drop_label0参数置为True，默认为False
    模型命名规则：昆虫_ROI数量_是否数据增强_backbone_优化器_batchsize
    """

    batch_size = 16

    save_name = 'jxsy_two_vgg16_' + str(batch_size)

    save_dir = os.path.join('model', save_name + '.pth')
    log_dir = 'log/' + save_name + '/'
    writer = SummaryWriter(log_dir=log_dir)

    train_data = MyDataset(r'G:\train_jxsy_two\train_info.json', transform, drop_label0=True)
    test_data = MyDataset(r'G:\val_jxsy_two\test_info.json')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=6, drop_last=False,
                             pin_memory=True)

    print('Number of steps per epoch: {}'.format(len(train_loader)))

    # net = ResNet50(num_class=8).to(device)
    # net = Paper2020().to(device)
    net = torchvision.models.vgg16(weights=None, progress=False, num_classes=8).to(device)
    # net = torchvision.models.resnet50(weights=None, progress=False, num_classes=8).to(device)

    # net.apply(net_init)
    # net.apply(set_bn)  # BN层的momentum修改为0.99，默认为0.9

    # check = torch.load('vgg16_bn-6c64b313.pth')
    # print(check.keys())
    # check.pop('classifier.6.weight')
    # check.pop('classifier.6.bias')

    # check = torch.load('resnet50-0676ba61.pth')
    # check.pop("fc.weight")
    # check.pop("fc.bias")

    # net.load_state_dict(check, strict=False)

    criterion = nn.CrossEntropyLoss().to(device)  # use a Classification Cross-Entropy loss

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 25], gamma=0.1)

    best_acc = 0
    for epoch in range(1, 31):
        print('Epoch: {}, learning rate: {}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        train_acc = train(net, device, train_loader, criterion, optimizer, epoch, writer)

        # state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict()}
        # torch.save(state, 'net_state.pth')

        scheduler.step()
        torch.cuda.empty_cache()

        val_acc = val(net, device, test_loader, criterion, epoch, writer)
        if val_acc > best_acc:
            print('Validation accuracy increased from {} to {}. Save the model to {}'.format(best_acc, val_acc, save_dir))
            best_acc = val_acc
            torch.save(net, save_dir)
            print('==================================================================================\n')
        else:
            print('Validation accuracy did not increase from {}'.format(best_acc))
            print('==================================================================================\n')
