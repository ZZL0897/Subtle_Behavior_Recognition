import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, strides=None, padding=None, first=False) -> None:
        super(Bottleneck, self).__init__()
        if padding is None:
            padding = [0, 1, 0]
        if strides is None:
            strides = [1, 1, 1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, strides[0], padding[0], bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, strides[1], padding[1], bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, 1, strides[2], padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4, momentum=0.01),
        )
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, strides[1], bias=False),
                nn.BatchNorm2d(out_channels * 4, momentum=0.01),
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, Bottleneck=Bottleneck, num_class=10) -> None:
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = nn.Sequential(
            Bottleneck(64, 64, first=True),
            Bottleneck(256, 64, first=False),
            Bottleneck(256, 64, first=False),
        )

        self.stage2 = nn.Sequential(
            Bottleneck(256, 128, strides=[1, 2, 1], first=True),
            Bottleneck(512, 128, first=False),
            Bottleneck(512, 128, first=False),
            Bottleneck(512, 128, first=False),
        )

        self.stage3 = nn.Sequential(
            Bottleneck(512, 256, strides=[1, 2, 1], first=True),
            Bottleneck(1024, 256, first=False),
            Bottleneck(1024, 256, first=False),
            Bottleneck(1024, 256, first=False),
            Bottleneck(1024, 256, first=False),
            Bottleneck(1024, 256, first=False),
        )

        self.stage4 = nn.Sequential(
            Bottleneck(1024, 512, strides=[1, 2, 1], first=True),
            Bottleneck(2048, 512, first=False),
            Bottleneck(2048, 512, first=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        # print(out.shape)
        out = self.avgpool(out)
        # print(out.shape)
        out = torch.flatten(out, 1)
        # print(out.shape)
        out = self.fc(out)
        # print(out)
        # print(out.shape)
        return out


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    x.cuda()
    res50 = ResNet50(num_class=8)
    # print(res50)
    print(summary(res50, input_size=(1, 3, 224, 224)))
    # res50(x)
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())
    # print(get_parameter_number(res50))
