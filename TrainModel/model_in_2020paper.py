import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary


class Paper2020(nn.Module):
    def __init__(self, num_classes=8):
        super(Paper2020, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(13 * 13 * 128, 512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    x = torch.randn((1, 3, 100, 100))
    x.cuda()
    net = Paper2020()
    print(net)
    print(summary(net, input_size=(1, 3, 100, 100)))
    # net(x)

