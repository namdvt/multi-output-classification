import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear(x)
        if x.shape[0] != 1:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    

class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv1 = Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2 = Conv2d(32, 32, kernel_size=3, padding=2)
        self.conv3 = Conv2d(32, 64, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = Conv2d(64, 80, kernel_size=1)
        self.conv5 = Conv2d(80, 192, kernel_size=3, padding=2)
        self.conv6 = Conv2d(192, 256, kernel_size=3, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class Category(nn.Module):
    def __init__(self, in_channels):
        super(Category, self).__init__()
        self.branch0 = Conv2d(in_channels, 32, kernel_size=(1, 1))
        self.branch1 = nn.Sequential(
            Conv2d(in_channels, 32, kernel_size=(1, 1)),
            Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch2 = nn.Sequential(
            Conv2d(in_channels, 32, kernel_size=(1, 1)),
            Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
            Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0))
        )
        self.conv = nn.Conv2d(96, 256, kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = torch.cat((x0, x1, x2), dim=1)
        x_cat = self.conv(x_cat)
        return self.relu(x + x_cat)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.stem = Stem(3)
        self.category = nn.Sequential(
            Category(256),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(256, 5),
            nn.Softmax(1)
        )
        self.color = nn.Sequential(
            Conv2d(256, 32, kernel_size=3),
            Conv2d(32, 32, kernel_size=3),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(32, 6),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.stem(x)
        color = self.color(x)
        category = self.category(x)
        return color, category
