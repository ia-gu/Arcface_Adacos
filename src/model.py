import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

class VGG_based(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(9*9*512, 2048),
            nn.ReLU(inplace = True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 9*9*512)
        x = self.linear(x)
        return x




class ResNet_based(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = nn.ModuleList([
            self._building_block(256) for _ in range(2)
        ])
        self.conv2 = nn.Conv2d(256, 512, 1, stride=(2, 2))
        self.block2 = nn.ModuleList([
            self._building_block(512) for _ in range(4)
        ])
        self.conv3 = nn.Conv2d(512, 1024, 1, stride=(2, 2))
        self.block3 = nn.ModuleList([
            self._building_block(1024) for _ in range(6)
        ])
        self.conv4 = nn.Conv2d(1024, 2048, 1, stride=(2, 2))
        self.block4 = nn.ModuleList([
            self._building_block(2048) for _ in range(3)
        ])
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 512)
        self.out = nn.Linear(512, 100)

    def pretrain(self, x):
        x = self.head(x)
        x = self.block0(x)
        for block in self.block1:
            x = block(x)
        x = self.conv2(x)
        for block in self.block2:
            x = block(x)
        x = self.conv3(x)
        for block in self.block3:
            x = block(x)
        x = self.conv4(x)
        for block in self.block4:
            x = block(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.out(x)
        return x

    # 順伝播
    def forward(self, x):
        x = self.head(x)
        x = self.block0(x)
        for block in self.block1:
            x = block(x)
        x = self.conv2(x)
        for block in self.block2:
            x = block(x)
        x = self.conv3(x)
        for block in self.block3:
            x = block(x)
        x = self.conv4(x)
        for block in self.block4:
            x = block(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = torch.relu(x)
        # x = self.out(x)
        # x = torch.log_softmax(x, dim=-1)
        return x


    def test(self, x1, x2):
        x1 = self.head(x1)
        x1 = self.block0(x1)
        for block in self.block1:
            x1 = block(x1)
        x1 = self.conv2(x1)
        for block in self.block2:
            x1 = block(x1)
        x1 = self.conv3(x1)
        for block in self.block3:
            x1 = block(x1)
        x1 = self.conv4(x1)
        for block in self.block4:
            x1 = block(x1)
        x1 = self.avg_pool(x1)
        x1 = self.fc(x1)
        x1 = torch.relu(x1)

        x2 = self.head(x2)
        x2 = self.block0(x2)
        for block in self.block1:
            x2 = block(x2)
        x2 = self.conv2(x2)
        for block in self.block2:
            x2 = block(x2)
        x2 = self.conv3(x2)
        for block in self.block3:
            x2 = block(x2)
        x2 = self.conv4(x2)
        for block in self.block4:
            x2 = block(x2)
        x2 = self.avg_pool(x2)
        x2 = self.fc(x2)
        x2 = torch.relu(x2)

        return x1, x2


    def _building_block(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return ResBlock(channel_in, channel_out)

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out

        self.block = nn.Sequential(
            nn.Conv2d(channel_in, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel_out, 1, padding=0),
            nn.BatchNorm2d(channel_out),
        )
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.block(x)
        shortcut = self.shortcut(x)
        x = self.relu(h + shortcut)
        return x
    
    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, 1, padding=0)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))







class PretrainedResNet(nn.Module):
    def __init__(self, embedding_size, pretrain):
        super().__init__()
        self.pretrained_resnet = models.resnet50(pretrained=pretrain)
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.pretrained_resnet.conv1(x)
        x = self.pretrained_resnet.bn1(x)
        x = self.pretrained_resnet.relu(x)
        x = self.pretrained_resnet.maxpool(x)
        x = self.pretrained_resnet.layer1(x)
        x = self.pretrained_resnet.layer2(x)
        x = self.pretrained_resnet.layer3(x)
        x = self.pretrained_resnet.layer4(x)
        x = self.pretrained_resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def test(self, x1, x2):
        x1 = self.pretrained_resnet.conv1(x1)
        x1 = self.pretrained_resnet.bn1(x1)
        x1 = self.pretrained_resnet.relu(x1)
        x1 = self.pretrained_resnet.maxpool(x1)
        x1 = self.pretrained_resnet.layer1(x1)
        x1 = self.pretrained_resnet.layer2(x1)
        x1 = self.pretrained_resnet.layer3(x1)
        x1 = self.pretrained_resnet.layer4(x1)
        x1 = self.pretrained_resnet.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.fc(x1)

        x2 = self.pretrained_resnet.conv1(x2)
        x2 = self.pretrained_resnet.bn1(x2)
        x2 = self.pretrained_resnet.relu(x2)
        x2 = self.pretrained_resnet.maxpool(x2)
        x2 = self.pretrained_resnet.layer1(x2)
        x2 = self.pretrained_resnet.layer2(x2)
        x2 = self.pretrained_resnet.layer3(x2)
        x2 = self.pretrained_resnet.layer4(x2)
        x2 = self.pretrained_resnet.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc(x2)

        return x1, x2