# _*_ coding : utf-8 _*_
# @Time : 2024/9/11 下午2:23
# @Author : Kmoon_Hs
# @File : se-resnet


import paddle
import paddle.nn as nn


# 定义Squeeze-and-Excitation模块
class SEBlock(nn.Layer):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2D(1)  # 全局平均池化
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze：全局平均池化
        b, c, _, _ = x.shape
        y = self.global_avgpool(x).reshape([b, c])

        # Excitation：全连接层和激活
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).reshape([b, c, 1, 1])

        # 通道加权
        return x * y


# 定义残差块，并集成SENet模块
class SEBasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)
        self.downsample = downsample
        self.se = SEBlock(out_channels, reduction)  # 加入SENet模块

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 通过SE模块进行特征重加权
        out = self.se(out)

        out += identity
        out = self.relu(out)

        return out


class SEResNet34(nn.Layer):
    def __init__(self, num_classes=10, reduction=16):
        super(SEResNet34, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 每一层包含的残差块个数，并使用SENet模块
        self.layer1 = self._make_layer(64, 3, reduction=reduction)
        self.layer2 = self._make_layer(128, 4, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, 6, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, 3, stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * SEBasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * SEBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.in_channels, out_channels * SEBasicBlock.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels * SEBasicBlock.expansion),
            )

        layers = []
        layers.append(SEBasicBlock(self.in_channels, out_channels, stride, downsample, reduction))
        self.in_channels = out_channels * SEBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(self.in_channels, out_channels, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x