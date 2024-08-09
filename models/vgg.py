# _*_ coding : utf-8 _*_
# @Time : 2024/8/7 下午12:16
# @Author : Kmoon_Hs
# @File : vgg

import paddle
import paddle.nn as nn
import paddle.vision.models as models


class VGG(nn.Layer):
    def __init__(self, version='vgg16', num_classes=10):
        super(VGG, self).__init__()

        if version == 'vgg16':
            self.model = models.vgg16(pretrained=False, num_classes=num_classes)
        elif version == 'vgg19':
            self.model = models.vgg19(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported VGG version: {version}")

    def forward(self, x):
        return self.model(x)
