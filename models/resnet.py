# _*_ coding : utf-8 _*_
# @Time : 2024/8/7 下午12:15
# @Author : Kmoon_Hs
# @File : resnet

import paddle
import paddle.nn as nn
import paddle.vision.models as models

class ResNet34(nn.Layer):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        # 使用PaddlePaddle自带的ResNet34模型
        self.model = models.resnet34(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
