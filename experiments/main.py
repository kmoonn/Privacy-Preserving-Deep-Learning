# _*_ coding : utf-8 _*_
# @Time : 2024/8/2 下午3:53
# @Author : Kmoon_Hs
# @File : main

import paddle
# 数据集
from datasets.mnist import MNIST
from datasets.cifar10 import CIFAR10
# 模型
# from models.mnist_model import MNISTModel
# from models.cifar10_model import CIFAR10Model
# 变换
from methods import Additive_Multiplicative_Matrix_Transformation

import config

def main():
    # 初始化数据集、模型、优化器等
    if config.dataset == 'MNIST':
        dataset = MNIST(method=Additive_Multiplicative_Matrix_Transformation())  # 选择变换
        # model = MNISTModel()
    # elif config.dataset == 'CIFAR10':
    #     dataset = CIFAR10(method=RotateTransform(angle=30))
    #     model = CIFAR10Model()

    # 数据加载
    train_loader = dataset.get_loader(batch_size=config.batch_size)

    # 训练和评估模型
    model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()), loss=paddle.nn.CrossEntropyLoss())
    model.fit(train_loader, epochs=config.epochs)

if __name__ == "__main__":
    main()