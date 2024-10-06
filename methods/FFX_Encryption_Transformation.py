# _*_ coding : utf-8 _*_
# @Time : 2024/8/7 下午12:09
# @Author : Kmoon_Hs
# @File : FFX_Encryption_Transformation


import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

import pyffx


class FFX_Encryption_Transformation:
    def __init__(self, block_size, seed, password, image):
        self.method_label = "FFXE"
        self.block_size = block_size
        self.channels, self.width, self.height = image.shape[1:]
        self.seed = seed
        self.password = password
        self.key = self.generate_key(binary=True)  # 二进制密钥
        self.lookup, self.relookup = self.generate_lookup(self.password)  # 使用给定的密码生成加密和解密的查找表
        self.blocks_axis0 = int(self.height / self.block_size)
        self.blocks_axis1 = int(self.width / self.block_size)
        self.image = image

    # 生成用于加密和解密的查找表。通过将整数范围的像素值进行加密和解密，创建查找表。
    def generate_lookup(self, password="password"):
        password = str.encode(password)
        fpe = pyffx.Integer(password, length=3)
        f = lambda x: fpe.encrypt(x)
        g = lambda x: fpe.decrypt(x)
        f = np.vectorize(f)
        g = np.vectorize(g)
        lookup = f(np.arange(256))
        relookup = g(np.arange(1000))
        lookup = torch.from_numpy(lookup).long()
        relookup = torch.from_numpy(relookup).long()
        return lookup, relookup

    def forward(self, X, decrypt=False):
        X = self.segment(X)
        if decrypt:  # 解密
            X = (X * self.lookup.max()).long()
            X[:, :, :, self.key] = self.relookup[X[:, :, :, self.key]]
            X = X.float()
            X = X / 255.0
        else:  # 加密
            # important: without it cuda trigerring devise assertion error with index out of bound
            X = torch.clamp(X, 0, 1)
            X = (X * 255).long()
            X[:, :, :, self.key] = self.lookup[X[:, :, :, self.key]].clone()
            X = X.float()
            X = X / self.lookup.max()
        X = self.integrate(X)
        return X.contiguous()

    def segment(self, X):
        """将输入张量 X 的维度重新排序，将通道维度从第二维移到最后一维。
        假设输入张量形状为 (batch_size, channels, height, width)，
        它将变为 (batch_size, height, width, channels)。"""
        X = X.permute(0, 2, 3, 1)
        '''将图像重新形状，分割为块。每个块的大小为 block_size x block_size，
        而图像被分割成 blocks_axis0 x blocks_axis1 个块。'''
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.block_size,
            self.blocks_axis1,
            self.block_size,
            self.channels,
        )

        X = X.permute(0, 1, 3, 2, 4, 5)
        '''将这些块展平成一维向量，每个块现在被表示为一维的像素值序列。'''
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.blocks_axis1,
            self.block_size * self.block_size * self.channels,
        )
        return X

    def integrate(self, X):
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.blocks_axis1,
            self.block_size,
            self.block_size,
            self.channels,
        )
        X = X.permute(0, 1, 3, 2, 4, 5)
        X = X.reshape(
            -1,
            self.blocks_axis0 * self.block_size,
            self.blocks_axis1 * self.block_size,
            self.channels,
        )
        X = X.permute(0, 3, 1, 2)
        return X

    def generate_key(self, binary=False):
        torch.manual_seed(self.seed)
        key = torch.randperm(self.block_size * self.block_size * self.channels)
        if binary:
            key = key > len(key) / 2
        return key

    def apply(self):
        return np.uint8((np.array(self.forward(self.image)[0].permute(1, 2, 0)) * 255)).reshape(1, self.width,
                                                                                                self.height)


# 定义配置类
# class Config:
#     def __init__(self, image, block_size=4, seed=2024, password="password"):
#         self.block_size = block_size
#         self.channels, self.width, self.height = image.shape[1:]
#         self.seed = seed
#         self.password = password


if __name__ == '__main__':
    # 加载数据集
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataset = 'mnist'  # 数据集
    for i in range(3000):
        image, label = mnist[i]
        image = image.unsqueeze(0)  # 增加批次维度
        # config = Config(image)
        method = FFX_Encryption_Transformation(
            block_size=4, seed=2024, password="password",
            image=image
        )

        transfer_image = method.apply()
        transfer_image = transfer_image.reshape(28, 28)  # MNIST
        img = Image.fromarray(transfer_image.astype('uint8'))
        img.save(r'data/transfer/{}_{}_{}_{}.png'.format(dataset, i, method.method_label, label), 'JPEG')
        # img.show()
