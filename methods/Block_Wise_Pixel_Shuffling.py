# _*_ coding : utf-8 _*_
# @Time : 2024/8/7 下午12:07
# @Author : Kmoon_Hs
# @File : Block-Wise_Pixel_Shuffling
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image


class Block_Wise_Pixel_Shuffling:
    def __init__(self, block_size, seed, image):
        self.method_label = "BWPS"
        self.block_size = block_size
        self.seed = seed
        self.channels, self.width, self.height = image.shape[1:]
        assert (
                self.height % self.block_size == 0 | self.width % self.block_size == 0
        ), "Image not divisible by block_size"
        self.blocks_axis0 = int(self.height / self.block_size)
        self.blocks_axis1 = int(self.width / self.block_size)
        self.key = self.generate_key(self.seed, binary=False)  # 排列密钥
        self.image = image

    def forward(self, X, decrypt=False):
        X = self.segment(X)
        if decrypt:  # 还原
            key = torch.argsort(self.key)
            X = X[:, :, :, key]
        else:  # 打乱
            X = X[:, :, :, self.key]
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

    def generate_key(self, seed, binary=False):
        torch.manual_seed(seed)
        key = torch.randperm(self.block_size * self.block_size * self.channels)
        if binary:
            key = key > len(key) / 2
        return key

    def apply(self):
        return self.forward(self.image)


# 定义配置类
# class Config:
#     def __init__(self, image, block_size=4, seed=2024):
#         self.block_size = block_size
#         self.channels, self.width, self.height = image.shape[1:]
#         self.seed = seed


if __name__ == '__main__':
    # 加载CIFAR-10数据集
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    dataset = 'cifar10'  # 数据集
    for i in range(400):
        image, label = cifar10[i]
        image = image.unsqueeze(0)  # 增加批次维度
        # config = Config(image)
        method = Block_Wise_Pixel_Shuffling(
            block_size=4,
            seed=2024,
            image=image
        )

        transfer_image = method.apply()
        transfer_image = np.array(transfer_image[0].permute(1, 2, 0)) * 255
        transfer_image = np.uint8(transfer_image)
        # transfer_image = transfer_image.reshape(28, 28)  # MNIST
        img = Image.fromarray(transfer_image)
        img.save(r'data/transfer/{}_{}_{}_{}.png'.format(dataset, i, method.method_label, label), 'JPEG')
        # img.show()
