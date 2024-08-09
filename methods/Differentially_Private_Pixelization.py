# _*_ coding : utf-8 _*_
# @Time : 2024/8/5 下午4:35
# @Author : Kmoon_Hs
# @File : Differentially_Private_Pixelization

import numpy as np
from PIL import Image
from scipy.stats import laplace

from paddle.vision.datasets import MNIST, Cifar10


class Differentially_Private_Pixelization:

    def __init__(self, image, block_size, m, epsilon):
        '''
        grid cell length b
        Privacy parameter epsilon
        Number of different pixels allowed m
        '''
        self.method_label = 'DPP'

        self.image = image
        self.block_size = block_size
        self.epsilon = epsilon
        self.global_sensitivity = 255 * m / pow(block_size, 2)

        if len(image.shape) == 3:
            self.height, self.width, self.channels = image.shape
        else:
            self.height, self.width = image.shape
            self.channels = 1

        self.block_num = int((self.width / block_size) * (self.height / block_size))

    # 分块
    def blocking_pixel_avg(self):
        blocks = []

        height = [i * self.block_size for i in range(int(self.height / self.block_size))]
        width = [i * self.block_size for i in range(int(self.width / self.block_size))]

        for i in height:
            for j in width:
                block = self.image[i:i + self.block_size, j:j + self.block_size]
                pixel_avg_value = np.mean(block)
                blocks.append(pixel_avg_value * np.ones_like(block))
        return blocks

    def add_laplace_noise(self, block_list):
        block_add_noise = []

        b = self.global_sensitivity / self.epsilon

        for i in range(self.block_num):
            noise = laplace.rvs(loc=0, scale=b, size=block_list[i].size).reshape(block_list[i].shape)
            block_add_noise.append(np.clip(block_list[i] + noise, 0, 255).astype(np.uint8))

        return block_add_noise

    def block2M(self, block_list):

        Row = []
        Column = []
        blocks = block_list

        for i in range(self.block_num):
            if (i + 1) % (self.width / self.block_size) != 0:
                Row.append(blocks[i])
            else:
                Row.append(blocks[i])
                Column.append(np.hstack(Row))
                Row = []

        return np.vstack(Column)

    def pixelization(self):
        return self.block2M(self.add_laplace_noise(self.blocking_pixel_avg()))

    def apply(self):
        return self.pixelization()


if __name__ == '__main__':
    # mnist = MNIST(mode='test', backend="cv2")
    cifar10 = Cifar10(mode='test', backend="cv2")
    dataset = 'cifar10'
    for i in range(400):
        image, label = cifar10[i]
        image = image.astype('uint8')
        method = Differentially_Private_Pixelization(
            image=image,
            block_size=4,
            m=4,
            epsilon=0.5
        )

        transfer_image = method.apply()
        img = Image.fromarray(transfer_image.astype('uint8'))
        img.save(r'data/transfer/{}_{}_{}_{}.png'.format(dataset, i, method.method_label, label), 'JPEG')
        # img.show()
