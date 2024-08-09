# _*_ coding : utf-8 _*_
# @Time : 2024/8/6 下午8:07
# @Author : Kmoon_Hs
# @File : Scalable_Image_Transformation

import random

import numpy as np
from PIL import Image

from paddle.vision.datasets import MNIST, Cifar10



class Scalable_Image_Transformation:
    def __init__(self, image, block_size, key1, key2, key3, key4, key5):

        self.method_label = "SIT"
        if len(image.shape) == 3:
            self.width, self.height, self.channels = image.shape
        else:
            self.width, self.height = image.shape
            self.channels = 1

        self.image_size = image.shape
        self.image = image.astype('uint8')
        self.key1 = key1
        self.key2 = key2
        self.key3 = key3
        self.key4 = key4
        self.key5 = key5

        self.block_size = block_size
        self.block_num = int((self.width / block_size) * (self.height / block_size))

    def ImagePartition(self, array):
        blocks = []
        width = [i * self.block_size for i in range(int(self.width / self.block_size))]
        hight = [i * self.block_size for i in range(int(self.height / self.block_size))]

        for i in hight:
            for j in width:
                blocks.append(array[i:i + self.block_size, j:j + self.block_size])

        return blocks

    def BlockRotation(self, array):
        rotated_blocks = []
        block_list = self.ImagePartition(array)

        for i in range(self.block_num):
            rotated_block = np.rot90(block_list[i], k=self.key1)
            rotated_blocks.append(rotated_block)

        return rotated_blocks

    def PixelAdjustment(self, array):
        adjusted_blocks = []
        block_list = self.BlockRotation(array)
        np.random.seed(self.key2)

        for i in range(self.block_num):
            adjusted_block = block_list[i] ^ np.random.choice([0, 255], size=(self.block_size, self.block_size),
                                                              p=[0.5, 0.5])
            adjusted_blocks.append(adjusted_block)

        return adjusted_blocks

    def BlockFlipping(self, array):
        flipped_blocks = []
        block_list = self.PixelAdjustment(array)

        for i in range(self.block_num):
            flipped_block = []
            if self.key3 == 0:
                flipped_block = block_list[i]
            elif self.key3 == 1:
                flipped_block = np.fliplr(block_list[i])  # 水平翻转
            elif self.key3 == 2:
                flipped_block = np.flipud(block_list[i])

            flipped_blocks.append(flipped_block)
        return flipped_blocks

    def ColorShuffling(self, r, g, b):
        color_shuffled_array = []

        if self.key4 == 0:
            color_shuffled_array = np.dstack((r, g, b))
        elif self.key4 == 1:
            color_shuffled_array = np.dstack((r, b, g))
        elif self.key4 == 2:
            color_shuffled_array = np.dstack((g, r, b))
        elif self.key4 == 3:
            color_shuffled_array = np.dstack((g, b, r))
        elif self.key4 == 4:
            color_shuffled_array = np.dstack((b, r, g))
        elif self.key4 == 5:
            color_shuffled_array = np.dstack((b, g, r))
        return color_shuffled_array

    def BlockShuffling(self, array):
        Row = []
        Column = []

        block_list = self.BlockFlipping(array)

        random.Random(self.key5).shuffle(block_list)

        for i in range(self.block_num):

            if (i + 1) % (self.image_size[1] / self.block_size) != 0:
                Row.append(block_list[i])
            else:
                Row.append(block_list[i])
                Column.append(np.hstack(Row))
                Row = []
        return np.vstack(Column)

    def MergeRGB(self, image_array):
        r = image_array[:, :, 0]
        g = image_array[:, :, 1]
        b = image_array[:, :, 2]

        encrypted_r = self.BlockShuffling(r)
        encrypted_g = self.BlockShuffling(g)
        encrypted_b = self.BlockShuffling(b)

        color_shuffled_array = self.ColorShuffling(encrypted_r, encrypted_g, encrypted_b)

        return color_shuffled_array

    def apply(self):
        if self.channels == 1:
            return self.BlockShuffling(self.image)
        else:
            return self.MergeRGB(self.image)


if __name__ == '__main__':
    # mnist = MNIST(mode='test', backend="cv2")
    cifar10 = Cifar10(mode='test', backend="cv2")
    dataset = 'cifar10'
    for i in range(400):
        image, label = cifar10[i]
        image = image.astype(np.uint8)
        method = Scalable_Image_Transformation(
            image=image,
            block_size=4,
            key1=random.choice([1, 2, 3, 4]),
            key2=2024,
            key3=random.choice([0, 1, 2]),
            key4=random.choice([0, 1, 2, 3, 4, 5]),
            key5=2024
        )

        transfer_image = method.apply()
        img = Image.fromarray(transfer_image.astype('uint8'))
        img.save(r'data/transfer/{}_{}_{}_{}.png'.format(dataset, i, method.method_label, label), 'JPEG')
        # img.show()
