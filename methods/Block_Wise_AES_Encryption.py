# _*_ coding : utf-8 _*_
# @Time : 2024/8/6 下午3:40
# @Author : Kmoon_Hs
# @File : Block-Wise_AES_Encryption

import math
import os
import random

import numpy as np
from Crypto.Cipher import AES as aes
from PIL import Image

from paddle.vision.datasets import MNIST, Cifar10


class Block_Wise_AES_Encryption:

    def __init__(self, image, block_size=(4, 4), One_cipher=True, Shuffle=False):

        self.method_label = "BWAE"
        if len(image.shape) == 3:
            self.width, self.height, self.channels = image.shape
        else:
            self.width, self.height = image.shape
            self.channels = 1
        self.image_size = image.shape
        self.image = image

        self.block_size = block_size
        # 块数量
        self.block_num = int((self.width / block_size[0]) * (self.height / block_size[1]))

        block_bytes = block_size[0] * block_size[1]  # 块的字节大小 16
        self.scale = [1, 1]  # 放大情况
        self.shuffle = Shuffle
        self.one_cipher = One_cipher
        if block_bytes < 16:
            # scale it 放大 块字节小于AES加密字节16
            if self.block_size[0] < self.block_size[1]:
                less = 0
            else:
                less = 1
            if block_bytes == 2:
                self.scale[less] = 4
                self.scale[(less + 1) % 2] = 2
            elif block_bytes == 4:
                if self.block_size == (2, 2):
                    self.scale = [2, 2]
                else:
                    self.scale[less] = 4
            elif block_bytes == 8:
                self.scale[less] = 2
            self.block_size = (4, 4)
        self.update_params()  # due to updated parameters 更新参数

        if not One_cipher:
            self.ciphers = [aes.new(os.urandom(16), aes.MODE_ECB) for i in range(self.block_num)]
        else:
            self.ciphers = aes.new(os.urandom(16), aes.MODE_ECB)

    # 准备参数
    def update_params(self):

        block_size = self.block_size  # 更新为4x4
        # 填充
        if self.width % block_size[0] != 0:
            p0 = block_size[0] - self.width % block_size[0]
            self.p0_left = int(p0 / 2)
            self.p0_right = p0 - self.p0_left
        else:
            p0 = 0
            self.p0_left = 0
            self.p0_right = 0

        if self.height % block_size[1] != 0:
            p1 = block_size[1] - self.height % block_size[1]
            self.p1_left = int(p1 / 2)
            self.p1_right = p1 - self.p1_left
        else:
            p1 = 0
            self.p1_left = 0
            self.p1_right = 0
        if p0 == 0 and p1 == 0:  # 无需填充
            self.pad = False
        else:
            self.pad = True

        self.block_num = math.ceil(self.width / block_size[0]) * math.ceil(self.height / block_size[1])
        # 向上取整

    def padding(self, img):

        if self.channels == 3:  # RGB图像

            assert ((img.shape[0], img.shape[1], img.shape[2]) == self.image_size)

        else:  # Grey图像

            assert ((img.shape[0], img.shape[1]) == self.image_size)

        if not self.pad:  # 无需填充 直接copy返回
            return img.copy()

        if self.channels == 3:
            img1 = np.zeros((img.shape[0] + self.p0_left + self.p0_right, img.shape[1] + self.p1_left + self.p1_right,
                             img.shape[2]))
            for c in range(img.shape[2]):
                img1[:, :, c] = np.pad(img[:, :, c], ((self.p0_left, self.p0_right), (self.p1_left, self.p1_right)))
        else:
            img1 = np.zeros((img.shape[0] + self.p0_left + self.p0_right, img.shape[1] + self.p1_left + self.p1_right))
            img1[:, :] = np.pad(img[:, :], ((self.p0_left, self.p0_right), (self.p1_left, self.p1_right)))

        return img1  # 返回填充后的图像

    def M2vector(self, block):
        vec = block.reshape((1, block.shape[0] * block.shape[1]))
        return vec

    def vector2M(self, vector):
        M = vector.reshape((self.block_size[0], self.block_size[1]))
        return M

    # 分块
    def M2block(self, array):
        h, r = array.shape[0:2]
        blocks = []
        hight = [i * self.block_size[0] for i in range(int(h / self.block_size[0]))]
        width = [i * self.block_size[1] for i in range(int(r / self.block_size[1]))]

        for i in hight:
            for j in width:
                blocks.append(array[i:i + self.block_size[0], j:j + self.block_size[1]])

        return blocks

    def scaleup(self, img):

        ''' img: w*h, each pixel duplicate to the corresponding 4x4 block'''

        assert (self.scale != [1, 1] and img.shape[0:2] == self.image_size)

        img1 = np.ones(img.shape)

        if len(img.shape) == 3:

            for c in range(img.shape[2]):

                for i in range(img.shape[0]):

                    for j in range(img.shape[1]):
                        img1[i * self.scale[0]:(i + 1) * self.scale[0], j * self.scale[1]:(j + 1) * self.scale[1], c] *= \
                            img[i, j, c]

        else:

            for i in range(img.shape[0]):

                for j in range(img.shape[1]):
                    img1[i * self.scale[0]:(i + 1) * self.scale[0], j * self.scale[1]:(j + 1) * self.scale[1]] *= img[
                        i, j]

        return img1.astype(np.byte)

    # 随机合并块
    def block2M(self, block_list, seed=1):
        Row = []
        Column = []
        blocks = block_list

        if self.shuffle:
            random.Random(seed).shuffle(block_list)

        for i in range(self.block_num):

            if (i + 1) % (self.height / self.block_size[1]) != 0:
                Row.append(blocks[i])
            else:
                Row.append(blocks[i])
                Column.append(np.hstack(Row))
                Row = []

        return np.vstack(Column)

    # 块加密
    def block_enc(self, block, cipher):

        block1 = self.M2vector(block)

        assert (block1.shape[1] % 16 == 0)

        for i in range(int(block1.shape[1] / 16)):
            bytes = cipher.encrypt(block1[:, i * 16:(i + 1) * 16].tobytes())

            block1[:, i * 16:(i + 1) * 16] = np.frombuffer(bytes, dtype=np.byte)

        return self.vector2M(block1)

    # 加密
    def apply(self):

        if self.scale != [1, 1]:
            img1 = self.scaleup(self.image)
        else:
            img1 = self.image

        img2 = self.padding(img1).astype(np.byte)

        if len(img2.shape) == 3:

            for c in range(img2.shape[2]):  # channels

                blocks = self.M2block(img2[:, :, c])

                if not self.one_cipher:

                    blocks_e = [self.block_enc(b, e) for b, e in zip(blocks, self.ciphers)]

                else:

                    blocks_e = [self.block_enc(b, self.ciphers) for b in blocks]

                img2[:, :, c] = self.block2M(blocks_e)

        else:

            blocks = self.M2block(img2[:, :])

            if not self.one_cipher:

                blocks_e = [self.block_enc(b, e) for b, e in zip(blocks, self.ciphers)]

            else:

                blocks_e = [self.block_enc(b, self.ciphers) for b in blocks]

            img2[:, :] = self.block2M(blocks_e)

        return img2


if __name__ == '__main__':
    mnist = MNIST(mode='test',backend="cv2" )
    # cifar10 = Cifar10(mode='train', backend="cv2")
    dataset = 'mnist'
    for i in range(3000):
        image, label = mnist[i]
        method = Block_Wise_AES_Encryption(
            image=image,
            block_size=(4, 4),
            One_cipher=True,
            Shuffle=True
        )

        transfer_image = method.apply()
        img = Image.fromarray(transfer_image.astype('uint8'))
        img.save(r'data/transfer/{}_{}_{}_{}.png'.format(dataset, i, method.method_label, label), 'JPEG')
        # img.show()
