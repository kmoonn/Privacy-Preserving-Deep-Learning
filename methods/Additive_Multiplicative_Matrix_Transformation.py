# _*_ coding : utf-8 _*_
# @Time : 2024/8/2 下午3:57
# @Author : Kmoon_Hs
# @File : Additive_Multiplicative_Matrix_Transformation
import random
import numpy as np
from PIL import Image

from paddle.vision.datasets import MNIST, Cifar10


class Additive_Multiplicative_Matrix_Transformation:
    def __init__(self, image, rise_v=100, max_v_add=256, max_v_mul=10):
        self.method_label = "AMMT"
        if len(image.shape) == 3:
            self.width, self.height, self.channels = image.shape
        else:
            self.width, self.height = image.shape
        self.image = image

        self.MAX_V_ADD, self.MAX_V_MUL = max_v_add, max_v_mul
        self.RISE_V = rise_v
        self.R_add = np.random.randint(1, self.MAX_V_ADD, size=self.image.shape)
        self.R_mul = np.random.randint(1, self.MAX_V_MUL, size=self.image.shape)

    def MAT(self):
        return self.image + self.R_add

    def MMT(self):
        if len(self.image.shape) == 2:
            return self.image * self.R_mul

        r = self.image[:, :, 0] * self.R_mul[:, :, 0]
        g = self.image[:, :, 1] * self.R_mul[:, :, 1]
        b = self.image[:, :, 2] * self.R_mul[:, :, 2]

        return np.dstack((r, g, b))

    def Rise_MMT(self):
        self.image = self.image + self.RISE_V
        if len(self.image.shape) == 2:
            return self.image * self.R_mul

        r = self.image[:, :, 0] * self.R_mul[:, :, 0]
        g = self.image[:, :, 1] * self.R_mul[:, :, 1]
        b = self.image[:, :, 2] * self.R_mul[:, :, 2]

        return np.dstack((r, g, b))

    def MAT_MMT(self):
        self.image = self.image + self.R_add

        if len(self.image.shape) == 2:
            return self.image * self.R_mul

        r = (self.image[:, :, 0]) * self.R_mul[:, :, 0]
        g = (self.image[:, :, 1]) * self.R_mul[:, :, 1]
        b = (self.image[:, :, 2]) * self.R_mul[:, :, 2]

        return np.dstack((r, g, b))

    def MMT_MAT(self):
        if len(self.image.shape) == 2:
            return self.image * self.R_mul + self.R_add

        r = self.image[:, :, 0] * self.R_mul[:, :, 0]
        g = self.image[:, :, 1] * self.R_mul[:, :, 1]
        b = self.image[:, :, 2] * self.R_mul[:, :, 2]

        return np.dstack((r, g, b)) + self.R_add

    def apply(self):
        choice = random.choice([self.MAT, self.MMT, self.Rise_MMT, self.MAT_MMT, self.MMT_MAT])
        return choice()


class test:
    def __init__(self, image):
        self.image = image

    def test(self):
        method = Additive_Multiplicative_Matrix_Transformation(
            image=self.image,
            rise_v=random.choice([100, 200, 300, 400]),
            max_v_add=random.choice([256, 512, 1024, 2048]),
            max_v_mul=random.choice([10, 100, 1000, 10000]))

        transfer_image = method.apply()
        return transfer_image


if __name__ == '__main__':
    # mnist = MNIST(mode='test',backend="cv2" )
    cifar10 = Cifar10(mode='test', backend="cv2")
    dataset = 'cifar10'  # 数据集
    for i in range(3000):
        image, label = cifar10[i]
        image = image.astype('uint8')
        method = Additive_Multiplicative_Matrix_Transformation(
            image=image,
            rise_v=random.choice([100, 200, 300, 400]),
            max_v_add=random.choice([256, 512, 1024, 2048]),
            max_v_mul=random.choice([10, 100, 1000, 10000]))

        transfer_image = method.apply()
        img = Image.fromarray(transfer_image.astype('uint8'))
        img.save(r'data/transfer/{}_{}_{}_{}.png'.format(dataset, i, 'AMMT', label), 'JPEG')
        # img.show()
