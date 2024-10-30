# _*_ coding : utf-8 _*_
# @Time : 2024/8/4 下午4:56
# @Author : Kmoon_Hs
# @File : Pixel_Negative-Positive_Transformation

import random
import numpy as np
from PIL import Image

from paddle.vision.datasets import MNIST, Cifar10


class Pixel_Negative_Positive_Transformation:
    def __init__(self, image, Kc, Ks):
        self.method_label = "PNPT"
        self.image = image.astype('uint8')
        if len(image.shape) == 3:
            self.width, self.height, self.channels = image.shape
            self.Kc = Kc
            self.Ks = Ks
        else:
            self.width, self.height = image.shape
            self.channels = 1
            self.Kc = Kc[0]

    def NP(self):
        if self.channels == 3:
            Kr, Kg, Kb = self.Kc
            np.random.seed(Kr)
            Ir = np.random.choice([0, 255], size=(self.width, self.height), p=[0.5, 0.5])
            np.random.seed(Kg)
            Ig = np.random.choice([0, 255], size=(self.width, self.height), p=[0.5, 0.5])
            np.random.seed(Kb)
            Ib = np.random.choice([0, 255], size=(self.width, self.height), p=[0.5, 0.5])

            r = self.image[:, :, 0]
            g = self.image[:, :, 1]
            b = self.image[:, :, 2]
            return np.dstack((r ^ Ir, g ^ Ig, b ^ Ib))
        else:
            Kg = self.Kc
            np.random.seed(Kg)
            I = np.random.choice([0, 255], size=(self.width, self.height), p=[0.5, 0.5])
            return I ^ self.image

    def Shuffle(self):
        image_np = self.NP()
        r = image_np[:, :, 0]
        g = image_np[:, :, 1]
        b = image_np[:, :, 2]
        if self.Ks == 0:
            return np.dstack((r, g, b))
        elif self.Ks == 1:
            return np.dstack((r, b, g))
        elif self.Ks == 2:
            return np.dstack((g, r, b))
        elif self.Ks == 3:
            return np.dstack((g, b, r))
        elif self.Ks == 4:
            return np.dstack((b, r, g))
        elif self.Ks == 5:
            return np.dstack((b, g, r))

    def apply(self):
        if self.channels == 3:
            choice = random.choice([self.Shuffle(), self.NP()])
            return choice
        else:
            return self.NP()


if __name__ == '__main__':
    # mnist = MNIST(mode='test', backend="cv2")
    cifar10 = Cifar10(mode='test', backend="cv2")
    dataset = 'cifar10'
    for i in range(3000):
        image, label = cifar10[i]
        image = image.astype('uint8')
        method = Pixel_Negative_Positive_Transformation(
            image=image,
            Kc=(2024, 2025, 2026),
            Ks=random.randint(0, 5)
        )

        transfer_image = method.apply()
        img = Image.fromarray(transfer_image.astype('uint8'))
        img.save(r'data/transfer/{}_{}_{}_{}.png'.format(dataset, i, method.method_label, label), 'JPEG')
        # img.show()
