import random
import numpy as np
from PIL import Image
import argparse

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=str, default='train')
    parser.add_argument('--d', type=str, default='mnist')
    args = parser.parse_args()

    if args.d == 'mnist':
        dataset = MNIST(mode=args.m, backend="cv2")
    else:
        dataset = Cifar10(mode=args.m, backend="cv2")

    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.astype('uint8')
        method = Additive_Multiplicative_Matrix_Transformation(
            image=image,
            rise_v=random.choice([100, 200, 300, 400]),
            max_v_add=random.choice([256, 512, 1024, 2048]),
            max_v_mul=random.choice([10, 100, 1000, 10000]))

        transfer_image = method.apply()
        img = Image.fromarray(transfer_image.astype('uint8'))
        img.save(r'../data/{}/{}/{}_{}_{}_{}.png'.format(args.m, args.d, args.d, i, method.method_label, label), 'JPEG')