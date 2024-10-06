# _*_ coding : utf-8 _*_
# @Time : 2024/8/7 下午12:05
# @Author : Kmoon_Hs
# @File : SVD-based_Image_Obfuscation

import numpy as np
from PIL import Image
from scipy.optimize import fsolve

from paddle.vision.datasets import MNIST,Cifar10


class SVD_Based_Image_Obfuscation:
    def __init__(self, image, k, epsilon):
        self.method_label = "SVDO"
        if len(image.shape) == 3:
            self.width, self.height, self.channels = image.shape
        else:
            self.width, self.height = image.shape
            self.channels = 1
        self.image = image
        self.k = k
        self.epsilon = epsilon

    def svd_compression(self):
        svd_image = np.zeros_like(self.image)
        for i in range(self.channels):
            # 进行奇异值分解, 从svd函数中得到的奇异值sigma 是从大到小排列的
            if self.channels == 1:
                U, Sigma, VT = np.linalg.svd(self.image[:, :])
                svd_image = U[:, :self.k].dot(np.diag(Sigma[:self.k])).dot(VT[:self.k, :])
            else:
                U, Sigma, VT = np.linalg.svd(self.image[:, :, i])
                svd_image[:, :, i] = U[:, :self.k].dot(np.diag(Sigma[:self.k])).dot(VT[:self.k, :])
            # img_remake = (U @ np.diag(Sigma) @ VT)
            # # if (img_remake == self.image[:, :, i]).any():
            # #     "we can't remake the original matrix using U,D,VT"

        return svd_image

    def sample_radial(self):
        """
        根据概率密度函数Dϵ,k(x0)(x)对径向坐标进行采样
        """

        def objective_func(r):
            return np.exp(-self.epsilon * r) - np.random.uniform(0, 1)

        r0 = np.linalg.norm(self.image)
        r = fsolve(objective_func, r0)[0]

        return r

    def sample_angular(self):
        """
        在单位(k-1)-球面上均匀采样一个点
        """
        angles = np.random.uniform(0, 2 * np.pi, self.k - 1)
        a = np.concatenate([np.cos(angles), [np.sin(angles[-1])]])
        for i in range(len(a) - 2, -1, -1):
            a[i] *= np.sin(angles[i])
        return a

    def private_sampling(self):
        """
        实现随机采样机制
        """
        # img.shape = (28,28,1)
        sampling_image = np.zeros_like(self.image)
        for i in range(self.channels):
            # 进行奇异值分解, 从svd函数中得到的奇异值sigma 是从大到小排列的
            U, Sigma, VT = np.linalg.svd(self.image[:, :, i])
            img_remake = (U @ np.diag(Sigma) @ VT)
            if (img_remake == self.image[:, :, i]).any():
                "we can't remake the original matrix using U,D,VT"

            r = self.sample_radial(Sigma)
            a = self.sample_angular()
            x = r * a

            sampling_image[:, :, i] = U[:, :k].dot(np.diag(x)).dot(VT[:k, :])

        return np.clip(sampling_image, 0, 255)

    def apply(self):
        return self.svd_compression()


class test:
    def __init__(self, image):
        self.image = image

    def test(self):
        method = SVD_Based_Image_Obfuscation(
            image=self.image,
            k=5,
            epsilon=0.5, )
        transfer_image = method.apply()
        return transfer_image


if __name__ == '__main__':
    mnist = MNIST(mode='test', backend="cv2")
    # cifar10 = Cifar10(mode='test', backend="cv2")
    dataset = 'mnist'  # 数据集
    for i in range(3000):
        image, label = mnist[i]
        # print(image)
        method = SVD_Based_Image_Obfuscation(
            image=image,
            k=4,
            epsilon=0.5
        )

        transfer_image = method.apply()
        img = Image.fromarray(transfer_image.astype('uint8'))
        img.save(r'data/transfer/{}_{}_{}_{}.png'.format(dataset, i, method.method_label, label), 'JPEG')
        # img.show()
