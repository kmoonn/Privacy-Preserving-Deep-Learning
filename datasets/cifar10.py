# _*_ coding : utf-8 _*_
# @Time : 2024/8/2 下午4:01
# @Author : Kmoon_Hs
# @File : cifat10


import os

import paddle
from paddle.vision import Compose, Normalize, Transpose

from experiments.utils import load_image_to_array


class CIFAR10(paddle.io.Dataset):
    def __init__(self, image_dir=None, transform=Compose([Normalize(mean=[127.5], std=[127.5])])):
        super(CIFAR10, self).__init__()
        self.transform = transform
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if
                            img.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_image_to_array(image_path)

        if self.transform:
            image = self.transform(image)

        image = Transpose([2, 0, 1])(image)  # 调整通道维度

        label = self._extract_label_from_filename(image_path)
        return image, label

    def _extract_label_from_filename(self, image_path):
        filename = os.path.basename(image_path)
        label = filename.split('_')[2]  # 提取标签部分
        label_mapping = {
            'RAW': 0,
            'AMMT': 1,
            'BWAE': 2,
            'BWPS': 3,
            'DPP': 4,
            'FFXE': 5,
            'PNPT': 6,
            'RMT': 7,
            'SIT': 8,
            'SVDO': 9,
        }  # 定义标签映射
        return label_mapping.get(label, -1)


def get_loader(image_dir, batch_size=64, shuffle=True):
    dataset = CIFAR10(image_dir=image_dir)
    return paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    train = CIFAR10(image_dir='../experiments/transformed/cifar10/train')
    test = CIFAR10(image_dir='../experiments/transformed/cifar10/test')
    print(f'Train size: {len(train)}')
    print(f'Test size: {len(test)}')
    img, label = train[0]
    print(f'Image shape: {img.shape}')
    print(f'Label: {label}')
