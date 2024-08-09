# _*_ coding : utf-8 _*_
# @Time : 2024/8/2 下午3:54
# @Author : Kmoon_Hs
# @File : config


# config.py

dataset = 'CIFAR10'  # or 'CIFAR10' or 'LFW'

train_image_dir = f'../experiments/transfer/{dataset}/train'  # 训练集路径
test_image_dir = f'../experiments/transfer/{dataset}/test'  # 测试集路径

model_name = 'resnet34'  # 可以选择 'resnet34', 'vgg16', 'vgg19'
batch_size = 64
epochs = 10
learning_rate = 0.001
num_classes = 9
save_path = 'resnet34_cifar10.pdparams' # or 'resnet34_mnist.pdparams'
