# _*_ coding : utf-8 _*_
# @Time : 2024/8/2 下午3:54
# @Author : Kmoon_Hs
# @File : config


# config.py

dataset = 'cifar10'

train_image_dir = f'../experiments/transfer/{dataset}/train'  # 训练集路径
test_image_dir = f'../experiments/transfer/{dataset}/test'  # 测试集路径

model_name = 'seresnet34'  # 可以选择 'resnet34'、'seresnet34'
batch_size = 64
epochs = 10
learning_rate = 0.001
num_classes = 10
save_path = f'{model_name}_{dataset}.pdparams'
