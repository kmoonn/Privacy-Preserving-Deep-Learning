# config.py

# 训练参数配置
TRAINING_CONFIG = {
    'batch_size': 64,
    'num_epochs': 10,
    'learning_rate': 1e-3,
    'optimizer': 'Adam',  # 可选: 'Adam', 'SGD'
    'loss_function': 'CrossEntropyLoss',
}

dataset = 'mnist'

# 数据集路径配置
DATA_CONFIG = {
    'train_data_dir': f'data/{dataset}/train',
    'test_data_dir': f'data/{dataset}/test',
    'dataset_type': f'{dataset}',
}

model = 'ResNet34'

# 模型配置
MODEL_CONFIG = {
    'model_name': model,  # 可选: 'ResNet34', 'SEResNet34'
    'num_classes': 10,
    'reduction': 16,  # SE模块的压缩率
}
