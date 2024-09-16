# _*_ coding : utf-8 _*_
# @Time : 2024/8/2 下午3:53
# @Author : Kmoon_Hs
# @File : main
import time
import warnings

warnings.filterwarnings("ignore", category=Warning)  # 过滤报警信息

import paddle
from paddle.metric import Accuracy
from paddle.nn import CrossEntropyLoss
from paddle.optimizer import Adam

import config
# 数据集
from datasets.mnist import get_loader
# from datasets.cifar10 import get_loader
# 模型
from models.resnet import ResNet34
from models.seresnet import SEResNet34


def train():
    # 数据加载
    train_loader = get_loader(image_dir=config.train_image_dir, batch_size=config.batch_size)

    # 初始化模型
    if config.model_name == 'resnet34':
        model = ResNet34(num_classes=config.num_classes)
    elif config.model_name == 'seresnet34':
        model = SEResNet34(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")

    # 定义损失函数和优化器
    criterion = CrossEntropyLoss()
    optimizer = Adam(parameters=model.parameters(), learning_rate=config.learning_rate)

    # 训练模型
    print('start training ... ')
    start = time.time()
    model.train()
    loss_list = []
    for epoch in range(config.epochs):
        for batch_id, (images, labels) in enumerate(train_loader()):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            if batch_id % 100 == 0:
                loss_list.append(loss.numpy())
                print(f"Epoch [{epoch + 1}/{config.epochs}], Batch [{batch_id}], Loss: {loss.numpy()}")

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), config.save_path)
    print(f"Model saved to {config.save_path}")
    end = time.time()
    print(f"Training time: {end - start}")
    return loss_list


def eval():
    # 加载测试集数据
    test_loader = get_loader(image_dir=config.test_image_dir, batch_size=config.batch_size, shuffle=True)

    # 初始化模型
    if config.model_name == 'resnet34':
        model = ResNet34(num_classes=config.num_classes)
    elif config.model_name == 'seresnet34':
        model = SEResNet34(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")

    # 加载模型参数
    model.set_state_dict(paddle.load(config.save_path))

    print('start evaling ... ')
    start = time.time()

    model.eval()  # 设置模型为评估模式
    accuracy = Accuracy()

    with paddle.no_grad():  # 评估过程中不需要梯度
        for batch_id, (images, labels) in enumerate(test_loader()):
            outputs = model(images)
            correct = accuracy.compute(outputs, labels)
            accuracy.update(correct)
            preds = paddle.argmax(outputs, axis=1)  # 取每个样本的最大概率对应的类别

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                if i % 100 == 0:
                    print(
                        f"Sample {batch_id * len(labels) + i + 1}: Predicted = {pred}, Actual = {label}, {'Correct' if pred == label else 'Incorrect'}")

    acc = accuracy.accumulate()  # 计算准确率
    print(f"Overall Test Accuracy: {acc}")
    end = time.time()
    print(f"Test time: {end - start}")


if __name__ == "__main__":
    # 模型训练
    loss_list = train()

    # 模型测试
    eval()
