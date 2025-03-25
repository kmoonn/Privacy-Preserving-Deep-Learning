import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.nn import CrossEntropyLoss

# 自定义损失函数
from loss import CustomLoss

from paddle.optimizer import SGD

import config

# 数据集
# from datasets.mnist import get_loader
from datasets.cifar10 import get_loader

# 模型
from models.resnet import ResNet34
from models.seresnet import SEResNet34


warnings.filterwarnings("ignore", category=Warning)  # 过滤报警信息


def train():
    train_loader = get_loader(image_dir=config.train_image_dir, batch_size=config.batch_size)

    # 初始化模型
    if config.model_name == 'resnet34':
        model = ResNet34(num_classes=config.num_classes)
    elif config.model_name == 'seresnet34':
        model = SEResNet34(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")

    # 损失函数设置
    criterion = CrossEntropyLoss()
    # 优化器
    optimizer = SGD(parameters=model.parameters(), learning_rate=config.learning_rate)

    print('start training ... ')
    start = time.time()
    model.train()

    train_loss_list = []
    train_acc_list = []

    # 训练轮数
    for epoch in range(config.epochs):
        epoch_loss = []
        correct_predictions = 0     # 正确预测
        total_samples = 0   # 样本总数

        # 训练批次
        for batch_id, (images, labels) in enumerate(train_loader()):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            # 计算 loss
            epoch_loss.append(loss.numpy().item())

            # 计算训练集上的 accuracy
            preds = paddle.argmax(outputs, axis=1)
            correct_predictions += (preds == labels).astype("float32").sum().numpy()
            total_samples += labels.shape[0]

            print(correct_predictions, total_samples)

        # 计算 loss 和 acc
        avg_loss = np.mean(epoch_loss)
        avg_acc = correct_predictions / total_samples

        train_loss_list.append(avg_loss)
        train_acc_list.append(avg_acc)

        print(f"Epoch [{epoch + 1}/{config.epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    paddle.save(model.state_dict(), config.save_path)
    print(f"Model saved to {config.save_path}")
    print(f"Training time: {time.time() - start:.2f}s")

    return train_loss_list, train_acc_list


def eval():
    test_loader = get_loader(image_dir=config.test_image_dir, batch_size=config.batch_size, shuffle=True)

    if config.model_name == 'resnet34':
        model = ResNet34(num_classes=config.num_classes)
    elif config.model_name == 'seresnet34':
        model = SEResNet34(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")

    model.set_state_dict(paddle.load(config.save_path))

    print('start evaling ... ')
    start = time.time()
    model.eval()

    test_loss_list = []
    test_acc_list = []

    criterion = CrossEntropyLoss()

    with paddle.no_grad():
        for epoch in range(config.epochs):
            epoch_loss = []
            correct_predictions = 0
            total_samples = 0

            for batch_id, (images, labels) in enumerate(test_loader()):
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 计算 loss
                epoch_loss.append(loss.numpy().item())

                # 计算测试集上的 accuracy
                preds = paddle.argmax(outputs, axis=1)
                correct_predictions += (preds == labels).astype("float32").sum().numpy()
                total_samples += labels.shape[0]

            # 计算 loss 和 acc
            avg_loss = np.mean(epoch_loss)
            avg_acc = correct_predictions / total_samples

            test_loss_list.append(avg_loss)
            test_acc_list.append(avg_acc)

            print(f"Epoch [{epoch + 1}/{config.epochs}], Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")

    print(f"Test time: {time.time() - start:.2f}s")

    return test_loss_list, test_acc_list


def show(train_loss, train_acc, test_loss, test_acc):
    # 绘制 Loss 和 Accuracy 曲线
    epochs = np.arange(config.epochs)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 绘制 Loss 曲线
    ax1.plot(epochs, train_loss, 'r-', label="Train Loss")
    ax1.plot(epochs, test_loss, 'r--', label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    # 绘制 Accuracy 曲线
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, 'b-', label="Train Accuracy")
    ax2.plot(epochs, test_acc, 'b--', label="Test Accuracy")
    ax2.set_ylabel("Accuracy", color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # 添加图例
    ax1.legend(loc="upper right")
    ax2.legend(loc="lower right")

    plt.title("Loss & Accuracy over Epochs")
    plt.show()


if __name__ == "__main__":
    # 训练
    train_loss, train_acc = train()

    # # 测试
    # test_loss, test_acc = eval()
    #
    # # 结果图
    # show(train_loss, train_acc, test_loss, test_acc)

