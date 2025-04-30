import paddle
from paddle.nn import CrossEntropyLoss
from paddle.optimizer import Adam
import matplotlib.pyplot as plt
from experiments.data.dataLoader import get_loader
from experiments.models.resnet import ResNet34
from experiments.models.seresnet import SEResNet34
from configs.config import TRAINING_CONFIG, DATA_CONFIG, MODEL_CONFIG


if MODEL_CONFIG['model_name'] == 'ResNet34':
    model = ResNet34(num_classes=10)
else:
    model = SEResNet34(num_classes=10)

# 定义损失函数和优化器
criterion = CrossEntropyLoss()
optimizer = Adam(parameters=model.parameters())

# 设置数据加载器
train_loader = get_loader(image_dir='data/mnist/train', dataset_type='mnist', batch_size=64, shuffle=True)
test_loader = get_loader(image_dir='data/mnist/test', dataset_type='mnist', batch_size=64, shuffle=False)

# 训练循环
num_epochs = 10

# 用于记录每个epoch的损失和准确率
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_id, (images, labels) in enumerate(train_loader()):
        optimizer.clear_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.numpy()

        _, predicted = paddle.argmax(outputs, axis=1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.shape[0]

        if batch_id % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_id}, Loss: {loss.numpy()}")

    # 记录训练集的平均损失和准确率
    train_loss = running_train_loss / len(train_loader())
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 评估模型
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    for images, labels in test_loader():
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_test_loss += loss.numpy()

        _, predicted = paddle.argmax(outputs, axis=1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.shape[0]

    # 记录测试集的平均损失和准确率
    test_loss = running_test_loss / len(test_loader())
    test_accuracy = correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# 绘制损失和准确率曲线
epochs = range(1, num_epochs + 1)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制训练和测试损失曲线
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue', linestyle='-')
ax1.plot(epochs, test_losses, label='Test Loss', color='tab:blue', linestyle='--')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 创建共享x轴的第二个y轴，用于绘制准确率曲线
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:orange')
ax2.plot(epochs, train_accuracies, label='Train Accuracy', color='tab:orange', linestyle='-')
ax2.plot(epochs, test_accuracies, label='Test Accuracy', color='tab:orange', linestyle='--')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# 设置图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 设置标题
plt.title('Training and Test Loss & Accuracy')

# 显示图形
plt.tight_layout()
plt.show()
