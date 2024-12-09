import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.net import Net
from utils.saver import load_checkpoint, save_checkpoint
from utils.signal import TimeSeriesDataset, read_from_csv



# 检查是否有CUDA支持的GPU可用，如果有，则使用第一个GPU；否则，使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_data = read_from_csv('../dataset/train.csv')

# 创建数据集和 DataLoader
train_dataset = TimeSeriesDataset(train_data, 205)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 模型初始化
input_size = 1  # 每个时间步的特征数量
kernel_size = 3
gru_units = 64
num_classes = 4  # 四分类任务

model = Net(input_size, kernel_size, gru_units, num_classes).cuda()  # 将模型移动到 GPU 上

# 优化器和损失函数
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 100
checkpoint_path = 'checkpoint.pth'  # 断点文件路径

# 1. 尝试加载已有的断点
start_epoch = 0
total_loss = 0
if torch.cuda.is_available() and os.path.exists(checkpoint_path):
    model, optimizer, start_epoch, total_loss = load_checkpoint(model, optimizer, checkpoint_path)

# 2. 开始训练
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels, lengths in train_loader:
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        # 由于 CNN 需要输入形状为 (batch_size, seq_len, input_size)，所以确保数据形状正确
        sequences = sequences.unsqueeze(-1)  # 添加输入特征维度: (batch_size, seq_len, 1)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(sequences, lengths)

        # 计算损失
        loss = criterion(outputs, labels.float())  # 标签要转换为 float 类型
        loss.backward()

        # 更新参数
        optimizer.step()

        # 计算准确度
        predicted = torch.sigmoid(outputs)  # 应用 sigmoid 函数，得到每个类别的概率
        predicted = (predicted > 0.5).float()  # 将概率转换为 0 或 1
        correct += (predicted == labels).sum().item()
        total += labels.size(0) * labels.size(1)  # 总样本数

        total_loss += loss.item()

    # 打印当前 epoch 的损失和准确率
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}')

    # 每 5 个 epoch 保存一次断点
    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, epoch, total_loss / len(train_loader), checkpoint_path)
