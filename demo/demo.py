import os

import torch
import pandas as pd
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset

from utils.net import Net
from utils.saver import load_checkpoint
from utils.test import TestDataset, read_from_csv

# 准备数据集和数据加载器
data = read_from_csv('../dataset/testA.csv')
dataset = TestDataset(data)
print(dataset)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

# 模型初始化
input_size = 1  # 每个时间步的特征数量
kernel_size = 3
gru_units = 64
num_classes = 4  # 四分类任务

model = Net(input_size, kernel_size, gru_units, num_classes)

# 优化器和损失函数
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.is_available() and os.path.exists('checkpoint.pth'):
    model, optimizer, start_epoch, total_loss = load_checkpoint(model, optimizer, 'checkpoint.pth')
# 加载模型的权重
model.eval()  # 切换到评估模式

# 预测并保存结果
output_data = []
with torch.no_grad():
    for idx, batch_data in enumerate(dataloader):
        # 对输入的 batch 进行预测
        ids, signal, length = batch_data
        outputs = model(signal.unsqueeze(-1), length)
        predicted = torch.sigmoid(outputs)  # 应用 sigmoid 函数，得到每个类别的概率
        predicted = (predicted > 0.5).float()  # 将概率转换为 0 或 1
        # 保存每个 id 和对应的输出
        for id_value, predicted_label in zip(ids, predicted):
            output_data.append([id_value.item()] + predicted_label.numpy().flatten().tolist())

# 7. 保存预测结果到新的 CSV 文件
output_df = pd.DataFrame(output_data, columns=['id', 'label_0', 'label_1', 'label_2', 'label_3'])
output_df.to_csv('predictions.csv', index=False)
