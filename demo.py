import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.net import TCNWithGRU
from utils.saver import load_checkpoint, save_checkpoint, load_config
from utils.signal import TimeSeriesDataset, read_from_csv
from utils.test import TestDataset
from utils.test import read_from_csv as read_from_csv_test

# 加载配置
config = load_config('config/config.yaml')

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 加载训练数据
train_data = read_from_csv(config["train_data_path"])
train_dataset = TimeSeriesDataset(train_data, config["sequence_length"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

# 初始化模型
model = TCNWithGRU(
    input_size=config["input_size"],
    tcn_channels=config["tcn_channel"],
    gru_hidden_size=config["gru_units"],
    output_size=config["num_classes"],
    kernel_size=config["kernel_size"],
    dropout=config["dropout"]
).to(device)

# 优化器和损失函数
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 训练过程
start_epoch = 0
total_loss = 0
if os.path.exists(config["checkpoint_path"]):
    model, optimizer, start_epoch, total_loss = load_checkpoint(model, optimizer, config["checkpoint_path"])

for epoch in range(start_epoch, config["num_epochs"]):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels, lengths in train_loader:
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        sequences = sequences.unsqueeze(-1).permute(0, 2, 1)

        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        predicted = torch.sigmoid(outputs) > 0.5
        correct += (predicted == labels).sum().item()
        total += labels.size(0) * labels.size(1)
        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}")

    if (epoch + 1) % config["save_interval"] == 0:
        save_checkpoint(model, optimizer, epoch, total_loss / len(train_loader), config["checkpoint_path"])

# 评估
eval_data = read_from_csv_test(config["test_data_path"])
eval_dataset = TestDataset(eval_data)
eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)

model.to("cpu").eval()
output_data = []

with torch.no_grad():
    for ids, sequences, length in eval_loader:
        sequences = sequences.unsqueeze(-1).permute(0, 2, 1)
        outputs = model(sequences, length)
        predicted = torch.sigmoid(outputs) > 0.5

        for id_value, predicted_label in zip(ids, predicted):
            output_data.append([id_value.item()] + predicted_label.numpy().flatten().tolist())

output_df = pd.DataFrame(output_data, columns=['id', 'label_0', 'label_1', 'label_2', 'label_3'])
output_df.to_csv('predictions.csv', index=False)
print("Evaluation ended. Prediction results appeared in the file 'predictions.csv'.")
