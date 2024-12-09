import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Net(nn.Module):
    def __init__(self, input_size, kernel_size, gru_units, num_classes):
        super(Net, self).__init__()

        # CNN 部分
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=input_size, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # GRU 部分
        self.gru = nn.GRU(input_size=input_size, hidden_size=gru_units, batch_first=True, bidirectional=True)

        # 分类部分
        self.fc = nn.Linear(gru_units * 2, num_classes)  # 因为 GRU 是双向的，所以乘以 2

    def forward(self, x, lengths):
        # CNN 部分
        x = x.transpose(1, 2)  # 从 (batch_size, seq_len, input_size) 转换为 (batch_size, input_size, seq_len)
        x = self.conv1(x)
        # x = self.pool(x)

        # GRU 部分
        x = x.transpose(1, 2)  # 再次转换为 (batch_size, seq_len, input_size)
        
        # 使用 pack_padded_sequence 来处理变长序列
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)

        # 解包 GRU 输出
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 取 GRU 的最后时间步输出
        final_output = output[torch.arange(output.size(0)), lengths - 1, :]  # 选取每个序列的最后一个有效时间步

        # 分类层
        out = self.fc(final_output)  # 输出四个类别的概率分布
        return out
