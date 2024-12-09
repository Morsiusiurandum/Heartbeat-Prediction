import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SignalTest:
    """
    自定义数据类,用于训练和预测
    """

    def __init__(self, id: int, heartbeat_signals: np.ndarray, length: int, label: np.ndarray):
        """
        初始化心跳信号实例.

        :param heartbeat_signals: 心跳信号序列
        :param label: 心跳信号类别（0, 1, 2, 3）
        """
        self.id = id
        self.heartbeat_signals = heartbeat_signals
        self.length = length
        self.label = label

    def __repr__(self):
        """
        返回心跳信号实例的字符串表示.
        """
        return (
            f"heartbeat_signals={self.heartbeat_signals.size}, "
            f"label={self.label})"
        )


def read_from_csv(path: str) -> np.ndarray:
    """
    读取 CSV 文件.
    :param path: CSV 文件路径
    :return: 保存数据的 np.ndarray
    """
    origin_data = pd.read_csv(path)

    new_signals = origin_data['heartbeat_signals'].str.split(',', expand=True)
    new_signals.columns = ['signals_' + str(x + 1) for x in range(205)]  # 重命名新生成的列名
    new_df = pd.DataFrame(new_signals, dtype=np.float64)  # 转化为数组形式

    new_df["id"] = origin_data["id"]  # 加入标签列

    ids = np.array(new_df["id"].astype(int).values)
    signals = new_df.iloc[:, :-1].values  # 提取 signals_* 列为 NumPy 数组

    new_data = [SignalTest(id, signal, len(np.trim_zeros(signal, 'b')), np.array([0, 0, 0, 0])) for id, signal in zip(ids, signals)]

    return np.array(new_data)


class TestDataset(Dataset):
    def __init__(self, data: np.array, max_len=None):
        """
        :param data: numpy ndarray, shape (N, 2), first column is label, second is signal
        :param max_len: Maximum length for padding.
        """
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.data[idx].id  # 标签
        signal = self.data[idx].heartbeat_signals  # 信号序列
        length = self.data[idx].length  # 信号序列长度

        # 返回信号序列（需要转为 Tensor）以及对应的标签（one-hot 编码）
        return id,torch.tensor(signal, dtype=torch.float32), length
