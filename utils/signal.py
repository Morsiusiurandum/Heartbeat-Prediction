import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Signal:
    """
    自定义数据类,用于训练和预测
    """

    def __init__(self, label: np.ndarray, heartbeat_signals: np.ndarray, length: int):
        """
        初始化心跳信号实例.
        
        :param heartbeat_signals: 心跳信号序列
        :param label: 心跳信号类别（0, 1, 2, 3）
        """
        self.heartbeat_signals = heartbeat_signals
        self.label = label
        self.length = length

    def __repr__(self):
        """
        返回心跳信号实例的字符串表示.
        """
        return (
            f"heartbeat_signals={self.heartbeat_signals.size}, "
            f"label={self.label})"
        )


def label_to_one_hot(label: int) -> np.ndarray:
    """
    将标签转换为 one-hot 编码
    :param label: 指定的标签
    :return: One-hot 编码
    """
    one_hot = np.zeros(4)
    one_hot[int(label)] = 1
    return one_hot


def read_from_csv(path: str) -> np.ndarray:
    """
    读取 CSV 文件.
    :param path: CSV 文件路径
    :return: 保存数据的 np.ndarray
    """
    origin_data = pd.read_csv(path)
    origin_data.head()

    new_signals = origin_data['heartbeat_signals'].str.split(',', expand=True)
    new_signals.columns = ['signals_' + str(x + 1) for x in range(205)]  # 重命名新生成的列名
    new_df = pd.DataFrame(new_signals, dtype=np.float64)  # 转化为数组形式

    new_df["label"] = origin_data["label"]  # 加入标签列

    labels = np.array([label_to_one_hot(label) for label in new_df["label"].astype(int).values])
    signals = new_df.iloc[:, :-1].values  # 提取 signals_* 列为 NumPy 数组

    new_data = [Signal(label, signal, len(np.trim_zeros(signal, 'b'))) for label, signal in zip(labels, signals)]

    return np.array(new_data)


class TimeSeriesDataset(Dataset):
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
        label = self.data[idx].label  # 标签
        signal = self.data[idx].heartbeat_signals  # 信号序列
        length = self.data[idx].length  # 信号序列长度

        # 返回信号序列（需要转为 Tensor）以及对应的标签（one-hot 编码）
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), length
