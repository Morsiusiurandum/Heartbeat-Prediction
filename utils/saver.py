import torch
from torch import nn
from torch.optim import Adam
import yaml


def load_checkpoint(model: nn.Module, optimizer: Adam, checkpoint_path: str) -> (nn.Module, Adam, int, float):
    """
    加载断点
    :param model: chp中的模型
    :param optimizer: chp中的优化器
    :param checkpoint_path: 断点路径
    :return: model, optimizer, start_epoch, total_loss
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
    total_loss = checkpoint['loss']  # 之前的损失值
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model, optimizer, start_epoch, total_loss


def save_checkpoint(model: nn.Module, optimizer: Adam, epoch: int, loss: float, checkpoint_path: str):
    """
    保存断点
    :param model: 待保存的模型
    :param optimizer: 待保存的优化器
    :param epoch: 已经训练的 epoch 数
    :param loss:  当前 epoch 的损失值
    :param checkpoint_path: 保存路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as f:
        return yaml.safe_load(f)
