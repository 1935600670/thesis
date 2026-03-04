# -*- coding: utf-8 -*-
"""绘制训练损失曲线与预测 vs 真实曲线，并保存到指定目录。"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


def _ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_dir: Union[str, Path] = "output/images",
    filename: str = "loss_curve.png",
) -> str:
    """
    绘制训练/验证损失曲线并保存。

    Args:
        train_losses: 每轮训练损失
        val_losses: 每轮验证损失（可选）
        save_dir: 保存目录
        filename: 文件名

    Returns:
        保存后的完整路径
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("可视化需要 matplotlib: pip install matplotlib")

    save_dir = _ensure_dir(save_dir)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-", label="Train Loss")
    if val_losses is not None and len(val_losses) == len(train_losses):
        plt.plot(epochs, val_losses, "r-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("LSTM Training Loss")
    plt.grid(True, alpha=0.3)
    out_path = Path(save_dir) / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_predictions(
    targets: np.ndarray,
    preds: np.ndarray,
    save_dir: Union[str, Path] = "output/images",
    filename: str = "predictions.png",
    max_points: int = 500,
) -> str:
    """
    绘制预测值 vs 真实值曲线并保存。

    Args:
        targets: 真实值一维数组
        preds: 预测值一维数组
        save_dir: 保存目录
        filename: 文件名
        max_points: 最多绘制点数（过多时下采样）

    Returns:
        保存后的完整路径
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("可视化需要 matplotlib: pip install matplotlib")

    save_dir = _ensure_dir(save_dir)
    n = len(targets)
    if n > max_points:
        step = n // max_points
        idx = np.arange(0, n, step)[:max_points]
        targets = targets[idx]
        preds = preds[idx]
    x = np.arange(len(targets))
    plt.figure(figsize=(10, 5))
    plt.plot(x, targets, "b-", alpha=0.7, label="True")
    plt.plot(x, preds, "r-", alpha=0.7, label="Pred")
    plt.xlabel("Sample")
    plt.ylabel("Close")
    plt.legend()
    plt.title("LSTM: Prediction vs True")
    plt.grid(True, alpha=0.3)
    out_path = Path(save_dir) / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)
