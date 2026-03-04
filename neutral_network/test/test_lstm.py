# -*- coding: utf-8 -*-
"""LSTM 测试：在测试集上评估并返回指标；提供 run_test（评估 + 保存预测图）。"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Union
from pathlib import Path

from neutral_network.net import ClosePredictorLSTM, MultiCategoryLSTM
from neutral_network.plot import plot_predictions


def evaluate_model(
    model: ClosePredictorLSTM,  # 或 MultiCategoryLSTM，接口一致
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    在测试集上评估模型，计算 MAE、RMSE、MAPE（百分比）。

    Returns:
        含 'mae', 'rmse', 'mape' 的字典
    """
    model = model.to(device)
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())

    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()

    mae = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    mask = np.abs(targets) > 1e-8
    mape = (np.abs((preds[mask] - targets[mask]) / targets[mask]).mean() * 100.0) if mask.any() else float("nan")

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def evaluate_model_and_predict(
    model: ClosePredictorLSTM,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """评估并返回指标及预测值、真实值（用于绘图）。"""
    model = model.to(device)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()
    mae = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    mask = np.abs(targets) > 1e-8
    mape = (np.abs((preds[mask] - targets[mask]) / targets[mask]).mean() * 100.0) if mask.any() else float("nan")
    metrics = {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
    return metrics, preds, targets


def run_test(
    model: Union[MultiCategoryLSTM, ClosePredictorLSTM],
    test_loader: DataLoader,
    img_save_dir: Union[str, Path],
    device: torch.device = None,
) -> Dict[str, float]:
    """
    在测试集上评估模型，并将预测 vs 真实曲线保存到 img_save_dir。

    Returns:
        评估指标字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, preds, targets = evaluate_model_and_predict(model, test_loader, device)
    plot_path = plot_predictions(
        targets, preds,
        save_dir=img_save_dir,
        filename="predictions.png",
    )
    print(f"预测图已保存: {plot_path}")
    return metrics


def load_model_from_checkpoint(
    model_path: str,
) -> Tuple[MultiCategoryLSTM, dict]:
    """从 run_train 保存的 checkpoint 加载模型与 meta。"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    model = MultiCategoryLSTM(
        category_dims=meta["category_dims"],
        hidden_per_category=meta.get("hidden_size", 64),
        num_layers=meta.get("num_layers", 2),
        dropout=0.1,
    )
    model.load_state_dict(ckpt["model_state"])
    return model, meta
