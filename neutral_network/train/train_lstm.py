# -*- coding: utf-8 -*-
"""LSTM 训练：在给定 DataLoader 上训练模型；并提供完整流程 run_train（构建数据、训练、保存模型与损失图）。"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

from neutral_network.net import ClosePredictorLSTM, MultiCategoryLSTM
from neutral_network.plot import plot_loss_curve


def train_model(
    model: ClosePredictorLSTM,  # 或 MultiCategoryLSTM，接口一致
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    val_loader: Optional[DataLoader] = None,
) -> Tuple[ClosePredictorLSTM, List[float], List[float]]:
    """
    训练 LSTM 预测下一日 close。

    Args:
        model: ClosePredictorLSTM 实例
        train_loader: 训练 DataLoader
        device: 设备 (cuda/cpu)
        epochs: 训练轮数
        lr: 学习率
        val_loader: 可选验证 DataLoader

    Returns:
        (训练好的 model, train_losses, val_losses)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    val_loss += criterion(pred, y).item()
                    n_val += 1
            avg_val = val_loss / max(n_val, 1)
            val_losses.append(avg_val)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")
        else:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_train:.6f}")

    return model, train_losses, val_losses


def run_train(
    csv_path: str,
    stock_yml_path: str,
    model_save_path: str,
    img_save_dir: str,
    seq_len: int = 60,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[MultiCategoryLSTM, DataLoader, Dict[str, Any]]:
    """
    完整训练流程：构建数据集 → 训练 → 保存模型到 model_save_path，损失曲线到 img_save_dir。

    Returns:
        (训练好的 model, test_loader, meta) 其中 meta 含 category_dims 等，供 run_test 加载模型用。
    """
    from neutral_network.dataset import build_datasets_by_category

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("构建数据集（按类型分 LSTM：60 天预测 1 天 close）...")
    train_ds, val_ds, test_ds, value_cols, category_dims, scaler_dict = build_datasets_by_category(
        csv_path,
        seq_len=seq_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        yml_path=stock_yml_path,
        target_col="close",
    )
    n_features = len(value_cols)
    print(f"总特征数: {n_features}, 各类别维度: {category_dims}")
    print(f"训练样本: {len(train_ds)}, 验证: {len(val_ds)}, 测试: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MultiCategoryLSTM(
        category_dims=category_dims,
        hidden_per_category=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
    )
    # 将最后一层 bias 初始化为训练集目标均值，减轻系统性高/低估（PyTorch 的 Linear 本身带 bias，默认从 0 初始化）
    target_mean = scaler_dict.get("target_mean")
    if target_mean is not None and hasattr(model.fc, "bias") and model.fc.bias is not None:
        with torch.no_grad():
            model.fc.bias.fill_(float(target_mean))
        print(f"输出层 bias 已初始化为训练集 close 均值: {target_mean:.4f}")

    print("开始训练...")
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        device,
        epochs=epochs,
        lr=lr,
        val_loader=val_loader,
    )

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "category_dims": category_dims,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }
    torch.save({"model_state": model.state_dict(), "meta": meta}, model_save_path)
    print(f"模型已保存: {model_save_path}")

    plot_path = plot_loss_curve(
        train_losses,
        val_losses=val_losses if val_losses else None,
        save_dir=img_save_dir,
        filename="loss_curve.png",
    )
    print(f"损失曲线已保存: {plot_path}")

    return model, test_loader, meta
