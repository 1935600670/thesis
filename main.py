# -*- coding: utf-8 -*-
"""
调试入口：读取 config/img_path.yml、config/model_path.yml，
用 data/stock_000001.csv 训练 LSTM（60 天预测 1 天 close），保存模型与可视化结果，并跑测试。

运行: python main.py
"""
from pathlib import Path

from loader import load_yml
from neutral_network.train import run_train
from neutral_network.test import run_test


def _project_root() -> Path:
    """项目根目录（包含 config 的目录）。"""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config").is_dir():
            return parent
    return current


def main():
    root = _project_root()
    csv_path = root / "data" / "stock_000001.csv"
    stock_yml = root / "config" / "stock_columns.yml"

    if not csv_path.exists():
        print(f"未找到数据文件: {csv_path}")
        return

    # 读取保存路径配置
    img_cfg = load_yml(root / "config" / "img_path.yml")
    model_cfg = load_yml(root / "config" / "model_path.yml")
    img_dir = img_cfg.get("lstm_path") or "output/images"
    model_path = model_cfg.get("lstm_path") or "output/model.pt"
    if not Path(img_dir).is_absolute():
        img_dir = str(root / img_dir)
    if not Path(model_path).is_absolute():
        model_path = str(root / model_path)

    print("开始训练（模型与图片路径由 config 指定）...")
    model, test_loader, _ = run_train(
        csv_path=str(csv_path),
        stock_yml_path=str(stock_yml),
        model_save_path=model_path,
        img_save_dir=img_dir,
        seq_len=60,
        batch_size=32,
        epochs=50,
        lr=1e-3,
        hidden_size=64,
        num_layers=2,
    )

    print("\n测试集评估:")
    metrics = run_test(model, test_loader, img_save_dir=img_dir)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if k != "mape" or v == v else f"  {k}: {v}")


if __name__ == "__main__":
    main()
