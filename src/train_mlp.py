from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from .config import get_path, load_config
    from .dataset_builder import load_processed_dataset
    from .evaluate import regression_metrics_with_ci
    from .mlp_model import BiofilmMLP
    from .visualize import plot_pred_vs_true, plot_residuals, save_fig
except ImportError:
    from config import get_path, load_config
    from dataset_builder import load_processed_dataset
    from evaluate import regression_metrics_with_ci
    from mlp_model import BiofilmMLP
    from visualize import plot_pred_vs_true, plot_residuals, save_fig


def _tensor_dataset(X, y):
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
    )


def train_mlp_model(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    target: str = "biofilm_thickness",
    config_path: str | Path | None = None,
    epochs: int = 300,
    batch_size: int = 32,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    训练 MLP 模型，含：
    - L2 正则化（weight_decay）
    - 学习率调度（ReduceLROnPlateau）
    - 早停（patience）
    - BatchNorm + Dropout
    - Bootstrap 置信区间评估
    """
    cfg = load_config(config_path)
    if X is None or y is None:
        _, X, y = load_processed_dataset(config_path=config_path, target=target)
    if len(X) < 6:
        raise ValueError("At least 6 samples are needed to train the MLP.")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model_dir = get_path("models", cfg)
    result_dir = get_path("results", cfg)
    fig_dir = get_path("figures", cfg)

    test_size = float(cfg.get("modeling", {}).get("test_size", 0.2))
    test_size = min(0.3, max(1 / len(X), test_size))
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    validation_size = float(cfg.get("modeling", {}).get("validation_size", 0.2))
    val_size = validation_size if len(X_train_val) >= 10 else max(1 / len(X_train_val), 0.25)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    drop_last = len(X_train_s) > batch_size and len(X_train_s) % batch_size == 1
    train_loader = DataLoader(
        _tensor_dataset(X_train_s, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    val_ds = _tensor_dataset(X_val_s, y_val)

    model = BiofilmMLP(input_dim=X.shape[1], dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            x_val_t, y_val_t = val_ds.tensors
            val_loss = float(criterion(model(x_val_t), y_val_t).cpu())

        train_loss = float(np.mean(train_losses)) if train_losses else val_loss
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "lr": current_lr})

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- 评估（含 Bootstrap CI）----------
    model.eval()
    with torch.no_grad():
        test_pred = model(
            torch.tensor(X_test_s, dtype=torch.float32)
        ).numpy().ravel()

    metrics = regression_metrics_with_ci(y_test, test_pred)
    pd.DataFrame([{"model": "mlp", "target": target, **metrics}]).to_csv(
        result_dir / f"mlp_{target}_metrics.csv", index=False
    )

    # ---------- 保存模型 ----------
    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "target": target,
        "feature_columns": list(X.columns),
        "hidden_dims": (128, 64, 32),
        "dropout": dropout,
    }
    torch.save(ckpt, model_dir / "mlp_model.pt")
    torch.save(ckpt, model_dir / f"mlp_model_{target}.pt")
    joblib.dump(scaler, model_dir / "mlp_scaler.joblib")
    joblib.dump(scaler, model_dir / f"mlp_scaler_{target}.joblib")

    # ---------- 训练曲线 ----------
    history_df = pd.DataFrame(history)
    history_df.to_csv(result_dir / "mlp_training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("MLP loss curve")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["lr"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")

    plt.tight_layout()
    for path in [fig_dir / "mlp_loss_curve.png", fig_dir / f"mlp_{target}_loss_curve.png"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_pred_vs_true(
        y_test, test_pred,
        "MLP: predicted vs true",
        fig_dir / f"mlp_{target}_pred_vs_true.png",
    )
    plot_residuals(
        y_test,
        test_pred,
        "MLP: residuals",
        fig_dir / f"mlp_{target}_residuals.png",
    )

    if len(X) < 20:
        warnings.warn("Small dataset: MLP results are mainly useful as a workflow smoke test.")
    return history_df


if __name__ == "__main__":
    train_mlp_model()
