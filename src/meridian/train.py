"""Train the baseline model, log everything to MLflow, register the artifact, and
promote it to the ``production`` alias if it beats the current production RMSE.

Run directly (``python -m meridian.train``) for the first baseline, or call
``train_and_register()`` from the drift monitor to trigger automated retraining.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn

from .config import settings
from .data import generate_baseline, load_real_csv
from .model import StandardScaler, TripDurationNet
from .pyfunc import MLflowTripModel


def _train_torch(x_train, y_train, x_val, y_val, *, epochs=30, hidden=64, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    net = TripDurationNet(n_features=x_train.shape[1], hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    xt = torch.from_numpy(x_train.astype("float32"))
    yt = torch.from_numpy(y_train.astype("float32"))
    xv = torch.from_numpy(x_val.astype("float32"))

    ds = torch.utils.data.TensorDataset(xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)

    for epoch in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_rmse = float(np.sqrt(mean_squared_error(y_val, net(xv).numpy())))
        mlflow.log_metric("val_rmse", val_rmse, step=epoch)
    return net


def _current_production_rmse(client: "mlflow.MlflowClient") -> float | None:
    try:
        mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    except Exception:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get("test_rmse")


def train_and_register(real_csv: str | None = None, *, reason: str = "baseline") -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.model_name)

    df = load_real_csv(real_csv) if real_csv else generate_baseline(seed=settings.seed)
    feature_cols = list(settings.feature_cols)
    x = df[feature_cols].to_numpy(dtype="float32")
    y = df[settings.target_col].to_numpy(dtype="float32")

    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.3, random_state=settings.seed)
    x_val, x_te, y_val, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=settings.seed)

    scaler = StandardScaler().fit(x_tr)
    x_tr_s, x_val_s, x_te_s = (scaler.transform(a) for a in (x_tr, x_val, x_te))

    hidden = 64
    with mlflow.start_run(run_name=f"train-{reason}") as run:
        mlflow.log_params(
            {"epochs": 30, "hidden": hidden, "lr": 1e-3, "n_rows": len(df),
             "source": "real_csv" if real_csv else "synthetic", "reason": reason}
        )
        net = _train_torch(x_tr_s, y_tr, x_val_s, y_val, hidden=hidden, seed=settings.seed)

        with torch.no_grad():
            preds = net(torch.from_numpy(x_te_s.astype("float32"))).numpy()
        test_rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        test_mae = float(mean_absolute_error(y_te, preds))
        mlflow.log_metrics({"test_rmse": test_rmse, "test_mae": test_mae})

        # Bundle weights + scaler + feature order as a pyfunc model.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            torch.save(net.state_dict(), tmp / "weights.pt")
            (tmp / "meta.json").write_text(json.dumps(
                {"feature_cols": feature_cols, "hidden": hidden, "scaler": scaler.state()}
            ))
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MLflowTripModel(),
                artifacts={"weights": str(tmp / "weights.pt"), "meta": str(tmp / "meta.json")},
                registered_model_name=settings.model_name,
                code_paths=[str(Path(__file__).parent)],
            )

        # Persist the training distribution as the drift reference.
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        df[feature_cols].to_parquet(settings.reference_path, index=False)

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{settings.model_name}'")
    version = max(versions, key=lambda mv: int(mv.version)).version
    prev = _current_production_rmse(client)
    promoted = prev is None or test_rmse <= prev
    if promoted:
        client.set_registered_model_alias(settings.model_name, settings.model_alias, version)

    result = {"version": version, "test_rmse": test_rmse, "test_mae": test_mae,
              "previous_rmse": prev, "promoted": promoted, "reason": reason}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train + register the trip-duration model.")
    p.add_argument("--real-csv", help="Path to a real NYC taxi export instead of synthetic data.")
    p.add_argument("--reason", default="baseline")
    args = p.parse_args()
    train_and_register(args.real_csv, reason=args.reason)
