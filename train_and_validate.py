"""
train_and_validate.py

Trains the amortized MJD estimator with a held-out validation set, keeps the
best checkpoint, and runs parameter-recovery diagnostics on an independent
test set.

Outputs:
    mjd_weights.pth   checkpoint (same format the main script loads)
    recovery.png                predicted vs true for each parameter + loss curves
    pd_check.png                predicted vs true default probability
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

import matplotlib
matplotlib.use("Agg")                     # render to file, no display needed
import matplotlib.pyplot as plt
import joblib

from Amortized_Inference_Data_Generator import generate_mjd_dataset
from model import SDEPredictor, clip_to_prior
from pricing import mjd_pd

# ----------------------------- Configuration ------------------------------ #
SEED        = 42
N_TOTAL     = 120_000
FRAC_VAL    = 0.10
FRAC_TEST   = 0.10
EPOCHS      = 30
BATCH       = 256
LR          = 1e-3
R_TRAIN     = 0.05                        # must match the generator's r
N_PD_CHECK  = 10_000                      # PD check is a scalar loop, so subsample
PARAM_NAMES = ["V/D", "sigma_V", "lambda", "mu_J", "sigma_J"]


# ------------------------------- Helpers ---------------------------------- #
def split(X, Y, rng):
    """Random train / validation / test split."""
    idx = rng.permutation(len(X))
    n_test, n_val = int(FRAC_TEST * len(X)), int(FRAC_VAL * len(X))
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (X[tr], Y[tr]), (X[va], Y[va]), (X[te], Y[te])


def to_loader(X, Y, shuffle, drop_last=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    # drop_last on training avoids a size-1 final batch, which BatchNorm rejects
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, drop_last=drop_last)


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)                 # BatchNorm behaves differently in eval
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n                      # epoch average, not last batch

@torch.no_grad()
def predict(model, X, Y_mean, Y_std):
    model.eval()
    out = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return clip_to_prior(out * Y_std + Y_mean)


def pd_from_params(p, floor=1e-4):
    """Risk-neutral PD in normalized units (D = 1), with the same clipping as inference."""
    V, s, lam, mj, sj = p
    return mjd_pd(V=max(V, floor), D=1.0, mu=R_TRAIN, sigma=max(s, floor),
                  lam=max(lam, 0.0), mu_j=mj, sigma_j=max(sj, floor))

def pd_report(pd_true, pd_pred, threshold=0.30):
    bins = [0, 1e-4, 1e-2, 0.1, 0.3, 1.0]
    df = pd.DataFrame({"true": pd_true, "pred": pd_pred})
    for col, label in [("pred", "PREDICTED PD (calibration: can an alert's number be trusted?)"),
                       ("true", "TRUE PD (recall: are risky firms caught?)")]:
        groups = df.groupby(pd.cut(df[col], bins, include_lowest=True), observed=True)
        print(f"\nBucketed by {label}")
        print(groups.agg(n=("true", "size"), mean_true=("true", "mean"), mean_pred=("pred", "mean")))

    risky, flagged = pd_true > threshold, pd_pred > threshold
    tp = int((risky & flagged).sum())
    print(f"\nAlerts at {threshold:.0%}: recall {tp / max(risky.sum(), 1):.1%} "
          f"({tp}/{risky.sum()} truly risky firms flagged), "
          f"precision {tp / max(flagged.sum(), 1):.1%} ({tp}/{flagged.sum()} alerts correct)")

################################### Main #############################################
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    window = [30, 60, 120,250]
    sweep = []
    for t in window:
      np.random.seed(SEED)
      # 1. Simulate and split
      print(f"Simulating {N_TOTAL:,} paths with a sample of {t:,} days...")
      X, Y = generate_mjd_dataset(n_samples=N_TOTAL, days=t, r=R_TRAIN)
      (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te) = split(X, Y, rng)

      # 2. Standardize targets using training statistics only (no leakage)
      Y_mean, Y_std = Y_tr.mean(axis=0), Y_tr.std(axis=0)
      standardize = lambda Y: (Y - Y_mean) / Y_std

      train_loader = to_loader(X_tr, standardize(Y_tr), shuffle=True, drop_last=True)
      val_loader   = to_loader(X_va, standardize(Y_va), shuffle=False)

      # 3. Train, keeping the best validation checkpoint
      model = SDEPredictor(t)
      criterion = nn.MSELoss()
      optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

      history = {"train": [], "val": []}
      best_val, best_state = np.inf, None

      for epoch in range(1, EPOCHS + 1):
          tr_loss = run_epoch(model, train_loader, criterion, optimizer)
          va_loss = run_epoch(model, val_loader, criterion)
          history["train"].append(tr_loss)
          history["val"].append(va_loss)

          flag = ""
          if va_loss < best_val:
              best_val = va_loss
              best_state = {k: v.clone() for k, v in model.state_dict().items()}
              flag = "  <- best"
          print(f"Epoch {epoch:2d} | train {tr_loss:.4f} | val {va_loss:.4f}{flag}")

      model.load_state_dict(best_state)

      # 4. Parameter recovery on the untouched test set
      Y_pred = predict(model, X_te, Y_mean, Y_std)

      resid = Y_pred - Y_te
      rmse = np.sqrt((resid ** 2).mean(axis=0))
      prior_rmse = Y_te.std(axis=0)         # error from always predicting the average
      r2 = 1 - (resid ** 2).sum(axis=0) / ((Y_te - Y_te.mean(axis=0)) ** 2).sum(axis=0)

      print("\nParameter recovery (test set)")
      print(f"{'param':>8} | {'R^2':>6} | {'RMSE':>8} | {'prior RMSE':>10} | {'skill':>6}")
      for i, name in enumerate(PARAM_NAMES):
          skill = 1 - rmse[i] / prior_rmse[i]
          print(f"{name:>8} | {r2[i]:6.3f} | {rmse[i]:8.4f} | {prior_rmse[i]:10.4f} | {skill:6.1%}")
      def jump_var(Y):
        return Y[:, 2] * (Y[:, 3]**2 + Y[:, 4]**2)

      jv_true, jv_pred = jump_var(Y_te), jump_var(Y_pred)
      r2_jv = 1 - ((jv_pred - jv_true)**2).sum() / ((jv_true - jv_true.mean())**2).sum()
      print(f"Total jump variance R^2: {r2_jv:.3f}")

      # 5. Recovery plots: five parameters + loss curves
      fig, axes = plt.subplots(2, 3, figsize=(15, 9))
      for i, (ax, name) in enumerate(zip(axes.flat, PARAM_NAMES)):
          lo, hi = Y_te[:, i].min(), Y_te[:, i].max()
          ax.hexbin(Y_te[:, i], Y_pred[:, i], gridsize=60, mincnt=1,
                    cmap="viridis", extent=(lo, hi, lo, hi))
          ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect recovery")
          ax.axhline(Y_te[:, i].mean(), color="orange", ls=":", lw=1, label="prior mean")
          ax.set_title(f"{name}   R² = {r2[i]:.3f}")
          ax.set_xlabel("true")
          ax.set_ylabel("predicted")
          ax.legend(loc="upper left", fontsize=8)

      ax = axes.flat[5]
      ax.plot(range(1, EPOCHS + 1), history["train"], label="train")
      ax.plot(range(1, EPOCHS + 1), history["val"], label="validation")
      ax.set_title("Loss (standardized MSE)")
      ax.set_xlabel("epoch")
      ax.legend()

      fig.tight_layout()
      fig.savefig(f"recovery_{t}d.png", dpi=150)
      plt.close(fig)
      # 6. What actually matters: does the network get the PD right?
      sub = rng.choice(len(X_te), size=N_PD_CHECK, replace=False)
      pd_true = np.array([pd_from_params(Y_te[i]) for i in sub])
      pd_pred = np.array([pd_from_params(Y_pred[i]) for i in sub])

      # Same prediction, but with jump parameters replaced by their prior means.
      # If this matches pd_pred, the predicted jump parameters add no information.
      jump_prior = Y_mean[2:]
      pd_prior_jumps = np.array([
          pd_from_params(np.concatenate([Y_pred[i, :2], jump_prior])) for i in sub
      ])

      rho_pred, _ = spearmanr(pd_true, pd_pred)
      rho_prior, _ = spearmanr(pd_true, pd_prior_jumps)
      print("\nDefault probability check (risk-neutral, test subsample)")
      print(f"  Mean |PD error|, predicted jumps:   {np.abs(pd_pred - pd_true).mean():.4%}")
      print(f"  Mean |PD error|, prior-mean jumps:  {np.abs(pd_prior_jumps - pd_true).mean():.4%}")
      print(f"  Rank correlation, predicted jumps:  {rho_pred:.3f}")
      print(f"  Rank correlation, prior-mean jumps: {rho_prior:.3f}")
      torch.save({
          "state_dict": model.state_dict(),
          "Y_mean": torch.tensor(Y_mean, dtype=torch.float32),
          "Y_std": torch.tensor(Y_std, dtype=torch.float32),
          "n_days": t,
      }, f"mjd_weights_{t}d.pth")
      fig, ax = plt.subplots(figsize=(6, 6))
      floor = 1e-6                          # keep zero PDs visible on log axes
      ax.scatter(np.maximum(pd_true, floor), np.maximum(pd_pred, floor), s=4, alpha=0.4)
      ax.plot([floor, 1], [floor, 1], "r--", lw=1)
      ax.set_xscale("log")
      ax.set_yscale("log")
      ax.set_xlabel("true PD")
      ax.set_ylabel("predicted PD")
      ax.set_title(f"PD recovery   Spearman ρ = {rho_pred:.3f}")
      fig.tight_layout()
      fig.savefig(f"pd_check_{t}d.png", dpi=150)
      plt.close(fig)
      print(f"Saved pd_check_{t}d.png")

      sweep.append({"days": t, **dict(zip(PARAM_NAMES, r2)),
                    "jump_var": r2_jv, "pd_spearman": rho_pred})

      Y_va_pred = predict(model, X_va, Y_mean, Y_std)
      sub_va = rng.choice(len(X_va), size=N_PD_CHECK, replace=False)
      pd_va_true = np.array([pd_from_params(Y_va[i]) for i in sub_va])
      pd_va_pred = np.array([pd_from_params(Y_va_pred[i]) for i in sub_va])

      iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pd_va_pred, pd_va_true)
      pd_report(pd_true, iso.predict(pd_pred))   # should close the gap in every bucket
      pd_report(pd_true, pd_pred)
      err = pd.DataFrame({"vd": Y_te[:, 0], "se": (Y_pred[:, 1] - Y_te[:, 1]) ** 2})
      err["b"] = pd.cut(err["vd"], [0.5, 0.8, 1.0, 1.5, 3, 10], include_lowest=True)
      print(err.groupby("b", observed=True)["se"].agg(n="size", rmse=lambda s: np.sqrt(s.mean())))

      joblib.dump(iso, f"pd_calibrator_{t}d.joblib")

    results = pd.DataFrame(sweep).set_index("days")
    print("\nWindow sweep summary (R^2 per parameter)")
    print(results.round(3))
    results.to_csv("window_sweep.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in PARAM_NAMES + ["jump_var", "pd_spearman"]:
        ax.plot(results.index, results[col], marker="o", label=col)
    ax.set_xlabel("window length (trading days)")
    ax.set_ylabel("R² (Spearman ρ for PD)")
    ax.set_title("Parameter recovery vs. window length")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("window_sweep.png", dpi=150)
    plt.close(fig)
