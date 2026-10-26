"""
robustness_drift.py

Sim-to-real check for the rate mismatch. The network and calibrator were
trained at r = 5%, but live scoring uses the current Treasury rate, and
PD_phys uses a higher physical drift. This script simulates fresh test sets at
other rates and measures how parameter recovery, PD ranking and calibration
degrade as the rate moves away from the training value.

The generator uses one r for both the asset drift and equity pricing, so this
tests the rate mismatch directly and gives an approximate read on drift
sensitivity for PD_phys. It does not separate the two.

Outputs:
    drift_robustness.csv
    drift_robustness.png
"""
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Amortized_Inference_Data_Generator import generate_mjd_dataset
from model import SDEPredictor
from pricing import mjd_pd
from train_and_validate import PARAM_NAMES, predict

SEED = 7
R_TRAIN = 0.05
WINDOWS = [30, 60, 120, 250]
# 0.12 approximates r = 4% plus THETA * sigma_V for a 20% asset volatility
DRIFTS = [0.00, 0.02, 0.05, 0.08, 0.12]
N_SAMPLES = 5_000
THRESHOLD = 0.30
FLOOR = 1e-4


def pd_batch(P, mu):
    """Normalized-unit PDs (D = 1) with the same floors used at inference."""
    return np.array([
        mjd_pd(V=max(p[0], FLOOR), D=1.0, mu=mu, sigma=max(p[1], FLOOR),
               lam=max(p[2], 0.0), mu_j=p[3], sigma_j=max(p[4], FLOOR))
        for p in P
    ])


rows = []
for t in WINDOWS:
    try:
        ckpt = torch.load(f"mjd_weights_{t}d.pth", weights_only=True)
        iso = joblib.load(f"pd_calibrator_{t}d.joblib")
    except FileNotFoundError as e:
        print(f"Skipping {t}d: {e}")
        continue
    model = SDEPredictor(int(ckpt["n_days"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    Y_mean, Y_std = ckpt["Y_mean"].numpy(), ckpt["Y_std"].numpy()

    for mu in DRIFTS:
        # Same seed at every rate, so parameter draws are paired where the
        # generator's random-number use does not depend on r.
        np.random.seed(SEED)
        X, Y = generate_mjd_dataset(n_samples=N_SAMPLES, days=t, r=mu)
        Y_pred = predict(model, X, Y_mean, Y_std)

        resid = Y_pred - Y
        r2 = 1 - (resid ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)

        pd_true = pd_batch(Y, mu)
        pd_raw = pd_batch(Y_pred, mu)
        pd_cal = iso.predict(pd_raw)

        risky, flagged = pd_true > THRESHOLD, pd_cal > THRESHOLD
        tp = int((risky & flagged).sum())
        rows.append({
            "days": t,
            "r": mu,
            **{f"R2_{n}": v for n, v in zip(PARAM_NAMES, r2)},
            "pd_spearman": spearmanr(pd_true, pd_raw)[0],
            "mae_raw": np.abs(pd_raw - pd_true).mean(),
            "mae_cal": np.abs(pd_cal - pd_true).mean(),
            # Calibration where alerts happen: mean true vs mean calibrated PD among flagged firms
            "alert_mean_true": pd_true[flagged].mean() if flagged.any() else np.nan,
            "alert_mean_cal": pd_cal[flagged].mean() if flagged.any() else np.nan,
            "recall": tp / max(risky.sum(), 1),
            "precision": tp / max(flagged.sum(), 1),
        })
        print(f"{t}d  r={mu:.2f}  spearman {rows[-1]['pd_spearman']:.3f}  "
              f"MAE cal {rows[-1]['mae_cal']:.4f}  recall {rows[-1]['recall']:.1%}  "
              f"precision {rows[-1]['precision']:.1%}")

results = pd.DataFrame(rows)
results.to_csv("drift_robustness.csv", index=False)

# Degradation relative to the training rate, per window
print("\nChange relative to r = 5% (training rate)")
for t, g in results.groupby("days"):
    base = g[np.isclose(g["r"], R_TRAIN)].iloc[0]
    for _, row in g.iterrows():
        print(f"{t:>4}d r={row['r']:.2f}: spearman {row['pd_spearman'] - base['pd_spearman']:+.3f}, "
              f"calibrated MAE {row['mae_cal'] - base['mae_cal']:+.4f}, "
              f"alert gap {row['alert_mean_cal'] - row['alert_mean_true']:+.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for t, g in results.groupby("days"):
    axes[0].plot(g["r"], g["pd_spearman"], marker="o", label=f"{t}d")
    axes[1].plot(g["r"], g["mae_cal"], marker="o", label=f"{t}d")
    axes[2].plot(g["r"], g["alert_mean_cal"] - g["alert_mean_true"], marker="o", label=f"{t}d")
for ax, title in zip(axes, ["PD rank correlation", "Calibrated PD mean abs error",
                            "Calibration gap among alerts (pred - true)"]):
    ax.axvline(R_TRAIN, color="grey", ls=":", lw=1)
    ax.set_xlabel("simulation rate r")
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes[2].axhline(0, color="red", ls="--", lw=1)
axes[0].legend()
fig.tight_layout()
fig.savefig("drift_robustness.png", dpi=150)
plt.close(fig)
print("Saved drift_robustness.csv and drift_robustness.png")
