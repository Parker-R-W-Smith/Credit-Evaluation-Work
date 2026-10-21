"""
ood.py

Out-of-distribution checks for the amortized MJD estimator.

The network was trained only on simulated equity/debt paths. When a real
firm's window looks unlike anything in training, the network extrapolates and
its output has no validated meaning. Two checks guard against this:

    check_window  compares summary features of the input window with the
                  training inputs (per-feature bounds plus a joint
                  Mahalanobis distance).
    check_params  flags parameters that clip_to_prior had to correct, i.e.
                  the network predicted values outside its own prior.

Assumes the generator's X holds equity/D level paths, which is what the
inference script feeds the network.
"""
import numpy as np

FEATURE_NAMES = ["log_start_level", "log_change", "daily_vol",
                 "max_abs_return", "excess_kurtosis"]
PARAM_NAMES = ["V/D", "sigma_V", "lambda", "mu_J", "sigma_J"]


def _np(v):
    """Checkpoint values are tensors (weights_only loading); work in NumPy."""
    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)


def window_features(X):
    """Summary features of one or more E/D paths. X: (days,) or (n, days)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    logX = np.log(np.maximum(X, 1e-8))
    rets = np.diff(logX, axis=1)
    centred = rets - rets.mean(axis=1, keepdims=True)
    var = (centred ** 2).mean(axis=1)
    kurt = (centred ** 4).mean(axis=1) / np.maximum(var ** 2, 1e-24) - 3.0
    return np.column_stack([
        logX[:, 0],                       # starting leverage (log E/D)
        logX[:, -1] - logX[:, 0],         # change over the window
        rets.std(axis=1, ddof=1),         # daily volatility
        np.abs(rets).max(axis=1),         # largest single-day move (jumps)
        kurt,                             # tail heaviness
    ])


def _mahalanobis(F, mean, prec):
    d = F - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))


def fit_ood_stats(X, q=0.005):
    """Bounds from training inputs. Stored as tensors so torch.load(weights_only=True) accepts them."""
    import torch
    F = window_features(X)
    F = F[np.isfinite(F).all(axis=1)]
    lo, hi = np.quantile(F, [q, 1 - q], axis=0)
    mean = F.mean(axis=0)
    prec = np.linalg.pinv(np.cov(F, rowvar=False))
    maha_max = np.quantile(_mahalanobis(F, mean, prec), 1 - q)
    as_t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    return {"names": list(FEATURE_NAMES), "lo": as_t(lo), "hi": as_t(hi),
            "mean": as_t(mean), "prec": as_t(prec), "maha_max": as_t(maha_max)}


def check_window(x, stats):
    """Plain-text flags for one window (empty list = in distribution)."""
    f = window_features(x)[0]
    if not np.isfinite(f).all():
        return ["window contains non-finite values"]
    flags = []
    for name, v, a, b in zip(stats["names"], f, _np(stats["lo"]), _np(stats["hi"])):
        if v < a or v > b:
            flags.append(f"{name}={v:.3g} outside training range [{a:.3g}, {b:.3g}]")
    d = _mahalanobis(f[None, :], _np(stats["mean"]), _np(stats["prec"]))[0]
    if d > float(_np(stats["maha_max"])):
        flags.append(f"feature combination atypical (Mahalanobis {d:.1f} > {float(_np(stats['maha_max'])):.1f})")
    return flags


def check_params(raw, clipped, rtol=1e-3):
    """Flag parameters the network predicted outside its prior."""
    flags = []
    for name, a, b in zip(PARAM_NAMES, np.asarray(raw, float), np.asarray(clipped, float)):
        if abs(a - b) > rtol * (1.0 + abs(a)):
            flags.append(f"{name} predicted {a:.3g}, clipped to {b:.3g}")
    return flags


def flag_rate(X, stats):
    """Share of windows flagged; on held-out simulated data this should be a few percent."""
    return float(np.mean([bool(check_window(x, stats)) for x in np.atleast_2d(X)]))
