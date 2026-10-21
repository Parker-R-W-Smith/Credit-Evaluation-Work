"""
add_ood_stats.py

Adds out-of-distribution statistics to existing checkpoints without
retraining. The bounds describe the generator's input distribution, so a fresh
simulated sample from the same prior gives the same bounds as the training set.
"""
import numpy as np
import torch

from Amortized_Inference_Data_Generator import generate_mjd_dataset
from ood import fit_ood_stats, flag_rate

SEED = 42
R_TRAIN = 0.05
N_FIT, N_CHECK = 20_000, 2_000
WINDOWS = [30, 60, 120, 250]

for t in WINDOWS:
    path = f"mjd_weights_{t}d.pth"
    ckpt = torch.load(path, weights_only=True)

    np.random.seed(SEED)
    X, _ = generate_mjd_dataset(n_samples=N_FIT + N_CHECK, days=t, r=R_TRAIN)
    stats = fit_ood_stats(X[:N_FIT])

    # Sanity check on simulated windows not used to fit the bounds.
    # Expect roughly 5%: several features each cut at the 0.5% / 99.5% quantiles.
    print(f"{t}d: held-out simulated windows flagged: {flag_rate(X[N_FIT:], stats):.1%}")

    ckpt["ood"] = stats
    torch.save(ckpt, path)
    print(f"Updated {path}")
