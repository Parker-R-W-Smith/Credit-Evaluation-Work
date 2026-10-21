import math
import numpy as np
import torch
import torch.nn as nn

# Prior support for (V/D, sigma_V, lambda, mu_J, sigma_J).
# Single source of truth: the data generator samples from these ranges and
# predictions are clipped to them, so the two can never drift apart.
PARAM_NAMES = ["V/D", "sigma_V", "lambda", "mu_J", "sigma_J"]
PRIOR_BOUNDS = np.array([
    [0.5,  10.0],   # V/D: below 1 is allowed, since Merton only tests default at T.
                    #      Raise the upper bound if real firms' E/D ratios exceed ~9.
    [0.05, 0.60],   # sigma_V
    [0.1,  5.0],    # lambda (jumps per year)
    [-0.3, 0.1],    # mu_J   (mean log jump size)
    [0.05, 0.25],   # sigma_J
])


def clip_to_prior(params):
    """Clip parameters to the prior's support. Works on shape (5,) or (N, 5)."""
    return np.clip(params, PRIOR_BOUNDS[:, 0], PRIOR_BOUNDS[:, 1])


class SDEPredictor(nn.Module):
    """Predicts standardized (V/D, sigma_V, lambda, mu_J, sigma_J) from an
    n_days window of equity-to-debt ratios. Optionally augments the raw window
    with log returns and realized/bipower variation summary statistics."""

    N_SUMMARY = 4    # realized var, bipower var, jump var, max |return|
    X_FLOOR = 1e-4   # smallest equity-to-debt ratio the features will see
    MAX_RET = 1.0    # cap on |daily log return|; keeps near-zero equity from exploding

    def __init__(self, n_days=30, use_features=True):
        super().__init__()
        self.n_days = n_days
        self.use_features = use_features
        n_in = n_days + ((n_days - 1) + self.N_SUMMARY if use_features else 0)

        self.input_norm = nn.BatchNorm1d(n_in)   # puts levels and returns on a common scale
        self.network = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5),
        )

    @classmethod
    def features(cls, x):
        r = torch.diff(torch.log(x.clamp_min(cls.X_FLOOR)), dim=1)  # daily log returns
        r = r.clamp(-cls.MAX_RET, cls.MAX_RET)                       # bound extreme moves
        rv = (r ** 2).sum(dim=1, keepdim=True)                       # realized variance
        bv = (math.pi / 2) * (r[:, 1:].abs() * r[:, :-1].abs()).sum(dim=1, keepdim=True)
        jv = (rv - bv).clamp_min(0.0)                                # jump variation
        max_abs = r.abs().max(dim=1, keepdim=True).values            # largest single move
        return torch.cat([r, rv, bv, jv, max_abs], dim=1)

    def forward(self, x):
        if self.use_features:
            x = torch.cat([x, self.features(x)], dim=1)
        return self.network(self.input_norm(x))
