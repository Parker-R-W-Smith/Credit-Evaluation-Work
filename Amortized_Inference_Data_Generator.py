# Import required libraries

## Numeric Libraries
from scipy.stats import norm, poisson
import numpy as np
from model import SDEPredictor
from pricing import merton76_call
# pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def generate_mjd_dataset(n_samples=100000, days=30, D_scaled=31.09, r=0.05):
    dt = 1 / 252

    # Sample Latent Parameters
    V_0 = np.random.uniform(D_scaled * 1.1, D_scaled * 10, n_samples)
    sigma_v = np.random.uniform(0.05, 0.60, n_samples)
    lambda_j = np.random.uniform(0.1, 5.0, n_samples)
    mu_j = np.random.uniform(-0.3, 0.1, n_samples)
    sigma_j = np.random.uniform(0.05, 0.25, n_samples)
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1 

    V_paths = np.zeros((n_samples, days))
    V_paths[:, 0] = V_0

    for t in range(1, days):
        dW = np.random.normal(0, np.sqrt(dt), n_samples)
        dN = np.random.poisson(lambda_j * dt, n_samples)
        Z = np.random.normal(0, 1, n_samples)
        J = mu_j * dN + sigma_j * np.sqrt(dN) * Z

        drift = (r - 0.5 * sigma_v**2 - lambda_j * k) * dt
        diffusion = sigma_v * dW
        V_paths[:, t] = V_paths[:, t-1] * np.exp(drift + diffusion + J)

    Equity_paths = merton76_call(
    V_paths, D_scaled, T=1.0, r=r,
    sigma=sigma_v[:, None], lam=lambda_j[:, None],
    mu_j=mu_j[:, None], sigma_j=sigma_j[:, None],
)

    # Normalize Inputs and Outputs
    X = Equity_paths / D_scaled  # Network sees Equity-to-Debt ratio
    V_ratio = V_paths[:, -1] / D_scaled     # Network predicts V-to-Debt ratio

    Y = np.column_stack((V_ratio, sigma_v, lambda_j, mu_j, sigma_j))
    return X, Y


