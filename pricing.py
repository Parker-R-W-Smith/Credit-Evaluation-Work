# Pricing Functions

# Libraries
import numpy as np
from scipy.stats import norm, poisson

def mjd_pd(V, D, mu, sigma, lam, mu_j, sigma_j, T=1.0, tol=1e-12):
    """
    Default probability under Merton jump-diffusion asset dynamics.
    mu  : expected total return of assets (r for risk-neutral, r + premium for physical)
    lam : annualized jump intensity; mu_j, sigma_j: mean/sd of log jump size
    """
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1            # expected relative jump size
    n = np.arange(int(poisson.ppf(1 - tol, lam * T)) + 2)
    w = poisson.pmf(n, lam * T)
  

    # Conditional on n jumps, ln(V_T/D) is normal with this mean and sd
    mean = np.log(V / D) + (mu - 0.5 * sigma**2 - lam * k) * T + n * mu_j
    sd = np.sqrt(sigma**2 * T + n * sigma_j**2)

    return float(np.sum(w * norm.cdf(-mean / sd)))

def merton76_call(V, D, T, r, sigma, lam, mu_j, sigma_j, tol=1e-12):
    """
    Merton (1976) jump-diffusion call price: equity value as a call on firm assets.
    Broadcasts: V can be (N, days). Per-sample params should be (N, 1).
    """
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    lam_p = lam * (1 + k)

    n_max = int(poisson.ppf(1 - tol, np.max(lam_p) * T)) + 2
    price = np.zeros(np.broadcast(V, sigma).shape)

    for n in range(n_max):
        w = poisson.pmf(n, lam_p * T)
        sig_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
        r_n = r - lam * k + n * np.log1p(k) / T

        d1 = (np.log(V / D) + (r_n + 0.5 * sig_n**2) * T) / (sig_n * np.sqrt(T))
        d2 = d1 - sig_n * np.sqrt(T)
        price += w * (V * norm.cdf(d1) - D * np.exp(-r_n * T) * norm.cdf(d2))

    return price
