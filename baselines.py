"""
baselines.py

Classical Merton model calibrated with the iterative procedure of
Vassalou & Xing (2004) and Bharath & Shumway (2008): no jumps, no neural
network. It uses the same barrier (short-term debt + half of long-term debt),
the same horizon and the same physical-drift convention as the main model, so
any disagreement comes from the jumps and the amortized estimation.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

TRADING_DAYS = 252


def bs_call(V, D, r, sigma, T):
    """Equity as a European call on firm assets."""
    sqrt_t = np.sqrt(T)
    d1 = (np.log(V / D) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)


def merton_pd(V, D, mu, sigma, T=1.0):
    """Probability that assets end below the barrier at T under drift mu."""
    d2 = (np.log(V / D) + (mu - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(-d2))


def implied_asset_value(E, D, r, sigma, T):
    """Invert the call price for V. The call lies between V - D e^{-rT} and V, which brackets the root."""
    pv_debt = D * np.exp(-r * T)
    hi = (E + pv_debt) * (1 + 1e-9)
    return brentq(lambda V: bs_call(V, D, r, sigma, T) - E, E, hi, xtol=1e-12 * hi)


def merton_iterative(E, D, r, T=1.0, theta=0.4, tol=1e-5, max_iter=200):
    """
    E: daily equity values (same units as D), ideally one year.
    Returns asset value, asset volatility and risk-neutral / physical PD.
    """
    E = np.asarray(E, dtype=np.float64)
    dt = 1.0 / TRADING_DAYS
    sigma_e = np.diff(np.log(E)).std(ddof=1) / np.sqrt(dt)
    sigma = max(sigma_e * E[-1] / (E[-1] + D), 1e-4)      # standard starting guess

    converged, it = False, 0
    for it in range(1, max_iter + 1):
        V = np.array([implied_asset_value(e, D, r, sigma, T) for e in E])
        new_sigma = max(np.diff(np.log(V)).std(ddof=1) / np.sqrt(dt), 1e-4)
        if abs(new_sigma - sigma) < tol:
            sigma, converged = new_sigma, True
            break
        sigma = new_sigma

    V_now = implied_asset_value(E[-1], D, r, sigma, T)
    return {
        "V": V_now,
        "sigma": sigma,
        "pd_rn": merton_pd(V_now, D, r, sigma, T),
        "pd_phys": merton_pd(V_now, D, r + theta * sigma, sigma, T),
        "converged": converged,
        "iterations": it,
    }
