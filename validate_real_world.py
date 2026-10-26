"""
validate_real_world.py

Work in Progress. Access to bankrupt company data required to do a full comparison.

Checks the model against reality rather than simulation, in three parts:

    A. Pricing consistency: with no jumps, mjd_pd should reproduce the
       classical Merton PD.
    B. Cross-section today: model PD vs the classical iterative Merton model
       (same barrier, no jumps) and vs Altman Z-score, by rank correlation.
    C. Default case studies: firms that filed for Chapter 11, scored with
       point-in-time data 1 year, 6 months and 3 months before filing, and
       ranked against industry peers on the same dates.

Part C is case-study evidence, not a statistical backtest: the sample is small
and yfinance coverage of bankrupt firms is patchy/nonexistent. Find this first.

Outputs:
    pricing_check.csv, cross_section.csv, cross_section.png, default_cases.csv

Environment variable: FRED_API_KEY.
"""
import os
import traceback

import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import merton_iterative, merton_pd
from credit_engine import (SCALE, THETA, fetch_inputs, first_value, load_bundle,
                           score_window, statement_column)
from pricing import mjd_pd

WINDOW = 120
KMV_DAYS = 252          # the classical model is usually calibrated on one year

# A spread of sectors and credit quality makes rank correlations meaningful.
CROSS_SECTION = ["BA", "AAPL", "DB", "APLD", "DJT", "MARA", "TENX", "NRXP",
                 "F", "T", "VZ", "KO", "PFE", "INTC", "CCL", "AAL", "JBLU",
                 "KSS", "M", "GT", "RIVN", "LCID", "PLUG"]

# Verify each ticker loads the bankrupt company's own history before trusting
# results: tickers of delisted firms are often changed or reused.
CASES = [
    {"name": "Bed Bath & Beyond", "ticker": "BBBYQ", "filed": "2023-04-23",
     "peers": ["BBY", "WSM", "TGT", "KSS", "M"]},
    {"name": "Yellow Corp",       "ticker": "YELLQ", "filed": "2023-08-06",
     "peers": ["ODFL", "SAIA", "XPO", "ARCB"]},
    {"name": "Rite Aid",          "ticker": "RADCQ", "filed": "2023-10-15",
     "peers": ["CVS", "WMT", "KR"]},
    {"name": "WeWork",            "ticker": "WEWKQ", "filed": "2023-11-06",
     "peers": ["CBRE", "JLL", "BXP", "VNO"]},
    {"name": "Spirit Airlines",   "ticker": "SAVEQ", "filed": "2024-11-18",
     "peers": ["JBLU", "ALK", "LUV", "ULCC", "AAL"]},
]
HORIZONS = [365, 180, 90]   # days before filing


# ------------------------------- Helpers ---------------------------------- #
def altman_z(tk, as_of, market_equity):
    """Original Altman (1968) Z-score"""
    bs, _ = statement_column(tk.balance_sheet, as_of)
    inc, _ = statement_column(tk.income_stmt, as_of)
    TA = first_value(bs, "Total Assets")
    TL = first_value(bs, "Total Liabilities Net Minority Interest", "Total Liabilities")
    WC = first_value(bs, "Working Capital")
    if np.isnan(WC):
        WC = first_value(bs, "Current Assets") - first_value(bs, "Current Liabilities")
    RE = first_value(bs, "Retained Earnings")
    EBIT = first_value(inc, "EBIT", "Operating Income")
    sales = first_value(inc, "Total Revenue")
    return (1.2 * WC / TA + 1.4 * RE / TA + 3.3 * EBIT / TA
            + 0.6 * market_equity / TL + 1.0 * sales / TA)


def score_firm(ticker, raw_r, bundle, as_of=None):
    """Model PD, classical Merton PD and Altman Z"""
    inp = fetch_inputs(ticker, raw_r, lookback_days=max(bundle["n_days"], KMV_DAYS), as_of=as_of)
    eq = inp["equity"].values
    res = score_window(eq, inp["D"], inp["r"].values, bundle, sector=inp["sector"])

    if len(eq) >= KMV_DAYS:
        k = merton_iterative(eq[-KMV_DAYS:], inp["D"], float(inp["r"].values[-1]), theta=THETA)
        res.update(KMV_PD_RN=k["pd_rn"], KMV_PD_Phys=k["pd_phys"],
                   KMV_Sigma=k["sigma"], KMV_Converged=k["converged"])
    try:
        res["Altman_Z"] = altman_z(inp["yf"], as_of, eq[-1] * SCALE)
    except Exception:
        res["Altman_Z"] = np.nan

    return {"Ticker": ticker, "Sector": inp["sector"], "E_over_D": eq[-1] / inp["D"],
            "Split_Factor": inp["split_factor"], **res}


def rank_corr(df, a, b, sign=1.0):
    sub = df[[a, b]].dropna()
    if len(sub) < 5:
        return np.nan, len(sub)
    return spearmanr(sub[a], sign * sub[b])[0], len(sub)


# ----------------------------- Part A: pricing ---------------------------- #
def pricing_check():
    rows = []
    for V in [0.8, 1.2, 2.0, 5.0]:
        for sigma in [0.1, 0.3, 0.6]:
            m = mjd_pd(V=V, D=1.0, mu=0.05, sigma=sigma, lam=0.0, mu_j=0.0, sigma_j=0.1, T=1.0)
            rows.append({"V/D": V, "sigma": sigma, "mjd_pd": m,
                         "merton_pd": merton_pd(V, 1.0, 0.05, sigma)})
    df = pd.DataFrame(rows)
    df["abs_diff"] = (df["mjd_pd"] - df["merton_pd"]).abs()
    df.to_csv("pricing_check.csv", index=False)
    print(f"\nA. Pricing check: max |mjd_pd - merton_pd| with no jumps = {df['abs_diff'].max():.2e}")
    if df["abs_diff"].max() > 1e-6:
        print("   Mismatch. Expected if mjd_pd uses a first-passage barrier rather than "
              "default at maturity. Otherwise look for a bug in pricing.py.")


# ------------------------- Part B: cross-section -------------------------- #
def cross_section(raw_r, bundle):
    rows = []
    for ticker in CROSS_SECTION:
        try:
            rows.append(score_firm(ticker, raw_r, bundle))
        except Exception as e:
            print(f"  {ticker}: skipped ({e})")
    df = pd.DataFrame(rows)
    df.to_csv("cross_section.csv", index=False)

    clean = df[~df["OOD"] & ~df["Financial"]]
    print("\nB. Cross-section rank agreement (Spearman)")
    for label, data in [("all firms", df), ("in-distribution, non-financial", clean)]:
        rho_k, n_k = rank_corr(data, "PD_Phys", "KMV_PD_Phys")
        rho_z, n_z = rank_corr(data, "PD_Phys", "Altman_Z", sign=-1.0)   # low Z = risky
        print(f"  {label:<32} vs classical Merton: {rho_k:.3f} (n={n_k})   "
              f"vs Altman Z: {rho_z:.3f} (n={n_z})")
    print(f"  {int(df['OOD'].sum())} of {len(df)} firms flagged out of distribution")

    fig, ax = plt.subplots(figsize=(7, 6))
    floor = 1e-8
    for ood, colour, label in [(False, "tab:blue", "in distribution"), (True, "tab:red", "OOD")]:
        sub = df[df["OOD"] == ood].dropna(subset=["KMV_PD_Phys"])
        ax.scatter(np.maximum(sub["KMV_PD_Phys"], floor), np.maximum(sub["PD_Phys"], floor),
                   c=colour, label=label)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"], (max(row["KMV_PD_Phys"], floor), max(row["PD_Phys"], floor)),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.plot([floor, 1], [floor, 1], "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("classical Merton PD (physical, no jumps)")
    ax.set_ylabel("model PD (physical, calibrated)")
    ax.set_title("Model vs classical Merton, current cross-section")
    ax.legend()
    fig.tight_layout()
    fig.savefig("cross_section.png", dpi=150)
    plt.close(fig)


# ------------------------- Part C: default cases -------------------------- #
def default_cases(raw_r, bundle):
    rows = []
    for case in CASES:
        filed = pd.Timestamp(case["filed"])
        for h in HORIZONS:
            as_of = filed - pd.Timedelta(days=h)
            try:
                target = score_firm(case["ticker"], raw_r, bundle, as_of=as_of)
            except Exception as e:
                print(f"  {case['name']} at -{h}d: could not score ({e}); check the ticker")
                continue
            peers = []
            for p in case["peers"]:
                try:
                    peers.append(score_firm(p, raw_r, bundle, as_of=as_of))
                except Exception as e:
                    print(f"    peer {p} at -{h}d skipped ({e})")
            group = pd.DataFrame([target] + peers)

            def rank_of_target(col, ascending=False):
                """1 = riskiest in the group."""
                if col not in group or pd.isna(group[col].iloc[0]):
                    return np.nan
                return int(group[col].rank(ascending=ascending, method="min").iloc[0])

            rows.append({
                "case": case["name"], "days_before": h, "as_of": as_of.date(),
                "group_size": len(group),
                "PD_Phys": target["PD_Phys"], "KMV_PD_Phys": target.get("KMV_PD_Phys", np.nan),
                "Altman_Z": target["Altman_Z"], "OOD": target["OOD"],
                "split_factor": target["Split_Factor"],
                "rank_model": rank_of_target("PD_Phys"),
                "rank_kmv": rank_of_target("KMV_PD_Phys"),
                "rank_altman": rank_of_target("Altman_Z", ascending=True),
            })

    df = pd.DataFrame(rows)
    df.to_csv("default_cases.csv", index=False)
    if df.empty:
        print("\nC. No default cases could be scored; check the tickers in CASES.")
        return
    print("\nC. Default case studies (rank 1 = riskiest in peer group)")
    print(df.drop(columns=["as_of"]).to_string(index=False))
    for col in ["rank_model", "rank_kmv", "rank_altman"]:
        valid = df[col].dropna()
        print(f"  {col:<12} riskiest in group: {(valid == 1).sum()}/{len(valid)}")
    if (df["split_factor"] != 1).any():
        print("  Note: split adjustments applied for some firms. Check that the implied market "
              "caps look right. If Yahoo already restated share counts, they are double-adjusted.")


# ---------------------------------- Main ----------------------------------- #
if __name__ == "__main__":
    fred = Fred(api_key=os.environ.get("FRED_API_KEY"))
    raw_r = fred.get_series("DGS5")
    bundle = load_bundle(WINDOW)

    for part in (pricing_check, lambda: cross_section(raw_r, bundle),
                 lambda: default_cases(raw_r, bundle)):
        try:
            part()
        except Exception:
            traceback.print_exc()
