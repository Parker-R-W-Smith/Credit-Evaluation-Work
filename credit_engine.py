"""
credit_engine.py

Data access and scoring shared by the daily evaluator and the validation
scripts.
"""
import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from model import SDEPredictor, clip_to_prior
from ood import check_params, check_window
from pricing import mjd_pd

SCALE = 1e9                  # report values in billions
THETA = 0.4                  # assumed asset Sharpe ratio: physical drift = r + THETA * sigma_V
FLOOR = 1e-4
REPORTING_LAG_DAYS = 90      # annual statements treated as public 90 days after fiscal year end
FINANCIAL_SECTORS = {"Financial Services"}


# ------------------------------ Model loading ------------------------------ #
def load_bundle(n_days):
    """Model, target scaling, calibrator and OOD statistics for one window length."""
    ckpt = torch.load(f"mjd_weights_{n_days}d.pth", weights_only=True)
    model = SDEPredictor(int(ckpt["n_days"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ood = ckpt.get("ood")
    if ood is None:
        print("Warning: checkpoint has no OOD statistics (run add_ood_stats.py). "
              "Input-distribution checks are disabled.")
    return {
        "model": model,
        "n_days": int(ckpt["n_days"]),
        "Y_mean": ckpt["Y_mean"],
        "Y_std": ckpt["Y_std"],
        "calibrator": joblib.load(f"pd_calibrator_{n_days}d.joblib"),
        "ood": ood,
    }


# ---------------------------- Statement helpers ---------------------------- #
def first_value(row, *names):
    """First non-missing value among several possible yfinance row names."""
    for name in names:
        value = row.get(name, np.nan)
        if pd.notna(value):
            return float(value)
    return np.nan


def statement_column(frame, as_of=None):
    """
    Most recent statement column. With as_of, only statements that would have
    been public by then (fiscal date + reporting lag), to avoid look-ahead.
    """
    if frame is None or frame.empty:
        raise ValueError("financial statement unavailable")
    dates = pd.to_datetime(frame.columns)
    if as_of is None:
        pos = int(np.argmax(dates))
    else:
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=REPORTING_LAG_DAYS)
        eligible = np.where(dates <= cutoff)[0]
        if len(eligible) == 0:
            raise ValueError(f"no statement public by {pd.Timestamp(as_of):%Y-%m-%d}")
        pos = int(eligible[np.argmax(dates[eligible])])
    return frame.iloc[:, pos], dates[pos]


def kmv_barrier(bs):
    """Default barrier: short-term debt + half of long-term debt (KMV convention)."""
    short = first_value(bs, "Current Debt")
    if np.isnan(short) or short == 0:
        short = first_value(bs, "Short Long Term Debt")
    long_term = first_value(bs, "Long Term Debt", "Total Debt")
    D = np.nan_to_num(short) + 0.5 * np.nan_to_num(long_term)
    if D <= 0:
        raise ValueError("no debt found on balance sheet")
    return D


def _info(tk):
    try:
        return tk.info or {}
    except Exception:
        return {}


def _splits_after(tk, date):
    """
    Product of split ratios after `date`. Yahoo prices are split-adjusted to
    today's share basis, while historical share counts are not, so the share
    count is multiplied by this factor to keep market cap consistent.
    """
    splits = tk.splits
    if splits is None or splits.empty:
        return 1.0
    idx = splits.index.tz_localize(None) if splits.index.tz is not None else splits.index
    after = splits.values[idx > date]
    return float(np.prod(after)) if len(after) else 1.0


# ------------------------------- Data fetch -------------------------------- #
def fetch_inputs(ticker, raw_r, lookback_days, as_of=None):
    """
    Equity path (bn), rate path, barrier (bn) and sector for one firm.
    as_of=None: live. as_of=date: point-in-time, using only data available then.
    """
    tk = yf.Ticker(ticker)
    info = _info(tk)
    end = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(lookback_days * 1.6) + 30)

    # auto_adjust=False. Close is split-adjusted but not dividend-adjusted, so Close * shares is market cap.
    hist = tk.history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=False)
    if hist.empty:
        raise ValueError("no price history (delisted tickers are often renamed or reused)")
    hist.index = hist.index.tz_localize(None)

    bs, bs_date = statement_column(tk.balance_sheet, as_of)
    D = kmv_barrier(bs)

    split_factor = 1.0
    if as_of is None:
        shares = info.get("sharesOutstanding")
    else:
        shares = first_value(bs, "Ordinary Shares Number", "Share Issued")
        split_factor = _splits_after(tk, bs_date)
        shares = shares * split_factor
    if shares is None or not np.isfinite(shares) or shares <= 0:
        raise ValueError("shares outstanding unavailable")

    r = (raw_r.dropna() / 100).reindex(hist.index, method="ffill")
    frame = pd.DataFrame({"equity": hist["Close"] * shares / SCALE, "r": r}).dropna()

    return {
        "ticker": ticker,
        "equity": frame["equity"],
        "r": frame["r"],
        "D": D / SCALE,
        "sector": info.get("sector"),
        "statement_date": bs_date,
        "split_factor": split_factor,
        "yf": tk,
    }


# --------------------------------- Scoring --------------------------------- #
def score_window(equity, D, r, bundle, sector=None):
    """Model PDs plus reliability flags for one firm's equity path."""
    n = bundle["n_days"]
    equity = np.asarray(equity, dtype=np.float64)
    if len(equity) < n:
        raise ValueError(f"Insufficient aligned data. Found {len(equity)} days, need {n}.")
    x = equity[-n:] / D
    X = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    # Predict, un-standardize, and clip exactly as in training
    with torch.no_grad():
        scaled = bundle["model"](X).squeeze(0)
    raw = (scaled * bundle["Y_std"] + bundle["Y_mean"]).numpy()
    clipped = clip_to_prior(raw[None, :])[0]

    # Reliability Checks
    flags = []
    if bundle["ood"] is not None:
        flags += check_window(x, bundle["ood"])
    flags += check_params(raw, clipped)
    financial = sector in FINANCIAL_SECTORS

    V_ratio, sigma_v, lam, mu_j, sigma_j = clipped
    V_ratio = max(V_ratio, FLOOR)
    sigma_v = max(sigma_v, FLOOR)
    lam = max(lam, 0.0)
    sigma_j = max(sigma_j, FLOOR)

    current_r = float(np.asarray(r)[-1])
    mu_phys = current_r + THETA * sigma_v
    params = dict(V=V_ratio * D, D=D, sigma=sigma_v, lam=lam,
                  mu_j=mu_j, sigma_j=sigma_j, T=1.0)
    pd_rn_raw = mjd_pd(mu=current_r, **params)
    pd_phys_raw = mjd_pd(mu=mu_phys, **params)

    # The isotonic calibrator was fit on risk-neutral PDs (r = 5%) from the
    # validation set. It corrects parameter-estimation error. Applying it to
    # PD_phys assumes that error does not depend on the drift used to compute
    # the PD. Note that robustness_drift.py tests this assumption.
    cal = bundle["calibrator"]
    pd_rn = float(cal.predict([pd_rn_raw])[0])
    pd_phys = float(cal.predict([pd_phys_raw])[0])

    notes = list(flags)
    if financial:
        notes.append("financial firm: debt-barrier model ill-suited to banks and insurers")

    return {
        "Est_Asset_Value_Bn": V_ratio * D,
        "Est_Volatility": sigma_v,
        "Jump_Intensity": lam,
        "PD_RN_Raw": pd_rn_raw,
        "PD_RN": pd_rn,
        "PD_Phys_Raw": pd_phys_raw,
        "PD_Phys": pd_phys,
        "OOD": bool(flags),
        "Financial": financial,
        "Flags": "; ".join(notes),
    }
