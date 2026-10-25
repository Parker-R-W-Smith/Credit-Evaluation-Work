# Credit-Evaluation-Work

A repository for continual work on a credit default tester and tracker.

This project estimates one-year default probabilities for publicly traded companies using a **Merton jump-diffusion (MJD) structural credit model**, with the model's hidden parameters inferred by a neural network trained entirely on simulated data (*amortized inference*). A daily evaluator scores a portfolio of tickers, flags results the model may not be qualified to produce, alerts on high-risk names, and posts a PDF tear sheet to Discord.

> **Status:** active work in progress. The simulation-based training and validation pipeline is complete; real-world validation against historical defaults is not (see [Real-world validation](#real-world-validation-speculative)). Nothing here is investment advice.

---

## How it works

Structural credit models treat a company's equity as a call option on its assets: if asset value falls below the firm's debt at the horizon, shareholders are wiped out and the firm defaults. Asset value and asset volatility can't be observed directly, so they have to be backed out from the equity price. This project extends the classical Merton model with **jumps** (sudden drops in asset value), which better capture the abrupt repricings seen before many real defaults.

The pipeline runs in five steps:

1. **Simulate.** `Amortized_Inference_Data_Generator.py` draws random parameter sets (asset-to-debt ratio, asset volatility, jump intensity, mean and spread of jump size), simulates asset paths under jump-diffusion, and converts them into equity paths with the Merton (1976) jump-diffusion option price.
2. **Learn the inverse.** `model.py` defines `SDEPredictor`, a small feed-forward network that takes a window of equity-to-debt ratios and predicts the five parameters `(V/D, σ_V, λ, μ_J, σ_J)`. Besides the raw window, it sees daily log returns plus realized variance, bipower variation, jump variation and the largest single move, which help separate diffusion from jumps.
3. **Price default.** `pricing.py` turns predicted parameters into a default probability by summing over the possible number of jumps in a year (a Poisson mixture of normal distributions).
4. **Calibrate.** An isotonic regression, fit on a held-out validation set, corrects systematic bias between the PD implied by the predicted parameters and the true PD.
5. **Score real firms.** `credit_engine.py` pulls equity prices and balance sheets from Yahoo Finance and the risk-free rate from FRED, feeds the most recent window through the network, and returns calibrated PDs together with reliability flags.

Because inference is a single forward pass, scoring a firm is near-instant once the network is trained, with no per-firm optimization loop.

## Repository layout

| File | Purpose |
|---|---|
| `Amortized_Inference_Data_Generator.py` | Simulates MJD asset paths and the corresponding equity paths used as training data. |
| `model.py` | `SDEPredictor` network, the parameter prior bounds (`PRIOR_BOUNDS`), and `clip_to_prior`. |
| `pricing.py` | `mjd_pd` (default probability under MJD) and `merton76_call` (equity as a jump-diffusion call on assets). |
| `train_and_validate.py` | Trains one model per window length (30, 60, 120, 250 trading days), runs parameter-recovery and PD diagnostics on a test set, fits the PD calibrator, and saves checkpoints. |
| `add_ood_stats.py` | Adds out-of-distribution statistics to existing checkpoints without retraining. |
| `ood.py` | Out-of-distribution checks: per-feature bounds and Mahalanobis distance on input windows, plus detection of predictions that had to be clipped to the prior. |
| `robustness_drift.py` | Tests how recovery, PD ranking and calibration degrade when the interest rate differs from the 5% used in training. |
| `baselines.py` | Classical Merton model (no jumps, no neural network) using the iterative calibration of Vassalou & Xing (2004) and Bharath & Shumway (2008). |
| `credit_engine.py` | Shared data access and scoring used by both the daily evaluator and the validation scripts. |
| `credit_evaluator.py` | Daily run: scores the portfolio, sends Discord alerts and generates the PDF tear sheet. |
| `validate_real_world.py` | **Speculative.** Real-world validation framework; see below. |
| `mjd_amortized_weights.pth` | Legacy checkpoint from an earlier version of the model; see [Note on the included checkpoint](#note-on-the-included-checkpoint). |
| `Requirements.txt` | Python dependencies. |

## Installation

Python 3.9 or later is recommended.

```bash
git clone https://github.com/Parker-R-W-Smith/Credit-Evaluation-Work.git
cd Credit-Evaluation-Work
pip install numpy pandas torch scipy scikit-learn matplotlib joblib fpdf fredapi yfinance requests
```

Note that `datetime` and `traceback` are part of the Python standard library and do not need to be installed, and scikit-learn is installed as `scikit-learn` (the `sklearn` package name on PyPI is deprecated).

### API keys

The live scripts read two environment variables:

| Variable | Used by | Purpose |
|---|---|---|
| `FRED_API_KEY` | `credit_evaluator.py`, `validate_real_world.py` | Downloads the 5-year Treasury yield (`DGS5`) as the risk-free rate. A free key is available from the [St. Louis Fed](https://fred.stlouisfed.org/docs/api/api_key.html). |
| `DISCORD_WEBHOOK` | `credit_evaluator.py` | Webhook URL for alerts and the daily tear sheet. If it is missing, Discord posting is skipped and everything else still runs. |

```bash
export FRED_API_KEY="your_key_here"
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
```

## Usage

### 1. Train the models

```bash
python train_and_validate.py
```

This simulates 120,000 paths for each window length, trains for 30 epochs, keeps the checkpoint with the best validation loss, and writes:

| Output | Contents |
|---|---|
| `mjd_weights_{30,60,120,250}d.pth` | Model weights, target scaling and window length. |
| `pd_calibrator_{30,60,120,250}d.joblib` | Isotonic PD calibrator for each window. |
| `recovery_{t}d.png` | Predicted vs true value for each parameter, plus loss curves. |
| `pd_check_{t}d.png` | Predicted vs true default probability on log axes. |
| `window_sweep.csv`, `window_sweep.png` | Recovery quality (R² per parameter, PD rank correlation) as a function of window length. |

The console output also reports skill over a predict-the-average baseline, whether the predicted jump parameters actually improve PD accuracy compared with plugging in prior means, PD calibration by bucket, and alert precision and recall at the 30% threshold.

### 2. Add out-of-distribution statistics

```bash
python add_ood_stats.py
```

This adds the input-distribution bounds to each checkpoint. Without them the evaluator still runs but prints a warning and skips the input checks. On held-out simulated data, roughly 5% of windows should be flagged.

### 3. (Optional) Check sensitivity to interest rates

```bash
python robustness_drift.py
```

The network and calibrator are trained at r = 5%, but live scoring uses the current Treasury yield and the physical PD uses a higher drift. This script re-simulates test sets at rates from 0% to 12% and writes `drift_robustness.csv` and `drift_robustness.png`.

### 4. Run the daily evaluator

```bash
python credit_evaluator.py
```

For each ticker in `PORTFOLIO`, the evaluator prints the estimates and any flags, posts a Discord alert if the physical PD exceeds the threshold, and finally uploads `Daily_Risk_Report.pdf`. The script is self-contained and can be scheduled with cron, Task Scheduler or a CI workflow.

## Configuration

The main settings are constants at the top of each file:

| Setting | File | Default | Meaning |
|---|---|---|---|
| `PORTFOLIO` | `credit_evaluator.py` | `BA, AAPL, DB, APLD, DJT, MARA, TENX, NRXP` | Tickers scored each day. |
| `WINDOW` | `credit_evaluator.py` | `120` | Trading-day window; must match a trained checkpoint. |
| `ALERT_THRESHOLD` | `credit_evaluator.py` | `0.30` | Physical PD above which an alert is sent. |
| `THETA` | `credit_engine.py` | `0.4` | Assumed asset Sharpe ratio; physical drift = r + θ·σ_V. |
| `REPORTING_LAG_DAYS` | `credit_engine.py` | `90` | Days after fiscal year end before a statement counts as public (point-in-time mode). |
| `PRIOR_BOUNDS` | `model.py` | see file | Parameter ranges for simulation and clipping. |

## Model details

**Default barrier.** Following the KMV convention, debt is short-term debt plus half of long-term debt, taken from the most recent annual balance sheet.

**Horizon.** Default is tested at a one-year horizon (T = 1), as in the original Merton model.

**Two default probabilities.** `PD_RN` is the risk-neutral PD, computed with the risk-free rate as the asset drift. `PD_Phys` is the physical, real-world PD, computed with drift r + θ·σ_V and generally lower. Alerts use `PD_Phys`. Both are reported raw and after calibration.

**Calibration caveat.** The isotonic calibrator was fit on risk-neutral PDs at r = 5%. Applying it to `PD_Phys` assumes the network's estimation error doesn't depend on the drift; `robustness_drift.py` gives an approximate read on whether that holds.

**Reliability flags.** Every score carries flags that appear in the console, in the Discord alert and in the tear sheet's notes section:

- **OOD**: the input window looks unlike the simulated training data (unusual leverage, volatility, largest move, tail heaviness, or an atypical combination of these), or the network predicted parameters outside its own prior. PDs for these firms are extrapolations.
- **FIN**: the firm is in the Financial Services sector. A debt-barrier model is poorly suited to banks and insurers, whose liabilities are their operating business rather than financing.

## Validation

### Simulation-based (complete)

`train_and_validate.py` and `robustness_drift.py` evaluate the estimator on simulated firms whose true parameters and PDs are known. This establishes how well the network recovers each parameter, how well it ranks and calibrates PDs, how alert precision and recall behave at the threshold, and how all of these change with window length and interest rate. It does **not** establish that the MJD model describes real companies well.

### Real-world validation (speculative)

`validate_real_world.py` is a framework for testing the model against reality. It is left in the repository as a **speculative piece and is not currently effective**: the key test requires historical market and financial-statement data for companies that went bankrupt, and that data sits behind premium providers (for example CRSP/Compustat via WRDS). Yahoo Finance's coverage of delisted and bankrupt firms is patchy or missing, and delisted tickers are often renamed or reused, so the results cannot be trusted as they stand.

The script is organized in three parts:

- **A. Pricing consistency.** Checks that `mjd_pd` with jumps switched off reproduces the classical Merton PD. This is a pure numerical check and needs no external data.
- **B. Cross-section today.** Compares the model's PD ranking for a list of current firms with the classical iterative Merton model and the Altman Z-score. This runs on live Yahoo Finance data, but measures agreement with other models, not accuracy against actual defaults.
- **C. Default case studies.** Scores firms that filed for Chapter 11 (Bed Bath & Beyond, Yellow, Rite Aid, WeWork, Spirit Airlines) using point-in-time data 12, 6 and 3 months before filing, and ranks each against industry peers on the same dates. **This is the part blocked by data access.**

Even with full data access, Part C as written would be case-study evidence rather than a statistical backtest; a proper backtest would also need a default database and a much larger sample. The code is kept so that it can be pointed at a premium data source if one becomes available.

## Limitations

- **Trained on simulation only.** The network has never seen a real firm. Its outputs are meaningful only to the extent that real equity behaves like the simulated MJD process.
- **Fixed training rate.** Training uses r = 5%; live rates differ.
- **Annual statements.** The barrier comes from annual balance sheets, so it can be up to a year out of date.
- **Data quality.** Yahoo Finance fields are sometimes missing or inconsistently named. Historical share counts are split-adjusted in point-in-time mode, which can double-adjust if Yahoo has already restated them.
- **Financial firms.** Scores are produced but flagged, and should not be relied on.

## References

- Merton, R. C. (1974). On the pricing of corporate debt: The risk structure of interest rates. *Journal of Finance*, 29(2).
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2).
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4).
- Vassalou, M., & Xing, Y. (2004). Default risk in equity returns. *Journal of Finance*, 59(2).
- Bharath, S. T., & Shumway, T. (2008). Forecasting default with the Merton distance to default model. *Review of Financial Studies*, 21(3).
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps. *Journal of Financial Econometrics*, 2(1).

## License

Released under the GNU General Public License v3.0. See [`LICENSE`](LICENSE).
