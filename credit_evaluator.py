"""
credit_evaluator.py

Daily credit-risk run: scores a portfolio with the amortized MJD estimator,
flags inputs the model was not trained on, alerts on high physical PD, and
posts a PDF tear sheet to Discord.

Environment variables: FRED_API_KEY, DISCORD_WEBHOOK.
"""
import datetime
import os
import traceback

import pandas as pd
import requests
from fpdf import FPDF
from fredapi import Fred

from credit_engine import fetch_inputs, load_bundle, score_window

WINDOW = 120             # trading days (chosen from window_sweep.csv)
ALERT_THRESHOLD = 0.30   # alert precision/recall at this level is reported by train_and_validate.py
PORTFOLIO = ["BA", "AAPL", "DB", "APLD", "DJT", "MARA", "TENX", "NRXP"]


# ----------------------------- Notifications ------------------------------ #
def post_discord(message, pdf_path=None):
    """Post a message, optionally with a PDF attachment. Discord returns 200 or 204 on success."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK")
    if not webhook_url or not webhook_url.startswith("http"):
        print("Error: invalid or missing DISCORD_WEBHOOK. Skipping Discord post.")
        return
    if pdf_path is None:
        response = requests.post(webhook_url, json={"content": message})
    else:
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            response = requests.post(webhook_url, data={"content": message}, files=files)
    if response.status_code not in (200, 204):
        print(f"Discord post failed. Status code: {response.status_code}")


# ------------------------------- Tear sheet -------------------------------- #
def generate_tear_sheet(df, filename="Daily_Risk_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    date_str = datetime.date.today().strftime("%B %d, %Y")
    pdf.cell(0, 10, txt=f"Quantitative Credit Risk Tear Sheet - {date_str}", ln=True, align="C")
    pdf.ln(10)

    cols = ["Ticker", "Asset Val (Bn)", "Volatility", "Jump Int.",
            "PD (RN)", "PD (Phys)", "PD (Phys, Raw)", "Flag"]
    widths = [20, 26, 22, 22, 26, 26, 28, 20]
    pdf.set_font("Arial", "B", 9)
    for col, width in zip(cols, widths):
        pdf.cell(width, 10, col, border=1, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 9)
    for _, row in df.iterrows():
        pdf.cell(widths[0], 10, str(row["Ticker"]), border=1, align="C")
        pdf.cell(widths[1], 10, f"{row['Est_Asset_Value_Bn']:.2f}", border=1, align="C")
        pdf.cell(widths[2], 10, f"{row['Est_Volatility']:.4f}", border=1, align="C")
        pdf.cell(widths[3], 10, f"{row['Jump_Intensity']:.4f}", border=1, align="C")
        pdf.cell(widths[4], 10, f"{row['PD_RN']:.4%}", border=1, align="C")

        if row["PD_Phys"] > ALERT_THRESHOLD:
            pdf.set_text_color(220, 53, 69)
            pdf.set_font("Arial", "B", 9)
        pdf.cell(widths[5], 10, f"{row['PD_Phys']:.4%}", border=1, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 9)

        pdf.cell(widths[6], 10, f"{row['PD_Phys_Raw']:.4%}", border=1, align="C")
        flag = "OOD" if row["OOD"] else ("FIN" if row["Financial"] else "")
        pdf.cell(widths[7], 10, flag, border=1, align="C")
        pdf.ln(10)

    # Reasons behind each flag
    flagged = df[df["Flags"] != ""]
    if not flagged.empty:
        pdf.ln(6)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, "Reliability notes", ln=True)
        pdf.set_font("Arial", "", 8)
        for _, row in flagged.iterrows():
            pdf.multi_cell(0, 5, f"{row['Ticker']}: {row['Flags']}")
        pdf.ln(2)
        pdf.multi_cell(0, 5, "OOD = input outside the simulated training distribution; "
                             "PDs for these firms are extrapolations. FIN = financial firm.")

    pdf.output(filename)
    return filename


# ---------------------------------- Main ----------------------------------- #
if __name__ == "__main__":
    fred = Fred(api_key=os.environ.get("FRED_API_KEY"))
    raw_r = fred.get_series("DGS5")          # 5-year Treasury constant maturity, percent

    bundle = load_bundle(WINDOW)

    results = []
    for ticker in PORTFOLIO:
        try:
            print(f"Analyzing {ticker}...")
            inputs = fetch_inputs(ticker, raw_r, lookback_days=bundle["n_days"])
            res = score_window(inputs["equity"].values, inputs["D"], inputs["r"].values,
                               bundle, sector=inputs["sector"])
            res = {"Ticker": ticker, **res}
            results.append(res)

            if res["Flags"]:
                print(f"  Flags: {res['Flags']}")
            if res["PD_Phys"] > ALERT_THRESHOLD:
                caveat = " (low confidence: input outside training distribution)" if res["OOD"] else ""
                post_discord(f"🚨 **ALERT!** {ticker} shows a {res['PD_Phys']:.2%} risk of default{caveat}")
        except Exception as e:
            print(f"Error processing {ticker}. Skipping. Reason: {e}")
            traceback.print_exc()

    summary_table = pd.DataFrame(results)
    if summary_table.empty:
        raise SystemExit("No tickers could be scored; nothing to report.")
    print(summary_table.drop(columns=["Flags"]))

    print("Generating PDF tear sheet...")
    report = generate_tear_sheet(summary_table)
    print("Uploading to Discord...")
    post_discord("📊 **Daily Credit Risk Tear Sheet Generated**", pdf_path=report)
