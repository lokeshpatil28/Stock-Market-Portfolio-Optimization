import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# =========================
# STOCK LIST
# =========================

stocks = [
    "HDFCBANK.NS",
    "TECHM.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "NESTLEIND.NS",
    "ITC.NS",
    "LT.NS",
    "INDIGO.NS",
    "ONGC.NS",
    "GOLDBEES.NS"
]

# =========================
# DATE RANGE (LAST 6 MONTHS)
# =========================

end = datetime.today()
start = end - timedelta(days=180)

# =========================
# SAVE DIRECTORY
# =========================

save_dir = "../data/prices/"
os.makedirs(save_dir, exist_ok=True)

# =========================
# DOWNLOAD LOOP
# =========================

for stock in stocks:

    print(f"\nDownloading {stock} ...")

    try:
        df = yf.download(
            stock,
            start=start,
            end=end,
            progress=False
        )

        if df.empty:
            print("⚠ No data")
            continue

        # keep only required columns
        df = df.reset_index()
        df = df[["Date", "Close"]]

        filename = stock.replace(".NS", "") + ".csv"
        filepath = os.path.join(save_dir, filename)

        df.to_csv(filepath, index=False)

        print(f"✅ Saved -> {filepath}")

    except Exception as e:
        print(f"❌ Error downloading {stock}: {e}")

print("\n🎯 All downloads complete.")