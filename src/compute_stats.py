import pandas as pd
import numpy as np
import glob
import os

# =========================
# LOAD ALL PRICE FILES
# =========================

price_files = glob.glob("../data/prices/*.csv")

dfs = []

for f in price_files:
    name = os.path.basename(f).replace(".csv", "")
    df = pd.read_csv(f)

    # keep Date + Close only
    df = df[["Date", "Close"]]
    df = df.rename(columns={"Close": name})

    dfs.append(df)

# merge on Date
data = dfs[0]
for df in dfs[1:]:
    data = pd.merge(data, df, on="Date", how="inner")

data = data.sort_values("Date")
data = data.set_index("Date")

# save combined prices
data.to_csv("../data/combined_prices.csv")

print("\nCombined price data shape:", data.shape)

# =========================
# RETURNS
# =========================

returns = data.pct_change().dropna()

returns.to_csv("../data/returns.csv")

# =========================
# STATISTICS
# =========================

# ⭐ annualised expected return
mu = returns.mean().values * 252

# ⭐ covariance (annualised)
Sigma = returns.cov().values * 252

std_dev = returns.std().values * np.sqrt(252)

# =========================
# SAVE
# =========================

np.save("../data/mu.npy", mu)
np.save("../data/cov.npy", Sigma)
np.save("../data/std.npy", std_dev)

print("\nExpected returns (annualised):\n", mu)
print("\nVolatility (annualised):\n", std_dev)
print("\nCovariance matrix:\n", Sigma)

print("\n✅ Statistics computed and saved.")