import numpy as np
import pandas as pd
import os

# ===== LOAD DATA =====
mu = np.load("../data/mu.npy")
Sigma = np.load("../data/cov.npy")

stock_names = [
    "GOLDBEES","HDFCBANK","INDIGO","ITC","LT",
    "NESTLEIND","ONGC","SUNPHARMA","TECHM","TITAN"
]

n = len(mu)

# ===== USER TARGET RETURN =====
target = 0.001   # you can change this

# ===== BUILD KKT SYSTEM =====
# minimize xT Σ x
# s.t sum x = 1
#     muT x = target

KKT = np.zeros((n+2, n+2))

KKT[:n, :n] = 2*Sigma
KKT[:n, n] = 1
KKT[:n, n+1] = mu

KKT[n, :n] = 1
KKT[n+1, :n] = mu

rhs = np.zeros(n+2)
rhs[n] = 1
rhs[n+1] = target

sol = np.linalg.solve(KKT, rhs)

x = sol[:n]

# ===== LONG ONLY CONSTRAINT =====
x[x < 0] = 0

# renormalise
x = x / x.sum()

# ===== RESULTS =====
ret = mu @ x
risk = x @ Sigma @ x

print("\n===== OPTIMAL PORTFOLIO =====\n")

for name, w in zip(stock_names, x):
    print(f"{name:12s} : {w*100:.2f}%")

print("\nTotal:", x.sum())
print("Expected return:", ret)
print("Variance:", risk)
print("Volatility:", np.sqrt(risk))

# ===== SAVE =====
os.makedirs("../results", exist_ok=True)

df = pd.DataFrame({
    "Stock": stock_names,
    "Weight": x
})

df.to_csv("../results/optimal_weights.csv", index=False)

summary = pd.DataFrame({
    "Return":[ret],
    "Variance":[risk],
    "Volatility":[np.sqrt(risk)]
})

summary.to_csv("../results/summary.csv", index=False)

print("\n✅ Results saved in results/ folder")