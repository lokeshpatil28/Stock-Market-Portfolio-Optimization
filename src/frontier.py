import numpy as np
import pandas as pd
import os

mu = np.load("../data/mu.npy")
Sigma = np.load("../data/cov.npy")

n = len(mu)

targets = np.linspace(min(mu)*0.8, max(mu)*1.2, 30)

frontier_ret = []
frontier_vol = []

for target in targets:

    KKT = np.zeros((n+2, n+2))

    KKT[:n, :n] = 2*Sigma
    KKT[:n, n] = 1
    KKT[:n, n+1] = mu

    KKT[n, :n] = 1
    KKT[n+1, :n] = mu

    rhs = np.zeros(n+2)
    rhs[n] = 1
    rhs[n+1] = target

    try:
        sol = np.linalg.solve(KKT, rhs)
    except:
        continue

    x = sol[:n]

    x[x < 0] = 0
    if x.sum() == 0:
        continue
    x = x / x.sum()

    ret = mu @ x
    var = x @ Sigma @ x

    frontier_ret.append(ret)
    frontier_vol.append(np.sqrt(var))

df = pd.DataFrame({
    "Return": frontier_ret,
    "Volatility": frontier_vol
})

os.makedirs("../results", exist_ok=True)
df.to_csv("../results/frontier.csv", index=False)

print("✅ Efficient frontier data saved")