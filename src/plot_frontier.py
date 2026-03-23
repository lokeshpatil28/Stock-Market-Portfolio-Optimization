import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/frontier.csv")

plt.figure(figsize=(8,6))
plt.plot(df["Volatility"], df["Return"], marker='o')
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.grid()

plt.savefig("../results/frontier.png")
plt.show()