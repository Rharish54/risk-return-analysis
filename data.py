import pandas as pd 
import yfinance as yf 
import numpy as np  

stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
          'NVDA', 'GOOG', 'AVGO', 'TSLA', 'BRK-B',
          'WMT', 'LLY', 'JPM', 'XOM', 'V',
          'JNJ', 'MU', 'MA', 'COST', 'ORCL',
          'CVX', 'NFLX', 'ABBV', 'PLTR', 'BAC',
          'PG', 'AMD', 'KO', 'UNH', 'MS']

cryptos = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD']

market = ["^GSPC"]  # S&P 500

all_assets = stocks + cryptos + market

# -----------------------------
# 2. Download weekly data
# -----------------------------
data = yf.download(
    all_assets,
    period="6mo",
    interval="1wk",
    auto_adjust=True
)["Close"]

# -----------------------------
# 3. Compute returns
# -----------------------------
returns = data.pct_change().dropna()

# -----------------------------
# 4. Volatility
# -----------------------------
volatility = returns.std()

# -----------------------------
# 5. Beta calculation
# -----------------------------
market_returns = returns["^GSPC"]

betas = {}

# Stocks (vs S&P 500)
for stock in stocks:
    cov = np.cov(returns[stock], market_returns)[0][1]
    var = np.var(market_returns)
    betas[stock] = cov / var

# Crypto (vs BTC)
btc_returns = returns["BTC-USD"]

for c in cryptos:
    if c != "BTC-USD":
        cov = np.cov(returns[c], btc_returns)[0][1]
        var = np.var(btc_returns)
        betas[c] = cov / var

# BTC beta = 1 by definition
betas["BTC-USD"] = 1.0

# -----------------------------
# 6. Average return
# -----------------------------
avg_return = returns.mean()

# -----------------------------
# 7. Max drawdown
# -----------------------------
def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

drawdowns = data.apply(max_drawdown)

# -----------------------------
# 8. Build dataset
# -----------------------------
summary = pd.DataFrame({
    "Asset": data.columns,
    "Avg_Return": avg_return,
    "Volatility": volatility,
    "Max_Drawdown": drawdowns
})

summary["Beta"] = summary["Asset"].map(betas)

# Add type (Stock vs Crypto)
summary["Type"] = summary["Asset"].apply(
    lambda x: "Crypto" if "USD" in x else "Stock"
)

# Remove market index row
summary = summary[summary["Asset"] != "^GSPC"]

# -----------------------------
# 9. Save to CSV (for SAS)
# -----------------------------
summary.to_csv("stat430_dataset.csv", index=False)

print(summary.head())
