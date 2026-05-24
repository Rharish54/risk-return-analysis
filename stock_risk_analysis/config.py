"""Project configuration for the stock and crypto risk-return analysis."""

from pathlib import Path

OUTPUT_DIR = Path("outputs")

STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "GOOG", "AVGO", "TSLA", "BRK-B",
    "WMT", "LLY", "JPM", "XOM", "V",
    "JNJ", "MU", "MA", "COST", "ORCL",
    "CVX", "NFLX", "ABBV", "PLTR", "BAC",
    "PG", "AMD", "KO", "UNH", "MS",
]

CRYPTOS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD"]
MARKET_INDEX = "^GSPC"

ALL_ASSETS = STOCKS + CRYPTOS + [MARKET_INDEX]

DOWNLOAD_PERIOD = "6mo"
DOWNLOAD_INTERVAL = "1wk"
CONFIDENCE_LEVEL = 0.95
