"""Functions for downloading and preparing weekly price data."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: list[str],
    period: str,
    interval: str,
) -> pd.DataFrame:
    """Download adjusted closing prices from Yahoo Finance."""
    raw_data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if "Close" not in raw_data:
        raise ValueError("Downloaded data does not contain closing prices.")

    prices = raw_data["Close"]
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna()
    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage returns from price data."""
    return prices.pct_change().dropna()


def validate_required_assets(returns: pd.DataFrame, market_index: str, benchmark_crypto: str) -> None:
    """Make sure benchmark columns are available before analysis runs."""
    if market_index not in returns.columns:
        raise ValueError(f"{market_index} data was not downloaded correctly.")

    if benchmark_crypto not in returns.columns:
        raise ValueError(f"{benchmark_crypto} data was not downloaded correctly.")
