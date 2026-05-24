"""Risk and return metric calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_max_drawdown(price_series: pd.Series) -> float:
    """Return the largest percentage drop from a prior peak."""
    rolling_max = price_series.cummax()
    drawdown = (price_series - rolling_max) / rolling_max
    return float(drawdown.min())


def calculate_betas(
    returns: pd.DataFrame,
    stocks: list[str],
    cryptos: list[str],
    market_index: str,
    benchmark_crypto: str = "BTC-USD",
) -> dict[str, float]:
    """Calculate stock betas against the market and crypto betas against BTC."""
    betas: dict[str, float] = {}
    market_returns = returns[market_index]
    crypto_benchmark_returns = returns[benchmark_crypto]

    for stock in stocks:
        if stock in returns.columns:
            covariance = np.cov(returns[stock], market_returns)[0][1]
            variance = np.var(market_returns)
            betas[stock] = covariance / variance

    for crypto in cryptos:
        if crypto in returns.columns:
            if crypto == benchmark_crypto:
                betas[crypto] = 1.0
            else:
                covariance = np.cov(returns[crypto], crypto_benchmark_returns)[0][1]
                variance = np.var(crypto_benchmark_returns)
                betas[crypto] = covariance / variance

    return betas


def build_summary_dataset(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    stocks: list[str],
    cryptos: list[str],
    market_index: str,
) -> pd.DataFrame:
    """Create the final asset-level dataset used for statistical analysis."""
    betas = calculate_betas(returns, stocks, cryptos, market_index)

    summary = pd.DataFrame(
        {
            "Asset": prices.columns,
            "Avg_Return": returns.mean(),
            "Volatility": returns.std(),
            "Max_Drawdown": prices.apply(calculate_max_drawdown),
        }
    ).reset_index(drop=True)

    summary["Beta"] = summary["Asset"].map(betas)
    summary["Type"] = summary["Asset"].apply(lambda asset: "Crypto" if asset in cryptos else "Stock")

    summary = summary[summary["Asset"] != market_index].copy()
    summary = summary.dropna(subset=["Beta", "Avg_Return", "Volatility", "Max_Drawdown"])

    median_beta = summary["Beta"].median()
    summary["Beta_Group"] = np.where(summary["Beta"] >= median_beta, "High Beta", "Low Beta")
    summary["Is_Crypto"] = np.where(summary["Type"] == "Crypto", 1, 0)

    return summary.sort_values("Asset").reset_index(drop=True)
