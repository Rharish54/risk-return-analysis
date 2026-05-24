"""Functions for saving project outputs and text reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_descriptive_statistics(stats_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save descriptive statistics tables as CSV files."""
    stats_tables["all"].to_csv(output_dir / "descriptive_statistics_all.csv")
    stats_tables["by_type"].to_csv(output_dir / "descriptive_statistics_by_type.csv")
    stats_tables["by_beta_group"].to_csv(output_dir / "descriptive_statistics_by_beta_group.csv")


def build_results_summary(
    summary: pd.DataFrame,
    median_beta: float,
    avg_return_ci: tuple[float, float, float],
    volatility_ci: tuple[float, float, float],
    t_test_results: dict[str, float],
    regression_model,
) -> str:
    """Build the plain-text project results summary."""
    avg_return_mean, avg_return_low, avg_return_high = avg_return_ci
    volatility_mean, volatility_low, volatility_high = volatility_ci

    return f"""
STAT430 Final Project Results Summary

Number of assets analyzed: {len(summary)}
Number of stocks: {(summary["Type"] == "Stock").sum()}
Number of cryptocurrencies: {(summary["Type"] == "Crypto").sum()}

Median beta used to split high-beta and low-beta groups: {median_beta:.4f}

Overall average weekly return:
Mean = {avg_return_mean:.6f}
95% CI = ({avg_return_low:.6f}, {avg_return_high:.6f})

Overall volatility:
Mean = {volatility_mean:.6f}
95% CI = ({volatility_low:.6f}, {volatility_high:.6f})

High-beta mean average weekly return: {t_test_results["high_beta_mean"]:.6f}
Low-beta mean average weekly return: {t_test_results["low_beta_mean"]:.6f}

Welch Two-Sample T-Test:
H0: mean return of high-beta assets <= mean return of low-beta assets
HA: mean return of high-beta assets > mean return of low-beta assets

t statistic = {t_test_results["t_statistic"]:.4f}
one-sided p-value = {t_test_results["p_one_sided"]:.4f}

Regression Model:
Avg_Return = b0 + b1(Beta) + b2(Volatility) + b3(Max_Drawdown) + b4(Is_Crypto)

R-squared = {regression_model.rsquared:.4f}
Adjusted R-squared = {regression_model.rsquared_adj:.4f}

Coefficients:
{regression_model.params.to_string()}

P-values:
{regression_model.pvalues.to_string()}
""".strip()


def save_text_report(text: str, output_path: Path) -> None:
    """Save text content to a file."""
    with output_path.open("w", encoding="utf-8") as file:
        file.write(text)
