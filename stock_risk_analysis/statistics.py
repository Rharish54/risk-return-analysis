"""Statistical tests and regression modeling."""

from __future__ import annotations

import pandas as pd
from scipy import stats
import statsmodels.api as sm


def mean_confidence_interval(series: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return mean, lower CI bound, and upper CI bound."""
    clean_series = series.dropna()
    n = len(clean_series)

    if n < 2:
        raise ValueError("At least two observations are required to calculate a confidence interval.")

    mean = clean_series.mean()
    standard_error = stats.sem(clean_series)
    margin = standard_error * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin


def descriptive_statistics(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return descriptive statistics for the full sample and grouped samples."""
    numeric_columns = ["Beta", "Avg_Return", "Volatility", "Max_Drawdown"]

    return {
        "all": summary[numeric_columns].describe(),
        "by_type": summary.groupby("Type")[numeric_columns].agg(
            ["mean", "median", "std", "min", "max", "count"]
        ),
        "by_beta_group": summary.groupby("Beta_Group")[[
            "Avg_Return", "Volatility", "Max_Drawdown", "Beta"
        ]].agg(["mean", "median", "std", "count"]),
    }


def welch_t_test_high_vs_low_beta(summary: pd.DataFrame) -> dict[str, float]:
    """Run a one-sided Welch two-sample t-test for high-beta vs low-beta returns."""
    high_beta_returns = summary.loc[summary["Beta_Group"] == "High Beta", "Avg_Return"]
    low_beta_returns = summary.loc[summary["Beta_Group"] == "Low Beta", "Avg_Return"]

    t_stat, p_two_sided = stats.ttest_ind(
        high_beta_returns,
        low_beta_returns,
        equal_var=False,
        nan_policy="omit",
    )

    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)

    return {
        "t_statistic": t_stat,
        "p_one_sided": p_one_sided,
        "high_beta_mean": high_beta_returns.mean(),
        "low_beta_mean": low_beta_returns.mean(),
    }


def fit_regression_model(summary: pd.DataFrame):
    """Fit OLS regression model predicting average return."""
    predictors = summary[["Beta", "Volatility", "Max_Drawdown", "Is_Crypto"]]
    predictors = sm.add_constant(predictors)
    response = summary["Avg_Return"]
    return sm.OLS(response, predictors).fit()
