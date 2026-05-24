"""Plot generation for the risk-return analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def save_all_figures(summary: pd.DataFrame, model, output_dir: Path) -> None:
    """Create and save all project figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_histogram(
        summary["Beta"],
        "Distribution of Beta",
        "Beta",
        output_dir / "figure1_distribution_of_beta.png",
    )

    _save_histogram(
        summary["Avg_Return"],
        "Distribution of Average Weekly Returns",
        "Average Weekly Return",
        output_dir / "figure2_distribution_of_avg_returns.png",
    )

    _save_scatter_by_type(
        summary,
        x_column="Beta",
        y_column="Volatility",
        title="Beta vs Volatility",
        x_label="Beta",
        y_label="Volatility",
        output_path=output_dir / "figure3_beta_vs_volatility.png",
    )

    _save_boxplot(
        summary,
        column="Avg_Return",
        by="Beta_Group",
        title="Average Weekly Return by Beta Group",
        x_label="Beta Group",
        y_label="Average Weekly Return",
        output_path=output_dir / "figure4_avg_return_by_beta_group.png",
    )

    _save_beta_return_regression_plot(
        summary,
        output_dir / "figure5_avg_return_and_beta_regression.png",
    )

    _save_residual_plot(model, output_dir / "figure6_regression_residual_plot.png")

    _save_boxplot(
        summary,
        column="Volatility",
        by="Type",
        title="Volatility by Asset Type",
        x_label="Asset Type",
        y_label="Volatility",
        output_path=output_dir / "figure7_volatility_by_asset_type.png",
    )


def _save_histogram(series: pd.Series, title: str, x_label: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(series, bins=8, edgecolor="black")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_scatter_by_type(
    summary: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    for asset_type in summary["Type"].unique():
        subset = summary[summary["Type"] == asset_type]
        plt.scatter(subset[x_column], subset[y_column], label=asset_type)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_boxplot(
    summary: pd.DataFrame,
    column: str,
    by: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    summary.boxplot(column=column, by=by)
    plt.title(title)
    plt.suptitle("")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_beta_return_regression_plot(summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for asset_type in summary["Type"].unique():
        subset = summary[summary["Type"] == asset_type]
        plt.scatter(subset["Beta"], subset["Avg_Return"], label=asset_type)

    simple_x = sm.add_constant(summary["Beta"])
    simple_model = sm.OLS(summary["Avg_Return"], simple_x).fit()
    x_values = np.linspace(summary["Beta"].min(), summary["Beta"].max(), 100)
    y_values = simple_model.predict(sm.add_constant(x_values))
    plt.plot(x_values, y_values)

    plt.title("Average Weekly Return and Beta")
    plt.xlabel("Beta")
    plt.ylabel("Average Weekly Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_residual_plot(model, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(0, linestyle="--")
    plt.title("Regression Residual Plot")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
