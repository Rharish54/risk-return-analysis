"""Main entry point for the stock and crypto risk-return analysis."""

from __future__ import annotations

import warnings

from stock_risk_analysis.config import (
    ALL_ASSETS,
    CONFIDENCE_LEVEL,
    CRYPTOS,
    DOWNLOAD_INTERVAL,
    DOWNLOAD_PERIOD,
    MARKET_INDEX,
    OUTPUT_DIR,
    STOCKS,
)
from stock_risk_analysis.data import calculate_returns, download_price_data, validate_required_assets
from stock_risk_analysis.metrics import build_summary_dataset
from stock_risk_analysis.plots import save_all_figures
from stock_risk_analysis.reporting import (
    build_results_summary,
    save_descriptive_statistics,
    save_text_report,
)
from stock_risk_analysis.statistics import (
    descriptive_statistics,
    fit_regression_model,
    mean_confidence_interval,
    welch_t_test_high_vs_low_beta,
)

warnings.filterwarnings("ignore")


def run_analysis() -> None:
    """Run the full project pipeline from data download to saved outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading weekly price data...")
    prices = download_price_data(ALL_ASSETS, DOWNLOAD_PERIOD, DOWNLOAD_INTERVAL)
    returns = calculate_returns(prices)
    validate_required_assets(returns, MARKET_INDEX, "BTC-USD")

    print("Calculating risk and return metrics...")
    summary = build_summary_dataset(prices, returns, STOCKS, CRYPTOS, MARKET_INDEX)
    median_beta = summary["Beta"].median()

    dataset_path = OUTPUT_DIR / "stat430_final_dataset.csv"
    summary.to_csv(dataset_path, index=False)

    print("Running descriptive statistics, confidence intervals, t-test, and regression...")
    stats_tables = descriptive_statistics(summary)
    save_descriptive_statistics(stats_tables, OUTPUT_DIR)

    avg_return_ci = mean_confidence_interval(summary["Avg_Return"], CONFIDENCE_LEVEL)
    volatility_ci = mean_confidence_interval(summary["Volatility"], CONFIDENCE_LEVEL)
    t_test_results = welch_t_test_high_vs_low_beta(summary)
    regression_model = fit_regression_model(summary)

    save_text_report(
        regression_model.summary().as_text(),
        OUTPUT_DIR / "linear_regression_summary.txt",
    )

    print("Saving figures...")
    save_all_figures(summary, regression_model, OUTPUT_DIR)

    results_summary = build_results_summary(
        summary=summary,
        median_beta=median_beta,
        avg_return_ci=avg_return_ci,
        volatility_ci=volatility_ci,
        t_test_results=t_test_results,
        regression_model=regression_model,
    )
    save_text_report(results_summary, OUTPUT_DIR / "project_results_summary.txt")

    print(results_summary)
    print(f"\nDataset saved to: {dataset_path}")
    print(f"Regression summary saved to: {OUTPUT_DIR / 'linear_regression_summary.txt'}")
    print(f"Results summary saved to: {OUTPUT_DIR / 'project_results_summary.txt'}")
    print(f"Figures saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_analysis()
