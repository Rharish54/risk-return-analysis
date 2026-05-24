# Stock and Crypto Risk-Return Analysis

This project analyzes the relationship between risk and short-term return across a sample of large-cap stocks and cryptocurrencies. It downloads weekly market data, calculates asset-level risk metrics, runs statistical tests, fits a multiple linear regression model, and saves figures and summary tables.

## Project Overview

The analysis compares stocks and cryptocurrencies using:

- Average weekly return
- Volatility
- Beta
- Maximum drawdown
- High-beta vs low-beta asset groups
- Welch two-sample t-test
- Multiple linear regression

Stocks are measured against the S&P 500. Cryptocurrencies are measured against Bitcoin as the crypto benchmark.

## Repository Structure

```text
stock-risk-return-analysis/
├── stock_risk_analysis/
│   ├── __init__.py
│   ├── config.py          # Tickers, output paths, and project settings
│   ├── data.py            # Data download and return calculation
│   ├── metrics.py         # Beta, volatility, drawdown, and summary dataset
│   ├── statistics.py      # Confidence intervals, t-test, and regression
│   ├── plots.py           # Figure generation
│   ├── reporting.py       # CSV and text report output
│   └── main.py            # Runs the full analysis pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python -m stock_risk_analysis.main
```

The script will create an `outputs/` folder containing:

- `stat430_final_dataset.csv`
- Descriptive statistics CSV files
- `linear_regression_summary.txt`
- `project_results_summary.txt`
- Seven figure PNG files

## Notes

This project uses Yahoo Finance data through `yfinance`, so results may change depending on the date the analysis is run and the availability of ticker data.
