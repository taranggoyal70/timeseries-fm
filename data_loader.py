"""
Data Loader Module for Chronos-2 Multivariate Forecasting Experiments

Downloads Magnificent-7 stocks and FRED interest rates data.
Per README: Stocks (K=7), Rates (K=10), Combined (K=17)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime


class DataLoader:
    def __init__(self):
        self.mag7_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        self.fred_series = {
            "DGS3MO": "3-Month", "DGS6MO": "6-Month", "DGS1": "1-Year",
            "DGS2": "2-Year", "DGS3": "3-Year", "DGS5": "5-Year",
            "DGS7": "7-Year", "DGS10": "10-Year", "DGS20": "20-Year", "DGS30": "30-Year"
        }

    def download_stocks(self, start_date="2000-01-01", end_date=None):
        """Download Magnificent-7 stock data from Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Downloading stocks ({start_date} to {end_date})...")
        df = yf.download(self.mag7_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df_close = df["Close"].copy()
        else:
            df_close = df.copy()

        df_close = df_close.reset_index().rename(columns={"Date": "timestamp"})
        df_close["timestamp"] = pd.to_datetime(df_close["timestamp"])
        df_close = df_close.set_index("timestamp").asfreq("B").ffill().reset_index()

        print(f"✓ {len(df_close)} rows, {len(self.mag7_tickers)} stocks")
        return df_close

    def download_interest_rates(self, start_date="2000-01-01", end_date=None):
        """Download FRED interest rates data"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"\nDownloading FRED rates ({start_date} to {end_date})...")
        all_rates = []

        for ticker, name in self.fred_series.items():
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
                df = pd.read_csv(url)
                date_col = 'observation_date' if 'observation_date' in df.columns else 'DATE'
                df = df.rename(columns={date_col: "timestamp", ticker: ticker})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index("timestamp")
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
                all_rates.append(df[ticker])
                print(f"✓ {name} ({ticker})")
            except Exception as e:
                print(f"✗ {ticker}: {str(e)[:60]}")

        if not all_rates:
            raise ValueError("No interest rate data downloaded")

        df_rates = pd.concat(all_rates, axis=1).reset_index()
        df_rates = df_rates.rename(columns={"index": "timestamp"})
        df_rates["timestamp"] = pd.to_datetime(df_rates["timestamp"])
        df_rates = df_rates.set_index("timestamp").asfreq("B").ffill().reset_index()

        print(f"✓ {len(df_rates)} rows, {len(all_rates)} rates")
        return df_rates

    def download_combined(self, start_date="2000-01-01", end_date=None):
        """Download combined stocks and interest rates dataset"""
        print("\n" + "="*60)
        print("DOWNLOADING COMBINED DATASET")
        print("="*60)
        stocks = self.download_stocks(start_date, end_date)
        rates = self.download_interest_rates(start_date, end_date)
        combined = pd.merge(stocks, rates, on="timestamp", how="inner")
        print(f"\n✓ {len(combined)} rows, {len(combined.columns)-1} series")
        return combined


if __name__ == "__main__":
    loader = DataLoader()
    
    stocks_df = loader.download_stocks(start_date="2000-01-01")
    rates_df = loader.download_interest_rates(start_date="2000-01-01")
    combined_df = loader.download_combined(start_date="2000-01-01")
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Stocks: {stocks_df.shape}")
    print(f"Rates: {rates_df.shape}")
    print(f"Combined: {combined_df.shape}")
    
    print("\nStocks sample:")
    print(stocks_df.head())
    print("\nRates sample:")
    print(rates_df.head())
