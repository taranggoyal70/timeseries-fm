"""
Data loader for Chronos-2 forecasting experiments.
Downloads and prepares stock and interest rate data per README specifications.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import os

class DataLoader:
    """Load and prepare financial time series data for Chronos-2 experiments."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Magnificent-7 stocks
        self.mag7_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        
        # FRED interest rate series (10 maturities per README)
        self.fred_series = {
            "DGS3MO": "3-Month",
            "DGS6MO": "6-Month",
            "DGS1": "1-Year",
            "DGS2": "2-Year",
            "DGS3": "3-Year",
            "DGS5": "5-Year",
            "DGS7": "7-Year",
            "DGS10": "10-Year",
            "DGS20": "20-Year",
            "DGS30": "30-Year"
        }
    
    def download_stocks(self, start_date="2000-01-01", end_date=None):
        """Download Magnificent-7 stock data from Yahoo Finance."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Downloading Magnificent-7 stocks ({start_date} to {end_date})...")
        
        try:
            df = yf.download(self.mag7_tickers, start=start_date, end=end_date, 
                           auto_adjust=True, progress=False)
            
            # Extract close prices
            if isinstance(df.columns, pd.MultiIndex):
                df_close = df["Close"].copy()
            else:
                df_close = df.copy()
            
            # Reset index and clean
            df_close = df_close.reset_index()
            df_close = df_close.rename(columns={"Date": "timestamp"})
            df_close["timestamp"] = pd.to_datetime(df_close["timestamp"])
            
            # Business day frequency with forward fill
            df_close = df_close.set_index("timestamp").asfreq("B").ffill().reset_index()
            
            print(f"✓ Downloaded {len(df_close)} rows for {len(self.mag7_tickers)} stocks")
            
            # Save to CSV
            filepath = os.path.join(self.data_dir, "stocks.csv")
            df_close.to_csv(filepath, index=False)
            print(f"✓ Saved to {filepath}")
            
            return df_close
            
        except Exception as e:
            print(f"✗ Error downloading stocks: {e}")
            raise
    
    def download_interest_rates(self, start_date="2000-01-01", end_date=None):
        """Download FRED interest rate data."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\nDownloading FRED interest rates ({start_date} to {end_date})...")
        
        all_rates = []
        
        for ticker, name in self.fred_series.items():
            try:
                # Try direct FRED CSV download
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
                df = pd.read_csv(url, parse_dates=["DATE"])
                df = df.rename(columns={"DATE": "timestamp", ticker: ticker})
                df = df.set_index("timestamp")
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                # Replace '.' with NaN
                df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
                
                all_rates.append(df[ticker])
                print(f"✓ Downloaded {name} ({ticker})")
                
            except Exception as e:
                print(f"✗ Failed to download {ticker}: {e}")
                # Try yfinance as backup
                try:
                    df_yf = yf.download(f"^{ticker}", start=start_date, end=end_date, 
                                       auto_adjust=True, progress=False)
                    if not df_yf.empty:
                        all_rates.append(df_yf["Close"].rename(ticker))
                        print(f"✓ Downloaded {name} ({ticker}) via yfinance")
                except:
                    print(f"✗ Backup download also failed for {ticker}")
        
        if not all_rates:
            raise ValueError("Could not download any interest rate data")
        
        # Combine all rates
        df_rates = pd.concat(all_rates, axis=1)
        df_rates = df_rates.reset_index()
        df_rates = df_rates.rename(columns={"index": "timestamp"})
        df_rates["timestamp"] = pd.to_datetime(df_rates["timestamp"])
        
        # Business day frequency with forward fill
        df_rates = df_rates.set_index("timestamp").asfreq("B").ffill().reset_index()
        
        print(f"✓ Combined {len(df_rates)} rows for {len(all_rates)} interest rates")
        
        # Save to CSV
        filepath = os.path.join(self.data_dir, "interest_rates.csv")
        df_rates.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath}")
        
        return df_rates
    
    def download_combined(self, start_date="2000-01-01", end_date=None):
        """Download and combine stocks + interest rates (K=17)."""
        print("\n" + "="*60)
        print("DOWNLOADING COMBINED DATASET (Stocks + Interest Rates)")
        print("="*60)
        
        stocks = self.download_stocks(start_date, end_date)
        rates = self.download_interest_rates(start_date, end_date)
        
        # Merge on timestamp
        combined = pd.merge(stocks, rates, on="timestamp", how="inner")
        
        print(f"\n✓ Combined dataset: {len(combined)} rows, {len(combined.columns)-1} series (K=17)")
        
        # Save to CSV
        filepath = os.path.join(self.data_dir, "combined.csv")
        combined.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath}")
        
        return combined
    
    def load_data(self, dataset_type="stocks"):
        """Load previously downloaded data."""
        filepath = os.path.join(self.data_dir, f"{dataset_type}.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}. Run download first.")
        
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        print(f"✓ Loaded {dataset_type}: {len(df)} rows, {len(df.columns)-1} series")
        
        return df


if __name__ == "__main__":
    # Test data download
    loader = DataLoader()
    
    # Download all datasets
    stocks = loader.download_stocks()
    rates = loader.download_interest_rates()
    combined = loader.download_combined()
    
    print("\n" + "="*60)
    print("DATA DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Stocks: {stocks.shape}")
    print(f"Interest Rates: {rates.shape}")
    print(f"Combined: {combined.shape}")
