import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class DataFetcher:
    
    def __init__(self):
        self.mag7_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        self.fred_tickers = {
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
    
    def fetch_stocks(self, start_date="2000-01-01", end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching Magnificent-7 stocks from {start_date} to {end_date}...")
        df = yf.download(self.mag7_tickers, start=start_date, end=end_date, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df_close = df["Close"].copy()
        else:
            df_close = df.copy()
        
        df_close = df_close.reset_index()
        df_close = df_close.rename(columns={"Date": "timestamp"})
        df_close["timestamp"] = pd.to_datetime(df_close["timestamp"])
        df_close = df_close.set_index("timestamp").asfreq("B").ffill().reset_index()
        
        print(f"Fetched {len(df_close)} rows for {len(self.mag7_tickers)} stocks")
        return df_close
    
    def fetch_interest_rates(self, start_date="2000-01-01", end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching FRED interest rates from {start_date} to {end_date}...")
        
        all_rates = []
        for ticker in self.fred_tickers.keys():
            try:
                df = yf.download(f"^{ticker}", start=start_date, end=end_date, auto_adjust=True)
                if not df.empty:
                    all_rates.append(df["Close"].rename(ticker))
            except Exception as e:
                print(f"Warning: Could not fetch {ticker}: {e}")
                try:
                    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
                    df = pd.read_csv(url, parse_dates=["DATE"])
                    df = df.rename(columns={"DATE": "timestamp", ticker: ticker})
                    df = df.set_index("timestamp")
                    all_rates.append(df[ticker])
                except Exception as e2:
                    print(f"Backup fetch also failed for {ticker}: {e2}")
        
        if not all_rates:
            raise ValueError("Could not fetch any interest rate data")
        
        df_rates = pd.concat(all_rates, axis=1)
        df_rates = df_rates.reset_index()
        df_rates = df_rates.rename(columns={"index": "timestamp"})
        df_rates["timestamp"] = pd.to_datetime(df_rates["timestamp"])
        df_rates = df_rates.set_index("timestamp").asfreq("B").ffill().reset_index()
        
        print(f"Fetched {len(df_rates)} rows for {len(all_rates)} interest rates")
        return df_rates
    
    def fetch_combined(self, start_date="2000-01-01", end_date=None):
        stocks = self.fetch_stocks(start_date, end_date)
        rates = self.fetch_interest_rates(start_date, end_date)
        
        combined = pd.merge(stocks, rates, on="timestamp", how="inner")
        print(f"Combined dataset: {len(combined)} rows, {len(combined.columns)-1} series")
        return combined
    
    def prepare_chronos_format(self, df, target_column, item_id="series_1"):
        df_formatted = df.copy()
        df_formatted["item_id"] = item_id
        
        required_cols = ["item_id", "timestamp", target_column]
        other_cols = [col for col in df_formatted.columns if col not in required_cols]
        
        df_formatted = df_formatted[required_cols + other_cols]
        
        return df_formatted
