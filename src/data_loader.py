import yfinance as yf
import numpy as np
import pandas as pd
import sys

def fetch_and_process(ticker, start, end):
    """
    Fetches data from Yahoo Finance and calculates log returns + volatility.
    Strictly fails if data cannot be downloaded.
    """
    print(f"Downloading {ticker} from {start} to {end}...")
    
    # auto_adjust=True makes 'Close' the adjusted price
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    
    # 1. Validation: Check if data is empty
    if data.empty:
        print(f"❌ Error: No data found for {ticker}. Check your internet or the ticker symbol.")
        sys.exit(1)

    # 2. Cleaning: Handle multi-level columns (common in newer yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    # 3. Selection: safely get the price column
    # With auto_adjust=True, 'Close' is the correct column to use
    if 'Close' in data.columns:
        df = data[['Close']].copy()
        df.rename(columns={'Close': 'price'}, inplace=True)
    elif 'Adj Close' in data.columns:
        df = data[['Adj Close']].copy()
        df.rename(columns={'Adj Close': 'price'}, inplace=True)
    else:
        # Fallback if neither standard name exists
        print(f"⚠️ Warning: Could not find 'Close' column. Using: {data.columns[0]}")
        df = data.iloc[:, 0].to_frame()
        df.columns = ['price']

    # 4. Feature Engineering
    df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
    
    # Realized Volatility (10-day rolling standard deviation)
    df['volatility'] = df['log_ret'].rolling(window=10).std()
    
    # Drop NaNs created by the shifting/rolling operations
    df.dropna(inplace=True)
    
    return df