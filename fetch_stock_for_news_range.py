"""Script to fetch stock data for the date range of news sentiment data.

This script reads the news sentiment JSON files, extracts the date range
from the published dates, and fetches corresponding stock data using yfinance.
Organizes output into separate folders per symbol.
"""

import json
import os
import shutil
from datetime import datetime

import yfinance as yf

# Configuration
SYMBOLS = ["AES", "AOS", "APA", "ARE", "CAG", "CPB", "DVA", "FRT", "GNRC", "HSIC", "LW", "MGM", "MOH", "MOS", "MTCH", "PAYC", "POOL", "SWKS", "TAP"]
SOURCE_DIR = "alpha_vantage_data"
OUTPUT_BASE_DIR = "alpha_vantage_data"


def get_date_range_from_news(symbol: str) -> tuple[str, str]:
    """Extract date range from news sentiment file.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Tuple of (oldest_date, newest_date) in YYYY-MM-DD format
    """
    news_file = os.path.join(SOURCE_DIR, f"{symbol}_news_sentiment.json")

    if not os.path.exists(news_file):
        raise FileNotFoundError(f"News sentiment file not found: {news_file}")

    with open(news_file, "r") as f:
        data = json.load(f)

    if "feed" not in data or len(data["feed"]) == 0:
        raise ValueError(f"No news items found in {news_file}")

    # Extract all publish dates
    dates = []
    for item in data["feed"]:
        if "time_published" in item:
            # Format: YYYYMMDDTHHMMSS
            date_str = item["time_published"][:8]  # Extract YYYYMMDD
            # Convert to YYYY-MM-DD
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            dates.append(date_obj)

    if not dates:
        raise ValueError(f"No valid dates found in {news_file}")

    oldest = min(dates)
    newest = max(dates)

    oldest_str = oldest.strftime("%Y-%m-%d")
    newest_str = newest.strftime("%Y-%m-%d")

    print(f"  News date range: {oldest_str} to {newest_str}")

    return oldest_str, newest_str


def fetch_stock_data_yfinance(symbol: str, start_date: str, end_date: str) -> dict:
    """Fetch stock data using yfinance for the specified date range.

    Args:
        symbol: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dictionary with stock data in JSON-serializable format
    """
    print(f"  Fetching stock data from yfinance...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No stock data available for {symbol}")

    # Convert DataFrame to JSON-serializable format
    stock_data = {
        "Meta Data": {
            "Symbol": symbol,
            "Start Date": start_date,
            "End Date": end_date,
            "Data Points": len(df),
            "Source": "yfinance"
        },
        "Time Series (Daily)": {}
    }

    # Convert each row to dict
    for date, row in df.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        stock_data["Time Series (Daily)"][date_str] = {
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": int(row["Volume"])
        }

    print(f"  Retrieved {len(df)} daily data points")

    return stock_data


def organize_data_by_symbol():
    """Organize news and stock data into separate folders per symbol."""
    print(f"\nOrganizing data into symbol-specific folders...")

    for symbol in SYMBOLS:
        symbol_dir = os.path.join(OUTPUT_BASE_DIR, symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        # Move news sentiment file if it exists
        news_file = os.path.join(SOURCE_DIR, f"{symbol}_news_sentiment.json")
        if os.path.exists(news_file):
            dest_file = os.path.join(symbol_dir, "news_sentiment.json")
            shutil.move(news_file, dest_file)
            print(f"  Moved {symbol} news data to {symbol_dir}/")

        # Move stock data file if it exists
        stock_file = os.path.join(SOURCE_DIR, f"{symbol}_stock_data.json")
        if os.path.exists(stock_file):
            dest_file = os.path.join(symbol_dir, "stock_data.json")
            shutil.move(stock_file, dest_file)
            print(f"  Moved {symbol} stock data to {symbol_dir}/")


def main():
    """Main function to fetch stock data for news date ranges."""
    print("Fetching stock data for news sentiment date ranges")
    print("="*60)

    for symbol in SYMBOLS:
        try:
            print(f"\n{symbol}:")

            # Get date range from news data
            start_date, end_date = get_date_range_from_news(symbol)

            # Fetch stock data using yfinance
            stock_data = fetch_stock_data_yfinance(symbol, start_date, end_date)

            # Save stock data temporarily
            temp_output_file = os.path.join(SOURCE_DIR, f"{symbol}_stock_data.json")
            with open(temp_output_file, "w") as f:
                json.dump(stock_data, f, indent=2)

            print(f"  Saved stock data")

        except FileNotFoundError as e:
            print(f"  Error: {e}")
            print(f"  Skipping {symbol} - run fetch_alpha_vantage.py first")
            continue
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            continue

    # Organize all data into symbol-specific folders
    organize_data_by_symbol()

    print(f"\n{'='*60}")
    print(f"Data organized in {OUTPUT_BASE_DIR}/[SYMBOL]/ directories:")
    print(f"  - news_sentiment.json (Alpha Vantage news data)")
    print(f"  - stock_data.json (yfinance stock data)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
