"""Stock data service using yfinance.

This module provides functions to fetch stock data, company info,
and basic metrics from Yahoo Finance.
"""

import atexit
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from config import APP

# Ticker cache with TTL to avoid creating multiple Ticker objects
_ticker_cache: dict[str, tuple[yf.Ticker, datetime]] = {}
_TICKER_CACHE_TTL = timedelta(minutes=5)


def get_ticker(symbol: str) -> yf.Ticker:
    """Get or create cached yfinance Ticker object.

    Caches Ticker objects for 5 minutes to reduce redundant
    network requests when multiple functions need the same ticker.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Cached or new yf.Ticker instance.
    """
    symbol = symbol.upper().strip()
    now = datetime.now()

    if symbol in _ticker_cache:
        ticker, cached_at = _ticker_cache[symbol]
        if now - cached_at < _TICKER_CACHE_TTL:
            return ticker

    ticker = yf.Ticker(symbol)
    _ticker_cache[symbol] = (ticker, now)
    return ticker


def clear_ticker_cache(symbol: Optional[str] = None) -> None:
    """Clear ticker cache.

    Args:
        symbol: Specific symbol to clear, or None for all.
    """
    global _ticker_cache
    if symbol:
        _ticker_cache.pop(symbol.upper().strip(), None)
    else:
        _ticker_cache.clear()


# Register cleanup on shutdown to prevent semaphore leaks from yfinance Ticker objects
atexit.register(clear_ticker_cache)


@dataclass
class StockInfo:
    """Company information and current metrics."""

    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    current_price: float
    previous_close: float
    day_change: float
    day_change_percent: float
    volume: int
    avg_volume: int
    fifty_two_week_high: float
    fifty_two_week_low: float
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]


def fetch_stock_data(
    symbol: str,
    period: str = APP.DEFAULT_PERIOD,
) -> pd.DataFrame:
    """Fetch historical stock data from yfinance.

    Args:
        symbol: Stock ticker symbol (e.g., "MSFT")
        period: Time period for historical data. Valid values:
            1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

    Returns:
        DataFrame with OHLCV data indexed by date. Columns:
            Open, High, Low, Close, Volume, Dividends, Stock Splits

    Raises:
        ValueError: If symbol is invalid or data unavailable
    """
    try:
        ticker = get_ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            raise ValueError(f"No data available for symbol: {symbol}")

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone for simplicity

        return df

    except Exception as e:
        raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}") from e


def get_stock_info(symbol: str) -> StockInfo:
    """Get company information and current metrics.

    Args:
        symbol: Stock ticker symbol (e.g., "MSFT")

    Returns:
        StockInfo dataclass with company details and metrics

    Raises:
        ValueError: If symbol is invalid or info unavailable
    """
    try:
        ticker = get_ticker(symbol)
        info = ticker.info

        if not info or "shortName" not in info:
            raise ValueError(f"No info available for symbol: {symbol}")

        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        previous_close = info.get("previousClose", 0)
        day_change = current_price - previous_close
        day_change_percent = (day_change / previous_close * 100) if previous_close else 0

        return StockInfo(
            symbol=symbol.upper(),
            name=info.get("shortName", symbol),
            sector=info.get("sector", "N/A"),
            industry=info.get("industry", "N/A"),
            market_cap=info.get("marketCap", 0),
            current_price=current_price,
            previous_close=previous_close,
            day_change=day_change,
            day_change_percent=day_change_percent,
            volume=info.get("volume", 0),
            avg_volume=info.get("averageVolume", 0),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh", 0),
            fifty_two_week_low=info.get("fiftyTwoWeekLow", 0),
            pe_ratio=info.get("trailingPE"),
            dividend_yield=info.get("dividendYield"),
        )

    except Exception as e:
        raise ValueError(f"Failed to get info for {symbol}: {str(e)}") from e


def validate_symbol(symbol: str) -> bool:
    """Check if a stock symbol is valid.

    Args:
        symbol: Stock ticker symbol to validate

    Returns:
        True if symbol exists and has data, False otherwise
    """
    try:
        ticker = get_ticker(symbol)
        # Try to get basic info - if it fails, symbol is invalid
        info = ticker.info
        return bool(info and info.get("shortName"))
    except Exception:
        return False


def get_multiple_stocks(
    symbols: list[str],
    period: str = APP.DEFAULT_PERIOD,
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple stocks.

    Args:
        symbols: List of stock ticker symbols
        period: Time period for historical data

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    result: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, period)
            result[symbol.upper()] = df
        except ValueError:
            # Skip invalid symbols
            continue

    return result


def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calculate basic performance metrics from price data.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary with performance metrics:
            - total_return: Percentage return over period
            - max_drawdown: Maximum peak-to-trough decline
            - volatility: Annualized standard deviation of returns
            - start_price: First price in period
            - end_price: Last price in period
    """
    if df.empty:
        return {}

    close = df["Close"]

    # Total return
    start_price = close.iloc[0]
    end_price = close.iloc[-1]
    total_return = ((end_price - start_price) / start_price) * 100

    # Daily returns
    daily_returns = close.pct_change().dropna()

    # Volatility (annualized)
    volatility = daily_returns.std() * (252**0.5) * 100  # 252 trading days

    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    return {
        "total_return": round(total_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "volatility": round(volatility, 2),
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
    }
