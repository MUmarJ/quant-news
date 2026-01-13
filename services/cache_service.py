"""DuckDB caching service for stock data.

This module provides a caching layer using DuckDB to minimize API calls
and enable fast local data access with SQL query capabilities.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from config import APP


class CacheService:
    """DuckDB-based caching service for stock data.

    Provides methods to cache and retrieve stock prices, company info,
    and news articles with manual refresh capability.

    Attributes:
        db_path: Path to the DuckDB database file.
        conn: DuckDB connection object.
    """

    def __init__(self, db_path: str = "cache/quant_news.duckdb") -> None:
        """Initialize the cache service.

        Args:
            db_path: Path to the DuckDB database file. Created if not exists.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create exports directory
        (self.db_path.parent / "exports").mkdir(exist_ok=True)

        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema if tables don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                dividends DOUBLE,
                stock_splits DOUBLE,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                market_cap BIGINT,
                current_price DOUBLE,
                previous_close DOUBLE,
                pe_ratio DOUBLE,
                dividend_yield DOUBLE,
                fifty_two_week_high DOUBLE,
                fifty_two_week_low DOUBLE,
                volume BIGINT,
                avg_volume BIGINT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                symbol VARCHAR NOT NULL,
                data_type VARCHAR NOT NULL,
                period VARCHAR,
                last_updated TIMESTAMP,
                record_count INTEGER,
                PRIMARY KEY (symbol, data_type)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news_cache (
                id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                title VARCHAR,
                source VARCHAR,
                url VARCHAR,
                published_at TIMESTAMP,
                summary VARCHAR,
                sentiment VARCHAR,
                sentiment_score DOUBLE,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: Add sentiment_score column if it doesn't exist (for existing DBs)
        try:
            self.conn.execute("ALTER TABLE news_cache ADD COLUMN sentiment_score DOUBLE")
        except Exception:
            pass  # Column already exists

    def _checkpoint(self) -> None:
        """Force WAL checkpoint to prevent corruption on unexpected shutdown.

        This flushes the Write-Ahead Log to the main database file,
        preventing corruption if the process is killed mid-write
        (common during Dash debug mode hot reloads).
        """
        try:
            self.conn.execute("CHECKPOINT")
        except Exception:
            pass

    def is_cached(
        self,
        symbol: str,
        data_type: str = "prices",
        period: Optional[str] = None,
    ) -> bool:
        """Check if data is cached for a symbol.

        Args:
            symbol: Stock ticker symbol.
            data_type: Type of data ('prices', 'info', 'news').
            period: Time period for prices (e.g., '1y').

        Returns:
            True if cached data exists, False otherwise.
        """
        result = self.conn.execute("""
            SELECT COUNT(*) FROM cache_metadata
            WHERE symbol = ? AND data_type = ?
        """, [symbol.upper(), data_type]).fetchone()

        return result[0] > 0 if result else False

    def get_cache_info(self, symbol: str, data_type: str = "prices") -> Optional[dict]:
        """Get cache metadata for a symbol.

        Args:
            symbol: Stock ticker symbol.
            data_type: Type of data.

        Returns:
            Dictionary with cache info or None if not cached.
        """
        result = self.conn.execute("""
            SELECT period, last_updated, record_count
            FROM cache_metadata
            WHERE symbol = ? AND data_type = ?
        """, [symbol.upper(), data_type]).fetchone()

        if result:
            return {
                "period": result[0],
                "last_updated": result[1],
                "record_count": result[2],
            }
        return None

    def get_cached_news(
        self,
        symbol: str,
        max_age_minutes: int = 15,
    ) -> Optional[list[dict]]:
        """Get cached news articles for a symbol.

        Args:
            symbol: Stock ticker symbol.
            max_age_minutes: Maximum cache age in minutes (default 15).

        Returns:
            List of article dictionaries or None if cache is stale/empty.
        """
        symbol = symbol.upper().strip()
        cutoff_time = datetime.now() - pd.Timedelta(minutes=max_age_minutes)

        result = self.conn.execute("""
            SELECT id, symbol, title, source, url, published_at,
                   summary, sentiment, sentiment_score, fetched_at
            FROM news_cache
            WHERE symbol = ? AND fetched_at >= ?
            ORDER BY published_at DESC
        """, [symbol, cutoff_time]).fetchall()

        if not result:
            return None

        return [
            {
                "id": row[0],
                "symbol": row[1],
                "title": row[2],
                "source": row[3],
                "url": row[4],
                "published_at": row[5],
                "summary": row[6],
                "sentiment": row[7],
                "sentiment_score": row[8],
                "fetched_at": row[9],
            }
            for row in result
        ]

    def cache_news(self, symbol: str, articles: list) -> None:
        """Store news articles in cache.

        Args:
            symbol: Stock ticker symbol.
            articles: List of NewsArticle objects or dicts.
        """
        symbol = symbol.upper().strip()
        now = datetime.now()

        # Clear old news for this symbol
        self.conn.execute("DELETE FROM news_cache WHERE symbol = ?", [symbol])

        for article in articles:
            # Handle both NewsArticle objects and dicts
            if hasattr(article, "id"):
                article_id = article.id
                title = article.title
                source = article.source
                url = article.url
                published_at = article.published_at
                summary = article.summary
                sentiment = article.sentiment
                sentiment_score = article.sentiment_score
            else:
                article_id = article.get("id")
                title = article.get("title")
                source = article.get("source")
                url = article.get("url")
                published_at = article.get("published_at")
                summary = article.get("summary")
                sentiment = article.get("sentiment")
                sentiment_score = article.get("sentiment_score")

            self.conn.execute("""
                INSERT OR REPLACE INTO news_cache
                (id, symbol, title, source, url, published_at, summary, sentiment, sentiment_score, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [article_id, symbol, title, source, url, published_at, summary, sentiment, sentiment_score, now])

        # Update metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO cache_metadata (symbol, data_type, period, last_updated, record_count)
            VALUES (?, 'news', NULL, ?, ?)
        """, [symbol, now, len(articles)])

        self._checkpoint()

    def _period_to_days(self, period: str) -> int:
        """Convert period string to approximate number of days.

        Args:
            period: Time period string (e.g., '1mo', '3mo', '1y').

        Returns:
            Approximate number of days for the period.
        """
        period_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }
        return period_map.get(period, 365)  # Default to 1 year

    def get_stock_prices(
        self,
        symbol: str,
        period: str = APP.DEFAULT_PERIOD,
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """Get stock prices from cache or fetch from API.

        Args:
            symbol: Stock ticker symbol.
            period: Time period (e.g., '1y', '6mo', '3mo').
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            Tuple of (DataFrame with OHLCV data indexed by date, metadata dict).
            Metadata contains: from_cache (bool), api_error (str or None), cache_time (datetime or None).

        Raises:
            ValueError: If symbol is invalid or data unavailable.
        """
        symbol = symbol.upper().strip()
        metadata = {"from_cache": False, "api_error": None, "cache_time": None}

        # Calculate the start date based on period
        days = self._period_to_days(period)
        start_date = datetime.now() - pd.Timedelta(days=days)

        # Check cache first (unless force refresh)
        if not force_refresh and self.is_cached(symbol, "prices"):
            # Check if cache has data going back far enough for the requested period
            oldest_date_result = self.conn.execute("""
                SELECT MIN(date) FROM stock_prices WHERE symbol = ?
            """, [symbol]).fetchone()

            oldest_cached = oldest_date_result[0] if oldest_date_result else None
            cache_has_enough_data = False

            if oldest_cached:
                oldest_cached_dt = pd.to_datetime(oldest_cached)
                # Allow 7 days tolerance for weekends/holidays
                cache_has_enough_data = oldest_cached_dt <= (start_date + pd.Timedelta(days=7))

            if cache_has_enough_data:
                df = self.conn.execute("""
                    SELECT date, open, high, low, close, volume, dividends, stock_splits
                    FROM stock_prices
                    WHERE symbol = ? AND date >= ?
                    ORDER BY date
                """, [symbol, start_date.strftime("%Y-%m-%d")]).fetchdf()

                if not df.empty:
                    df.set_index("date", inplace=True)
                    df.index = pd.to_datetime(df.index)
                    # Rename columns to match yfinance format
                    df = df.rename(columns={
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume",
                        "dividends": "Dividends", "stock_splits": "Stock Splits"
                    })
                    # Get cache metadata
                    cache_info = self.get_cache_info(symbol, "prices")
                    metadata["from_cache"] = True
                    metadata["cache_time"] = cache_info.get("last_updated") if cache_info else None
                    return df, metadata

        # Fetch from API
        from services.stock_data import fetch_stock_data

        try:
            df = fetch_stock_data(symbol, period)
            # Store in cache
            self._cache_prices(symbol, df, period)
            metadata["from_cache"] = False
            return df, metadata
        except Exception as e:
            # API failed - try to return cached data if available
            if self.is_cached(symbol, "prices"):
                days = self._period_to_days(period)
                start_date = datetime.now() - pd.Timedelta(days=days)

                df = self.conn.execute("""
                    SELECT date, open, high, low, close, volume, dividends, stock_splits
                    FROM stock_prices
                    WHERE symbol = ? AND date >= ?
                    ORDER BY date
                """, [symbol, start_date.strftime("%Y-%m-%d")]).fetchdf()

                if not df.empty:
                    df.set_index("date", inplace=True)
                    df.index = pd.to_datetime(df.index)
                    df = df.rename(columns={
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume",
                        "dividends": "Dividends", "stock_splits": "Stock Splits"
                    })
                    cache_info = self.get_cache_info(symbol, "prices")
                    metadata["from_cache"] = True
                    metadata["api_error"] = str(e)
                    metadata["cache_time"] = cache_info.get("last_updated") if cache_info else None
                    return df, metadata

            # No cache available, re-raise the error
            raise

    def _cache_prices(self, symbol: str, df: pd.DataFrame, period: str) -> None:
        """Store price data in cache.

        Args:
            symbol: Stock ticker symbol.
            df: DataFrame with OHLCV data.
            period: Time period string.
        """
        if df.empty:
            return

        # Prepare data for insertion - select only the columns we need
        # yfinance may return varying columns, so be explicit
        cache_df = df.reset_index()

        # The index becomes the first column (date)
        # Map the columns we need, handling variations in yfinance output
        column_mapping = {}
        for col in cache_df.columns:
            col_lower = str(col).lower()
            if col_lower in ("date", "index"):
                column_mapping[col] = "date"
            elif col_lower == "open":
                column_mapping[col] = "open"
            elif col_lower == "high":
                column_mapping[col] = "high"
            elif col_lower == "low":
                column_mapping[col] = "low"
            elif col_lower == "close":
                column_mapping[col] = "close"
            elif col_lower == "volume":
                column_mapping[col] = "volume"
            elif col_lower == "dividends":
                column_mapping[col] = "dividends"
            elif col_lower in ("stock splits", "stock_splits"):
                column_mapping[col] = "stock_splits"

        cache_df = cache_df.rename(columns=column_mapping)

        # Ensure required columns exist, add with default values if missing
        required_cols = ["date", "open", "high", "low", "close", "volume", "dividends", "stock_splits"]
        for col in required_cols:
            if col not in cache_df.columns:
                cache_df[col] = 0.0 if col != "date" else None

        # Select only the columns we need in the right order
        cache_df = cache_df[required_cols]
        cache_df["symbol"] = symbol
        cache_df["fetched_at"] = datetime.now()

        # Delete existing data for this symbol
        self.conn.execute("DELETE FROM stock_prices WHERE symbol = ?", [symbol])

        # Insert new data
        self.conn.execute("""
            INSERT INTO stock_prices
            (symbol, date, open, high, low, close, volume, dividends, stock_splits, fetched_at)
            SELECT symbol, date, open, high, low, close, volume, dividends, stock_splits, fetched_at
            FROM cache_df
        """)

        # Update metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO cache_metadata (symbol, data_type, period, last_updated, record_count)
            VALUES (?, 'prices', ?, ?, ?)
        """, [symbol, period, datetime.now(), len(df)])

        self._checkpoint()

    def get_stock_info(self, symbol: str, force_refresh: bool = False) -> Optional[dict]:
        """Get company info from cache or fetch from API.

        Args:
            symbol: Stock ticker symbol.
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            Dictionary with company info or None if unavailable.
        """
        symbol = symbol.upper().strip()

        # Check cache first
        if not force_refresh and self.is_cached(symbol, "info"):
            result = self.conn.execute("""
                SELECT name, sector, industry, market_cap, current_price,
                       previous_close, pe_ratio, dividend_yield,
                       fifty_two_week_high, fifty_two_week_low, volume, avg_volume
                FROM stock_info
                WHERE symbol = ?
            """, [symbol]).fetchone()

            if result:
                return {
                    "symbol": symbol,
                    "name": result[0],
                    "sector": result[1],
                    "industry": result[2],
                    "market_cap": result[3],
                    "current_price": result[4],
                    "previous_close": result[5],
                    "pe_ratio": result[6],
                    "dividend_yield": result[7],
                    "fifty_two_week_high": result[8],
                    "fifty_two_week_low": result[9],
                    "volume": result[10],
                    "avg_volume": result[11],
                }

        # Fetch from API
        from services.stock_data import get_stock_info

        info = get_stock_info(symbol)

        # Store in cache
        self._cache_info(info)

        return {
            "symbol": info.symbol,
            "name": info.name,
            "sector": info.sector,
            "industry": info.industry,
            "market_cap": info.market_cap,
            "current_price": info.current_price,
            "previous_close": info.previous_close,
            "pe_ratio": info.pe_ratio,
            "dividend_yield": info.dividend_yield,
            "fifty_two_week_high": info.fifty_two_week_high,
            "fifty_two_week_low": info.fifty_two_week_low,
            "volume": info.volume,
            "avg_volume": info.avg_volume,
        }

    def _cache_info(self, info: "StockInfo") -> None:
        """Store company info in cache.

        Args:
            info: StockInfo dataclass from stock_data service.
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO stock_info
            (symbol, name, sector, industry, market_cap, current_price,
             previous_close, pe_ratio, dividend_yield, fifty_two_week_high,
             fifty_two_week_low, volume, avg_volume, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            info.symbol, info.name, info.sector, info.industry,
            info.market_cap, info.current_price, info.previous_close,
            info.pe_ratio, info.dividend_yield, info.fifty_two_week_high,
            info.fifty_two_week_low, info.volume, info.avg_volume,
            datetime.now()
        ])

        # Update metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO cache_metadata (symbol, data_type, period, last_updated, record_count)
            VALUES (?, 'info', NULL, ?, 1)
        """, [info.symbol, datetime.now()])

        self._checkpoint()

    def get_multiple_stocks(
        self,
        symbols: list[str],
        period: str = APP.DEFAULT_PERIOD,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Get data for multiple stocks.

        Args:
            symbols: List of stock ticker symbols.
            period: Time period for historical data.
            force_refresh: If True, bypass cache for all symbols.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        result: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                df = self.get_stock_prices(symbol, period, force_refresh)
                result[symbol.upper()] = df
            except ValueError:
                # Skip invalid symbols
                continue

        return result

    def export_to_parquet(
        self,
        symbol: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Export cached data to Parquet file.

        Args:
            symbol: Stock ticker symbol.
            output_path: Optional custom output path.

        Returns:
            Path to the exported file.

        Raises:
            ValueError: If no cached data exists for symbol.
        """
        symbol = symbol.upper()

        if not self.is_cached(symbol, "prices"):
            raise ValueError(f"No cached data for {symbol}")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = f"cache/exports/{symbol}_{timestamp}.parquet"

        self.conn.execute(f"""
            COPY (
                SELECT * FROM stock_prices WHERE symbol = '{symbol}'
            ) TO '{output_path}' (FORMAT PARQUET)
        """)

        return output_path

    def get_cache_status(self) -> list[dict]:
        """Get status of all cached data.

        Returns:
            List of dictionaries with cache info for each symbol.
        """
        result = self.conn.execute("""
            SELECT symbol, data_type, period, last_updated, record_count
            FROM cache_metadata
            ORDER BY symbol, data_type
        """).fetchall()

        return [
            {
                "symbol": row[0],
                "data_type": row[1],
                "period": row[2],
                "last_updated": row[3],
                "record_count": row[4],
            }
            for row in result
        ]

    def get_all_cached_symbols(self) -> list[str]:
        """Get list of all cached symbols.

        Returns:
            List of unique stock symbols in cache.
        """
        result = self.conn.execute("""
            SELECT DISTINCT symbol FROM cache_metadata
            WHERE data_type = 'prices'
            ORDER BY symbol
        """).fetchall()

        return [row[0] for row in result]

    def clear_symbol(self, symbol: str) -> None:
        """Clear all cached data for a symbol.

        Args:
            symbol: Stock ticker symbol to clear.
        """
        symbol = symbol.upper()
        self.conn.execute("DELETE FROM stock_prices WHERE symbol = ?", [symbol])
        self.conn.execute("DELETE FROM stock_info WHERE symbol = ?", [symbol])
        self.conn.execute("DELETE FROM news_cache WHERE symbol = ?", [symbol])
        self.conn.execute("DELETE FROM cache_metadata WHERE symbol = ?", [symbol])
        self._checkpoint()

    def clear_all(self) -> None:
        """Clear all cached data."""
        self.conn.execute("DELETE FROM stock_prices")
        self.conn.execute("DELETE FROM stock_info")
        self.conn.execute("DELETE FROM news_cache")
        self.conn.execute("DELETE FROM cache_metadata")
        self._checkpoint()

    def get_raw_data(self, symbol: str) -> pd.DataFrame:
        """Get raw cached data as DataFrame for display.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            DataFrame with all cached price data.
        """
        return self.conn.execute("""
            SELECT date, open, high, low, close, volume, fetched_at
            FROM stock_prices
            WHERE symbol = ?
            ORDER BY date DESC
        """, [symbol.upper()]).fetchdf()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> "CacheService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# Singleton instance for app-wide use
_cache_instance: Optional[CacheService] = None


def get_cache() -> CacheService:
    """Get the singleton cache service instance.

    Returns:
        CacheService instance.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance
