"""News service for fetching and aggregating stock news.

This module provides functions to fetch news from multiple sources:
- yfinance (built-in, no API key needed)
- Alpha Vantage News API (requires free API key)
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

from config import API


@dataclass
class NewsArticle:
    """Represents a news article."""

    id: str
    symbol: str
    title: str
    source: str
    url: str
    published_at: datetime
    summary: Optional[str] = None
    sentiment: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    sentiment_score: Optional[float] = None
    impact: Optional[str] = None  # How the stock is affected (e.g., "price target raised")
    price_change_percent: Optional[float] = None  # Current stock price change %


def _extract_stock_impact(title: str, summary: str = "") -> Optional[str]:
    """Extract how the stock is affected from the article title and summary.

    Args:
        title: Article title.
        summary: Article summary (optional).

    Returns:
        Description of the stock impact, or None if not determinable.
    """
    text = f"{title} {summary}".lower()

    # Define impact patterns (order matters - more specific first)
    impact_patterns = [
        # Price targets and ratings
        (["price target raised", "raises price target", "ups price target"], "Price target raised"),
        (["price target lowered", "lowers price target", "cuts price target"], "Price target lowered"),
        (["price target"], "Price target updated"),
        (["upgrade", "upgraded"], "Stock upgraded"),
        (["downgrade", "downgraded"], "Stock downgraded"),
        (["initiates coverage", "starts coverage", "begins coverage"], "Coverage initiated"),
        (["buy rating", "outperform rating", "overweight"], "Positive rating"),
        (["sell rating", "underperform rating", "underweight"], "Negative rating"),
        # Earnings and financials
        (["beats estimates", "beats expectations", "tops estimates", "earnings beat"], "Earnings beat"),
        (["misses estimates", "misses expectations", "earnings miss"], "Earnings miss"),
        (["revenue growth", "sales growth", "revenue up", "sales up"], "Revenue growth"),
        (["revenue decline", "sales decline", "revenue down", "sales down"], "Revenue decline"),
        (["profit increase", "profit growth", "net income up"], "Profit growth"),
        (["profit decline", "net income down", "loss reported"], "Profit decline"),
        (["guidance raised", "raises guidance", "raises outlook"], "Guidance raised"),
        (["guidance lowered", "lowers guidance", "cuts outlook"], "Guidance lowered"),
        (["dividend increase", "raises dividend", "dividend hike"], "Dividend increased"),
        (["dividend cut", "suspends dividend", "dividend reduced"], "Dividend cut"),
        (["stock buyback", "share repurchase", "buyback program"], "Stock buyback announced"),
        # Corporate actions
        (["acquisition", "acquires", "to acquire", "buyout"], "Acquisition news"),
        (["merger", "to merge", "merging with"], "Merger news"),
        (["ipo", "initial public offering", "goes public"], "IPO news"),
        (["stock split", "split announced"], "Stock split"),
        (["spinoff", "spin-off", "spins off"], "Spinoff announced"),
        # Legal and regulatory
        (["lawsuit", "sued", "legal action", "litigation"], "Legal action"),
        (["sec investigation", "regulatory probe", "investigation"], "Regulatory investigation"),
        (["fda approval", "drug approved", "receives approval"], "FDA/Regulatory approval"),
        (["fda rejection", "drug rejected", "approval denied"], "FDA/Regulatory rejection"),
        (["settles", "settlement", "agrees to pay"], "Legal settlement"),
        # Business operations
        (["layoffs", "job cuts", "workforce reduction", "cutting jobs"], "Layoffs announced"),
        (["hiring", "adding jobs", "workforce expansion"], "Hiring expansion"),
        (["new product", "product launch", "launches", "unveils"], "Product launch"),
        (["partnership", "partners with", "collaboration"], "Partnership announced"),
        (["contract win", "wins contract", "awarded contract"], "Contract awarded"),
        (["loses contract", "contract loss"], "Contract lost"),
        (["expansion", "expands into", "new market"], "Market expansion"),
        (["restructuring", "reorganization"], "Restructuring"),
        # Market sentiment
        (["insider buying", "insiders buy", "ceo buys"], "Insider buying"),
        (["insider selling", "insiders sell", "ceo sells"], "Insider selling"),
        (["short interest", "heavily shorted", "short squeeze"], "Short interest news"),
        (["analyst bullish", "analysts optimistic"], "Bullish analyst sentiment"),
        (["analyst bearish", "analysts pessimistic"], "Bearish analyst sentiment"),
        # External factors
        (["tariff", "trade war", "import duty"], "Trade/Tariff impact"),
        (["supply chain", "chip shortage", "supply shortage"], "Supply chain impact"),
        (["recall", "product recall"], "Product recall"),
    ]

    for keywords, impact in impact_patterns:
        if any(keyword in text for keyword in keywords):
            return impact

    return None


def _generate_article_id(title: str, source: str, published_at: datetime) -> str:
    """Generate unique ID for an article.

    Args:
        title: Article title.
        source: News source name.
        published_at: Publication timestamp.

    Returns:
        MD5 hash string as unique identifier.
    """
    content = f"{title}|{source}|{published_at.isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()


def fetch_yfinance_news(symbol: str, max_articles: int = 10) -> list[NewsArticle]:
    """Fetch news from yfinance.

    Args:
        symbol: Stock ticker symbol.
        max_articles: Maximum number of articles to return.

    Returns:
        List of NewsArticle objects.
    """
    try:
        from services.stock_data import get_ticker

        ticker = get_ticker(symbol)
        news = ticker.news

        if not news:
            return []

        articles = []
        for item in news[:max_articles]:
            # Parse timestamp
            pub_time = datetime.fromtimestamp(item.get("providerPublishTime", 0))

            title = item.get("title", "")
            article = NewsArticle(
                id=_generate_article_id(
                    title,
                    item.get("publisher", ""),
                    pub_time,
                ),
                symbol=symbol.upper(),
                title=title,
                source=item.get("publisher", "Unknown"),
                url=item.get("link", ""),
                published_at=pub_time,
                summary=None,  # yfinance doesn't provide summaries
                sentiment=None,
                impact=_extract_stock_impact(title),
            )
            articles.append(article)

        return articles

    except Exception:
        return []


def fetch_alpha_vantage_news(
    symbol: str,
    max_articles: int = 10,
    topics: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "LATEST",
) -> list[NewsArticle]:
    """Fetch news from Alpha Vantage News API.

    Includes AI-powered sentiment scores.

    Args:
        symbol: Stock ticker symbol.
        max_articles: Maximum number of articles to return.
        topics: Comma-separated topics to filter by (e.g., "earnings,technology").
                Available topics: earnings, ipo, mergers_and_acquisitions,
                financial_markets, economy_fiscal, economy_monetary, economy_macro,
                energy_transportation, finance, life_sciences, manufacturing,
                real_estate, retail_wholesale, technology
        time_from: Start time in YYYYMMDDTHHMM format (e.g., "20260101T0000")
        time_to: End time in YYYYMMDDTHHMM format (e.g., "20260115T2359")
        sort: Sort order - "LATEST", "EARLIEST", or "RELEVANCE" (default: LATEST)

    Returns:
        List of NewsArticle objects with sentiment data.
    """
    if not API.ALPHA_VANTAGE_API_KEY:
        return []

    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol.upper(),
            "limit": max_articles,
            "apikey": API.ALPHA_VANTAGE_API_KEY,
            "sort": sort,
        }

        # Add optional filters
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        response = requests.get(
            API.ALPHA_VANTAGE_BASE_URL,
            params=params,
            timeout=API.DEFAULT_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()

        if "feed" not in data:
            return []

        articles = []
        for item in data["feed"][:max_articles]:
            # Parse timestamp (format: 20240115T120000)
            time_str = item.get("time_published", "")
            try:
                pub_time = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
            except ValueError:
                pub_time = datetime.now()

            # Get ticker-specific sentiment
            ticker_sentiment = None
            sentiment_score = None
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol.upper():
                    sentiment_score = float(ts.get("ticker_sentiment_score", 0))
                    label = ts.get("ticker_sentiment_label", "")
                    if "bullish" in label.lower():
                        ticker_sentiment = "bullish"
                    elif "bearish" in label.lower():
                        ticker_sentiment = "bearish"
                    else:
                        ticker_sentiment = "neutral"
                    break

            # If no ticker-specific sentiment, use overall
            if ticker_sentiment is None:
                overall_score = float(item.get("overall_sentiment_score", 0))
                if overall_score > 0.15:
                    ticker_sentiment = "bullish"
                elif overall_score < -0.15:
                    ticker_sentiment = "bearish"
                else:
                    ticker_sentiment = "neutral"
                sentiment_score = overall_score

            title = item.get("title", "")
            summary = item.get("summary", "")
            article = NewsArticle(
                id=_generate_article_id(
                    title,
                    item.get("source", ""),
                    pub_time,
                ),
                symbol=symbol.upper(),
                title=title,
                source=item.get("source", "Unknown"),
                url=item.get("url", ""),
                published_at=pub_time,
                summary=summary,
                sentiment=ticker_sentiment,
                sentiment_score=sentiment_score,
                impact=_extract_stock_impact(title, summary),
            )
            articles.append(article)

        return articles

    except Exception:
        return []


def _get_stock_price_change(symbol: str) -> Optional[float]:
    """Get the current day's price change percentage for a stock.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Price change percentage, or None if unavailable.
    """
    try:
        from services.stock_data import get_stock_info

        info = get_stock_info(symbol)
        return round(info.day_change_percent, 2)
    except Exception:
        return None


def fetch_news(
    symbol: str,
    max_articles: int = 10,
    prefer_alpha_vantage: bool = True,
    include_price_change: bool = True,
    topics: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "LATEST",
) -> list[NewsArticle]:
    """Fetch news from available sources.

    Tries Alpha Vantage first (if API key available), falls back to yfinance.

    Args:
        symbol: Stock ticker symbol.
        max_articles: Maximum number of articles to return.
        prefer_alpha_vantage: If True, try Alpha Vantage first.
        include_price_change: If True, fetch and include current stock price change.
        topics: Comma-separated topics to filter by (Alpha Vantage only).
        time_from: Start time in YYYYMMDDTHHMM format (Alpha Vantage only).
        time_to: End time in YYYYMMDDTHHMM format (Alpha Vantage only).
        sort: Sort order - "LATEST", "EARLIEST", or "RELEVANCE" (Alpha Vantage only).

    Returns:
        List of NewsArticle objects, sorted by date (newest first).
    """
    articles: list[NewsArticle] = []

    # Try Alpha Vantage first (has sentiment)
    if prefer_alpha_vantage and API.ALPHA_VANTAGE_API_KEY:
        articles = fetch_alpha_vantage_news(
            symbol,
            max_articles,
            topics=topics,
            time_from=time_from,
            time_to=time_to,
            sort=sort,
        )

    # Fallback to yfinance if no results
    if not articles:
        articles = fetch_yfinance_news(symbol, max_articles)

    # Fetch current stock price change if requested
    if include_price_change and articles:
        price_change = _get_stock_price_change(symbol)
        if price_change is not None:
            for article in articles:
                article.price_change_percent = price_change

    # Sort by date (newest first)
    articles.sort(key=lambda x: x.published_at, reverse=True)

    return articles[:max_articles]


def fetch_news_cached(
    symbol: str,
    max_articles: int = 10,
    cache_minutes: int = 15,
) -> list[NewsArticle]:
    """Fetch news with DuckDB caching support.

    Checks cache first and returns cached articles if fresh enough.
    Otherwise fetches from API and caches the results.

    Args:
        symbol: Stock ticker symbol.
        max_articles: Maximum number of articles.
        cache_minutes: Cache validity in minutes (default 15).

    Returns:
        List of NewsArticle objects.
    """
    from services.cache_service import get_cache

    cache = get_cache()

    # Try cache first
    cached = cache.get_cached_news(symbol, cache_minutes)
    if cached:
        articles = [
            NewsArticle(
                id=a["id"],
                symbol=a["symbol"],
                title=a["title"],
                source=a["source"],
                url=a["url"],
                published_at=a["published_at"],
                summary=a.get("summary"),
                sentiment=a.get("sentiment"),
                sentiment_score=a.get("sentiment_score"),
                impact=a.get("impact"),
            )
            for a in cached[:max_articles]
        ]
        # Always fetch fresh price change even for cached articles
        price_change = _get_stock_price_change(symbol)
        if price_change is not None:
            for article in articles:
                article.price_change_percent = price_change
        return articles

    # Fetch fresh data
    articles = fetch_news(symbol, max_articles)

    # Cache the results
    if articles:
        cache.cache_news(symbol, articles)

    return articles


def fetch_news_multiple(
    symbols: list[str],
    max_per_symbol: int = 5,
) -> dict[str, list[NewsArticle]]:
    """Fetch news for multiple symbols.

    Args:
        symbols: List of stock ticker symbols.
        max_per_symbol: Maximum articles per symbol.

    Returns:
        Dictionary mapping symbol to list of articles.
    """
    result: dict[str, list[NewsArticle]] = {}

    for symbol in symbols:
        articles = fetch_news(symbol, max_per_symbol)
        result[symbol.upper()] = articles

    return result


def get_sentiment_summary(articles: list[NewsArticle]) -> dict:
    """Calculate sentiment summary from articles.

    Args:
        articles: List of NewsArticle objects.

    Returns:
        Dictionary with sentiment counts and overall assessment.
    """
    if not articles:
        return {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
            "total": 0,
            "overall": "neutral",
            "score": 0.0,
        }

    bullish = sum(1 for a in articles if a.sentiment == "bullish")
    bearish = sum(1 for a in articles if a.sentiment == "bearish")
    neutral = sum(1 for a in articles if a.sentiment == "neutral" or a.sentiment is None)
    total = len(articles)

    # Calculate average sentiment score
    scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Determine overall sentiment
    if bullish > bearish and bullish > neutral:
        overall = "bullish"
    elif bearish > bullish and bearish > neutral:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "total": total,
        "overall": overall,
        "score": round(avg_score, 3),
    }


def format_time_ago(dt: datetime) -> str:
    """Format datetime as relative time string.

    Args:
        dt: Datetime to format.

    Returns:
        Human-readable relative time (e.g., "2h ago", "3d ago").
    """
    now = datetime.now()
    diff = now - dt

    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days}d ago"
    else:
        weeks = int(seconds / 604800)
        return f"{weeks}w ago"
