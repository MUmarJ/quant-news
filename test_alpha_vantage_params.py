"""Test Alpha Vantage premium parameters for better news analysis."""

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

from services.news_service import fetch_alpha_vantage_news

# Load environment variables
load_dotenv()


def test_basic_fetch():
    """Test basic news fetch (no filters)."""
    print("=" * 70)
    print("TEST 1: Basic fetch (no filters)")
    print("=" * 70)
    articles = fetch_alpha_vantage_news("AAPL", max_articles=5)
    print(f"Retrieved {len(articles)} articles")
    if articles:
        print(f"\nLatest article:")
        print(f"  Title: {articles[0].title}")
        print(f"  Source: {articles[0].source}")
        print(f"  Sentiment: {articles[0].sentiment} ({articles[0].sentiment_score})")
        print(f"  Impact: {articles[0].impact}")
    print()


def test_topic_filter():
    """Test with topic filtering."""
    print("=" * 70)
    print("TEST 2: Topic filter (earnings only)")
    print("=" * 70)
    articles = fetch_alpha_vantage_news(
        "AAPL", max_articles=5, topics="earnings,financial_markets"
    )
    print(f"Retrieved {len(articles)} earnings-related articles")
    for i, article in enumerate(articles[:3], 1):
        print(f"\n{i}. {article.title[:60]}...")
        print(f"   Sentiment: {article.sentiment}")
    print()


def test_time_range():
    """Test with time range filtering."""
    print("=" * 70)
    print("TEST 3: Time range filter (last 7 days)")
    print("=" * 70)

    # Last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    time_from = start_date.strftime("%Y%m%dT%H%M")
    time_to = end_date.strftime("%Y%m%dT%H%M")

    print(f"Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    articles = fetch_alpha_vantage_news(
        "NVDA", max_articles=10, time_from=time_from, time_to=time_to
    )
    print(f"Retrieved {len(articles)} articles in date range")

    if articles:
        print(f"\nFirst article: {articles[0].published_at}")
        print(f"Last article:  {articles[-1].published_at}")
    print()


def test_relevance_sort():
    """Test with relevance sorting."""
    print("=" * 70)
    print("TEST 4: Relevance sorting")
    print("=" * 70)
    articles = fetch_alpha_vantage_news("MSFT", max_articles=5, sort="RELEVANCE")
    print(f"Retrieved {len(articles)} articles (sorted by relevance)")
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article.title[:60]}...")
        print(f"   Source: {article.source} | Sentiment: {article.sentiment}")
    print()


def test_combined_filters():
    """Test with multiple filters combined."""
    print("=" * 70)
    print("TEST 5: Combined filters (topic + time + sort)")
    print("=" * 70)

    # Last 30 days, earnings + tech topics, sorted by relevance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    articles = fetch_alpha_vantage_news(
        "TSLA",
        max_articles=10,
        topics="earnings,technology,financial_markets",
        time_from=start_date.strftime("%Y%m%dT%H%M"),
        time_to=end_date.strftime("%Y%m%dT%H%M"),
        sort="RELEVANCE",
    )

    print(f"Retrieved {len(articles)} articles")
    print("Filters: earnings/tech topics, last 30 days, sorted by relevance")

    # Analyze sentiment distribution
    bullish = sum(1 for a in articles if a.sentiment == "bullish")
    bearish = sum(1 for a in articles if a.sentiment == "bearish")
    neutral = sum(1 for a in articles if a.sentiment == "neutral")

    print(f"\nSentiment breakdown:")
    print(f"  Bullish: {bullish}")
    print(f"  Bearish: {bearish}")
    print(f"  Neutral: {neutral}")

    if articles:
        avg_score = sum(
            a.sentiment_score for a in articles if a.sentiment_score is not None
        ) / len([a for a in articles if a.sentiment_score is not None])
        print(f"  Average score: {avg_score:.3f}")
    print()


def main():
    """Run all tests."""
    if not os.getenv("ALPHA_VANTAGE_API_KEY"):
        print("Error: ALPHA_VANTAGE_API_KEY not found in .env file")
        return

    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   Alpha Vantage Premium Parameters Test Suite                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")

    test_basic_fetch()
    test_topic_filter()
    test_time_range()
    test_relevance_sort()
    test_combined_filters()

    print("=" * 70)
    print("All tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
