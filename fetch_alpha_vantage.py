"""Script to fetch 10 years of news sentiment items from Alpha Vantage.

This script retrieves news sentiment data from Alpha Vantage API
with automatic chunking for dates with >1000 articles.
Fetches from today back to 10 years ago (20260115 to 20160115).
"""

import json
import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
SYMBOLS = ["AES", "AOS", "APA", "ARE", "CAG", "CPB", "DVA", "FRT", "GNRC", "HSIC", "LW", "MGM", "MOH", "MOS", "MTCH", "PAYC", "POOL", "SWKS", "TAP"]
OUTPUT_DIR = "alpha_vantage_data"
BASE_URL = "https://www.alphavantage.co/query"

# Date range: 10 years from today
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=10 * 365)  # 10 years ago


def fetch_news_sentiment(
    symbol: str,
    limit: int = 1000,
    time_from: str = None,
    time_to: str = None,
    topics: str = None,
    sort: str = "LATEST",
) -> dict:
    """Fetch news sentiment data from Alpha Vantage and save to JSON file.

    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of news items to fetch (default 1000, max 1000)
        time_from: Start time in YYYYMMDDTHHMM format (optional)
        time_to: End time in YYYYMMDDTHHMM format (optional)
        topics: Comma-separated topics to filter by (optional)
                Examples: "earnings", "ipo", "mergers_and_acquisitions",
                         "financial_markets", "technology", "energy_transportation"
        sort: Sort order - "LATEST", "EARLIEST", or "RELEVANCE" (default: LATEST)

    Returns:
        The JSON response data
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": API_KEY,
        "limit": limit,
        "sort": sort,
    }

    # Add optional parameters if specified
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to
    if topics:
        params["topics"] = topics

    print(f"Fetching news sentiment for {symbol}...")
    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}")

    data = response.json()

    # Check for API errors
    if "Error Message" in data:
        raise Exception(f"API Error: {data['Error Message']}")

    if "Note" in data:
        raise Exception(f"API Rate Limit: {data['Note']}")

    if "Information" in data:
        print(f"  Info: {data['Information']}")

    # Save JSON response
    output_file = os.path.join(OUTPUT_DIR, f"{symbol}_news_sentiment.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved to {output_file}")

    # Count news items
    if "feed" in data:
        num_items = len(data["feed"])
        print(f"  Retrieved {num_items} news items")

        # Show sample of first article if available
        if num_items > 0:
            first_article = data["feed"][0]
            print(f"  Latest: {first_article.get('title', 'N/A')[:60]}...")
            if "overall_sentiment_score" in first_article:
                print(f"  Sentiment: {first_article['overall_sentiment_score']}")

    return data


def fetch_all_historical_data(symbol: str, topics: str = None) -> list[dict]:
    """Fetch all available data for a symbol using cursor-based pagination.

    Alpha Vantage has a 1000 article limit per request. This function
    fetches 1000 items at a time, using the oldest article's date from
    each batch as the time_to for the next request.

    Args:
        symbol: Stock ticker symbol
        topics: Optional topic filters

    Returns:
        List of all article dictionaries
    """
    all_articles = []
    target_start = START_DATE
    time_from = target_start.strftime("%Y%m%dT0000")  # Start of target date
    time_to = None  # Start with no upper bound (fetches from latest)

    batch_num = 1
    print(f"\nFetching historical data back to {START_DATE.strftime('%Y-%m-%d')}")

    while True:
        print(f"  Batch {batch_num}: fetching up to 1000 articles" +
              (f" before {time_to[:8]}" if time_to else " (latest)"))

        try:
            data = fetch_news_sentiment(
                symbol,
                limit=1000,
                time_from=time_from,
                time_to=time_to,
                topics=topics,
                sort="LATEST",
            )

            if "feed" not in data or len(data["feed"]) == 0:
                print(f"    No more articles found")
                break

            articles = data["feed"]
            num_articles = len(articles)
            all_articles.extend(articles)

            print(f"    Retrieved {num_articles} articles (total: {len(all_articles)})")

            # Get the oldest article's date to use as time_to for next batch
            oldest_article = articles[-1]  # Last item when sorted by LATEST
            oldest_date_str = oldest_article.get("time_published", "")

            if not oldest_date_str:
                print(f"    ⚠ Could not get date from oldest article, stopping")
                break

            # Check if we've reached our target start date
            oldest_date = datetime.strptime(oldest_date_str[:8], "%Y%m%d")
            if oldest_date <= target_start:
                print(f"    ✓ Reached target date {target_start.strftime('%Y-%m-%d')}")
                break

            # If we got fewer than 1000, there are no more articles
            if num_articles < 1000:
                print(f"    ✓ Fetched all available articles")
                break

            # Use oldest date as the upper bound for next request
            # Format: YYYYMMDDTHHMM (truncate seconds from YYYYMMDDTHHMMSS)
            time_to = oldest_date_str[:13]  # e.g., "20211101T1234"

            batch_num += 1

        except Exception as e:
            print(f"    Error in batch: {e}")
            time.sleep(5)
            # Try to continue if we have a time_to to work with
            if time_to:
                continue
            break

    return all_articles


def main():
    """Main function to fetch 10 years of news sentiment for all symbols."""
    if not API_KEY:
        print("Error: ALPHA_VANTAGE_API_KEY not found in .env file")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  Alpha Vantage Historical Data Fetch (10 Years)                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"\nDate range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Symbols: {len(SYMBOLS)}")
    print("="*70)

    # Optional: Add topic filters for more focused analysis
    # Uncomment and modify as needed:
    # topics = "earnings,financial_markets,technology"
    topics = None

    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] Processing {symbol}")
        print("-" * 70)

        try:
            # Fetch all historical data with automatic chunking
            all_articles = fetch_all_historical_data(symbol, topics=topics)

            if all_articles:
                # Save combined data (generic filename, not time-specific)
                output_file = os.path.join(OUTPUT_DIR, f"{symbol}_news_sentiment.json")
                combined_data = {
                    "symbol": symbol,
                    "time_from": START_DATE.strftime("%Y%m%dT%H%M"),
                    "time_to": END_DATE.strftime("%Y%m%dT%H%M"),
                    "total_articles": len(all_articles),
                    "feed": all_articles,
                }

                with open(output_file, "w") as f:
                    json.dump(combined_data, f, indent=2)

                print(f"\n  ✓ Saved {len(all_articles)} total articles to {output_file}")

                # Show date range of articles
                if all_articles:
                    first_date = all_articles[0].get("time_published", "Unknown")
                    last_date = all_articles[-1].get("time_published", "Unknown")
                    print(f"  Date range: {last_date[:8]} to {first_date[:8]}")
            else:
                print(f"\n  ⚠ No articles found for {symbol}")

        except Exception as e:
            print(f"\n  ✗ Error processing {symbol}: {e}")
            continue


    print("\n" + "="*70)
    print(f"✓ All data saved to {OUTPUT_DIR}/ directory")
    print("="*70)


if __name__ == "__main__":
    main()
