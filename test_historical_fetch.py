"""Test the historical data fetching with one symbol."""

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Import from fetch script
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fetch_alpha_vantage import fetch_all_historical_data, START_DATE, END_DATE

load_dotenv()

def test_single_symbol():
    """Test fetching 10 years of data for one symbol."""
    if not os.getenv("ALPHA_VANTAGE_API_KEY"):
        print("Error: ALPHA_VANTAGE_API_KEY not found")
        return

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  Test: 10-Year Historical Data Fetch (Single Symbol)              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"\nDate range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print("="*70)

    # Test with AAPL (should have lots of articles)
    symbol = "AAPL"
    print(f"\nTesting with {symbol}...")

    try:
        all_articles = fetch_all_historical_data(symbol, topics=None)

        if all_articles:
            print(f"\n{'='*70}")
            print(f"✓ SUCCESS: Retrieved {len(all_articles)} total articles for {symbol}")
            print(f"{'='*70}")

            # Analyze results
            print(f"\nAnalysis:")
            print(f"  Total articles: {len(all_articles)}")

            # Date range
            if all_articles:
                dates = [a.get("time_published", "") for a in all_articles if a.get("time_published")]
                if dates:
                    earliest = min(dates)
                    latest = max(dates)
                    print(f"  Date range: {earliest[:8]} to {latest[:8]}")

            # Sentiment breakdown
            sentiments = {}
            for article in all_articles:
                for ts in article.get("ticker_sentiment", []):
                    if ts.get("ticker") == symbol:
                        label = ts.get("ticker_sentiment_label", "Unknown")
                        sentiments[label] = sentiments.get(label, 0) + 1

            if sentiments:
                print(f"\n  Sentiment breakdown:")
                for label, count in sorted(sentiments.items(), key=lambda x: -x[1]):
                    pct = (count / len(all_articles)) * 100
                    print(f"    {label}: {count} ({pct:.1f}%)")

            # Sample articles
            print(f"\n  Sample articles:")
            for i, article in enumerate(all_articles[:3], 1):
                print(f"\n  {i}. {article.get('title', 'N/A')[:60]}...")
                print(f"     Source: {article.get('source', 'N/A')}")
                print(f"     Date: {article.get('time_published', 'N/A')[:8]}")

        else:
            print(f"\n⚠ No articles found for {symbol}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("Test complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_single_symbol()
