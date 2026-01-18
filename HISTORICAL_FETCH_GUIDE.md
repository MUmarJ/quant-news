# 10-Year Historical Data Fetch Guide

Your `fetch_alpha_vantage.py` script now automatically fetches 10 years of news sentiment data with intelligent chunking.

## What It Does

- **Date Range**: Automatically fetches from today back to 10 years ago (e.g., 2026-01-15 to 2016-01-15)
- **Smart Chunking**: Handles the 1000-article limit by breaking the time period into smaller chunks
- **Adaptive**: If a chunk has 1000+ articles, it automatically reduces chunk size
- **Resume-Safe**: Continues fetching even if errors occur in individual chunks

## Test Results (AAPL)

Successfully retrieved **3,496 articles** spanning 10 years:
- Date range: 2016-01-19 to 2026-01-15
- 21 chunks processed
- Sentiment: 39% Neutral, 37% Somewhat-Bullish, 12% Bullish, 9% Somewhat-Bearish, 3% Bearish

## Usage

### Run Full Fetch (All Symbols)

```bash
conda activate quant-news
python fetch_alpha_vantage.py
```

This will:
1. Process all 19 symbols in the `SYMBOLS` list
2. Fetch 10 years of data for each
3. Save to `alpha_vantage_data/{SYMBOL}_news_sentiment_10y.json`

### Test Single Symbol

```bash
python test_historical_fetch.py
```

This tests with AAPL to verify everything works before running the full fetch.

## Output Format

Each file contains:

```json
{
  "symbol": "AAPL",
  "time_from": "20160118T0000",
  "time_to": "20260115T2349",
  "total_articles": 3496,
  "feed": [
    {
      "title": "Article title",
      "source": "Source name",
      "time_published": "20260115T123456",
      "url": "https://...",
      "summary": "Article summary...",
      "overall_sentiment_score": 0.123,
      "overall_sentiment_label": "Bullish",
      "ticker_sentiment": [
        {
          "ticker": "AAPL",
          "ticker_sentiment_score": "0.334618",
          "ticker_sentiment_label": "Somewhat-Bullish"
        }
      ],
      "topics": [...]
    }
  ]
}
```

## Configuration

In `fetch_alpha_vantage.py`:

### 1. Change Symbols

```python
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]  # Edit this list
```

### 2. Add Topic Filters

```python
# In main() function, line ~202
topics = "earnings,technology,financial_markets"
```

### 3. Adjust Date Range

```python
# At top of file, line ~26-27
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5 * 365)  # 5 years instead of 10
```

### 4. Adjust Chunk Size

```python
# In fetch_all_historical_data(), line ~131
chunk_start = max(current_end - timedelta(days=90), target_start)  # 3 months instead of 6
```

### 5. Adjust Rate Limiting

```python
# Between chunks (line ~172)
time.sleep(2)  # Adjust based on your API plan

# Between symbols (line ~243)
time.sleep(5)  # Adjust based on your API plan
```

## How Chunking Works

1. **Initial Chunk**: Starts with 6-month periods
2. **Hit Limit**: If a chunk returns 1000 articles, it reduces to 1-month chunks
3. **Backward Progress**: Works backwards from today to 10 years ago
4. **Automatic Merging**: All chunks are combined into a single JSON file per symbol

### Example Flow

```
Chunk 1: 2025-07-19 to 2026-01-15 → 1000 articles (hit limit, reduce size)
Chunk 2: 2025-02-19 to 2025-08-18 → 144 articles (normal)
Chunk 3: 2024-08-23 to 2025-02-19 → 228 articles (normal)
...continues until reaching 2016-01-18...
Total: 3496 articles across 21 chunks
```

## Rate Limits & Timing

With your **premium API key**:
- Higher rate limits than free tier (check your plan)
- Script uses 2-second delays between chunks
- 5-second delays between symbols
- Adjust these in the code based on your plan limits

**Estimated time for 19 symbols**:
- ~20-25 chunks per symbol on average
- ~2 seconds per chunk
- ~5 seconds between symbols
- **Total: ~20-30 minutes** for all 19 symbols

## Error Handling

The script handles:
- **API errors**: Logs error and continues with next chunk
- **Rate limits**: Waits 5 seconds if rate limited
- **Network issues**: Catches exceptions and continues
- **Empty results**: Logs warning and moves to next chunk

## Post-Processing Ideas

Once you have the data, you can:

1. **Sentiment Analysis Over Time**
   ```python
   # Analyze how sentiment changed before/after earnings
   ```

2. **Topic Analysis**
   ```python
   # What topics correlate with price movements?
   ```

3. **Source Analysis**
   ```python
   # Which news sources are most reliable?
   ```

4. **Correlation Studies**
   ```python
   # Does sentiment predict stock price changes?
   ```

## Files

- `fetch_alpha_vantage.py` - Main script (now with 10-year support)
- `test_historical_fetch.py` - Test with single symbol
- `alpha_vantage_data/` - Output directory (created automatically)

## Next Steps

1. **Test first**: Run `test_historical_fetch.py` to verify your API key works
2. **Configure**: Adjust symbols, topics, and date range as needed
3. **Run**: Execute `fetch_alpha_vantage.py` for full fetch
4. **Analyze**: Use the JSON files for historical sentiment analysis

## Troubleshooting

**"Hit 1000 article limit" on every chunk**
- Stock has very high news volume
- Script will automatically reduce chunk size to 1 month

**API rate limit errors**
- Increase sleep times between requests
- Check your plan's rate limits

**Missing data for some periods**
- Alpha Vantage may not have data for all periods
- Older articles may be limited

**Out of memory**
- Script processes one symbol at a time to minimize memory
- Each symbol's data is saved independently
