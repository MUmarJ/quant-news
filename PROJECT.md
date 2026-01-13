# Quant News Tracker

A Python quantitative stock tracking application that provides real-time market analysis, news summaries with AI-powered insights, and comprehensive technical indicators.

## Features

- **Stock Symbol Input**: Track multiple equities (MSFT, AAPL, NVDA, etc.)
- **1-Year Performance Analysis**: Historical price data with key metrics
- **Technical Indicators**: MACD, RSI, Bollinger Bands, and more
- **AI-Powered News Summaries**: Using LM Studio (local) or OpenAI
- **Sentiment Analysis**: Market sentiment from news sources

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **UI Framework** | Plotly Dash + Bootstrap | Production-ready, callback-based, excellent charting |
| **Data Source** | yfinance | Free, reliable, actively maintained |
| **Data Cache** | DuckDB | Single-file DB, SQL queries, native Parquet I/O, fast analytics |
| **Technical Analysis** | `ta` library | Comprehensive indicator calculations |
| **LLM** | LM Studio / OpenAI | Configurable, OpenAI-compatible API |
| **News** | Alpha Vantage + yfinance | Multi-source for comprehensive coverage |

---

## Architecture

```
quant-news/
├── app.py                     # Main Dash app entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── config.py                 # Configuration settings
│
├── cache/                     # Local data cache
│   ├── quant_news.duckdb     # DuckDB database (auto-created)
│   └── exports/              # User-exported Parquet files
│
├── assets/
│   └── styles.css            # Custom CSS styling
│
├── layouts/
│   ├── main_layout.py        # Main dashboard layout
│   └── components.py         # Reusable UI components (cards, inputs)
│
├── callbacks/
│   ├── stock_callbacks.py    # Stock data fetching callbacks
│   ├── chart_callbacks.py    # Chart update callbacks
│   └── news_callbacks.py     # News section callbacks
│
├── services/
│   ├── stock_data.py         # yfinance wrapper
│   ├── cache_service.py      # DuckDB caching layer
│   ├── news_service.py       # News fetching & aggregation
│   ├── llm_service.py        # LM Studio/OpenAI integration
│   └── analytics.py          # Technical indicators (MACD, RSI, etc.)
│
└── utils/
    └── helpers.py            # Utility functions
```

---

## Data Caching Strategy

### Overview
- **Database**: DuckDB (single-file, embedded, fast analytics)
- **Data Granularity**: Monthly ticker data only (minimize API calls)
- **Cache Invalidation**: Manual refresh only (user-triggered)
- **Export Format**: Parquet files for user downloads

### Why DuckDB
1. **Single file** - No server, portable `quant_news.duckdb`
2. **SQL interface** - Easy queries across cached data
3. **Columnar storage** - Fast aggregations on financial data
4. **Native Parquet I/O** - Direct export without conversion
5. **Python integration** - Returns pandas DataFrames directly

### Database Schema

```sql
-- Stock price data (monthly granularity)
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
);

-- Company information cache
CREATE TABLE IF NOT EXISTS stock_info (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    market_cap BIGINT,
    pe_ratio DOUBLE,
    dividend_yield DOUBLE,
    fifty_two_week_high DOUBLE,
    fifty_two_week_low DOUBLE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cache metadata for tracking freshness
CREATE TABLE IF NOT EXISTS cache_metadata (
    symbol VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,  -- 'prices', 'info', 'news'
    period VARCHAR,              -- '1y', '6mo', etc.
    last_updated TIMESTAMP,
    record_count INTEGER,
    PRIMARY KEY (symbol, data_type)
);

-- News articles cache
CREATE TABLE IF NOT EXISTS news_cache (
    id VARCHAR PRIMARY KEY,      -- Hash of title+source+date
    symbol VARCHAR NOT NULL,
    title VARCHAR,
    source VARCHAR,
    url VARCHAR,
    published_at TIMESTAMP,
    summary VARCHAR,
    sentiment VARCHAR,           -- 'bullish', 'bearish', 'neutral'
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Cache Flow

```
User enters symbols → Check DuckDB cache
                           ↓
                    Cache hit? ──Yes──→ Return cached data
                           ↓ No
                    Fetch from yfinance
                           ↓
                    Store in DuckDB
                           ↓
                    Return data + update metadata
```

### API Usage

```python
from services.cache_service import CacheService

cache = CacheService()

# Fetch with caching (returns from cache if available)
df = cache.get_stock_prices("MSFT", period="1y")

# Force refresh (bypass cache)
df = cache.get_stock_prices("MSFT", period="1y", force_refresh=True)

# Get multiple symbols
data = cache.get_multiple_stocks(["MSFT", "AAPL", "NVDA"], period="1y")

# Export to Parquet
cache.export_to_parquet("MSFT", "cache/exports/MSFT_1y.parquet")

# View cache status
status = cache.get_cache_status()  # Returns dict of cached symbols + dates

# Clear cache for symbol
cache.clear_symbol("MSFT")

# Clear all cache
cache.clear_all()
```

### UI Features for Data Management

1. **View Raw Data** button - Opens modal with DataTable of cached data
2. **Export Parquet** button - Downloads selected symbol data as .parquet
3. **Refresh Data** button - Force re-fetch from API (manual invalidation)
4. **Cache Status** indicator - Shows last update time per symbol

---

## Technical Indicators

### Trend Indicators
- **SMA** (Simple Moving Average): 20, 50, 200 day
- **EMA** (Exponential Moving Average): 12, 26 day
- **MACD** (12, 26, 9): Line, signal, and histogram

### Momentum Indicators
- **RSI** (Relative Strength Index): 14 day
- **Stochastic Oscillator**: %K, %D
- **ROC** (Rate of Change)

### Volatility Indicators
- **Bollinger Bands**: 20 day, 2 standard deviations
- **ATR** (Average True Range): 14 day
- **Rolling Standard Deviation**

### Volume Indicators
- **Volume Bars**: Color-coded (green/red for up/down days)
- **OBV** (On-Balance Volume)
- **Volume Moving Average**: 20 day

### Performance Metrics
- 1-Year Return %
- Maximum Drawdown
- Sharpe Ratio (configurable risk-free rate)

---

## Dashboard Components

### Stock Input
- Multi-symbol input (comma-separated: "MSFT, AAPL, NVDA")
- Symbol validation
- Quick-add buttons for popular stocks
- Recent searches history

### Charts
- **Price Chart**: Candlestick with MA/Bollinger overlays
- **MACD Chart**: Line, signal, histogram subplot
- **RSI Chart**: With overbought (70) / oversold (30) zones
- **Volume Chart**: Color-coded bars with OBV overlay
- **Comparison Mode**: Normalized % overlay for multiple stocks

### News Section
- Recent headlines per ticker
- AI-generated summary of key themes
- Sentiment badges (Bullish / Bearish / Neutral)
- Source attribution and timestamps

---

## Implementation Phases

### Phase 1: Core Setup
1. Initialize project structure
2. Create `requirements.txt`
3. Set up configuration (.env support)

### Phase 2: Data Layer
4. Build yfinance stock data service
   - Fetch 1-year historical data
   - Get company info and basic metrics
5. Implement news aggregation
   - yfinance news (built-in)
   - Alpha Vantage API (optional)

### Phase 3: Analytics
6. Implement technical indicators using `ta` library
   - All trend, momentum, volatility, volume indicators
   - Performance metrics calculations

### Phase 4: LLM Integration
7. Create LLM service
   - Auto-detect LM Studio (localhost:1234)
   - Fallback to OpenAI if configured
   - Generate news summaries
   - Create sentiment analysis

### Phase 5: UI Development
8. Build Dash app structure
   - Bootstrap theme (dash-bootstrap-components)
   - Responsive sidebar + main content layout
9. Implement callbacks
   - Stock selection → data fetch
   - Timeframe → chart update
   - Indicator toggles
10. Create Plotly charts
    - Price, MACD, RSI, Volume charts
11. Build news components
    - Bootstrap cards, sentiment badges

### Phase 6: Polish
12. Loading spinners and skeleton states
13. Error handling with alerts
14. Custom CSS styling
15. Responsive design

---

## Dependencies

```txt
dash>=2.15.0
dash-bootstrap-components>=1.5.0
plotly>=5.18.0
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.24.0
ta>=0.11.0
openai>=1.12.0
python-dotenv>=1.0.0
requests>=2.31.0
```

---

## Configuration

### Environment Variables (.env)

```env
# LLM Configuration
LM_STUDIO_URL=http://localhost:1234/v1
OPENAI_API_KEY=sk-...  # Optional fallback

# News API (optional)
ALPHA_VANTAGE_API_KEY=your_key_here

# App Settings
DEBUG=true
PORT=8050
```

---

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Access at http://localhost:8050
```

---

## Verification

1. **Unit Tests**: Test each service independently
2. **Integration Test**: Fetch data for MSFT, verify all components render
3. **LLM Test**: Verify LM Studio connection, test OpenAI fallback
4. **Manual Testing**:
   - Enter multiple symbols
   - Verify charts display correctly
   - Check news summaries generate
   - Test error handling (invalid symbols, API failures)

---

## Code Standards & Best Practices

### DRY Principles
- **No code duplication**: Extract common logic into reusable functions/classes
- **Single Source of Truth**: All constants defined once in `config.py` or dedicated constants modules
- **Modular structure**: Each module has a single responsibility

### Constants Management
```python
# config.py - Single source of truth for all constants
class IndicatorDefaults:
    SMA_PERIODS = [20, 50, 200]
    EMA_PERIODS = [12, 26]
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2

class APIConfig:
    LM_STUDIO_URL = "http://localhost:1234/v1"
    DEFAULT_TIMEOUT = 30
```

### PEP 8 Compliance
- **Line length**: Max 88 characters (Black default)
- **Imports**: Grouped and sorted (stdlib, third-party, local)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Spacing**: 2 blank lines between top-level definitions

### Type Hints
Every function must include type hints:
```python
def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical stock data from yfinance.

    Args:
        symbol: Stock ticker symbol (e.g., "MSFT")
        period: Time period for historical data (default: "1y")

    Returns:
        DataFrame with OHLCV data indexed by date

    Raises:
        ValueError: If symbol is invalid or data unavailable
    """
    ...
```

### Docstrings
- **Google style** docstrings for all public functions and classes
- Include: Args, Returns, Raises sections
- Brief description on first line

### Linting & Formatting
```bash
# Pre-commit hooks (run before each commit)
black .                    # Code formatting
isort .                    # Import sorting
flake8 .                   # Style checking
mypy .                     # Type checking
```

### Project Structure (Updated)
```
quant-news/
├── app.py                     # Main entry point
├── config.py                  # Constants & configuration (SINGLE SOURCE OF TRUTH)
├── requirements.txt
├── requirements-dev.txt       # Dev dependencies (black, flake8, mypy, pytest)
├── .env.example
├── pyproject.toml            # Black, isort, mypy config
├── .pre-commit-config.yaml   # Pre-commit hooks
│
├── assets/
│   └── styles.css
│
├── layouts/                   # UI layouts
│   ├── __init__.py
│   ├── main_layout.py
│   └── components.py
│
├── callbacks/                 # Dash callbacks
│   ├── __init__.py
│   ├── stock_callbacks.py
│   ├── chart_callbacks.py
│   └── news_callbacks.py
│
├── services/                  # Core business logic
│   ├── __init__.py
│   ├── stock_data.py
│   ├── news_service.py
│   ├── llm_service.py
│   └── analytics.py
│
├── utils/                     # Shared utilities
│   ├── __init__.py
│   ├── helpers.py
│   └── validators.py
│
└── tests/                     # Test suite
    ├── __init__.py
    ├── conftest.py           # Pytest fixtures
    ├── test_stock_data.py
    ├── test_analytics.py
    ├── test_news_service.py
    └── test_llm_service.py
```

### Development Dependencies
```txt
# requirements-dev.txt
black>=24.0.0
isort>=5.13.0
flake8>=7.0.0
mypy>=1.8.0
pytest>=8.0.0
pytest-cov>=4.1.0
pre-commit>=3.6.0
```

---

## UI/UX Design System

### Design Philosophy

Inspired by **Bloomberg Terminal** (information density), **FinViz** (data visualization), and **Robinhood** (clean modern aesthetic).

**Core Principles:**
- Information density without visual clutter
- 5-second rule: answer key questions at a glance
- Color communicates meaning, not decoration
- Every element serves a function

---

### Color Palette (Dark Theme)

```
BACKGROUND
  --bg-primary:     #0D0D0D    /* Deep black - main background */
  --bg-secondary:   #1A1A1A    /* Elevated surfaces, cards */
  --bg-tertiary:    #242424    /* Hover states, subtle highlights */

BORDERS & DIVIDERS
  --border-subtle:  #2A2A2A    /* Card borders, dividers */
  --border-focus:   #3D3D3D    /* Focus states */

TEXT
  --text-primary:   #FFFFFF    /* Headlines, key metrics */
  --text-secondary: #A0A0A0    /* Labels, descriptions */
  --text-muted:     #666666    /* Timestamps, tertiary info */

SEMANTIC COLORS (Robinhood-inspired)
  --positive:       #00C805    /* Gains, bullish, success */
  --positive-muted: #0A3D0A    /* Positive backgrounds */
  --negative:       #FF5000    /* Losses, bearish, errors */
  --negative-muted: #3D1A0A    /* Negative backgrounds */
  --neutral:        #FFD700    /* Warnings, neutral sentiment */

ACCENT
  --accent-primary: #00D4AA    /* Interactive elements, CTAs */
  --accent-hover:   #00E5BB    /* Hover state */

CHART COLORS (Sequential for indicators)
  --chart-1:        #00D4AA    /* Primary line (price) */
  --chart-2:        #7B61FF    /* Secondary (MA-20) */
  --chart-3:        #FF6B6B    /* Tertiary (MA-50) */
  --chart-4:        #4ECDC4    /* Quaternary (MA-200) */
  --chart-5:        #FFE66D    /* Bollinger bands */
  --volume-up:      #00C805    /* Green volume bars */
  --volume-down:    #FF5000    /* Red volume bars */
```

---

### Typography

**Font Stack:**
```css
--font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
--font-mono: 'JetBrains Mono', 'SF Mono', monospace;  /* For numbers/data */
```

**Scale:**
```
--text-xs:    11px / 1.4    /* Timestamps, micro labels */
--text-sm:    13px / 1.5    /* Secondary text, table cells */
--text-base:  15px / 1.5    /* Body text, descriptions */
--text-lg:    18px / 1.4    /* Section headers */
--text-xl:    24px / 1.3    /* Card titles, stock symbols */
--text-2xl:   32px / 1.2    /* Key metrics, prices */
--text-3xl:   48px / 1.1    /* Hero numbers */
```

**Best Practices:**
- Use monospace font for all numerical data (prices, percentages, volumes)
- Numbers should be tabular (fixed-width) for easy scanning
- Stock symbols: ALL CAPS, bold, slightly larger

---

### Layout Grid System

**Desktop Layout (1440px+):**
```
+------------------+----------------------------------------+------------------+
|                  |                                        |                  |
|    SIDEBAR       |           MAIN CONTENT AREA            |   CONTEXT PANEL  |
|    (240px)       |              (flexible)                |     (320px)      |
|                  |                                        |                  |
|  - Stock Input   |  +----------------------------------+  |  - News Feed     |
|  - Watchlist     |  |     SUMMARY CARDS ROW            |  |  - AI Summary    |
|  - Quick Actions |  +----------------------------------+  |  - Sentiment     |
|                  |                                        |                  |
|                  |  +----------------------------------+  |                  |
|                  |  |                                  |  |                  |
|                  |  |     PRICE CHART (60% height)     |  |                  |
|                  |  |                                  |  |                  |
|                  |  +----------------------------------+  |                  |
|                  |                                        |                  |
|                  |  +---------------+------------------+  |                  |
|                  |  | MACD SUBPLOT  |  RSI SUBPLOT     |  |                  |
|                  |  +---------------+------------------+  |                  |
|                  |                                        |                  |
|                  |  +----------------------------------+  |                  |
|                  |  |     VOLUME CHART                 |  |                  |
|                  |  +----------------------------------+  |                  |
+------------------+----------------------------------------+------------------+
```

**Spacing Scale:**
```
--space-1:  4px     /* Tight grouping */
--space-2:  8px     /* Related elements */
--space-3:  12px    /* Default padding */
--space-4:  16px    /* Card padding */
--space-5:  24px    /* Section gaps */
--space-6:  32px    /* Major sections */
--space-8:  48px    /* Page margins */
```

---

### Component Specifications

#### Summary Metric Cards
```
+------------------------------------------+
|  CURRENT PRICE                           |
|  $425.32           +2.34 (+0.55%)        |
|  ^^^^^^^^          ^^^^^^^^^^^^^^^^^     |
|  text-3xl          text-lg, --positive   |
|  --text-primary    with arrow indicator  |
+------------------------------------------+

Specs:
- Background: --bg-secondary
- Border: 1px solid --border-subtle
- Border-radius: 12px
- Padding: --space-4
- Subtle shadow: 0 2px 8px rgba(0,0,0,0.3)
- Hover: border-color transitions to --accent-primary
```

#### Stock Symbol Input (Robinhood-style)
```
+--------------------------------------------------+
|  [AAPL] [x]  [MSFT] [x]  [NVDA] [x]  | + Add    |
+--------------------------------------------------+

- Chip/tag style for selected symbols
- Each chip: --bg-tertiary background, rounded-full
- Remove button (x) appears on hover
- Input field expands on focus
- Autocomplete dropdown with company names
```

#### Chart Cards
```
+--------------------------------------------------+
|  MSFT - Microsoft Corporation          1Y | 6M | 1M |
|  ------------------------------------------------|
|                                                   |
|     [Interactive Plotly Chart Area]               |
|     - Crosshair on hover                         |
|     - Tooltip with OHLCV data                    |
|     - Smooth animations                          |
|                                                   |
+--------------------------------------------------+

Chart Styling:
- Grid lines: --border-subtle, dashed, 0.5 opacity
- Axis labels: --text-muted, text-xs
- Hover tooltip: --bg-secondary with --border-focus
- Candlestick: --positive for up, --negative for down
```

#### News Card (FinViz-inspired density)
```
+--------------------------------------------------+
|  [BULLISH]  Reuters  |  2h ago                   |
|  Microsoft Reports Record Cloud Revenue          |
|  Azure growth accelerates to 29% as AI...        |
+--------------------------------------------------+

Specs:
- Sentiment badge: small pill, colored background
  - Bullish: --positive-muted bg, --positive text
  - Bearish: --negative-muted bg, --negative text
  - Neutral: --bg-tertiary bg, --text-secondary text
- Source + time: --text-muted, text-xs
- Headline: --text-primary, text-base, font-medium
- Preview: --text-secondary, text-sm, 2-line clamp
```

---

### Interactive States

```
HOVER
  - Cards: border-color transition, subtle lift (translateY -1px)
  - Buttons: background lightens 10%
  - Links: underline appears

FOCUS
  - Ring: 2px solid --accent-primary, 2px offset
  - High contrast for accessibility

ACTIVE/PRESSED
  - Scale down slightly (0.98)
  - Background darkens

LOADING
  - Skeleton: animated gradient from --bg-secondary to --bg-tertiary
  - Pulse animation for metrics
  - Chart: subtle shimmer overlay

DISABLED
  - Opacity: 0.5
  - Cursor: not-allowed
```

---

### Data Visualization Guidelines

**Charts (Plotly):**
```python
CHART_THEME = {
    "paper_bgcolor": "#0D0D0D",
    "plot_bgcolor": "#0D0D0D",
    "font": {"color": "#A0A0A0", "family": "Inter"},
    "xaxis": {
        "gridcolor": "#2A2A2A",
        "linecolor": "#2A2A2A",
        "tickfont": {"size": 11}
    },
    "yaxis": {
        "gridcolor": "#2A2A2A",
        "linecolor": "#2A2A2A",
        "tickfont": {"size": 11},
        "side": "right"  # Bloomberg-style
    },
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": "#1A1A1A",
        "bordercolor": "#3D3D3D",
        "font": {"family": "JetBrains Mono", "size": 12}
    }
}
```

**Indicator Colors:**
- Price line: --chart-1 (teal)
- SMA 20: --chart-2 (purple)
- SMA 50: --chart-3 (coral)
- SMA 200: --chart-4 (cyan)
- Bollinger Bands: --chart-5 (gold) with 0.1 opacity fill
- MACD Line: --chart-1
- Signal Line: --chart-3
- Histogram: --positive / --negative based on value
- RSI: --chart-2, with horizontal lines at 30/70

---

### Responsive Breakpoints

```
DESKTOP XL:   1440px+   (Full 3-column layout)
DESKTOP:      1024px+   (Collapsible context panel)
TABLET:       768px+    (Sidebar becomes top nav, 2-column)
MOBILE:       < 768px   (Single column, bottom nav)
```

---

### Accessibility Requirements

- Minimum contrast ratio: 4.5:1 for text
- All interactive elements keyboard accessible
- Focus indicators visible
- Chart data available in tabular format
- Screen reader announcements for live data updates
- Respect prefers-reduced-motion

---

### Animation Timing

```
--duration-fast:    100ms   /* Hover states */
--duration-normal:  200ms   /* Transitions */
--duration-slow:    300ms   /* Page transitions */
--easing-default:   cubic-bezier(0.4, 0, 0.2, 1)
--easing-bounce:    cubic-bezier(0.68, -0.55, 0.265, 1.55)
```

---

## Design Research Sources

### UI/UX Best Practices
- [Bloomberg Terminal UX Design](https://www.bloomberg.com/company/stories/how-bloomberg-terminal-ux-designers-conceal-complexity/) - Information density principles
- [Bloomberg Consistency Principles](https://www.bloomberg.com/ux/2020/08/11/consistency-more-than-just-a-buzzword/) - Predictability in complex interfaces
- [Robinhood UI Secrets](https://itexus.com/robinhood-ui-secrets-how-to-design-a-sky-rocket-trading-app/) - Clean minimalist financial UI
- [Robinhood Material Design](https://design.google/library/robinhood-investing-material) - Google's analysis of Robinhood
- [TradingView UI Documentation](https://www.tradingview.com/charting-library-docs/latest/ui_elements/) - Professional charting interface

### Dashboard Design
- [Dashboard Design Principles 2025 - UXPin](https://www.uxpin.com/studio/blog/dashboard-design-principles/) - Modern dashboard patterns
- [10 UI/UX Dashboard Principles 2025](https://medium.com/@farazjonanda/10-best-ui-ux-dashboard-design-principles-for-2025-2f9e7c21a454) - Speed of understanding
- [Top Dashboard Trends 2025](https://fuselabcreative.com/top-dashboard-design-trends-2025/) - Real-time data, dark mode
- [Card UI Design Examples](https://bricxlabs.com/blogs/card-ui-design-examples) - Card-based layouts
- [Glassmorphism 2025](https://onecodesoft.com/blogs/beyond-flat-5-ui-trends-dominating-screens-in-2025) - Subtle depth effects

### Financial Visualization
- [FinViz Heatmap Guide](https://finviz.blog/the-power-of-the-finviz-heatmap-a-comprehensive-guide/) - Color-coded data visualization
- [FinViz Screener Analysis](https://www.luxalgo.com/blog/finviz-market-screener-analysis/) - Filter UI patterns
- [Financial Dashboard Color Palettes](https://www.phoenixstrategy.group/blog/best-color-palettes-for-financial-dashboards) - Professional color schemes
- [Data Visualization Color Guide](https://blog.datawrapper.de/colors-for-data-vis-style-guides/) - Sequential palettes

### Typography
- [Font Strategies for Fintech](https://www.telerik.com/blogs/font-strategies-fintech-websites-apps) - Sans-serif dominance
- [Finance Font Pairings](https://typ.io/tags/finance) - Inter, Lato, Work Sans recommendations
- [Fonts in Financial Services](https://www.gate39media.com/blog/design-spotlight-fonts-in-financial-services) - Legibility principles

### Plotly/Dash Resources
- [Dash Finance Examples](https://plotly.com/examples/finance/) - Quantitative analysis dashboards
- [Plotly Dark Theme](https://plotly.com/python/templates/) - plotly_dark template
- [Dash Bootstrap Theme Explorer](https://hellodash.pythonanywhere.com/) - Dark theme components
- [Dash Design Kit](https://plotly.com/videos/making-dash-apps-beautiful-dash-design-kit/) - Polished dashboard styling

### Technology Comparisons
- [Streamlit vs Dash 2025](https://docs.kanaries.net/topics/Streamlit/streamlit-vs-dash) - Framework comparison
- [yfinance PyPI](https://pypi.org/project/yfinance/) - Stock data library
- [LM Studio OpenAI Compatibility](https://lmstudio.ai/docs/developer/openai-compat) - Local LLM integration
- [Alpha Vantage API](https://www.alphavantage.co/) - News and sentiment data

---

## Future Enhancements

- User authentication (Flask-Login or Dash Enterprise)
- Saved watchlists and portfolios
- Price alerts (email/push notifications)
- Export to PDF/Excel
- Historical sentiment trends
- Sector/industry comparisons
- Backtesting integration
