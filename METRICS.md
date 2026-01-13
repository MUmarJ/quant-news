# QuantNews Metrics Reference

This document lists all technical indicators and performance metrics used in QuantNews, along with their default settings.

## Technical Indicators

### Trend Indicators

| Indicator | Parameter | Default Value | Description |
|-----------|-----------|---------------|-------------|
| **SMA (Simple Moving Average)** | Periods | 20, 50, 200 | Average of closing prices over N days |
| **EMA (Exponential Moving Average)** | Periods | 12, 26 | Weighted average giving more weight to recent prices |

### MACD (Moving Average Convergence Divergence)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Fast EMA | 12 days | Short-term exponential moving average |
| Slow EMA | 26 days | Long-term exponential moving average |
| Signal Line | 9 days | EMA of the MACD line |

**Calculation:**
- MACD Line = EMA(12) - EMA(26)
- Signal Line = EMA(9) of MACD Line
- Histogram = MACD Line - Signal Line

### Momentum Indicators

| Indicator | Parameter | Default Value | Description |
|-----------|-----------|---------------|-------------|
| **RSI (Relative Strength Index)** | Period | 14 days | Measures speed and magnitude of price changes |
| | Overbought | 70 | Level above which asset is considered overbought |
| | Oversold | 30 | Level below which asset is considered oversold |
| **Stochastic Oscillator** | %K Period | 14 days | Lookback period for highest high/lowest low |
| | %D Period | 3 days | Smoothing period for %K (signal line) |
| **ROC (Rate of Change)** | Period | 12 days | Percentage change from N periods ago |

### Volatility Indicators

| Indicator | Parameter | Default Value | Description |
|-----------|-----------|---------------|-------------|
| **Bollinger Bands** | Period | 20 days | Moving average period |
| | Standard Deviation | 2.0 | Number of std deviations for upper/lower bands |
| **ATR (Average True Range)** | Period | 14 days | Average of true ranges over N periods |
| **Rolling Standard Deviation** | Period | 20 days | Standard deviation of closing prices |

### Volume Indicators

| Indicator | Parameter | Default Value | Description |
|-----------|-----------|---------------|-------------|
| **OBV (On-Balance Volume)** | N/A | N/A | Cumulative volume based on price direction |
| **Volume MA** | Period | 20 days | Simple moving average of volume |

---

## Performance Metrics

These metrics are calculated based on the selected time period.

| Metric | Calculation | Description |
|--------|-------------|-------------|
| **Total Return** | `(End Price - Start Price) / Start Price * 100` | Percentage gain/loss over the period |
| **Volatility** | `Daily Returns Std Dev * sqrt(252) * 100` | Annualized volatility (252 trading days/year) |
| **Sharpe Ratio** | `Annualized Return / Annualized Volatility` | Risk-adjusted return (assumes 0% risk-free rate) |
| **Max Drawdown** | `(Trough - Peak) / Peak * 100` | Largest peak-to-trough decline |
| **Win Rate** | `Positive Days / Total Days * 100` | Percentage of days with positive returns |
| **Best Day** | `max(Daily Returns) * 100` | Best single-day return |
| **Worst Day** | `min(Daily Returns) * 100` | Worst single-day return |

---

## Signal Generation

### RSI Signals
- **Overbought**: RSI > 70
- **Oversold**: RSI < 30
- **Neutral**: 30 <= RSI <= 70

### MACD Signals
- **Bullish Crossover**: MACD crosses above Signal line
- **Bearish Crossover**: MACD crosses below Signal line
- **Bullish**: MACD > Signal line
- **Bearish**: MACD < Signal line

### Trend Signals
- **Above SMA-50**: Price > SMA(50) - Bullish
- **Below SMA-50**: Price < SMA(50) - Bearish
- **Above SMA-200**: Price > SMA(200) - Long-term bullish
- **Below SMA-200**: Price < SMA(200) - Long-term bearish

### Cross Signals
- **Golden Cross**: SMA(50) crosses above SMA(200) - Major bullish signal
- **Death Cross**: SMA(50) crosses below SMA(200) - Major bearish signal

### Bollinger Band Signals
- **Above Upper Band**: Price > Upper Band - Potentially overbought/momentum breakout
- **Below Lower Band**: Price < Lower Band - Potentially oversold/breakdown
- **Within Bands**: Normal trading range

---

## Configuration

All indicator defaults are defined in `config.py` under `IndicatorDefaults`. To modify these values, update the corresponding constants:

```python
@dataclass(frozen=True)
class IndicatorDefaults:
    # Moving Averages
    SMA_PERIODS: tuple[int, ...] = (20, 50, 200)
    EMA_PERIODS: tuple[int, ...] = (12, 26)

    # MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # RSI
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30

    # Bollinger Bands
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0

    # ATR
    ATR_PERIOD: int = 14

    # Stochastic
    STOCHASTIC_K: int = 14
    STOCHASTIC_D: int = 3

    # Volume
    VOLUME_MA_PERIOD: int = 20
```

---

## Data Source

- **Price Data**: Yahoo Finance (via `yfinance` library)
- **Cache**: DuckDB local database (`cache/quant_news.duckdb`)
- **Trading Days**: 252 days/year (standard US market assumption)
