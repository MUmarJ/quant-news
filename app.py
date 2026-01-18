"""Quant News Tracker - Main Application Entry Point.

A quantitative stock tracking dashboard with technical analysis,
news aggregation, and AI-powered insights.
"""

import atexit
import json
from datetime import datetime
from io import StringIO

import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, html, ALL, dcc
from dash.exceptions import PreventUpdate

from callbacks.chart_callbacks import (
    create_comparison_chart,
    create_empty_chart,
    create_macd_chart,
    create_price_chart,
    create_rsi_chart,
    create_volume_chart,
)
from config import APP, COLORS
from layouts.components import (
    calculate_period_label,
    create_metric_card,
    create_news_card,
    create_symbol_tag,
)
from layouts.main_layout import create_layout
from services.analytics import (
    add_indicators_to_df,
    calculate_performance_metrics,
    get_latest_signals,
)
from services.cache_service import get_cache
from services.llm_service import get_llm
from services.news_service import (
    NewsArticle,
    fetch_news,
    fetch_news_cached,
    format_time_ago,
    get_sentiment_summary,
)

# Initialize Dash app with Bootstrap dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="QuantNews - Stock Analysis",
)

# Set layout
app.layout = create_layout()


# =============================================================================
# CLEANUP ON EXIT (prevents DuckDB WAL corruption)
# =============================================================================


def cleanup_on_exit():
    """Cleanup resources on application exit."""
    try:
        cache = get_cache()
        cache.close()
    except Exception:
        pass


atexit.register(cleanup_on_exit)


# =============================================================================
# SYMBOL MANAGEMENT CALLBACKS
# =============================================================================


@callback(
    Output("selected-symbols", "data"),
    Output("symbol-input", "value"),
    Input("add-symbol-btn", "n_clicks"),
    Input({"type": "quick-add", "symbol": ALL}, "n_clicks"),
    Input({"type": "remove-symbol", "symbol": ALL}, "n_clicks"),
    State("symbol-input", "value"),
    State("selected-symbols", "data"),
    prevent_initial_call=True,
)
def manage_symbols(add_click, quick_clicks, remove_clicks, input_value, current_symbols):
    """Handle adding and removing stock symbols."""
    current_symbols = current_symbols or []

    # Get the triggered context safely
    if not ctx.triggered:
        raise PreventUpdate

    triggered = ctx.triggered_id

    # Guard against None triggered_id
    if triggered is None:
        raise PreventUpdate

    # Handle add button
    if triggered == "add-symbol-btn":
        if not add_click or not input_value:
            raise PreventUpdate
        # Parse comma-separated symbols
        new_symbols = [s.strip().upper() for s in input_value.split(",") if s.strip()]
        for sym in new_symbols:
            if sym and sym not in current_symbols:
                current_symbols.append(sym)
        return current_symbols, ""

    # Handle quick-add buttons
    if isinstance(triggered, dict) and triggered.get("type") == "quick-add":
        # Find which button was clicked by checking n_clicks values
        clicked_any = any(c and c > 0 for c in quick_clicks)
        if not clicked_any:
            raise PreventUpdate
        symbol = triggered["symbol"]
        if symbol not in current_symbols:
            current_symbols.append(symbol)
        return current_symbols, dash.no_update

    # Handle remove buttons
    if isinstance(triggered, dict) and triggered.get("type") == "remove-symbol":
        # Find which button was clicked by checking n_clicks values
        clicked_any = any(c and c > 0 for c in remove_clicks)
        if not clicked_any:
            raise PreventUpdate
        symbol = triggered["symbol"]
        if symbol in current_symbols:
            current_symbols = [s for s in current_symbols if s != symbol]
        return current_symbols, dash.no_update

    raise PreventUpdate


@callback(
    Output("symbol-tags", "children"),
    Input("selected-symbols", "data"),
)
def update_symbol_tags(symbols):
    """Update the symbol tags display."""
    if not symbols:
        return html.Div(
            [
                html.I(className="bi bi-info-circle me-2"),
                html.Span("Add stocks using the input above or quick add buttons"),
            ],
            className="text-muted",
            style={"fontSize": "12px", "padding": "8px 0"},
        )

    return [create_symbol_tag(sym) for sym in symbols]


# =============================================================================
# DATA FETCHING CALLBACKS
# =============================================================================


@callback(
    Output("stock-data-store", "data"),
    Output("cache-status", "children"),
    Output("data-source-indicator", "children"),
    Input("selected-symbols", "data"),
    Input("current-period", "data"),
    Input("refresh-data-btn", "n_clicks"),
    Input("cache-enabled", "data"),
    prevent_initial_call=True,
)
def fetch_stock_data_callback(symbols, period, refresh_click, cache_enabled):
    """Fetch stock data for selected symbols."""
    if not symbols:
        return {}, "No data", ""

    cache = get_cache()
    triggered = ctx.triggered_id
    # Force refresh if button clicked OR if cache is disabled
    force_refresh = triggered == "refresh-data-btn" or not cache_enabled

    data = {}
    all_from_cache = True
    any_api_error = None

    for symbol in symbols:
        try:
            df, metadata = cache.get_stock_prices(symbol, period, force_refresh=force_refresh)
            # Track data source info
            if not metadata.get("from_cache"):
                all_from_cache = False
            if metadata.get("api_error"):
                any_api_error = metadata["api_error"]

            # Add technical indicators
            df = add_indicators_to_df(df)
            # Convert to JSON-serializable format
            data[symbol] = {
                "prices": df.to_json(date_format="iso"),
                "metrics": calculate_performance_metrics(df),
                "signals": get_latest_signals(df),
                "from_cache": metadata.get("from_cache", False),
                "api_error": metadata.get("api_error"),
            }
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

    # Update cache status
    cache_status = cache.get_cache_status()
    if cache_status:
        last = cache_status[-1]
        status_text = f"Last update: {last['last_updated'].strftime('%H:%M')}"
    else:
        status_text = "No cached data"

    # Create data source indicator
    if any_api_error:
        source_indicator = html.Span(
            [
                html.I(className="bi bi-exclamation-triangle-fill me-1"),
                "API Error - Using cached data",
            ],
            className="data-source-badge data-source-error",
            title=f"API Error: {any_api_error}",
        )
    elif all_from_cache and data:
        source_indicator = html.Span(
            [
                html.I(className="bi bi-database me-1"),
                "Cached",
            ],
            className="data-source-badge data-source-cache",
        )
    elif data:
        source_indicator = html.Span(
            [
                html.I(className="bi bi-cloud-download me-1"),
                "Live",
            ],
            className="data-source-badge data-source-live",
        )
    else:
        source_indicator = ""

    return data, status_text, source_indicator


# =============================================================================
# CHART UPDATE CALLBACKS
# =============================================================================


@callback(
    Output("price-chart", "figure"),
    Output("chart-title", "children"),
    Output("chart-subtitle", "children"),
    Input("stock-data-store", "data"),
    Input("indicator-toggles", "value"),
    State("selected-symbols", "data"),
)
def update_price_chart(stock_data, indicators, symbols):
    """Update the main price chart."""
    if not stock_data or not symbols:
        return create_empty_chart("Select stocks to view price chart"), "Price Chart", "Add stocks to get started"

    # Use first symbol for main chart, or comparison for multiple
    if len(symbols) == 1:
        symbol = symbols[0]
        if symbol not in stock_data:
            return create_empty_chart(f"No data for {symbol}"), symbol, ""

        df = pd.read_json(StringIO(stock_data[symbol]["prices"]))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        metrics = stock_data[symbol].get("metrics", {})
        subtitle = f"{metrics.get('start_date', '')} to {metrics.get('end_date', '')}"

        fig = create_price_chart(df, symbol, indicators)
        return fig, symbol, subtitle
    else:
        # Multiple stocks - create comparison chart
        data_dict = {}
        for symbol in symbols:
            if symbol in stock_data:
                df = pd.read_json(StringIO(stock_data[symbol]["prices"]))
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                data_dict[symbol] = df

        fig = create_comparison_chart(data_dict)
        return fig, "Comparison", f"{len(symbols)} stocks"


@callback(
    Output("macd-chart", "figure"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_macd_chart(stock_data, symbols):
    """Update MACD chart."""
    if not stock_data or not symbols:
        return create_empty_chart("Select stocks to view MACD")

    symbol = symbols[0]
    if symbol not in stock_data:
        return create_empty_chart("No MACD data available")

    try:
        df = pd.read_json(StringIO(stock_data[symbol]["prices"]))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return create_macd_chart(df)
    except Exception as e:
        print(f"Error creating MACD chart: {e}")
        return create_empty_chart("Error loading MACD data")


@callback(
    Output("rsi-chart", "figure"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_rsi_chart(stock_data, symbols):
    """Update RSI chart."""
    if not stock_data or not symbols:
        return create_empty_chart("Select stocks to view RSI")

    symbol = symbols[0]
    if symbol not in stock_data:
        return create_empty_chart("No RSI data available")

    try:
        df = pd.read_json(StringIO(stock_data[symbol]["prices"]))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return create_rsi_chart(df)
    except Exception as e:
        print(f"Error creating RSI chart: {e}")
        return create_empty_chart("Error loading RSI data")


@callback(
    Output("volume-chart", "figure"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_volume_chart(stock_data, symbols):
    """Update volume chart."""
    if not stock_data or not symbols:
        return create_empty_chart("Select stocks to view volume")

    symbol = symbols[0]
    if symbol not in stock_data:
        return create_empty_chart("No volume data available")

    try:
        df = pd.read_json(StringIO(stock_data[symbol]["prices"]))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return create_volume_chart(df)
    except Exception as e:
        print(f"Error creating volume chart: {e}")
        return create_empty_chart("Error loading volume data")


# =============================================================================
# SUMMARY CARDS CALLBACK
# =============================================================================


@callback(
    Output("summary-cards", "children"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_summary_cards(stock_data, symbols):
    """Update summary metric cards."""
    if not stock_data or not symbols:
        return html.Div(
            [
                html.Div(
                    [
                        html.I(className="bi bi-graph-up", style={"fontSize": "32px", "opacity": "0.5"}),
                        html.P("Select stocks to view metrics", className="mt-2 mb-0"),
                    ],
                    className="empty-state",
                )
            ],
            style={"gridColumn": "1 / -1"},
        )

    cards = []
    for symbol in symbols:
        if symbol not in stock_data:
            continue

        metrics = stock_data[symbol].get("metrics", {})
        signals = stock_data[symbol].get("signals", {})

        # Current price card with dynamic period label
        price = metrics.get("end_price", 0)
        total_return = metrics.get("total_return", 0)
        start_date = metrics.get("start_date", "")
        end_date = metrics.get("end_date", "")

        # Calculate dynamic period label
        period_label = calculate_period_label(start_date, end_date)

        cards.append(
            create_metric_card(
                title=symbol,
                value=f"${price:,.2f}",
                change=f"{total_return:+.2f}%",
                change_positive=total_return > 0,
                subtitle=period_label,
            )
        )

    # Add aggregate cards if multiple stocks
    if len(symbols) > 1 and cards:
        avg_return = sum(
            stock_data[s].get("metrics", {}).get("total_return", 0)
            for s in symbols if s in stock_data
        ) / len([s for s in symbols if s in stock_data])

        cards.append(
            create_metric_card(
                title="AVG RETURN",
                value=f"{avg_return:+.1f}%",
                change_positive=avg_return > 0,
            )
        )

    if not cards:
        return html.Div(
            [
                html.Div(
                    [
                        html.I(className="bi bi-exclamation-circle", style={"fontSize": "32px", "opacity": "0.5"}),
                        html.P("Unable to load stock data", className="mt-2 mb-0"),
                    ],
                    className="empty-state",
                )
            ],
            style={"gridColumn": "1 / -1"},
        )
    return cards


# =============================================================================
# NEWS & AI CALLBACKS
# =============================================================================


@callback(
    Output("cache-enabled", "data"),
    Input("cache-toggle", "value"),
)
def update_cache_enabled(toggle_value):
    """Sync cache toggle with store."""
    return toggle_value


@callback(
    Output("news-data-store", "data"),
    Input("selected-symbols", "data"),
    Input("refresh-data-btn", "n_clicks"),
    Input("cache-enabled", "data"),
    prevent_initial_call=True,
)
def fetch_news_data(symbols, refresh_click, cache_enabled):
    """Fetch news once and store for reuse by other callbacks.

    This eliminates duplicate API calls - news is fetched once here,
    then consumed by both update_news_feed and update_ai_summary.
    """
    if not symbols:
        return {}

    symbol = symbols[0]

    # Use cached or uncached fetch based on toggle
    if cache_enabled:
        articles = fetch_news_cached(symbol, max_articles=5)
    else:
        articles = fetch_news(symbol, max_articles=5)

    if not articles:
        return {"symbol": symbol, "articles": [], "fetched_at": datetime.now().isoformat()}

    # Serialize articles for JSON storage
    articles_data = [
        {
            "id": a.id,
            "symbol": a.symbol,
            "title": a.title,
            "source": a.source,
            "url": a.url,
            "published_at": a.published_at.isoformat(),
            "summary": a.summary,
            "sentiment": a.sentiment,
            "sentiment_score": a.sentiment_score,
            "impact": a.impact,
            "price_change_percent": a.price_change_percent,
        }
        for a in articles
    ]

    return {
        "symbol": symbol,
        "articles": articles_data,
        "fetched_at": datetime.now().isoformat(),
    }


@callback(
    Output("news-feed", "children"),
    Output("sentiment-display", "children"),
    Output("news-meta", "children"),
    Input("news-data-store", "data"),
)
def update_news_feed(news_data):
    """Update news feed and sentiment display from cached news data."""
    if not news_data or not news_data.get("articles"):
        empty_news = html.Div(
            [
                html.I(className="bi bi-newspaper", style={"fontSize": "24px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("Select stocks to view latest news"),
            ],
            className="empty-state",
            style={"padding": "24px 16px"},
        )
        empty_sentiment = html.Div(
            [
                html.Div(
                    [
                        html.Div("-", className="sentiment-count", style={"color": COLORS.TEXT_MUTED}),
                        html.Div("Bullish", className="sentiment-label"),
                    ],
                    className="sentiment-item",
                ),
                html.Div(
                    [
                        html.Div("-", className="sentiment-count", style={"color": COLORS.TEXT_MUTED}),
                        html.Div("Neutral", className="sentiment-label"),
                    ],
                    className="sentiment-item",
                ),
                html.Div(
                    [
                        html.Div("-", className="sentiment-count", style={"color": COLORS.TEXT_MUTED}),
                        html.Div("Bearish", className="sentiment-label"),
                    ],
                    className="sentiment-item",
                ),
            ],
            className="sentiment-display",
        )
        return empty_news, empty_sentiment, ""

    # Deserialize articles from store
    articles = [
        NewsArticle(
            id=a["id"],
            symbol=a["symbol"],
            title=a["title"],
            source=a["source"],
            url=a["url"],
            published_at=datetime.fromisoformat(a["published_at"]),
            summary=a.get("summary"),
            sentiment=a.get("sentiment"),
            sentiment_score=a.get("sentiment_score"),
            impact=a.get("impact"),
            price_change_percent=a.get("price_change_percent"),
        )
        for a in news_data["articles"]
    ]

    if not articles:
        no_news = html.Div(
            [
                html.I(className="bi bi-inbox", style={"fontSize": "24px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("No recent news available"),
            ],
            className="empty-state",
            style={"padding": "24px 16px"},
        )
        return no_news, "", ""

    # Create news cards
    news_cards = [
        create_news_card(
            title=article.title,
            source=article.source,
            time_ago=format_time_ago(article.published_at),
            summary=article.summary,
            sentiment=article.sentiment,
            url=article.url,
            impact=article.impact,
            price_change_percent=article.price_change_percent,
            symbol=article.symbol,
        )
        for article in articles
    ]

    # Calculate sentiment summary
    sentiment = get_sentiment_summary(articles)

    sentiment_display = html.Div(
        [
            html.Div(
                [
                    html.Div(str(sentiment["bullish"]), className="sentiment-count", style={"color": COLORS.POSITIVE}),
                    html.Div("Bullish", className="sentiment-label"),
                ],
                className="sentiment-item",
            ),
            html.Div(
                [
                    html.Div(str(sentiment["neutral"]), className="sentiment-count", style={"color": COLORS.TEXT_SECONDARY}),
                    html.Div("Neutral", className="sentiment-label"),
                ],
                className="sentiment-item",
            ),
            html.Div(
                [
                    html.Div(str(sentiment["bearish"]), className="sentiment-count", style={"color": COLORS.NEGATIVE}),
                    html.Div("Bearish", className="sentiment-label"),
                ],
                className="sentiment-item",
            ),
        ],
        className="sentiment-display",
    )

    # Build news meta info (article count and date range)
    article_count = len(articles)
    if articles:
        oldest = min(a.published_at for a in articles)
        newest = max(a.published_at for a in articles)
        # Format date range
        if oldest.date() == newest.date():
            date_range = newest.strftime("%b %d")
        else:
            date_range = f"{oldest.strftime('%b %d')} - {newest.strftime('%b %d')}"
        news_meta = f"{article_count} articles | {date_range}"
    else:
        news_meta = ""

    return news_cards, sentiment_display, news_meta


@callback(
    Output("ai-summary", "children"),
    Output("llm-status", "children"),
    Input("news-data-store", "data"),
)
def update_ai_summary(news_data):
    """Update AI-generated summary from cached news data."""
    llm = get_llm()

    if not llm.is_available():
        status = "LLM: Offline"
        return html.Div("LLM not available. Start LM Studio or configure OpenAI API key.", className="text-muted"), status

    if not news_data or not news_data.get("articles"):
        return "", f"LLM: {llm.provider}"

    symbol = news_data.get("symbol", "")

    # Use articles directly from the store (already in dict format)
    article_dicts = [
        {"title": a["title"], "summary": a.get("summary") or ""}
        for a in news_data["articles"]
    ]

    if not article_dicts:
        return html.Div("No news to summarize", className="text-muted"), f"LLM: {llm.provider}"

    summary = llm.summarize_news(article_dicts, symbol)

    if summary:
        return dcc.Markdown(summary, className="ai-summary-content"), f"LLM: {llm.provider}"
    else:
        return html.Div("Could not generate summary", className="text-muted"), f"LLM: {llm.provider}"


@callback(
    Output("signals-display", "children"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_signals_display(stock_data, symbols):
    """Update technical signals display."""
    if not stock_data or not symbols:
        return html.Div(
            [
                html.I(className="bi bi-activity", style={"fontSize": "20px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("Select stocks to view signals"),
            ],
            className="empty-state",
            style={"padding": "16px", "fontSize": "13px"},
        )

    symbol = symbols[0]
    if symbol not in stock_data:
        return html.Div(
            [
                html.I(className="bi bi-exclamation-triangle", style={"fontSize": "20px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("No signal data available"),
            ],
            className="empty-state",
            style={"padding": "16px", "fontSize": "13px"},
        )

    signals = stock_data[symbol].get("signals", {})

    if not signals:
        return html.Div(
            [
                html.I(className="bi bi-dash-circle", style={"fontSize": "20px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("No signals available for this stock"),
            ],
            className="empty-state",
            style={"padding": "16px", "fontSize": "13px"},
        )

    # Tooltips explaining each signal type
    signal_tooltips = {
        "rsi": "RSI (14-day): Overbought >70, Oversold <30",
        "macd": "MACD (12/26/9): Bullish when MACD crosses above Signal",
        "trend_50": "Price position relative to 50-day SMA",
        "trend_200": "Price position relative to 200-day SMA (long-term)",
        "cross": "Golden Cross = SMA 50 > SMA 200 (Bullish), Death Cross = opposite",
        "bollinger": "Price position relative to Bollinger Bands (20-day, ±2σ)",
        "stochastic": "Stochastic (14/3): Overbought >80, Oversold <20",
        "momentum": "Price momentum based on rate of change",
    }

    signal_items = []

    for key, val in signals.items():
        if isinstance(val, dict):
            signal_text = val.get("signal", str(val))
            is_bullish = val.get("bullish", None)

            if is_bullish is True:
                color_class = "signal-bullish"
            elif is_bullish is False:
                color_class = "signal-bearish"
            else:
                # Check signal text for sentiment
                if "bullish" in signal_text.lower() or "above" in signal_text.lower():
                    color_class = "signal-bullish"
                elif "bearish" in signal_text.lower() or "below" in signal_text.lower():
                    color_class = "signal-bearish"
                else:
                    color_class = "signal-neutral"

            # Create unique ID for tooltip target
            signal_id = f"signal-{key}"

            # Get tooltip text for this signal type
            tooltip_text = signal_tooltips.get(key, f"{key.replace('_', ' ').title()} indicator")

            signal_items.append(
                html.Div(
                    [
                        html.Span(
                            key.replace("_", " ").title(),
                            className="signal-name",
                            id=signal_id,
                        ),
                        dbc.Tooltip(tooltip_text, target=signal_id, placement="left"),
                        html.Span(signal_text.replace("_", " ").title(), className=f"signal-value {color_class}"),
                    ],
                    className="signal-item",
                )
            )

    if not signal_items:
        return html.Div(
            [
                html.I(className="bi bi-dash-circle", style={"fontSize": "20px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("No signals detected"),
            ],
            className="empty-state",
            style={"padding": "16px", "fontSize": "13px"},
        )
    return signal_items


# =============================================================================
# DATA MODAL CALLBACKS
# =============================================================================


@callback(
    Output("data-modal", "is_open"),
    Output("data-table-container", "children"),
    Input("view-data-btn", "n_clicks"),
    Input("modal-close-btn", "n_clicks"),
    State("selected-symbols", "data"),
    State("data-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_data_modal(view_click, close_click, symbols, is_open):
    """Toggle data modal and populate table."""
    triggered = ctx.triggered_id

    if triggered == "modal-close-btn":
        return False, dash.no_update

    if triggered == "view-data-btn":
        if not symbols:
            return False, html.Div("No stocks selected", className="text-muted")

        cache = get_cache()
        symbol = symbols[0]

        try:
            df = cache.get_raw_data(symbol)
            if df.empty:
                return True, html.Div("No cached data", className="text-muted")

            # Create simple table
            table = dbc.Table.from_dataframe(
                df.head(50),
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                className="table-dark",
            )
            return True, table
        except Exception as e:
            return True, html.Div(f"Error loading data: {e}", className="text-danger")

    return is_open, dash.no_update


@callback(
    Output("download-data", "data"),
    Input("export-data-btn", "n_clicks"),
    Input("modal-export-btn", "n_clicks"),
    State("selected-symbols", "data"),
    prevent_initial_call=True,
)
def export_data(export_click, modal_export_click, symbols):
    """Export data to Parquet file."""
    if not symbols:
        raise PreventUpdate

    cache = get_cache()
    symbol = symbols[0]

    try:
        filepath = cache.export_to_parquet(symbol)
        return dcc.send_file(filepath)
    except Exception as e:
        print(f"Export error: {e}")
        raise PreventUpdate


# =============================================================================
# PERIOD SELECTOR CALLBACK
# =============================================================================


@callback(
    Output("current-period", "data"),
    Output({"type": "period-btn", "period": ALL}, "color"),
    Output({"type": "period-btn", "period": ALL}, "outline"),
    Input({"type": "period-btn", "period": ALL}, "n_clicks"),
    State("current-period", "data"),
    prevent_initial_call=True,
)
def update_period(clicks, current_period):
    """Update selected time period."""
    triggered = ctx.triggered_id

    if isinstance(triggered, dict) and triggered.get("type") == "period-btn":
        new_period = triggered["period"]
    else:
        new_period = current_period

    # Update button styles
    periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    colors = ["primary" if p == new_period else "secondary" for p in periods]
    outlines = [p != new_period for p in periods]

    return new_period, colors, outlines


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    app.run(
        debug=APP.DEBUG,
        host=APP.HOST,
        port=APP.PORT,
    )
