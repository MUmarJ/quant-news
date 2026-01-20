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
    create_overview_empty_state,
    create_recommendation_banner,
    create_sentiment_breakdown,
    create_symbol_tag,
    create_top_headlines,
    create_news_quick_stats,
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
    fetch_news,
    fetch_news_cached,
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
    """Fetch news for ALL selected symbols.

    Returns a dict with articles organized by symbol for per-symbol tabs.
    """
    if not symbols:
        return {}

    articles_by_symbol = {}

    for symbol in symbols:
        # Use cached or uncached fetch based on toggle
        if cache_enabled:
            articles = fetch_news_cached(symbol, max_articles=10)
        else:
            articles = fetch_news(symbol, max_articles=10)

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
            for a in (articles or [])
        ]

        articles_by_symbol[symbol] = articles_data

    return {
        "symbols": symbols,
        "articles_by_symbol": articles_by_symbol,
        "fetched_at": datetime.now().isoformat(),
    }


@callback(
    Output("ai-analysis-store", "data"),
    Input("news-data-store", "data"),
    State("selected-symbols", "data"),
)
def generate_ai_analysis(news_data, symbols):
    """Generate structured AI analysis for each symbol and overall.

    Uses ThreadPoolExecutor to parallelize LLM calls for better performance.

    Returns:
        {
            "overall": {...combined analysis...},
            "by_symbol": {
                "AAPL": {...per-symbol analysis...},
                "GOOGL": {...},
            }
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not news_data or not news_data.get("articles_by_symbol"):
        return {}

    llm = get_llm()
    articles_by_symbol = news_data.get("articles_by_symbol", {})

    result = {
        "overall": None,
        "by_symbol": {},
        "generated_at": datetime.now().isoformat(),
    }

    # Collect all articles for overall analysis
    all_articles = []

    # Prepare tasks for parallel execution
    symbol_tasks = {}
    for symbol, articles in articles_by_symbol.items():
        if not articles:
            continue

        all_articles.extend(articles)

        # Prepare articles for LLM
        article_dicts = [
            {
                "title": a["title"],
                "summary": a.get("summary") or "",
                "sentiment": a.get("sentiment") or "neutral",
            }
            for a in articles
        ]
        symbol_tasks[symbol] = article_dicts

    # Prepare overall analysis task
    overall_dicts = None
    if all_articles:
        overall_dicts = [
            {
                "title": a["title"],
                "summary": a.get("summary") or "",
                "sentiment": a.get("sentiment") or "neutral",
            }
            for a in all_articles
        ]

    # Execute LLM calls in parallel
    with ThreadPoolExecutor(max_workers=min(len(symbol_tasks) + 1, 5)) as executor:
        futures = {}

        # Submit per-symbol analysis tasks
        for symbol, article_dicts in symbol_tasks.items():
            future = executor.submit(llm.summarize_news_structured, article_dicts, [symbol])
            futures[future] = ("symbol", symbol)

        # Submit overall analysis task
        if overall_dicts:
            future = executor.submit(llm.summarize_news_structured, overall_dicts, symbols or [])
            futures[future] = ("overall", None)

        # Collect results as they complete
        for future in as_completed(futures):
            task_type, symbol = futures[future]
            try:
                analysis = future.result()
                if analysis:
                    if task_type == "symbol":
                        result["by_symbol"][symbol] = analysis
                    else:
                        result["overall"] = analysis
            except Exception as e:
                print(f"Error in LLM analysis for {task_type} {symbol}: {e}")

    return result


@callback(
    Output("symbol-tabs-container", "children"),
    Input("news-data-store", "data"),
    Input("ai-analysis-store", "data"),
    Input("selected-symbols", "data"),
)
def update_symbol_tabs(news_data, ai_analysis, symbols):
    """Build dynamic tabs based on selected symbols.

    Creates an "Overall" tab (always first, selected by default) plus one tab per symbol.
    Each tab contains: recommendation banner, key developments, headlines, sentiment breakdown.
    """
    # Handle empty state
    if not symbols:
        return create_overview_empty_state()

    # Show loading state if symbols selected but no news data yet
    if not news_data or not news_data.get("articles_by_symbol"):
        return _create_loading_state(symbols)

    articles_by_symbol = news_data.get("articles_by_symbol", {}) if news_data else {}
    analysis_by_symbol = ai_analysis.get("by_symbol", {}) if ai_analysis else {}
    overall_analysis = ai_analysis.get("overall", {}) if ai_analysis else {}

    # Build tabs list
    tabs = []

    # --- Overall Tab (always first) ---
    all_articles = []
    for sym_articles in articles_by_symbol.values():
        all_articles.extend(sym_articles or [])

    overall_content = _build_overall_tab_content(
        articles_by_symbol=articles_by_symbol,
        analysis_by_symbol=analysis_by_symbol,
        overall_analysis=overall_analysis,
        symbols=symbols,
    )

    tabs.append(
        dbc.Tab(
            overall_content,
            label="Overall",
            tab_id="tab-overall",
            className="context-tab",
        )
    )

    # --- Per-Symbol Tabs ---
    for symbol in symbols:
        sym_articles = articles_by_symbol.get(symbol, [])
        sym_analysis = analysis_by_symbol.get(symbol, {})

        tab_content = _build_tab_content(
            articles=sym_articles,
            analysis=sym_analysis,
            symbols=[symbol],
            is_overall=False,
        )

        tabs.append(
            dbc.Tab(
                tab_content,
                label=symbol,
                tab_id=f"tab-{symbol}",
                className="context-tab",
            )
        )

    return dbc.Tabs(
        tabs,
        id="symbol-tabs",
        active_tab="tab-overall",
        className="symbol-tabs",
    )


def _create_loading_state(symbols: list) -> html.Div:
    """Create loading state while fetching news data.

    Args:
        symbols: List of symbols being loaded

    Returns:
        Loading state component with spinner and status message
    """
    symbols_text = ", ".join(symbols) if len(symbols) <= 3 else f"{len(symbols)} stocks"

    return html.Div(
        [
            html.Div(
                [
                    # Spinner
                    html.Div(
                        [
                            html.Div(className="loading-spinner"),
                        ],
                        className="loading-spinner-container",
                    ),
                    # Status text
                    html.Div(
                        f"Fetching news for {symbols_text}...",
                        className="loading-status-text",
                    ),
                    # Sub-text
                    html.Div(
                        "Analyzing sentiment and generating insights",
                        className="loading-subtext",
                    ),
                ],
                className="news-loading-state",
            ),
        ],
        className="news-loading-container",
    )


def _build_overall_tab_content(
    articles_by_symbol: dict,
    analysis_by_symbol: dict,
    overall_analysis: dict,
    symbols: list,
) -> html.Div:
    """Build content for the Overall tab with summary table and combined analysis.

    Args:
        articles_by_symbol: Dict mapping symbol -> list of articles
        analysis_by_symbol: Dict mapping symbol -> analysis dict
        overall_analysis: Combined analysis across all symbols
        symbols: List of all selected symbols

    Returns:
        Overall tab content component
    """
    children = []

    # Collect all articles for stats
    all_articles = []
    for sym_articles in articles_by_symbol.values():
        all_articles.extend(sym_articles or [])

    # -- AI Summary (digest of all symbols) --
    # Build a comprehensive summary from per-symbol analyses
    summary_parts = []

    if analysis_by_symbol:
        # Count recommendations
        rec_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for symbol in symbols:
            sym_analysis = analysis_by_symbol.get(symbol, {})
            rec = sym_analysis.get("recommendation", "").lower()
            if "bullish" in rec:
                rec_counts["bullish"] += 1
            elif "bearish" in rec:
                rec_counts["bearish"] += 1
            else:
                rec_counts["neutral"] += 1

        # Build summary text
        total_symbols = len(symbols)
        if rec_counts["bullish"] > 0:
            summary_parts.append(f"{rec_counts['bullish']} of {total_symbols} stocks show bullish signals")
        if rec_counts["bearish"] > 0:
            summary_parts.append(f"{rec_counts['bearish']} of {total_symbols} stocks show bearish signals")
        if rec_counts["neutral"] > 0 and rec_counts["bullish"] == 0 and rec_counts["bearish"] == 0:
            summary_parts.append(f"All {total_symbols} stocks show neutral sentiment")

    # Add overall key developments if available
    if overall_analysis and overall_analysis.get("key_developments"):
        summary_parts.append(overall_analysis.get("key_developments", ""))

    if summary_parts:
        ai_summary = html.Div(
            [
                html.Div("AI Summary", className="section-title"),
                html.Div(
                    ". ".join(summary_parts) if len(summary_parts) > 1 else summary_parts[0],
                    className="key-developments-content",
                ),
            ],
            className="key-developments",
        )
        children.append(ai_summary)

    # -- Per-Symbol Recommendations Table --
    if symbols and analysis_by_symbol:
        table_rows = []
        for symbol in symbols:
            sym_analysis = analysis_by_symbol.get(symbol, {})
            sym_articles = articles_by_symbol.get(symbol, [])

            rec = sym_analysis.get("recommendation", "—")
            confidence = sym_analysis.get("confidence")
            article_count = len(sym_articles)

            # Determine color class for recommendation
            rec_lower = rec.lower() if rec != "—" else ""
            if "bullish" in rec_lower:
                rec_class = "rec-bullish"
            elif "bearish" in rec_lower:
                rec_class = "rec-bearish"
            else:
                rec_class = "rec-neutral"

            # Format recommendation display
            rec_display = rec.replace("_", " ") if rec != "—" else "—"

            # Format confidence
            conf_display = f"{int(confidence * 100)}%" if confidence else "—"

            table_rows.append(
                html.Tr(
                    [
                        html.Td(symbol, className="symbol-cell"),
                        html.Td(
                            rec_display,
                            className=f"recommendation-cell {rec_class}",
                        ),
                        html.Td(conf_display, className="confidence-cell"),
                        html.Td(str(article_count), className="articles-cell"),
                    ]
                )
            )

        recommendations_table = html.Div(
            [
                html.Div("Recommendations by Symbol", className="section-title"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Symbol"),
                                    html.Th("Recommendation"),
                                    html.Th("Confidence"),
                                    html.Th("Articles"),
                                ]
                            )
                        ),
                        html.Tbody(table_rows),
                    ],
                    className="recommendations-table",
                ),
            ],
            className="recommendations-section",
        )
        children.append(recommendations_table)

    # -- Aggregated Sentiment Breakdown --
    sentiment_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for a in all_articles:
        s = (a.get("sentiment") or "neutral").lower()
        if "bullish" in s:
            sentiment_counts["bullish"] += 1
        elif "bearish" in s:
            sentiment_counts["bearish"] += 1
        else:
            sentiment_counts["neutral"] += 1

    if any(sentiment_counts.values()):
        sentiment_breakdown = create_sentiment_breakdown(
            bullish=sentiment_counts["bullish"],
            neutral=sentiment_counts["neutral"],
            bearish=sentiment_counts["bearish"],
        )
        children.append(sentiment_breakdown)

    # -- Quick Stats --
    if all_articles:
        sources = list(set(a.get("source", "") for a in all_articles if a.get("source")))
        quick_stats = create_news_quick_stats(
            article_count=len(all_articles),
            source_count=len(sources),
            date_range=_get_date_range(all_articles),
            symbols=symbols,
        )
        children.append(quick_stats)

    # Handle empty state
    if not children:
        children.append(
            html.Div(
                [
                    html.I(className="bi bi-newspaper", style={"fontSize": "24px", "opacity": "0.5", "marginBottom": "8px"}),
                    html.P("No news available for selected stocks", style={"color": "#6B7280", "margin": "0"}),
                ],
                className="tab-empty-state",
                style={"textAlign": "center", "padding": "32px 16px"},
            )
        )

    return html.Div(children, className="tab-content-inner")


def _build_tab_content(
    articles: list,
    analysis: dict,
    symbols: list,
    is_overall: bool = False,
) -> html.Div:
    """Build content for a single tab (overall or per-symbol).

    Args:
        articles: List of article dictionaries for this tab
        analysis: AI analysis dictionary for this tab
        symbols: List of symbols (single symbol for per-symbol tab)
        is_overall: True if this is the Overall tab

    Returns:
        Tab content component
    """
    children = []

    # -- Recommendation Banner --
    if analysis and analysis.get("recommendation"):
        rec_banner = create_recommendation_banner(
            recommendation=analysis.get("recommendation", "NEUTRAL"),
            confidence=analysis.get("confidence"),
            article_count=len(articles),
            date_range=_get_date_range(articles),
        )
    elif articles:
        # Show loading state while waiting for AI analysis
        rec_banner = create_recommendation_banner(recommendation="LOADING")
    else:
        rec_banner = None

    if rec_banner:
        children.append(rec_banner)

    # -- Key Developments --
    if analysis and analysis.get("key_developments"):
        key_dev = html.Div(
            [
                html.Div("Key Developments", className="section-title"),
                html.Div(
                    analysis.get("key_developments", ""),
                    className="key-developments-content",
                ),
            ],
            className="key-developments",
        )
        children.append(key_dev)

    # -- Top Headlines --
    if articles:
        top_headlines = create_top_headlines(articles, max_count=5)
        children.append(top_headlines)

    # -- Sentiment Breakdown --
    sentiment_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for a in articles:
        s = (a.get("sentiment") or "neutral").lower()
        if "bullish" in s:
            sentiment_counts["bullish"] += 1
        elif "bearish" in s:
            sentiment_counts["bearish"] += 1
        else:
            sentiment_counts["neutral"] += 1

    if any(sentiment_counts.values()):
        sentiment_breakdown = create_sentiment_breakdown(
            bullish=sentiment_counts["bullish"],
            neutral=sentiment_counts["neutral"],
            bearish=sentiment_counts["bearish"],
        )
        children.append(sentiment_breakdown)

    # -- Quick Stats --
    if articles:
        sources = list(set(a.get("source", "") for a in articles if a.get("source")))
        quick_stats = create_news_quick_stats(
            article_count=len(articles),
            source_count=len(sources),
            date_range=_get_date_range(articles),
            symbols=symbols if is_overall else None,
        )
        children.append(quick_stats)

    # Handle empty state for this tab
    if not children:
        label = "all stocks" if is_overall else symbols[0] if symbols else "this stock"
        children.append(
            html.Div(
                [
                    html.I(className="bi bi-newspaper", style={"fontSize": "24px", "opacity": "0.5", "marginBottom": "8px"}),
                    html.P(f"No news available for {label}", style={"color": "#6B7280", "margin": "0"}),
                ],
                className="tab-empty-state",
                style={"textAlign": "center", "padding": "32px 16px"},
            )
        )

    return html.Div(children, className="tab-content-inner")


def _get_date_range(articles: list) -> str:
    """Get formatted date range from articles list."""
    if not articles:
        return ""

    dates = []
    for a in articles:
        pub = a.get("published_at")
        if pub:
            try:
                if isinstance(pub, str):
                    dt = datetime.fromisoformat(pub)
                else:
                    dt = pub
                dates.append(dt)
            except (ValueError, TypeError):
                continue

    if not dates:
        return ""

    oldest = min(dates)
    newest = max(dates)

    if oldest.date() == newest.date():
        return newest.strftime("%b %d")
    else:
        return f"{oldest.strftime('%b %d')} - {newest.strftime('%b %d')}"


@callback(
    Output("llm-status", "children"),
    Input("news-data-store", "data"),
)
def update_llm_status(news_data):
    """Update LLM status indicator in panel header."""
    llm = get_llm()

    if not llm.is_available():
        return "LLM: Offline"

    return f"LLM: {llm.provider}"


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
