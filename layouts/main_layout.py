"""Main dashboard layout for Quant News Tracker.

This module defines the overall page structure following the
3-column layout specified in PROJECT.md.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from layouts.components import (
    create_data_actions,
    create_indicator_toggles,
    create_period_selector,
    create_stock_input,
)


def create_sidebar() -> html.Div:
    """Create the left sidebar with stock input and controls.

    Returns:
        Sidebar div component.
    """
    return html.Div(
        [
            # Logo/Title
            html.Div(
                [
                    html.H1("QuantNews", className="app-title"),
                    html.P("Stock Analysis Dashboard", className="app-subtitle"),
                ],
                className="sidebar-header",
            ),
            html.Hr(className="sidebar-divider"),

            # Stock Input
            create_stock_input(),

            html.Hr(className="sidebar-divider"),

            # Period Selector
            html.Div(
                [
                    html.Label("Time Period", className="input-label"),
                    create_period_selector("1y"),
                ],
                className="period-section",
            ),

            html.Hr(className="sidebar-divider"),

            # Indicator Toggles
            html.Div(
                [
                    html.Label("Indicators", className="input-label"),
                    create_indicator_toggles(),
                ],
                className="indicator-section",
            ),

            html.Hr(className="sidebar-divider"),

            # Data Actions
            html.Div(
                [
                    html.Label("Data", className="input-label"),
                    create_data_actions(),
                ],
                className="data-section",
            ),

            # Spacer
            html.Div(className="sidebar-spacer"),

            # Footer
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Cache: ", className="cache-label"),
                            html.Span(id="cache-status", className="cache-status-text"),
                            dbc.Switch(
                                id="cache-toggle",
                                value=True,
                                className="cache-toggle-switch",
                                style={"marginLeft": "auto"},
                            ),
                        ],
                        className="cache-status-row",
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(id="data-source-indicator", className="data-source-row"),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
        id="sidebar",
    )


def create_main_content() -> html.Div:
    """Create the main content area with charts.

    Returns:
        Main content div component.
    """
    return html.Div(
        [
            # Summary Cards Row
            html.Div(
                id="summary-cards",
                className="summary-cards-row",
            ),

            # Main Price Chart
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(id="chart-title", className="chart-title"),
                            html.Div(id="chart-subtitle", className="chart-subtitle"),
                        ],
                        className="chart-header",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="price-chart",
                            className="main-chart",
                            style={"height": "380px"},
                            config={
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": [
                                    "lasso2d",
                                    "select2d",
                                ],
                                "displaylogo": False,
                                "responsive": False,
                            },
                        ),
                        type="circle",
                        color="#00D4AA",
                    ),
                ],
                className="chart-container price-chart-container",
            ),

            # Technical Indicators Row
            html.Div(
                [
                    # MACD Chart
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("MACD", className="subplot-title"),
                                    html.I(
                                        className="bi bi-info-circle ms-2 info-icon",
                                        id="macd-info-icon",
                                    ),
                                    dbc.Tooltip(
                                        "Moving Average Convergence Divergence: "
                                        "MACD Line = EMA(12) - EMA(26), "
                                        "Signal Line = 9-day EMA of MACD. "
                                        "Bullish when MACD crosses above Signal.",
                                        target="macd-info-icon",
                                        placement="top",
                                    ),
                                ],
                                className="subplot-title-row",
                            ),
                            dcc.Loading(
                                dcc.Graph(
                                    id="macd-chart",
                                    className="subplot-chart",
                                    style={"height": "140px"},
                                    config={"displayModeBar": False, "responsive": False},
                                ),
                                type="circle",
                                color="#00D4AA",
                            ),
                        ],
                        className="subplot-container",
                    ),
                    # RSI Chart
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("RSI", className="subplot-title"),
                                    html.I(
                                        className="bi bi-info-circle ms-2 info-icon",
                                        id="rsi-info-icon",
                                    ),
                                    dbc.Tooltip(
                                        "Relative Strength Index (14-day): "
                                        "Measures momentum on a 0-100 scale. "
                                        "Overbought: >70, Oversold: <30.",
                                        target="rsi-info-icon",
                                        placement="top",
                                    ),
                                ],
                                className="subplot-title-row",
                            ),
                            dcc.Loading(
                                dcc.Graph(
                                    id="rsi-chart",
                                    className="subplot-chart",
                                    style={"height": "140px"},
                                    config={"displayModeBar": False, "responsive": False},
                                ),
                                type="circle",
                                color="#00D4AA",
                            ),
                        ],
                        className="subplot-container",
                    ),
                ],
                className="indicators-row",
            ),

            # Volume Chart
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Volume", className="subplot-title"),
                            html.I(
                                className="bi bi-info-circle ms-2 info-icon",
                                id="volume-info-icon",
                            ),
                            dbc.Tooltip(
                                "Trading Volume: Number of shares traded. "
                                "Green = up day, Red = down day. "
                                "Line shows 20-day moving average.",
                                target="volume-info-icon",
                                placement="top",
                            ),
                        ],
                        className="subplot-title-row",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="volume-chart",
                            className="volume-chart",
                            style={"height": "110px"},
                            config={"displayModeBar": False, "responsive": False},
                        ),
                        type="circle",
                        color="#00D4AA",
                    ),
                ],
                className="chart-container volume-container",
            ),
        ],
        className="main-content",
        id="main-content",
    )


def create_context_panel() -> html.Div:
    """Create the right context panel with news and insights.

    Returns:
        Context panel div component.
    """
    return html.Div(
        [
            # News Section
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Latest News", className="panel-title"),
                            html.Span(id="news-meta", className="news-meta-info"),
                        ],
                        className="panel-title-row",
                    ),
                    dcc.Loading(
                        html.Div(id="news-feed", className="news-feed"),
                        type="circle",
                        color="#00D4AA",
                    ),
                ],
                className="news-section",
            ),

            html.Hr(className="panel-divider"),

            # AI Summary Section
            html.Div(
                [
                    html.H3("AI Summary", className="panel-title"),
                    html.Div(
                        [
                            html.Span(id="llm-status", className="llm-status-badge"),
                        ],
                        className="llm-status-row",
                    ),
                    dcc.Loading(
                        html.Div(id="ai-summary", className="ai-summary"),
                        type="circle",
                        color="#00D4AA",
                    ),
                ],
                className="ai-section",
            ),

            html.Hr(className="panel-divider"),

            # Sentiment Section
            html.Div(
                [
                    html.H3("Sentiment", className="panel-title"),
                    html.Div(id="sentiment-display", className="sentiment-display"),
                ],
                className="sentiment-section",
            ),

            html.Hr(className="panel-divider"),

            # Technical Signals
            html.Div(
                [
                    html.H3("Signals", className="panel-title"),
                    html.Div(id="signals-display", className="signals-display"),
                ],
                className="signals-section",
            ),
        ],
        className="context-panel",
        id="context-panel",
    )


def create_data_modal() -> dbc.Modal:
    """Create the raw data view modal.

    Returns:
        Modal component for viewing data.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Raw Data"),
                close_button=True,
            ),
            dbc.ModalBody(
                [
                    html.Div(id="data-table-container"),
                ],
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Export Parquet",
                        id="modal-export-btn",
                        color="primary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Close",
                        id="modal-close-btn",
                        color="secondary",
                    ),
                ],
            ),
        ],
        id="data-modal",
        size="xl",
        is_open=False,
    )


def create_layout() -> html.Div:
    """Create the complete dashboard layout.

    Returns:
        Root layout component.
    """
    return html.Div(
        [
            # Store for selected symbols
            dcc.Store(id="selected-symbols", data=[]),
            dcc.Store(id="current-period", data="1y"),
            dcc.Store(id="stock-data-store", data={}),
            dcc.Store(id="news-data-store", data={}),
            dcc.Store(id="cache-enabled", data=True),

            # Download component for exports
            dcc.Download(id="download-data"),

            # Main Layout Grid
            html.Div(
                [
                    create_sidebar(),
                    create_main_content(),
                    create_context_panel(),
                ],
                className="dashboard-grid",
            ),

            # Modals
            create_data_modal(),

            # Toast notifications
            html.Div(id="toast-container"),
        ],
        className="app-container",
    )
