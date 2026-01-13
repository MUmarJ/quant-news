"""Chart generation and update callbacks.

This module provides functions to create Plotly charts
following the design system in PROJECT.md.
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CHART_THEME, COLORS, INDICATOR_COLORS


def apply_chart_theme(fig: go.Figure, **overrides) -> None:
    """Apply chart theme with optional overrides.

    This helper merges CHART_THEME with any overrides to avoid
    duplicate keyword argument errors.
    """
    # Start with base theme
    layout_args = {
        "paper_bgcolor": CHART_THEME.get("paper_bgcolor"),
        "plot_bgcolor": CHART_THEME.get("plot_bgcolor"),
        "font": CHART_THEME.get("font"),
        "xaxis": CHART_THEME.get("xaxis"),
        "yaxis": CHART_THEME.get("yaxis"),
        "hovermode": CHART_THEME.get("hovermode"),
        "hoverlabel": CHART_THEME.get("hoverlabel"),
        "margin": CHART_THEME.get("margin"),
    }
    # Apply overrides
    layout_args.update(overrides)
    fig.update_layout(**layout_args)


def create_price_chart(
    df: pd.DataFrame,
    symbol: str,
    indicators: Optional[list[str]] = None,
) -> go.Figure:
    """Create the main price chart with optional indicators.

    Args:
        df: DataFrame with OHLCV and indicator columns.
        symbol: Stock ticker symbol.
        indicators: List of indicators to overlay.

    Returns:
        Plotly Figure object.
    """
    if df.empty:
        return create_empty_chart("No data available")

    indicators = indicators or []

    fig = go.Figure()

    # Candlestick chart - use .tolist() to ensure proper array serialization
    fig.add_trace(
        go.Candlestick(
            x=df.index.tolist(),
            open=df["Open"].tolist(),
            high=df["High"].tolist(),
            low=df["Low"].tolist(),
            close=df["Close"].tolist(),
            name=symbol,
            increasing_line_color=COLORS.POSITIVE,
            decreasing_line_color=COLORS.NEGATIVE,
            increasing_fillcolor=COLORS.POSITIVE,
            decreasing_fillcolor=COLORS.NEGATIVE,
        )
    )

    # Add SMA overlays with hover templates
    if "sma_20" in indicators and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                name="SMA 20",
                line=dict(color=INDICATOR_COLORS["sma_20"], width=1),
                opacity=0.8,
                hovertemplate=(
                    "<b>SMA 20</b><br>"
                    "Value: $%{y:.2f}<br>"
                    "<i>Average of last 20 closing prices</i>"
                    "<extra></extra>"
                ),
            )
        )

    if "sma_50" in indicators and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                name="SMA 50",
                line=dict(color=INDICATOR_COLORS["sma_50"], width=1),
                opacity=0.8,
                hovertemplate=(
                    "<b>SMA 50</b><br>"
                    "Value: $%{y:.2f}<br>"
                    "<i>Average of last 50 closing prices</i>"
                    "<extra></extra>"
                ),
            )
        )

    if "sma_200" in indicators and "SMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_200"],
                name="SMA 200",
                line=dict(color=INDICATOR_COLORS["sma_200"], width=1.5),
                opacity=0.8,
                hovertemplate=(
                    "<b>SMA 200</b><br>"
                    "Value: $%{y:.2f}<br>"
                    "<i>Average of last 200 closing prices (long-term trend)</i>"
                    "<extra></extra>"
                ),
            )
        )

    # Bollinger Bands with hover templates
    if "bollinger" in indicators:
        if "BB_Upper" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_Upper"],
                    name="BB Upper",
                    line=dict(color=INDICATOR_COLORS["bollinger_upper"], width=1, dash="dot"),
                    opacity=0.5,
                    hovertemplate=(
                        "<b>Bollinger Upper</b><br>"
                        "Value: $%{y:.2f}<br>"
                        "<i>= SMA(20) + 2 × Std Dev</i>"
                        "<extra></extra>"
                    ),
                )
            )
        if "BB_Lower" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_Lower"],
                    name="BB Lower",
                    line=dict(color=INDICATOR_COLORS["bollinger_lower"], width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(255, 230, 109, 0.1)",
                    opacity=0.5,
                    hovertemplate=(
                        "<b>Bollinger Lower</b><br>"
                        "Value: $%{y:.2f}<br>"
                        "<i>= SMA(20) - 2 × Std Dev</i>"
                        "<extra></extra>"
                    ),
                )
            )

    # Apply theme
    fig.update_layout(
        **CHART_THEME,
        title=None,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        xaxis_rangeslider_visible=False,
        autosize=False,
        height=380,
    )

    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    """Create MACD indicator chart.

    Args:
        df: DataFrame with MACD columns.

    Returns:
        Plotly Figure object.
    """
    if df.empty or "MACD" not in df.columns:
        return create_empty_chart("MACD data not available")

    fig = go.Figure()

    # Compute trend labels for histogram
    hist_trends = []
    if "MACD_Histogram" in df.columns:
        for i, val in enumerate(df["MACD_Histogram"]):
            if i == 0:
                hist_trends.append("Neutral")
            elif val > df["MACD_Histogram"].iloc[i - 1]:
                hist_trends.append("Bullish momentum")
            else:
                hist_trends.append("Bearish momentum")

    # MACD Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            line=dict(color=INDICATOR_COLORS["macd_line"], width=1.5),
            hovertemplate=(
                "<b>MACD Line</b><br>"
                "Value: %{y:.3f}<br>"
                "<i>= EMA(12) - EMA(26)</i>"
                "<extra></extra>"
            ),
        )
    )

    # Signal Line
    if "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                name="Signal",
                line=dict(color=INDICATOR_COLORS["macd_signal"], width=1.5),
                hovertemplate=(
                    "<b>Signal Line</b><br>"
                    "Value: %{y:.3f}<br>"
                    "<i>= 9-day EMA of MACD</i>"
                    "<extra></extra>"
                ),
            )
        )

    # Histogram
    if "MACD_Histogram" in df.columns:
        colors = [
            COLORS.POSITIVE if val >= 0 else COLORS.NEGATIVE
            for val in df["MACD_Histogram"]
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Histogram"],
                name="Histogram",
                marker_color=colors,
                opacity=0.7,
                customdata=hist_trends,
                hovertemplate=(
                    "<b>MACD Histogram</b><br>"
                    "Value: %{y:.3f}<br>"
                    "Trend: %{customdata}<br>"
                    "<i>= MACD - Signal</i>"
                    "<extra></extra>"
                ),
            )
        )

    # Apply theme
    apply_chart_theme(
        fig,
        title=None,
        showlegend=False,
        margin=dict(l=10, r=60, t=10, b=30),
        autosize=False,
        height=140,
    )

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS.BORDER_FOCUS,
        line_width=1,
    )

    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Create RSI indicator chart.

    Args:
        df: DataFrame with RSI column.

    Returns:
        Plotly Figure object.
    """
    if df.empty or "RSI" not in df.columns:
        return create_empty_chart("RSI data not available")

    fig = go.Figure()

    # Compute RSI signal labels
    rsi_signals = []
    for val in df["RSI"]:
        if pd.isna(val):
            rsi_signals.append("N/A")
        elif val > 70:
            rsi_signals.append("Overbought")
        elif val < 30:
            rsi_signals.append("Oversold")
        else:
            rsi_signals.append("Neutral")

    # RSI Line with hover template
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI",
            line=dict(color=INDICATOR_COLORS["rsi"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(123, 97, 255, 0.1)",
            customdata=rsi_signals,
            hovertemplate=(
                "<b>RSI (14-day)</b><br>"
                "Value: %{y:.1f}<br>"
                "Signal: %{customdata}<br>"
                "<i>Overbought: &gt;70 | Oversold: &lt;30</i>"
                "<extra></extra>"
            ),
        )
    )

    # Apply theme - build yaxis separately to avoid duplicate key error
    yaxis_config = {**CHART_THEME.get("yaxis", {}), "range": [0, 100]}
    fig.update_layout(
        paper_bgcolor=CHART_THEME.get("paper_bgcolor"),
        plot_bgcolor=CHART_THEME.get("plot_bgcolor"),
        font=CHART_THEME.get("font"),
        xaxis=CHART_THEME.get("xaxis"),
        hovermode=CHART_THEME.get("hovermode"),
        hoverlabel=CHART_THEME.get("hoverlabel"),
        title=None,
        showlegend=False,
        margin=dict(l=10, r=60, t=10, b=30),
        yaxis=yaxis_config,
        autosize=False,
        height=140,
    )

    # Add overbought/oversold lines
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color=COLORS.NEGATIVE,
        line_width=1,
        annotation_text="70",
        annotation_position="right",
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=COLORS.POSITIVE,
        line_width=1,
        annotation_text="30",
        annotation_position="right",
    )
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color=COLORS.BORDER_FOCUS,
        line_width=1,
    )

    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Create volume bar chart.

    Args:
        df: DataFrame with Volume column.

    Returns:
        Plotly Figure object.
    """
    if df.empty or "Volume" not in df.columns:
        return create_empty_chart("Volume data not available")

    # Determine colors and volume comparison based on price change
    colors = []
    vol_comparisons = []
    for i in range(len(df)):
        if i == 0:
            colors.append(COLORS.TEXT_MUTED)
            vol_comparisons.append("N/A")
        elif df["Close"].iloc[i] >= df["Close"].iloc[i - 1]:
            colors.append(COLORS.VOLUME_UP)
            vol_comparisons.append("Up day")
        else:
            colors.append(COLORS.VOLUME_DOWN)
            vol_comparisons.append("Down day")

    # Add volume vs MA comparison
    vol_vs_ma = []
    if "Volume_MA" in df.columns:
        for i in range(len(df)):
            vol = df["Volume"].iloc[i]
            ma = df["Volume_MA"].iloc[i]
            if pd.isna(ma):
                vol_vs_ma.append("N/A")
            elif vol > ma * 1.5:
                vol_vs_ma.append("High (>150% of avg)")
            elif vol > ma:
                vol_vs_ma.append("Above average")
            elif vol > ma * 0.5:
                vol_vs_ma.append("Below average")
            else:
                vol_vs_ma.append("Low (<50% of avg)")
    else:
        vol_vs_ma = ["N/A"] * len(df)

    fig = go.Figure()

    # Volume bars with hover template
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.8,
            customdata=list(zip(vol_comparisons, vol_vs_ma)),
            hovertemplate=(
                "<b>Volume</b><br>"
                "Value: %{y:,.0f}<br>"
                "Day: %{customdata[0]}<br>"
                "vs MA: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

    # Volume MA overlay with hover template
    if "Volume_MA" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Volume_MA"],
                name="Vol MA",
                line=dict(color=COLORS.CHART_4, width=1.5),
                hovertemplate=(
                    "<b>Volume MA (20-day)</b><br>"
                    "Value: %{y:,.0f}<br>"
                    "<i>Average of last 20 trading days</i>"
                    "<extra></extra>"
                ),
            )
        )

    # Apply theme
    apply_chart_theme(
        fig,
        title=None,
        showlegend=False,
        margin=dict(l=10, r=60, t=10, b=30),
        autosize=False,
        height=110,
    )

    return fig


def create_empty_chart(message: str = "No data") -> go.Figure:
    """Create an empty chart with message.

    Args:
        message: Message to display.

    Returns:
        Plotly Figure with empty state.
    """
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS.TEXT_MUTED),
    )

    # Build layout without conflicting xaxis/yaxis from CHART_THEME
    fig.update_layout(
        paper_bgcolor=COLORS.BG_PRIMARY,
        plot_bgcolor=COLORS.BG_PRIMARY,
        font=CHART_THEME.get("font", {}),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        autosize=False,
        height=380,
    )

    return fig


def create_comparison_chart(
    data: dict[str, pd.DataFrame],
    normalize: bool = True,
) -> go.Figure:
    """Create a comparison chart for multiple stocks.

    Args:
        data: Dictionary mapping symbol to DataFrame.
        normalize: If True, normalize to percentage change.

    Returns:
        Plotly Figure with overlaid lines.
    """
    if not data:
        return create_empty_chart("No stocks selected")

    fig = go.Figure()

    colors = [COLORS.CHART_1, COLORS.CHART_2, COLORS.CHART_3,
              COLORS.CHART_4, COLORS.CHART_5]

    for i, (symbol, df) in enumerate(data.items()):
        if df.empty:
            continue

        y_values = df["Close"]

        if normalize:
            # Normalize to percentage change from first value
            y_values = ((y_values / y_values.iloc[0]) - 1) * 100

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=y_values,
                name=symbol,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig.update_layout(
        **CHART_THEME,
        title=None,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        yaxis_title="% Change" if normalize else "Price",
        autosize=False,
        height=380,
    )

    if normalize:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=COLORS.BORDER_FOCUS,
            line_width=1,
        )

    return fig
