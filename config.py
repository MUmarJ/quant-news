"""Configuration and constants for Quant News Tracker.

This module serves as the SINGLE SOURCE OF TRUTH for all constants,
configuration values, and default settings used throughout the application.
"""

import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# TECHNICAL INDICATOR DEFAULTS
# =============================================================================


@dataclass(frozen=True)
class IndicatorDefaults:
    """Default parameters for technical indicators."""

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


INDICATORS: Final = IndicatorDefaults()


# =============================================================================
# API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """API endpoints and settings."""

    # LM Studio (local LLM)
    LM_STUDIO_URL: str = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

    # OpenAI (fallback)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # Alpha Vantage (news)
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    ALPHA_VANTAGE_BASE_URL: str = "https://www.alphavantage.co/query"

    # Request settings
    DEFAULT_TIMEOUT: int = 30
    MAX_RETRIES: int = 3


API: Final = APIConfig()


# =============================================================================
# APP CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AppConfig:
    """Application settings."""

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8050"))
    HOST: str = os.getenv("HOST", "127.0.0.1")

    # Default data period
    DEFAULT_PERIOD: str = "1y"

    # Popular stocks for quick-add
    POPULAR_STOCKS: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "JPM",
    )


APP: Final = AppConfig()


# =============================================================================
# UI THEME - DESIGN SYSTEM
# =============================================================================


@dataclass(frozen=True)
class Colors:
    """Color palette for the dark theme UI."""

    # Background
    BG_PRIMARY: str = "#0D0D0D"
    BG_SECONDARY: str = "#1A1A1A"
    BG_TERTIARY: str = "#242424"

    # Borders
    BORDER_SUBTLE: str = "#2A2A2A"
    BORDER_FOCUS: str = "#3D3D3D"

    # Text
    TEXT_PRIMARY: str = "#FFFFFF"
    TEXT_SECONDARY: str = "#A0A0A0"
    TEXT_MUTED: str = "#666666"

    # Semantic (Robinhood-inspired)
    POSITIVE: str = "#00C805"
    POSITIVE_MUTED: str = "#0A3D0A"
    NEGATIVE: str = "#FF5000"
    NEGATIVE_MUTED: str = "#3D1A0A"
    NEUTRAL: str = "#FFD700"

    # Accent
    ACCENT_PRIMARY: str = "#00D4AA"
    ACCENT_HOVER: str = "#00E5BB"

    # Chart colors (sequential)
    CHART_1: str = "#00D4AA"  # Primary (price)
    CHART_2: str = "#7B61FF"  # MA-20
    CHART_3: str = "#FF6B6B"  # MA-50
    CHART_4: str = "#4ECDC4"  # MA-200
    CHART_5: str = "#FFE66D"  # Bollinger
    VOLUME_UP: str = "#00C805"
    VOLUME_DOWN: str = "#FF5000"


COLORS: Final = Colors()


@dataclass(frozen=True)
class Typography:
    """Typography settings."""

    FONT_PRIMARY: str = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
    FONT_MONO: str = "'JetBrains Mono', 'SF Mono', monospace"

    # Font sizes (px)
    TEXT_XS: int = 11
    TEXT_SM: int = 13
    TEXT_BASE: int = 15
    TEXT_LG: int = 18
    TEXT_XL: int = 24
    TEXT_2XL: int = 32
    TEXT_3XL: int = 48


TYPOGRAPHY: Final = Typography()


@dataclass(frozen=True)
class Spacing:
    """Spacing scale (px)."""

    SPACE_1: int = 4
    SPACE_2: int = 8
    SPACE_3: int = 12
    SPACE_4: int = 16
    SPACE_5: int = 24
    SPACE_6: int = 32
    SPACE_8: int = 48


SPACING: Final = Spacing()


@dataclass(frozen=True)
class Layout:
    """Layout dimensions."""

    SIDEBAR_WIDTH: int = 240
    CONTEXT_PANEL_WIDTH: int = 320
    CARD_BORDER_RADIUS: int = 12

    # Breakpoints
    BREAKPOINT_MOBILE: int = 768
    BREAKPOINT_TABLET: int = 1024
    BREAKPOINT_DESKTOP: int = 1440


LAYOUT: Final = Layout()


# =============================================================================
# PLOTLY CHART THEME
# =============================================================================


CHART_THEME: Final[dict] = {
    "paper_bgcolor": COLORS.BG_PRIMARY,
    "plot_bgcolor": COLORS.BG_PRIMARY,
    "font": {
        "color": COLORS.TEXT_SECONDARY,
        "family": TYPOGRAPHY.FONT_PRIMARY,
    },
    "xaxis": {
        "gridcolor": COLORS.BORDER_SUBTLE,
        "linecolor": COLORS.BORDER_SUBTLE,
        "tickfont": {"size": TYPOGRAPHY.TEXT_XS},
        "showgrid": True,
        "gridwidth": 1,
        "griddash": "dot",
    },
    "yaxis": {
        "gridcolor": COLORS.BORDER_SUBTLE,
        "linecolor": COLORS.BORDER_SUBTLE,
        "tickfont": {"size": TYPOGRAPHY.TEXT_XS},
        "side": "right",  # Bloomberg-style
        "showgrid": True,
        "gridwidth": 1,
        "griddash": "dot",
    },
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": COLORS.BG_SECONDARY,
        "bordercolor": COLORS.BORDER_FOCUS,
        "font": {
            "family": TYPOGRAPHY.FONT_MONO,
            "size": TYPOGRAPHY.TEXT_SM,
        },
    },
    "margin": {"l": 10, "r": 60, "t": 40, "b": 40},
}


# =============================================================================
# INDICATOR COLOR MAP
# =============================================================================


INDICATOR_COLORS: Final[dict[str, str]] = {
    "price": COLORS.CHART_1,
    "sma_20": COLORS.CHART_2,
    "sma_50": COLORS.CHART_3,
    "sma_200": COLORS.CHART_4,
    "ema_12": COLORS.CHART_2,
    "ema_26": COLORS.CHART_3,
    "bollinger_upper": COLORS.CHART_5,
    "bollinger_lower": COLORS.CHART_5,
    "bollinger_mid": COLORS.CHART_5,
    "macd_line": COLORS.CHART_1,
    "macd_signal": COLORS.CHART_3,
    "rsi": COLORS.CHART_2,
    "obv": COLORS.CHART_4,
}
