"""LLM service for news summarization and analysis.

This module provides integration with LM Studio (local) and OpenAI (cloud)
for generating news summaries and sentiment analysis.
"""

from typing import Optional

import requests
from openai import OpenAI

from config import API


class LLMService:
    """Service for LLM-powered text generation.

    Supports LM Studio (local) with OpenAI API fallback.

    Attributes:
        client: OpenAI client instance.
        provider: Current provider ('lm_studio' or 'openai').
    """

    def __init__(self) -> None:
        """Initialize LLM service with auto-detection of LM Studio."""
        self.client: Optional[OpenAI] = None
        self.provider: Optional[str] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize OpenAI client with appropriate base URL."""
        # Try LM Studio first
        if self._is_lm_studio_available():
            self.client = OpenAI(
                base_url=API.LM_STUDIO_URL,
                api_key="not-needed",  # LM Studio doesn't require API key
            )
            self.provider = "lm_studio"
            return

        # Fallback to OpenAI
        if API.OPENAI_API_KEY:
            self.client = OpenAI(
                base_url=API.OPENAI_BASE_URL,
                api_key=API.OPENAI_API_KEY,
            )
            self.provider = "openai"
            return

        # No LLM available
        self.provider = None

    def _is_lm_studio_available(self) -> bool:
        """Check if LM Studio is running.

        Returns:
            True if LM Studio server is responding, False otherwise.
        """
        try:
            response = requests.get(
                f"{API.LM_STUDIO_URL}/models",
                timeout=2,
            )
            return response.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if any LLM provider is available.

        Returns:
            True if an LLM provider is configured and responding.
        """
        return self.provider is not None

    def get_provider_info(self) -> dict:
        """Get information about the current LLM provider.

        Returns:
            Dictionary with provider name and status.
        """
        return {
            "provider": self.provider or "none",
            "available": self.is_available(),
            "lm_studio_url": API.LM_STUDIO_URL,
            "has_openai_key": bool(API.OPENAI_API_KEY),
        }

    def _get_model(self) -> str:
        """Get the appropriate model name for the provider.

        Returns:
            Model identifier string.
        """
        if self.provider == "lm_studio":
            # LM Studio uses the loaded model
            return "local-model"
        elif self.provider == "openai":
            return "gpt-3.5-turbo"
        return ""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Generate text using the LLM.

        Args:
            prompt: User prompt/question.
            system_prompt: Optional system instructions.
            max_tokens: Maximum tokens in response.
            temperature: Creativity parameter (0-1).

        Returns:
            Generated text or None if unavailable.
        """
        if not self.is_available() or self.client is None:
            return None

        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self._get_model(),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"LLM generation error: {e}")
            return None

    def summarize_news(
        self,
        articles: list[dict],
        symbol: str,
    ) -> Optional[str]:
        """Generate a summary of news articles.

        Args:
            articles: List of article dictionaries with 'title' and 'summary'.
            symbol: Stock symbol for context.

        Returns:
            AI-generated summary or None if unavailable.
        """
        if not articles:
            return None

        # Build context from articles
        article_text = "\n".join([
            f"- {a.get('title', '')}: {a.get('summary', '')[:200]}"
            for a in articles[:10]  # Limit to 10 articles
        ])

        system_prompt = """You are an objective financial news analyst. Synthesize news into a clear, actionable conclusion.

RULES:
- Only state facts from the provided articles. Never fabricate.
- Commit to ONE clear recommendation. No hedging or "on the other hand" statements.
- If news is mixed, pick the direction with stronger evidence.
- Be concise. No filler. No caveats after your recommendation.
- Use the EXACT markdown format provided. Do not deviate."""

        prompt = f"""Analyze recent news for {symbol}:

{article_text}

Respond using this EXACT markdown format:

### Key Developments

[2-3 sentences on the most important news. Be specific about events, numbers, or catalysts mentioned.]

---

### Market Sentiment

**[BULLISH / BEARISH / NEUTRAL]**

[One sentence explaining why based on news tone and content.]

---

### Recommendation

> **[BULLISH / CAUTIOUS BULLISH / NEUTRAL / CAUTIOUS BEARISH / BEARISH]**

[One sentence explaining your recommendation. No counterpoints.]"""

        return self.generate(prompt, system_prompt, max_tokens=450)

    def analyze_sentiment(
        self,
        text: str,
    ) -> Optional[dict]:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with sentiment analysis or None.
        """
        system_prompt = """You are a sentiment analysis expert for financial text.
Analyze the sentiment and respond ONLY with a JSON object containing:
- sentiment: "bullish", "bearish", or "neutral"
- confidence: a number between 0 and 1
- reasoning: a brief one-sentence explanation"""

        prompt = f"""Analyze the sentiment of this financial text:

"{text}"

Respond with JSON only."""

        response = self.generate(prompt, system_prompt, max_tokens=150, temperature=0.3)

        if response:
            try:
                # Try to parse JSON from response
                import json
                # Handle potential markdown code blocks
                clean_response = response.strip()
                if clean_response.startswith("```"):
                    clean_response = clean_response.split("```")[1]
                    if clean_response.startswith("json"):
                        clean_response = clean_response[4:]
                return json.loads(clean_response)
            except Exception:
                return {"sentiment": "neutral", "confidence": 0.5, "raw": response}

        return None

    def generate_market_insight(
        self,
        symbol: str,
        price_data: dict,
        signals: dict,
        news_sentiment: Optional[str] = None,
    ) -> Optional[str]:
        """Generate comprehensive market insight.

        Args:
            symbol: Stock symbol.
            price_data: Dictionary with price metrics.
            signals: Dictionary with technical signals.
            news_sentiment: Optional news sentiment summary.

        Returns:
            AI-generated market insight or None.
        """
        system_prompt = """You are a professional market analyst.
Provide clear, concise insights based on technical and fundamental data.
Avoid making specific predictions or recommendations.
Focus on factual observations and key levels to watch."""

        # Build context
        price_context = f"""
Symbol: {symbol}
Current Price: ${price_data.get('end_price', 'N/A')}
1Y Return: {price_data.get('total_return', 'N/A')}%
Volatility: {price_data.get('volatility', 'N/A')}%
Max Drawdown: {price_data.get('max_drawdown', 'N/A')}%
"""

        signal_context = ""
        if signals:
            signal_lines = []
            for key, val in signals.items():
                if isinstance(val, dict):
                    signal_lines.append(f"- {key}: {val.get('signal', str(val))}")
            signal_context = "\nTechnical Signals:\n" + "\n".join(signal_lines)

        news_context = ""
        if news_sentiment:
            news_context = f"\nNews Sentiment: {news_sentiment}"

        prompt = f"""Based on the following data, provide a brief market insight for {symbol}:

{price_context}{signal_context}{news_context}

Provide a 3-4 sentence analysis covering:
1. Current technical position
2. Key levels to watch
3. Notable observations"""

        return self.generate(prompt, system_prompt, max_tokens=400)


# Singleton instance
_llm_instance: Optional[LLMService] = None


def get_llm() -> LLMService:
    """Get the singleton LLM service instance.

    Returns:
        LLMService instance.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMService()
    return _llm_instance
