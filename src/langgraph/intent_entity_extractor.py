"""
Intent and Entity Extractor
Real-Time Financial Advisory System - Phase 5 Component 1

This extractor prefers deterministic parsing over free-form LLM output so the
rest of the NLP-SQL pipeline gets stable, schema-aware inputs.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import config as settings
from src.langgraph.sql_generator import DatabaseSchemaKnowledge

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency at runtime
    HumanMessage = None
    SystemMessage = None
    ChatGoogleGenerativeAI = None


class IntentEntityExtractor:
    """Extract intent and entities from a user query."""

    COMPANY_ALIASES: Dict[str, str] = {
        "apple": "AAPL",
        "amazon": "AMZN",
        "alphabet": "GOOGL",
        "google": "GOOGL",
        "microsoft": "MSFT",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "salesforce": "CRM",
        "infosys": "INFY",
        "meta": "META",
        "netflix": "NFLX",
    }

    SECTOR_KEYWORDS = {
        "technology": "Technology",
        "financial": "Financial Services",
        "healthcare": "Healthcare",
        "consumer": "Consumer Defensive",
        "energy": "Energy",
        "industrials": "Industrials",
    }

    METRIC_KEYWORDS = {
        "price": "close_price",
        "latest price": "close_price",
        "volume": "volume",
        "rsi": "rsi_14",
        "macd": "macd",
        "macd signal": "macd_signal",
        "volatility": "volatility_30d",
        "beta": "beta",
        "sharpe": "sharpe_ratio",
        "sortino": "sortino_ratio",
        "var": "value_at_risk_95",
        "value at risk": "value_at_risk_95",
        "sentiment": "sentiment_score",
        "news": "sentiment_score",
        "drawdown": "max_drawdown",
        "return": "return_30d",
    }

    def __init__(self):
        self.settings = settings
        self.known_symbols = set(DatabaseSchemaKnowledge.KNOWN_SYMBOLS)
        self.llm = None

        if ChatGoogleGenerativeAI is not None:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    google_api_key=self.settings.GEMINI_API_KEY,
                    temperature=0.0,
                    max_tokens=800,
                )
            except Exception:
                self.llm = None

    async def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        text = user_query.strip()
        lowered = text.lower()

        symbols, unknown_mentions = self._extract_symbols(text)
        metrics = self._extract_metrics(lowered)
        time_period, date_context = self._extract_time_period(lowered)
        sectors = self._extract_sectors(lowered)
        limit = self._extract_limit(lowered)
        sentiment_filter = self._extract_sentiment_filter(lowered)
        intent = self._detect_intent(lowered, symbols, metrics, sectors)

        result = {
            "intent": intent,
            "entities": {
                "stocks": symbols,
                "unknown_symbols": unknown_mentions,
                "metrics": metrics,
                "time_period": time_period,
                "date_context": date_context,
                "sectors": sectors,
                "limit": limit,
                "portfolio_id": self._extract_portfolio_id(text),
                "comparison": "compare" in lowered or " vs " in lowered or "versus" in lowered,
                "sentiment_filter": sentiment_filter,
                "trend": "trend" in lowered,
                "raw_query": text,
            },
            "confidence": self._estimate_confidence(symbols, metrics, intent),
            "notes": [],
        }

        if unknown_mentions:
            result["notes"].append(
                "Some ticker mentions were unresolved; the SQL layer should fall back to symbol lookup."
            )

        if intent == "general_inquiry" and self.llm is not None:
            llm_result = await self._try_llm_fallback(text)
            if llm_result:
                return llm_result

        return result

    def _extract_symbols(self, text: str) -> Tuple[List[str], List[str]]:
        symbols: List[str] = []
        unknown_mentions: List[str] = []
        upper_tokens = re.findall(r"\b[A-Z]{1,5}(?:-[A-Z])?\b", text)

        for token in upper_tokens:
            if token in self.known_symbols and token not in symbols:
                symbols.append(token)
            elif token not in unknown_mentions and token not in {"RSI", "MACD", "VAR"}:
                unknown_mentions.append(token)

        lowered = text.lower()
        for company, symbol in self.COMPANY_ALIASES.items():
            if company in lowered and symbol not in symbols:
                symbols.append(symbol)

        if not symbols:
            quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
            for left, right in quoted:
                phrase = (left or right).strip()
                if phrase and phrase not in unknown_mentions:
                    unknown_mentions.append(phrase)

        return symbols, unknown_mentions

    def _extract_metrics(self, lowered: str) -> List[str]:
        metrics: List[str] = []
        for phrase, metric_name in self.METRIC_KEYWORDS.items():
            if phrase in lowered and metric_name not in metrics:
                metrics.append(metric_name)
        return metrics

    def _extract_time_period(self, lowered: str) -> Tuple[Optional[str], Dict[str, Any]]:
        mappings = [
            ("today", ("latest", {"kind": "latest"})),
            ("latest", ("latest", {"kind": "latest"})),
            ("current", ("latest", {"kind": "latest"})),
            ("this week", ("1w", {"kind": "relative", "days": 7})),
            ("last week", ("1w", {"kind": "relative", "days": 7})),
            ("this month", ("1m", {"kind": "relative", "days": 31})),
            ("last month", ("1m", {"kind": "relative", "days": 31})),
            ("this year", ("1y", {"kind": "relative", "days": 365})),
            ("last year", ("1y", {"kind": "relative", "days": 365})),
        ]
        for phrase, value in mappings:
            if phrase in lowered:
                return value

        compact = re.search(r"\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\b", lowered)
        if compact:
            quantity = int(compact.group(1))
            unit = compact.group(2)
            if "day" in unit:
                days = quantity
            elif "week" in unit:
                days = quantity * 7
            elif "month" in unit:
                days = quantity * 30
            else:
                days = quantity * 365
            return f"{quantity}{unit[0]}", {"kind": "relative", "days": days}

        return None, {}

    def _extract_sectors(self, lowered: str) -> List[str]:
        sectors: List[str] = []
        for phrase, sector in self.SECTOR_KEYWORDS.items():
            if phrase in lowered and sector not in sectors:
                sectors.append(sector)
        return sectors

    def _extract_limit(self, lowered: str) -> Optional[int]:
        match = re.search(r"\btop\s+(\d+)\b", lowered)
        if match:
            return int(match.group(1))

        match = re.search(r"\bhighest\s+(\d+)\b", lowered)
        if match:
            return int(match.group(1))

        return None

    def _extract_sentiment_filter(self, lowered: str) -> Optional[str]:
        if "negative sentiment" in lowered or "negative" in lowered:
            return "negative"
        if "positive sentiment" in lowered or "positive" in lowered:
            return "positive"
        return None

    def _extract_portfolio_id(self, text: str) -> Optional[str]:
        match = re.search(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            text,
            re.IGNORECASE,
        )
        return match.group(0) if match else None

    def _detect_intent(
        self,
        lowered: str,
        symbols: List[str],
        metrics: List[str],
        sectors: List[str],
    ) -> str:
        if any(keyword in lowered for keyword in ("portfolio", "holdings", "allocation", "rebalance", "optimize my portfolio", "portfolio optimization", "diversification")):
            return "portfolio_review"
        if sectors or "sector" in lowered or "with low volatility" in lowered:
            return "screening_query"
        if any(keyword in lowered for keyword in ("sentiment", "news", "headline")):
            return "sentiment_analysis"
        if any(keyword in lowered for keyword in ("rsi", "macd", "sma", "ema", "overbought", "oversold")):
            return "technical_analysis"
        if any(keyword in lowered for keyword in ("volatility", "beta", "sharpe", "sortino", "risk", "value at risk", "var")):
            return "risk_assessment"
        if any(keyword in lowered for keyword in ("compare", "versus", " vs ")):
            return "comparative_analysis"
        if any(keyword in lowered for keyword in ("top ", "highest", "lowest", "rank")):
            return "ranking_query"
        if symbols or any(metric in metrics for metric in ("close_price", "volume")):
            return "stock_analysis"
        return "general_inquiry"

    def _estimate_confidence(self, symbols: List[str], metrics: List[str], intent: str) -> str:
        score = 0
        if intent != "general_inquiry":
            score += 1
        if symbols:
            score += 1
        if metrics:
            score += 1
        if score >= 3:
            return "high"
        if score == 2:
            return "medium"
        return "low"

    async def _try_llm_fallback(self, user_query: str) -> Optional[Dict[str, Any]]:
        if self.llm is None:
            return None

        prompt = (
            "Extract intent and entities for a financial SQL agent.\n"
            "Use only these intents: stock_analysis, technical_analysis, "
            "risk_assessment, sentiment_analysis, comparative_analysis, "
            "ranking_query, screening_query, portfolio_review, general_inquiry.\n"
            "Return JSON only with keys intent, entities, confidence, notes."
        )

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=user_query),
            ]
        )
        payload = self._parse_json(self._extract_content(response.content))
        return payload or None

    def _extract_content(self, response_content: Any) -> str:
        content = response_content
        if isinstance(content, list):
            content = " ".join(part if isinstance(part, str) else str(part) for part in content)
        return str(content)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


async def main():
    extractor = IntentEntityExtractor()
    test_queries = [
        "What's the current price and volume of Apple stock?",
        "What are the latest risk metrics for Tesla?",
        "Show me recent news sentiment for Microsoft.",
        "Compare the trading volume of Google and Amazon over the last month.",
        "What are the RSI and MACD indicators for Netflix?",
    ]

    for query in test_queries:
        result = await extractor.extract_intent_and_entities(query)
        print(f"Query: {query}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
