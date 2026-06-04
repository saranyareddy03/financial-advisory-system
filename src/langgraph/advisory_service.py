"""
Hybrid financial advisory orchestration.

This service keeps the verified SQL path as the primary route, then falls back
to external providers when the database cannot answer the question. Structured
market data comes from FMP when available; web-backed context comes from Tavily
and is summarized by Gemini with explicit provenance.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from sqlalchemy import text

from src.config.settings import config as settings
from src.analytics.portfolio_advisory import PortfolioAdvisoryService
from src.database.connection import db_manager
from src.langgraph.intent_entity_extractor import IntentEntityExtractor
from src.langgraph.query_executor import (
    DatabaseQueryExecutor,
    ExecutionStatus,
    QueryExecutionResult,
)
from src.langgraph.response_formatter import (
    ConfidenceLevel,
    FormattedResponse,
    ResponseStyle,
    format_financial_response,
)
from src.langgraph.sql_generator import SQLQueryResult, generate_sql_query
from src.utils.groq_client import GroqChatClient

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional runtime dependency
    HumanMessage = None
    SystemMessage = None
    ChatGoogleGenerativeAI = None


logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when an external provider request fails."""


@dataclass
class PriceResult:
    symbol: str
    as_of: Optional[str]
    close_price: Optional[float]
    open_price: Optional[float]
    high_price: Optional[float]
    low_price: Optional[float]
    volume: Optional[int]
    source: str
    source_type: str


@dataclass
class NewsResult:
    symbol: str
    published_at: Optional[str]
    headline: str
    source: str
    url: Optional[str]
    source_type: str
    content: Optional[str] = None
    relevance_score: Optional[float] = None


@dataclass
class AdvisoryServiceResult:
    intent: str
    entities: Dict[str, Any]
    sql_result: SQLQueryResult
    execution_result: QueryExecutionResult
    formatted_response: FormattedResponse
    route: str
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def fetch_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 25,
) -> Dict[str, Any]:
    data = None
    headers = {"User-Agent": "financial-advisory-system/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {exc.code}: {body[:400]}") from exc
    except URLError as exc:
        raise ProviderError(f"Network error: {exc}") from exc


class SupabaseRepository:
    """Read-only DB helpers for fallback decisions."""

    def get_company_name(self, symbol: str) -> Optional[str]:
        query = text("SELECT company_name FROM stocks WHERE symbol = :symbol LIMIT 1")
        with db_manager.get_connection() as conn:
            row = conn.execute(query, {"symbol": symbol.upper()}).mappings().first()
            return row["company_name"] if row else None

    def get_latest_price(self, symbol: str) -> Optional[PriceResult]:
        query = text(
            """
            SELECT sp.symbol, sp.date, sp.close_price, sp.open_price, sp.high_price,
                   sp.low_price, sp.volume
            FROM stock_prices sp
            WHERE sp.symbol = :symbol
            ORDER BY sp.date DESC
            LIMIT 1
            """
        )
        with db_manager.get_connection() as conn:
            row = conn.execute(query, {"symbol": symbol.upper()}).mappings().first()
            if not row:
                return None
            return PriceResult(
                symbol=row["symbol"],
                as_of=row["date"].isoformat() if row["date"] else None,
                close_price=_to_float(row["close_price"]),
                open_price=_to_float(row["open_price"]),
                high_price=_to_float(row["high_price"]),
                low_price=_to_float(row["low_price"]),
                volume=_to_int(row["volume"]),
                source="supabase",
                source_type="db_stock_prices",
            )

    def get_recent_news(self, symbol: str, limit: int = 5) -> List[NewsResult]:
        query = text(
            """
            SELECT ss.symbol, fn.published_at, fn.headline, fn.source, fn.url, fn.content
            FROM sentiment_scores ss
            JOIN financial_news fn ON fn.id = ss.news_id
            WHERE ss.symbol = :symbol
            ORDER BY fn.published_at DESC
            LIMIT :limit
            """
        )
        with db_manager.get_connection() as conn:
            rows = conn.execute(
                query, {"symbol": symbol.upper(), "limit": limit}
            ).mappings().all()
            return [
                NewsResult(
                    symbol=row["symbol"],
                    published_at=row["published_at"].isoformat() if row["published_at"] else None,
                    headline=row["headline"],
                    source=row["source"] or "supabase",
                    url=row["url"],
                    source_type="db_news",
                    content=(row["content"] or "")[:1200] if row.get("content") else None,
                )
                for row in rows
            ]


class FMPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def get_quote(self, symbol: str) -> Optional[PriceResult]:
        if not self.enabled:
            return None
        url = (
            f"https://financialmodelingprep.com/stable/quote?"
            f"symbol={symbol.upper()}&apikey={self.api_key}"
        )
        payload = fetch_json(url)
        if not isinstance(payload, list) or not payload:
            return None
        row = payload[0]
        as_of = None
        if row.get("timestamp"):
            as_of = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc).isoformat()
        return PriceResult(
            symbol=row.get("symbol", symbol.upper()),
            as_of=as_of,
            close_price=_to_float(row.get("price")),
            open_price=_to_float(row.get("open")),
            high_price=_to_float(row.get("dayHigh")),
            low_price=_to_float(row.get("dayLow")),
            volume=_to_int(row.get("volume")),
            source="fmp",
            source_type="quote",
        )

    def get_history(self, symbol: str, limit: int = 90) -> List[PriceResult]:
        if not self.enabled:
            return []
        query = urlencode({"symbol": symbol.upper(), "apikey": self.api_key})
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/light?{query}"
        payload = fetch_json(url)
        if not isinstance(payload, list):
            return []
        rows: List[PriceResult] = []
        for item in payload[:limit]:
            rows.append(
                PriceResult(
                    symbol=item.get("symbol", symbol.upper()),
                    as_of=item.get("date"),
                    close_price=_to_float(item.get("close")),
                    open_price=_to_float(item.get("open")),
                    high_price=_to_float(item.get("high")),
                    low_price=_to_float(item.get("low")),
                    volume=_to_int(item.get("volume")),
                    source="fmp",
                    source_type="historical_eod",
                )
            )
        return rows

    def get_stock_news(self, symbol: str, limit: int = 5) -> List[NewsResult]:
        if not self.enabled:
            return []
        query = urlencode({"symbols": symbol.upper(), "limit": limit, "apikey": self.api_key})
        url = f"https://financialmodelingprep.com/stable/news/stock?{query}"
        payload = fetch_json(url)
        if not isinstance(payload, list):
            return []
        return [
            NewsResult(
                symbol=symbol.upper(),
                published_at=item.get("publishedDate"),
                headline=item.get("title", ""),
                source=item.get("site", "fmp"),
                url=item.get("url"),
                source_type="fmp_stock_news",
                content=(item.get("text") or item.get("content") or "")[:1200] or None,
            )
            for item in payload[:limit]
        ]


class TavilyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search_news(
        self,
        query: str,
        limit: int = 5,
        company_hint: Optional[str] = None,
    ) -> List[NewsResult]:
        if not self.enabled:
            return []

        payload = fetch_json(
            "https://api.tavily.com/search",
            method="POST",
            payload={
                "api_key": self.api_key,
                "query": query,
                "topic": "news",
                "max_results": limit,
                "search_depth": "advanced",
                "include_raw_content": False,
            },
        )

        results: List[NewsResult] = []
        for item in payload.get("results", [])[:limit]:
            title = item.get("title", "")
            content = item.get("content", "")
            if not _looks_relevant(title, content, query, company_hint):
                continue
            results.append(
                NewsResult(
                    symbol=_extract_symbol_from_query(query) or "GENERAL",
                    published_at=item.get("published_date"),
                    headline=title,
                    source=_domain_from_url(item.get("url")),
                    url=item.get("url"),
                    source_type="tavily_search",
                    content=(content or "")[:1600] or None,
                    relevance_score=_to_float(item.get("score")),
                )
            )
        return results

    def extract(self, urls: Sequence[str]) -> Dict[str, str]:
        if not self.enabled or not urls:
            return {}
        try:
            payload = fetch_json(
                "https://api.tavily.com/extract",
                method="POST",
                payload={
                    "api_key": self.api_key,
                    "urls": list(urls),
                },
            )
        except ProviderError:
            return {}

        content_by_url: Dict[str, str] = {}
        for item in payload.get("results", []) or payload.get("data", []):
            url = item.get("url")
            raw_content = item.get("raw_content") or item.get("content") or ""
            if url and raw_content:
                content_by_url[url] = raw_content[:3000]
        return content_by_url


class ExternalFallbackService:
    """Provider-aware fallback service."""

    def __init__(self):
        self.repo = SupabaseRepository()
        self.fmp = FMPClient(settings.FMP_API_KEY)
        self.tavily = TavilyClient(settings.TAVILY_API_KEY)

    def collect_context(
        self,
        symbol: Optional[str],
        user_query: str,
        intent: str,
        max_news: Optional[int] = None,
    ) -> Dict[str, Any]:
        max_news = max_news or settings.EXTERNAL_NEWS_LIMIT
        attempts: List[str] = []
        company_name = self.repo.get_company_name(symbol) if symbol else None

        latest_price = self.repo.get_latest_price(symbol) if symbol else None
        history: List[PriceResult] = []
        price_path = "db" if latest_price and not _is_stale(
            latest_price.as_of,
            settings.EXTERNAL_PRICE_MAX_AGE_DAYS,
        ) else None

        if symbol and price_path is None:
            try:
                attempts.append("fmp_quote")
                latest_price = self.fmp.get_quote(symbol)
                if latest_price and latest_price.close_price is not None:
                    price_path = "fmp_quote"
            except Exception as exc:  # pragma: no cover - network/runtime
                attempts.append(f"fmp_quote_error:{exc}")

        if symbol:
            try:
                attempts.append("fmp_history")
                history = self.fmp.get_history(symbol, limit=90)
            except Exception as exc:  # pragma: no cover - network/runtime
                attempts.append(f"fmp_history_error:{exc}")

        if symbol and latest_price is None and history:
            latest_price = history[0]
            price_path = "fmp_history"

        news = self.repo.get_recent_news(symbol, limit=max_news) if symbol else []
        news_path = "db" if news else None
        if symbol and not news:
            try:
                attempts.append("fmp_news")
                news = self.fmp.get_stock_news(symbol, limit=max_news)
                if news:
                    news_path = "fmp_news"
            except Exception as exc:  # pragma: no cover - network/runtime
                attempts.append(f"fmp_news_error:{exc}")

        tavily_docs: List[NewsResult] = []
        if (symbol and (not news or _needs_web_context(intent, user_query))) or not symbol:
            query = _build_tavily_query(symbol, company_name, user_query)
            try:
                attempts.append("tavily_search")
                tavily_docs = self.tavily.search_news(
                    query=query,
                    limit=settings.EXTERNAL_TAVILY_RESULT_LIMIT,
                    company_hint=company_name,
                )
                if tavily_docs and not news_path:
                    news_path = "tavily_search"
            except Exception as exc:  # pragma: no cover - network/runtime
                attempts.append(f"tavily_search_error:{exc}")

            extractable_urls = [doc.url for doc in tavily_docs if doc.url][:3]
            if extractable_urls:
                attempts.append("tavily_extract")
                extracted = self.tavily.extract(extractable_urls)
                for doc in tavily_docs:
                    if doc.url in extracted:
                        doc.content = extracted[doc.url]

        derived_metrics = _compute_derived_metrics(history)
        external_rows = _build_external_rows(
            symbol=symbol,
            company_name=company_name,
            latest_price=latest_price,
            derived_metrics=derived_metrics,
            news=news or tavily_docs,
        )

        return {
            "symbol": symbol,
            "company_name": company_name,
            "latest_price": asdict(latest_price) if latest_price else None,
            "history_points": [asdict(item) for item in history[:30]],
            "derived_metrics": derived_metrics,
            "news": [asdict(item) for item in news],
            "tavily_documents": [asdict(item) for item in tavily_docs],
            "attempts": attempts,
            "price_path": price_path or "unavailable",
            "news_path": news_path or "unavailable",
            "rows": external_rows,
        }


class FinancialAdvisoryService:
    """Shared app service for SQL-first and external-fallback answers."""

    def __init__(self):
        self.extractor = IntentEntityExtractor()
        self.executor = DatabaseQueryExecutor()
        self.external = ExternalFallbackService()
        self.portfolio_advisory = PortfolioAdvisoryService()
        self.llm = None
        self.groq_client = GroqChatClient(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_REASONING_MODEL,
        )

        if ChatGoogleGenerativeAI is not None and settings.GEMINI_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0.1,
                    max_tokens=1200,
                )
            except Exception:  # pragma: no cover - runtime dependency issue
                self.llm = None

    async def process_query(
        self,
        user_query: str,
        style: ResponseStyle = ResponseStyle.CONVERSATIONAL,
    ) -> AdvisoryServiceResult:
        extraction = await self.extractor.extract_intent_and_entities(user_query)
        intent = extraction.get("intent", "general_inquiry")
        entities = extraction.get("entities", {})

        if intent == "portfolio_review":
            portfolio_result = self.portfolio_advisory.answer_query(
                user_query=user_query,
                portfolio_id=entities.get("portfolio_id"),
            )
            if portfolio_result is not None:
                entities["portfolio_id"] = portfolio_result["portfolio_id"]
                return AdvisoryServiceResult(
                    intent=intent,
                    entities=entities,
                    sql_result=portfolio_result["sql_result"],
                    execution_result=portfolio_result["execution_result"],
                    formatted_response=portfolio_result["formatted_response"],
                    route="portfolio_analytics",
                    source_path="portfolio_analytics",
                    metadata={
                        "extraction": extraction,
                        "question_type": portfolio_result["question_type"],
                        "snapshot": {
                            "portfolio_id": portfolio_result["snapshot"].portfolio_id,
                            "portfolio_name": portfolio_result["snapshot"].portfolio_name,
                            "total_value": portfolio_result["snapshot"].total_value,
                        },
                        "optimized": portfolio_result["optimized"],
                    },
                )

        sql_result = await generate_sql_query(intent, entities, user_query)
        execution_result = await self.executor.execute_query(sql_result)

        if self._can_answer_from_sql(execution_result):
            formatted = await self._format_sql_response(
                user_query,
                intent,
                entities,
                sql_result,
                execution_result,
                style,
            )
            return AdvisoryServiceResult(
                intent=intent,
                entities=entities,
                sql_result=sql_result,
                execution_result=execution_result,
                formatted_response=formatted,
                route="sql",
                source_path="database",
                metadata={"extraction": extraction},
            )

        if self._should_try_external_fallback(intent, entities, user_query):
            fallback = await self._build_external_response(
                user_query=user_query,
                intent=intent,
                entities=entities,
                sql_result=sql_result,
                db_execution_result=execution_result,
                style=style,
            )
            if fallback is not None:
                return fallback

        formatted = self._build_no_data_response(user_query, intent, entities, execution_result)
        return AdvisoryServiceResult(
            intent=intent,
            entities=entities,
            sql_result=sql_result,
            execution_result=execution_result,
            formatted_response=formatted,
            route="sql_no_data",
            source_path="database",
            metadata={"extraction": extraction},
        )

    def _can_answer_from_sql(self, execution_result: QueryExecutionResult) -> bool:
        if execution_result.status != ExecutionStatus.SUCCESS:
            return False
        if execution_result.row_count == 0 or not execution_result.data:
            return False
        sample = execution_result.data[0]
        meaningful = [
            value for key, value in sample.items()
            if key not in {"symbol", "company_name"} and value is not None
        ]
        return bool(meaningful)

    def _should_try_external_fallback(
        self,
        intent: str,
        entities: Dict[str, Any],
        user_query: str,
    ) -> bool:
        symbols = entities.get("stocks") or []
        if intent in {"portfolio_review", "ranking_query", "screening_query", "comparative_analysis"}:
            return not symbols and bool(settings.TAVILY_API_KEY)
        if not symbols:
            return bool(settings.TAVILY_API_KEY)
        if intent in {"stock_analysis", "sentiment_analysis", "technical_analysis", "risk_assessment", "general_inquiry"}:
            return True
        return "good buy" in user_query.lower()

    async def _format_sql_response(
        self,
        user_query: str,
        intent: str,
        entities: Dict[str, Any],
        sql_result: SQLQueryResult,
        execution_result: QueryExecutionResult,
        style: ResponseStyle,
    ) -> FormattedResponse:
        try:
            return await format_financial_response(
                user_query=user_query,
                intent=intent,
                entities=entities,
                sql_result=sql_result,
                execution_result=execution_result,
                style=style,
            )
        except Exception as exc:
            logger.warning("Falling back to deterministic SQL response formatting: %s", exc)
            return self._build_sql_fallback_response(intent, execution_result)

    async def _build_external_response(
        self,
        user_query: str,
        intent: str,
        entities: Dict[str, Any],
        sql_result: SQLQueryResult,
        db_execution_result: QueryExecutionResult,
        style: ResponseStyle,
    ) -> Optional[AdvisoryServiceResult]:
        symbol = (entities.get("stocks") or [None])[0]
        context = self.external.collect_context(symbol, user_query, intent)
        rows = context.get("rows") or []
        if not rows and not context.get("news") and not context.get("tavily_documents"):
            return None

        execution_result = _build_external_execution_result(
            rows=rows,
            warnings=[
                "Primary SQL route returned no usable data.",
                f"External fallback used: price={context['price_path']}, news={context['news_path']}",
            ] + context.get("attempts", []),
        )

        formatted = await self._synthesize_external_response(
            user_query=user_query,
            intent=intent,
            entities=entities,
            context=context,
            style=style,
        )
        metadata = {
            "db_execution_status": db_execution_result.status.value,
            "db_row_count": db_execution_result.row_count,
            "fallback_context": context,
        }
        return AdvisoryServiceResult(
            intent=intent,
            entities=entities,
            sql_result=sql_result,
            execution_result=execution_result,
            formatted_response=formatted,
            route="external_fallback",
            source_path=f"{context['price_path']}|{context['news_path']}",
            metadata=metadata,
        )

    async def _synthesize_external_response(
        self,
        user_query: str,
        intent: str,
        entities: Dict[str, Any],
        context: Dict[str, Any],
        style: ResponseStyle,
    ) -> FormattedResponse:
        if self.llm is None and not self.groq_client.enabled:
            return _build_external_response_without_llm(user_query, intent, context)

        prompt = (
            "You are a financial assistant using verified structured fallback data and retrieved web news.\n"
            "Rules:\n"
            "1. Use only the supplied context.\n"
            "2. If a verified price is missing, say so explicitly.\n"
            "3. Do not present retrieved web context as database-backed fact.\n"
            "4. Keep recommendations cautious and conditional.\n"
            "5. Return JSON only with keys: natural_language, confidence, key_insights, recommendations, "
            "risk_warnings, explanation, follow_up_suggestions, disclaimer.\n"
        )
        user_payload = json.dumps(
            {
                "user_query": user_query,
                "intent": intent,
                "style": style.value,
                "entities": entities,
                "context": context,
            },
            default=str,
        )

        try:
            payload = _parse_json_payload(
                await self._invoke_reasoning_llm(
                    system_prompt=prompt,
                    user_payload=user_payload,
                )
            )
            return FormattedResponse(
                natural_language=payload.get("natural_language", "I used external fallback data to answer your request."),
                confidence=_parse_confidence(payload.get("confidence")),
                key_insights=_ensure_list(payload.get("key_insights")),
                recommendations=_ensure_list(payload.get("recommendations")),
                risk_warnings=_ensure_list(payload.get("risk_warnings")),
                data_summary={
                    "route": "external_fallback",
                    "price_path": context.get("price_path"),
                    "news_path": context.get("news_path"),
                    "symbol": context.get("symbol"),
                },
                explanation=payload.get("explanation", "Response synthesized from external fallback context."),
                follow_up_suggestions=_ensure_list(payload.get("follow_up_suggestions")),
                technical_details={
                    "fallback_context": context,
                    "provider_paths": {
                        "price": context.get("price_path"),
                        "news": context.get("news_path"),
                    },
                },
                disclaimer=payload.get(
                    "disclaimer",
                    "External provider and web-retrieved context were used because the database could not fully answer this query.",
                ),
            )
        except Exception as exc:
            logger.warning("External LLM synthesis failed, using deterministic fallback: %s", exc)
            return _build_external_response_without_llm(user_query, intent, context)

    def _build_sql_fallback_response(
        self,
        intent: str,
        execution_result: QueryExecutionResult,
    ) -> FormattedResponse:
        first_row = execution_result.data[0] if execution_result.data else {}
        insights = [
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in list(first_row.items())[:4]
            if value is not None
        ]
        return FormattedResponse(
            natural_language="I found structured database results, but the advanced formatter was unavailable. The key values are summarized below.",
            confidence=ConfidenceLevel.MEDIUM,
            key_insights=insights or ["Structured data was returned from the database."],
            recommendations=[],
            risk_warnings=execution_result.warnings or [],
            data_summary={"intent": intent, "row_count": execution_result.row_count},
            explanation="Deterministic fallback response built from SQL output.",
            follow_up_suggestions=[
                "Show the latest available data for this symbol",
                "Compare this stock with another ticker",
            ],
            technical_details={"sample_row": first_row},
            disclaimer="This summary is informational and should not be treated as personalized financial advice.",
        )

    def _build_no_data_response(
        self,
        user_query: str,
        intent: str,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
    ) -> FormattedResponse:
        symbol = ", ".join(entities.get("stocks") or []) or "the available database coverage"
        warnings = execution_result.warnings or []
        if execution_result.error_message:
            warnings = warnings + [execution_result.error_message]
        return FormattedResponse(
            natural_language=(
                f"I could not find enough structured data to answer '{user_query}' for {symbol}. "
                "Try a different ticker, a broader date range, or a simpler market-data question."
            ),
            confidence=ConfidenceLevel.INSUFFICIENT,
            key_insights=["The verified database returned no usable rows for this request."],
            recommendations=[
                "Try asking for the latest available price or recent sentiment",
                "Use a ticker that exists in the supported stock universe",
            ],
            risk_warnings=warnings,
            data_summary={"intent": intent, "row_count": execution_result.row_count},
            explanation="No structured database or provider fallback path produced enough evidence.",
            follow_up_suggestions=[
                "What symbols are available in the database?",
                "Show the latest price for a supported ticker",
            ],
            technical_details={"entities": entities},
            disclaimer="This system can only answer from available market data and retrieved evidence.",
        )

    async def _invoke_reasoning_llm(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> str:
        if self.llm is not None:
            try:
                response = await self.llm.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_payload),
                    ]
                )
                return _extract_text_content(response.content)
            except Exception:
                pass

        if not self.groq_client.enabled:
            raise RuntimeError("No reasoning LLM is available.")

        return await asyncio.to_thread(
            self.groq_client.generate,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            model=settings.GROQ_REASONING_MODEL,
            temperature=0.1,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )


def _extract_text_content(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(item if isinstance(item, str) else str(item) for item in content)
    return str(content)


def _parse_json_payload(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _build_external_response_without_llm(
    user_query: str,
    intent: str,
    context: Dict[str, Any],
) -> FormattedResponse:
    latest_price = context.get("latest_price") or {}
    symbol = context.get("symbol") or "the requested symbol"
    derived = context.get("derived_metrics") or {}
    news_docs = context.get("news") or context.get("tavily_documents") or []

    key_insights: List[str] = []
    if latest_price.get("close_price") is not None:
        key_insights.append(
            f"Latest available price for {symbol} is {latest_price['close_price']} from {context.get('price_path')}."
        )
    if derived.get("rsi_14") is not None:
        key_insights.append(f"Derived RSI(14) is {derived['rsi_14']:.2f}.")
    if derived.get("volatility_30d") is not None:
        key_insights.append(f"Derived 30-day annualized volatility is {derived['volatility_30d']:.2f}%.")
    if news_docs:
        key_insights.append(f"Collected {len(news_docs)} recent news items from {context.get('news_path')}.")

    natural_language = (
        f"I used external fallback sources because the database could not fully answer '{user_query}'. "
        f"The answer is based on {context.get('price_path')} for market data and {context.get('news_path')} for recent news."
    )
    if latest_price.get("close_price") is None:
        natural_language += " I could not verify a structured latest price from the fallback providers."

    return FormattedResponse(
        natural_language=natural_language,
        confidence=ConfidenceLevel.LOW if context.get("news_path") == "tavily_search" else ConfidenceLevel.MEDIUM,
        key_insights=key_insights or ["External fallback returned partial context only."],
        recommendations=[
            "Treat web-backed conclusions as directional, not definitive",
            "Re-run the query after the database is refreshed for a stronger answer",
        ],
        risk_warnings=[
            "External fallback was used because structured database coverage was incomplete.",
            "Retrieved web content may be noisier than database-backed market data.",
        ],
        data_summary={
            "intent": intent,
            "price_path": context.get("price_path"),
            "news_path": context.get("news_path"),
        },
        explanation="Deterministic fallback summary built from external provider data and retrieved news.",
        follow_up_suggestions=[
            f"What is the latest available price for {symbol}?",
            f"Show recent sentiment for {symbol}",
        ],
        technical_details={"fallback_context": context},
        disclaimer="This response may include external provider and retrieved web evidence, not just database-backed analytics.",
    )


def _build_external_execution_result(
    rows: List[Dict[str, Any]],
    warnings: List[str],
) -> QueryExecutionResult:
    column_info = []
    if rows:
        column_info = [{"name": key, "type": type(value).__name__} for key, value in rows[0].items()]
    return QueryExecutionResult(
        status=ExecutionStatus.SUCCESS,
        data=rows,
        row_count=len(rows),
        warnings=warnings,
        column_info=column_info,
    )


def _build_external_rows(
    symbol: Optional[str],
    company_name: Optional[str],
    latest_price: Optional[PriceResult],
    derived_metrics: Dict[str, Optional[float]],
    news: List[NewsResult],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if latest_price:
        rows.append(
            {
                "symbol": symbol,
                "company_name": company_name,
                "date": latest_price.as_of,
                "close_price": latest_price.close_price,
                "open_price": latest_price.open_price,
                "high_price": latest_price.high_price,
                "low_price": latest_price.low_price,
                "volume": latest_price.volume,
                **derived_metrics,
            }
        )
    if not rows and news:
        rows.extend(
            {
                "symbol": item.symbol,
                "published_at": item.published_at,
                "headline": item.headline,
                "source": item.source,
                "url": item.url,
            }
            for item in news[:5]
        )
    return rows


def _compute_derived_metrics(history: List[PriceResult]) -> Dict[str, Optional[float]]:
    closes = [item.close_price for item in reversed(history) if item.close_price is not None]
    if len(closes) < 15:
        return {"rsi_14": None, "macd": None, "macd_signal": None, "volatility_30d": None, "return_30d": None}

    series = pd.Series(closes, dtype="float64")
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + relative_strength))

    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    returns = series.pct_change().dropna()
    volatility_30d = None
    if len(returns) >= 20:
        volatility_30d = float(returns.tail(30).std() * sqrt(252) * 100)

    return_30d = None
    if len(series) >= 31 and series.iloc[-31] != 0:
        return_30d = float(((series.iloc[-1] / series.iloc[-31]) - 1) * 100)

    return {
        "rsi_14": _to_float(rsi.iloc[-1]),
        "macd": _to_float(macd.iloc[-1]),
        "macd_signal": _to_float(macd_signal.iloc[-1]),
        "volatility_30d": volatility_30d,
        "return_30d": return_30d,
    }


def _to_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_confidence(value: Any) -> ConfidenceLevel:
    lowered = str(value or "").strip().lower()
    if lowered == "high":
        return ConfidenceLevel.HIGH
    if lowered == "medium":
        return ConfidenceLevel.MEDIUM
    if lowered == "insufficient":
        return ConfidenceLevel.INSUFFICIENT
    return ConfidenceLevel.LOW


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            normalized = _normalize_list_item(item)
            if normalized:
                items.append(normalized)
        return items
    if value is None:
        return []
    text = _normalize_list_item(value)
    return [text] if text else []


def _normalize_list_item(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key in ("insight", "title", "explanation", "text"):
            item = value.get(key)
            if item is not None and str(item).strip():
                parts.append(str(item).strip())
        if parts:
            return ": ".join(parts[:2])
        return ", ".join(
            f"{key}: {item}" for key, item in value.items() if item is not None and str(item).strip()
        )

    text = str(value).strip()
    if not text:
        return ""

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        return _normalize_list_item(parsed)

    return text


def _domain_from_url(url: Optional[str]) -> str:
    if not url:
        return "unknown"
    without_scheme = url.split("://", 1)[-1]
    return without_scheme.split("/", 1)[0]


def _looks_relevant(title: str, content: str, query: str, company_hint: Optional[str]) -> bool:
    lowered = " ".join((title, content, query, company_hint or "")).lower()
    keywords = [part for part in query.lower().split() if len(part) > 2]
    return any(keyword in lowered for keyword in keywords[:4])


def _extract_symbol_from_query(query: str) -> str:
    for token in query.split():
        cleaned = token.strip("() ,.")
        if cleaned.isupper() and 1 <= len(cleaned) <= 5:
            return cleaned
    return ""


def _build_tavily_query(symbol: Optional[str], company_name: Optional[str], user_query: str) -> str:
    if symbol and company_name:
        return f"{company_name} ({symbol}) {user_query} market news analyst outlook"
    if symbol:
        return f"{symbol} {user_query} market news analyst outlook"
    return f"{user_query} financial market context analyst explanation"


def _needs_web_context(intent: str, user_query: str) -> bool:
    lowered = user_query.lower()
    return intent in {"sentiment_analysis", "general_inquiry"} or "good buy" in lowered or "outlook" in lowered


def _is_stale(as_of: Optional[str], max_age_days: int) -> bool:
    if not as_of:
        return True
    try:
        if "T" in as_of:
            dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return dt < datetime.now(timezone.utc) - timedelta(days=max_age_days)
