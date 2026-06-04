"""
Schema-aware SQL Query Generator
Real-Time Financial Advisory System - Phase 5 Component 2

This module generates SQL from verified schema knowledge first, with an LLM
prompt available as a constrained system prompt for future use.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config.settings import config as settings
from src.utils.groq_client import GroqAPIError, GroqChatClient

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency at runtime
    HumanMessage = None
    SystemMessage = None
    ChatGoogleGenerativeAI = None


class QueryComplexity(Enum):
    """Query complexity levels for different reasoning strategies."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


@dataclass
class SQLQueryResult:
    """Result structure for SQL generation."""

    sql: str
    parameters: Dict[str, Any]
    reasoning: str
    complexity: QueryComplexity
    estimated_execution_time: float
    tables_involved: List[str]
    validation_errors: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    expected_output_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sql": self.sql,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "complexity": self.complexity.value,
            "estimated_execution_time": self.estimated_execution_time,
            "tables_involved": self.tables_involved,
            "validation_errors": self.validation_errors,
            "optimization_suggestions": self.optimization_suggestions,
            "expected_output_columns": self.expected_output_columns,
            "metadata": self.metadata,
        }


class DatabaseSchemaKnowledge:
    """Exact verified public-schema knowledge for SQL generation."""

    VERIFIED_SCHEMA: Dict[str, Dict[str, str]] = {
        "financial_news": {
            "id": "uuid",
            "headline": "text",
            "content": "text",
            "publisher": "character varying",
            "published_at": "timestamp with time zone",
            "url": "text",
            "source": "character varying",
            "category": "character varying",
            "created_at": "timestamp with time zone",
        },
        "ml_features": {
            "id": "integer",
            "stock_id": "uuid",
            "date": "date",
            "adjusted_close": "numeric",
            "volume": "bigint",
            "return_1d": "numeric",
            "log_return": "numeric",
            "volatility_5d": "numeric",
            "volatility_20d": "numeric",
            "momentum_5d": "numeric",
            "price_vs_sma50": "numeric",
            "rsi_14": "numeric",
            "macd": "numeric",
            "macd_signal": "numeric",
            "macd_histogram": "numeric",
            "bb_upper": "numeric",
            "bb_lower": "numeric",
            "bb_width": "numeric",
            "volume_sma_20": "numeric",
            "volatility_30d": "numeric",
            "beta": "numeric",
            "sharpe_ratio": "numeric",
            "value_at_risk_95": "numeric",
            "max_drawdown": "numeric",
            "sentiment_score": "numeric",
            "relevance_score": "numeric",
            "news_volume": "integer",
            "target_return_1d": "numeric",
            "target_return_5d": "numeric",
            "target_direction": "integer",
            "created_at": "timestamp without time zone",
        },
        "news_stock_mentions": {
            "id": "uuid",
            "news_id": "uuid",
            "stock_id": "uuid",
            "symbol": "character varying",
            "mention_context": "text",
            "relevance_score": "numeric",
            "created_at": "timestamp with time zone",
        },
        "portfolio_holdings": {
            "id": "uuid",
            "portfolio_id": "uuid",
            "stock_id": "uuid",
            "symbol": "character varying",
            "shares": "numeric",
            "avg_cost_basis": "numeric",
            "current_price": "numeric",
            "market_value": "numeric",
            "unrealized_gain_loss": "numeric",
            "weight_percentage": "numeric",
            "purchase_date": "date",
            "last_updated": "timestamp with time zone",
        },
        "portfolios": {
            "id": "uuid",
            "user_id": "uuid",
            "name": "character varying",
            "description": "text",
            "total_value": "numeric",
            "cash_balance": "numeric",
            "is_default": "boolean",
            "created_at": "timestamp with time zone",
            "updated_at": "timestamp with time zone",
        },
        "risk_metrics": {
            "id": "uuid",
            "stock_id": "uuid",
            "symbol": "character varying",
            "calculation_date": "date",
            "beta": "numeric",
            "volatility_30d": "numeric",
            "volatility_90d": "numeric",
            "max_drawdown": "numeric",
            "value_at_risk_95": "numeric",
            "sharpe_ratio": "numeric",
            "sortino_ratio": "numeric",
            "return_1d": "numeric",
            "return_7d": "numeric",
            "return_30d": "numeric",
            "return_90d": "numeric",
            "return_1y": "numeric",
            "created_at": "timestamp with time zone",
        },
        "sentiment_scores": {
            "id": "uuid",
            "news_id": "uuid",
            "stock_id": "uuid",
            "symbol": "character varying",
            "sentiment_label": "character varying",
            "sentiment_score": "numeric",
            "confidence_score": "numeric",
            "model_version": "character varying",
            "processed_at": "timestamp with time zone",
        },
        "stock_prices": {
            "id": "uuid",
            "stock_id": "uuid",
            "symbol": "character varying",
            "date": "date",
            "open_price": "numeric",
            "high_price": "numeric",
            "low_price": "numeric",
            "close_price": "numeric",
            "volume": "bigint",
            "adjusted_close": "numeric",
            "created_at": "timestamp with time zone",
        },
        "stocks": {
            "id": "uuid",
            "symbol": "character varying",
            "company_name": "character varying",
            "sector": "character varying",
            "industry": "character varying",
            "market_cap": "bigint",
            "currency": "character varying",
            "exchange": "character varying",
            "country": "character varying",
            "is_active": "boolean",
            "created_at": "timestamp with time zone",
            "updated_at": "timestamp with time zone",
        },
        "technical_indicators": {
            "id": "uuid",
            "stock_id": "uuid",
            "symbol": "character varying",
            "date": "date",
            "sma_20": "numeric",
            "sma_50": "numeric",
            "sma_200": "numeric",
            "ema_12": "numeric",
            "ema_26": "numeric",
            "volatility_20": "numeric",
            "bollinger_upper": "numeric",
            "bollinger_lower": "numeric",
            "rsi_14": "numeric",
            "macd": "numeric",
            "macd_signal": "numeric",
            "macd_histogram": "numeric",
            "volume_sma_20": "bigint",
            "created_at": "timestamp with time zone",
        },
        "user_queries": {
            "id": "uuid",
            "user_id": "uuid",
            "session_id": "character varying",
            "query_text": "text",
            "intent": "character varying",
            "extracted_symbols": "ARRAY",
            "sql_generated": "text",
            "response_text": "text",
            "response_time_ms": "integer",
            "created_at": "timestamp with time zone",
        },
        "users": {
            "id": "uuid",
            "email": "character varying",
            "username": "character varying",
            "full_name": "character varying",
            "risk_tolerance": "character varying",
            "investment_horizon": "character varying",
            "age_group": "character varying",
            "annual_income_range": "character varying",
            "investment_experience": "character varying",
            "preferred_sectors": "ARRAY",
            "created_at": "timestamp with time zone",
            "updated_at": "timestamp with time zone",
            "last_login": "timestamp with time zone",
            "is_active": "boolean",
        },
        "v_daily_sentiment": {
            "symbol": "character varying",
            "company_name": "character varying",
            "sentiment_date": "date",
            "avg_sentiment": "numeric",
            "news_count": "bigint",
            "avg_confidence": "numeric",
        },
        "v_latest_stock_prices": {
            "symbol": "character varying",
            "company_name": "character varying",
            "sector": "character varying",
            "date": "date",
            "close_price": "numeric",
            "volume": "bigint",
            "open_price": "numeric",
            "high_price": "numeric",
            "low_price": "numeric",
        },
        "v_portfolio_summary": {
            "portfolio_id": "uuid",
            "portfolio_name": "character varying",
            "user_email": "character varying",
            "holdings_count": "bigint",
            "total_market_value": "numeric",
            "total_unrealized_pl": "numeric",
            "avg_weight": "numeric",
        },
    }

    KNOWN_SYMBOLS: Tuple[str, ...] = (
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMZN", "AVGO", "AXP",
        "BAC", "BRK-B", "BX", "COST", "CRM", "CSCO", "CVX", "DIS", "GE",
        "GOOG", "GOOGL", "HD", "IBM", "INFY", "ISRG", "JNJ", "JPM", "KO",
        "LIN", "LLY", "MA", "MCD", "META", "MRK", "MS", "MSFT", "NFLX",
        "NOW", "NVDA", "ORCL", "PEP", "PG", "PLTR", "PM", "TMO", "TMUS",
        "TSLA", "UNH", "V", "WFC", "WMT", "XOM",
    )

    @classmethod
    def get_allowed_relations(cls) -> List[str]:
        return sorted(cls.VERIFIED_SCHEMA.keys())

    @classmethod
    def get_columns(cls, relation_name: str) -> List[str]:
        return list(cls.VERIFIED_SCHEMA.get(relation_name, {}).keys())

    @classmethod
    def is_known_symbol(cls, symbol: str) -> bool:
        return symbol.upper() in cls.KNOWN_SYMBOLS

    @classmethod
    def build_system_prompt(cls) -> str:
        schema_lines: List[str] = []
        for relation_name, columns in cls.VERIFIED_SCHEMA.items():
            column_desc = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
            relation_kind = "view" if relation_name.startswith("v_") else "table"
            schema_lines.append(f"- {relation_name} ({relation_kind}): {column_desc}")

        rules = [
            "Use only the relations and columns listed below.",
            "The stock identifier column is symbol, not ticker.",
            "There is no table named sentiment_analysis.",
            "Join stock-linked tables on stock_id = stocks.id.",
            "Join sentiment_scores.news_id = financial_news.id.",
            "technical_indicators does not have a raw volume column; use volume_sma_20 there, or join stock_prices for raw volume.",
            "Always use table aliases.",
            "Never emit SQL comments.",
            "Return one valid PostgreSQL SELECT query only.",
            "Use latest-available data with MAX(date), MAX(calculation_date), or MAX(sentiment_date) instead of assuming CURRENT_DATE data exists.",
            "When filtering by a symbol, use the exact symbol string from the request.",
            "If a request asks for rankings, include ORDER BY and LIMIT.",
            "If a request asks for time trends, include ORDER BY ascending on the date column.",
            "Output JSON only with keys: sql, tables, expected_columns, notes.",
        ]

        patterns = [
            "Price queries: use v_latest_stock_prices or stock_prices with latest date per stock.",
            "Sentiment queries: use v_daily_sentiment for trends, or sentiment_scores joined to financial_news for article-level detail.",
            "Technical queries: use technical_indicators with latest date per stock. For raw trading volume trends, join stock_prices; for indicator-based volume context, use technical_indicators.volume_sma_20.",
            "Risk queries: use risk_metrics with latest calculation_date per stock.",
            "Portfolio queries: use portfolios, portfolio_holdings, users, or v_portfolio_summary.",
            "Comparison queries: filter with IN (...) and keep one row per symbol.",
            "Screening queries: join stocks to the latest risk or technical snapshot and filter by sector and metric thresholds.",
        ]

        return (
            "You are a PostgreSQL SQL generator for a real-time financial advisory system.\n\n"
            "STRICT RULES:\n"
            + "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))
            + "\n\nVERIFIED SCHEMA:\n"
            + "\n".join(schema_lines)
            + "\n\nCOMMON QUERY PATTERNS:\n"
            + "\n".join(f"- {pattern}" for pattern in patterns)
        )

    @classmethod
    def get_schema_context(cls) -> str:
        return cls.build_system_prompt()


class AdvancedSQLGenerator:
    """SQL generator that prefers deterministic templates over free-form prompts."""

    def __init__(self):
        self.settings = settings
        self.schema = DatabaseSchemaKnowledge()
        self.system_prompt = self.schema.build_system_prompt()
        self.llm = None
        self.groq_client = GroqChatClient(
            api_key=self.settings.GROQ_API_KEY,
            model=self.settings.GROQ_SQL_MODEL,
        )

        if ChatGoogleGenerativeAI is not None:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    google_api_key=self.settings.GEMINI_API_KEY,
                    temperature=0.0,
                    max_tokens=1200,
                )
            except Exception:
                self.llm = None

    async def generate_sql(
        self,
        intent: str,
        entities: Dict[str, Any],
        user_query: str,
    ) -> SQLQueryResult:
        complexity = self._analyze_complexity(intent, entities, user_query)
        use_template = self._matches_template(intent, entities, user_query)

        if use_template:
            sql_result = self._build_template(intent, entities, user_query, complexity)
            sql_result.sql = self._apply_known_repairs(sql_result.sql)
            validation_errors = self._validate_sql(sql_result.sql)
            sql_result.validation_errors.extend(validation_errors)

            if validation_errors and self.llm is not None:
                llm_candidate = await self._generate_via_llm_with_repair(
                    intent,
                    entities,
                    user_query,
                    complexity,
                    validation_errors,
                )
                if llm_candidate is not None:
                    return llm_candidate
                sql_result.optimization_suggestions.append(
                    "Template SQL failed validation and LLM repair could not produce a safe query."
                )

            if sql_result.validation_errors:
                sql_result.optimization_suggestions.append(
                    "Review symbol coverage or broaden the query intent before execution."
                )
            sql_result.metadata["generation_mode"] = "template"
            return sql_result

        llm_candidate = await self._generate_via_llm_with_repair(
            intent,
            entities,
            user_query,
            complexity,
            [],
        )
        if llm_candidate is not None:
            llm_candidate.metadata["generation_mode"] = "llm"
            return llm_candidate

        fallback_result = self._build_fallback_lookup_query(entities, complexity)
        fallback_result.sql = self._apply_known_repairs(fallback_result.sql)
        fallback_result.validation_errors.extend(self._validate_sql(fallback_result.sql))
        fallback_result.optimization_suggestions.append(
            "No template matched and LLM SQL generation did not yield a safe query; returned a lookup query instead."
        )
        fallback_result.metadata["generation_mode"] = "fallback_lookup"
        return fallback_result

    def _analyze_complexity(
        self,
        intent: str,
        entities: Dict[str, Any],
        user_query: str,
    ) -> QueryComplexity:
        symbols = self._get_symbols(entities)
        metrics = entities.get("metrics", [])
        lowered = user_query.lower()

        if "compare" in lowered or len(symbols) > 1:
            return QueryComplexity.MODERATE
        if intent in {"screening_query", "ranking_query"}:
            return QueryComplexity.MODERATE
        if intent == "portfolio_review":
            return QueryComplexity.COMPLEX
        if any(metric in {"sharpe_ratio", "value_at_risk_95"} for metric in metrics):
            return QueryComplexity.MODERATE
        return QueryComplexity.SIMPLE

    def _build_template(
        self,
        intent: str,
        entities: Dict[str, Any],
        user_query: str,
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        query_text = user_query.lower()
        symbols = self._get_symbols(entities)
        unresolved = self._get_unresolved_mentions(entities)
        params: Dict[str, Any] = {}

        if not symbols and unresolved:
            lookup = unresolved[0]
            sql = (
                "SELECT s.symbol, s.company_name, s.sector "
                "FROM stocks s "
                "WHERE s.symbol ILIKE :lookup_symbol OR s.company_name ILIKE :lookup_company "
                "ORDER BY s.symbol "
                "LIMIT 10"
            )
            params = {
                "lookup_symbol": f"%{lookup.upper()}%",
                "lookup_company": f"%{lookup.title()}%",
            }
            return self._result(
                sql,
                params,
                "Fallback symbol lookup for an unknown or unresolved ticker mention.",
                complexity,
                ["stocks"],
                ["symbol", "company_name", "sector"],
            )

        if intent == "portfolio_review":
            return self._build_portfolio_query(entities, complexity)

        if intent == "sentiment_analysis":
            if "trend" in query_text or entities.get("trend"):
                return self._build_sentiment_trend_query(symbols, complexity)
            return self._build_recent_sentiment_query(symbols, entities, complexity)

        if intent == "technical_analysis":
            return self._build_technical_query(symbols, complexity)

        if intent == "risk_assessment":
            if "highest sharpe" in query_text or "sharpe ratio" in query_text and ("highest" in query_text or "top" in query_text):
                return self._build_sharpe_ranking_query(entities, complexity)
            if "compare" in query_text or len(symbols) > 1:
                return self._build_risk_comparison_query(symbols, complexity)
            return self._build_risk_snapshot_query(symbols, complexity)

        if intent == "ranking_query":
            if "volume" in query_text:
                return self._build_volume_ranking_query(entities, complexity)
            return self._build_sharpe_ranking_query(entities, complexity)

        if intent in {"screening_query", "comparative_analysis"}:
            if "volatility" in query_text:
                if len(symbols) > 1:
                    return self._build_risk_comparison_query(symbols, complexity)
                return self._build_low_volatility_sector_screen(entities, complexity)

        if "good buy" in query_text:
            return self._build_buy_signal_query(symbols, complexity)
        if "latest price" in query_text or "current price" in query_text or "price" in query_text:
            return self._build_latest_price_query(symbols, complexity)
        if "macd" in query_text:
            return self._build_technical_query(symbols, complexity)
        if "rsi" in query_text:
            return self._build_technical_query(symbols, complexity)
        if "sentiment" in query_text:
            return self._build_recent_sentiment_query(symbols, entities, complexity)
        if "volatility" in query_text:
            return self._build_risk_comparison_query(symbols, complexity)

        return self._build_latest_price_query(symbols, complexity)

    def _matches_template(
        self,
        intent: str,
        entities: Dict[str, Any],
        user_query: str,
    ) -> bool:
        query_text = user_query.lower()
        symbols = self._get_symbols(entities)
        unresolved = self._get_unresolved_mentions(entities)
        advanced_terms = (
            "rising",
            "falling",
            "increasing",
            "decreasing",
            "trend over",
            "over the last",
            "over last",
            "crossover",
            "cross above",
            "cross below",
            "relative to",
            "versus average",
        )

        if not symbols and unresolved:
            return True
        if intent == "portfolio_review":
            return True
        if intent == "sentiment_analysis":
            return True
        if intent == "technical_analysis":
            if any(term in query_text for term in advanced_terms):
                return False
            return any(term in query_text for term in ("rsi", "macd", "sma", "ema", "overbought", "oversold"))
        if intent == "risk_assessment":
            if any(term in query_text for term in advanced_terms):
                return False
            return any(
                term in query_text
                for term in ("volatility", "beta", "sharpe", "sortino", "value at risk", "var", "risk")
            )
        if intent == "ranking_query":
            return any(term in query_text for term in ("top", "highest", "lowest", "rank", "volume", "sharpe"))
        if intent in {"screening_query", "comparative_analysis"}:
            return True
        if any(term in query_text for term in ("good buy", "latest price", "current price", "price", "macd", "rsi", "sentiment", "volatility")):
            return True
        if symbols and entities.get("metrics"):
            return True
        return False

    async def _generate_via_llm_with_repair(
        self,
        intent: str,
        entities: Dict[str, Any],
        user_query: str,
        complexity: QueryComplexity,
        prior_errors: List[str],
    ) -> Optional[SQLQueryResult]:
        if self.llm is None and not self.groq_client.enabled:
            return None

        try:
            candidate = await self._generate_with_llm(intent, entities, user_query, complexity)
        except Exception:
            return None

        candidate.sql = self._apply_known_repairs(candidate.sql)
        validation_errors = self._validate_sql(candidate.sql)
        semantic_errors = self._validate_semantics(user_query, entities, candidate.sql)
        candidate.validation_errors.extend(validation_errors + semantic_errors)
        if not validation_errors and not semantic_errors:
            return candidate

        repaired = await self._repair_with_llm(
            intent=intent,
            entities=entities,
            user_query=user_query,
            complexity=complexity,
            sql=candidate.sql,
            errors=prior_errors + validation_errors + semantic_errors,
        )
        if repaired is None:
            return None

        repaired.sql = self._apply_known_repairs(repaired.sql)
        repaired_errors = self._validate_sql(repaired.sql)
        repaired_semantic_errors = self._validate_semantics(user_query, entities, repaired.sql)
        repaired.validation_errors.extend(repaired_errors + repaired_semantic_errors)
        if repaired_errors or repaired_semantic_errors:
            return None
        repaired.metadata["repair_applied"] = True
        return repaired

    def _build_fallback_lookup_query(
        self,
        entities: Dict[str, Any],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        unresolved = self._get_unresolved_mentions(entities)
        lookup_value = unresolved[0] if unresolved else ""
        params = {
            "lookup_symbol": f"%{lookup_value.upper()}%",
            "lookup_company": f"%{lookup_value.title()}%",
        }
        sql = (
            "SELECT s.symbol, s.company_name, s.sector, s.industry "
            "FROM stocks s "
            "WHERE s.symbol ILIKE :lookup_symbol OR s.company_name ILIKE :lookup_company "
            "ORDER BY s.symbol "
            "LIMIT 10"
        )
        return self._result(
            sql,
            params,
            "Fallback lookup query used when no template matched and safe LLM SQL was unavailable.",
            complexity,
            ["stocks"],
            ["symbol", "company_name", "sector", "industry"],
        )

    def _build_latest_price_query(
        self,
        symbols: Sequence[str],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        params: Dict[str, Any] = {}
        sql = (
            "SELECT vlsp.symbol, vlsp.company_name, vlsp.date, vlsp.close_price, vlsp.volume "
            "FROM v_latest_stock_prices vlsp"
        )
        if symbols:
            sql += " WHERE vlsp.symbol = :symbol_0"
            params["symbol_0"] = symbols[0]
        sql += " ORDER BY vlsp.symbol"
        return self._result(
            sql,
            params,
            "Latest price lookup using the verified latest-price view.",
            complexity,
            ["v_latest_stock_prices"],
            ["symbol", "company_name", "date", "close_price", "volume"],
        )

    def _build_buy_signal_query(
        self,
        symbols: Sequence[str],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        params = {"symbol_0": symbols[0]} if symbols else {}
        sql = (
            "WITH latest_price AS ("
            " SELECT sp.symbol, sp.date, sp.close_price, sp.volume"
            " FROM stock_prices sp"
            " WHERE sp.symbol = :symbol_0"
            "   AND sp.date = ("
            "     SELECT MAX(sp2.date) FROM stock_prices sp2 WHERE sp2.symbol = sp.symbol"
            "   )"
            "), latest_ti AS ("
            " SELECT ti.symbol, ti.date, ti.rsi_14, ti.macd, ti.macd_signal, ti.macd_histogram"
            " FROM technical_indicators ti"
            " WHERE ti.symbol = :symbol_0"
            "   AND ti.date = ("
            "     SELECT MAX(ti2.date) FROM technical_indicators ti2 WHERE ti2.symbol = ti.symbol"
            "   )"
            "), latest_risk AS ("
            " SELECT rm.symbol, rm.calculation_date, rm.volatility_30d, rm.beta, rm.sharpe_ratio"
            " FROM risk_metrics rm"
            " WHERE rm.symbol = :symbol_0"
            "   AND rm.calculation_date = ("
            "     SELECT MAX(rm2.calculation_date) FROM risk_metrics rm2 WHERE rm2.symbol = rm.symbol"
            "   )"
            "), recent_sentiment AS ("
            " SELECT vds.symbol, AVG(vds.avg_sentiment) AS avg_sentiment, SUM(vds.news_count) AS news_count"
            " FROM v_daily_sentiment vds"
            " WHERE vds.symbol = :symbol_0"
            "   AND vds.sentiment_date >= ("
            "     SELECT MAX(vds2.sentiment_date) - INTERVAL '6 days' FROM v_daily_sentiment vds2 WHERE vds2.symbol = vds.symbol"
            "   )"
            " GROUP BY vds.symbol"
            ") "
            "SELECT lp.symbol, lp.date AS price_date, lp.close_price, lp.volume, "
            "lt.rsi_14, lt.macd, lt.macd_signal, lt.macd_histogram, "
            "lr.calculation_date, lr.volatility_30d, lr.beta, lr.sharpe_ratio, "
            "rs.avg_sentiment, rs.news_count "
            "FROM latest_price lp "
            "LEFT JOIN latest_ti lt ON lt.symbol = lp.symbol "
            "LEFT JOIN latest_risk lr ON lr.symbol = lp.symbol "
            "LEFT JOIN recent_sentiment rs ON rs.symbol = lp.symbol"
        )
        return self._result(
            sql,
            params,
            "Buy-signal snapshot using latest price, technical, risk, and recent sentiment data.",
            QueryComplexity.COMPLEX,
            ["stock_prices", "technical_indicators", "risk_metrics", "v_daily_sentiment"],
            [
                "symbol",
                "price_date",
                "close_price",
                "volume",
                "rsi_14",
                "macd",
                "macd_signal",
                "macd_histogram",
                "calculation_date",
                "volatility_30d",
                "beta",
                "sharpe_ratio",
                "avg_sentiment",
                "news_count",
            ],
        )

    def _build_sharpe_ranking_query(
        self,
        entities: Dict[str, Any],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        limit = int(entities.get("limit") or 10)
        params = {"limit": max(1, min(limit, 25))}
        sql = (
            "SELECT rm.symbol, rm.calculation_date, rm.sharpe_ratio, rm.volatility_30d, rm.beta "
            "FROM risk_metrics rm "
            "WHERE rm.calculation_date = ("
            "  SELECT MAX(rm2.calculation_date) FROM risk_metrics rm2 WHERE rm2.symbol = rm.symbol"
            ") "
            "ORDER BY rm.sharpe_ratio DESC NULLS LAST "
            "LIMIT :limit"
        )
        return self._result(
            sql,
            params,
            "Ranking query over the latest available risk snapshot per symbol.",
            complexity,
            ["risk_metrics"],
            ["symbol", "calculation_date", "sharpe_ratio", "volatility_30d", "beta"],
        )

    def _build_technical_query(
        self,
        symbols: Sequence[str],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        params = {"symbol_0": symbols[0]} if symbols else {}
        sql = (
            "SELECT ti.symbol, ti.date, ti.rsi_14, ti.macd, ti.macd_signal, ti.macd_histogram, "
            "ti.sma_20, ti.sma_50, ti.sma_200, ti.ema_12, ti.ema_26 "
            "FROM technical_indicators ti "
        )
        if symbols:
            sql += (
                "WHERE ti.symbol = :symbol_0 "
                "AND ti.date = ("
                "  SELECT MAX(ti2.date) FROM technical_indicators ti2 WHERE ti2.symbol = ti.symbol"
                ") "
            )
        sql += "ORDER BY ti.symbol, ti.date DESC"
        return self._result(
            sql,
            params,
            "Latest technical-indicator snapshot per requested symbol.",
            complexity,
            ["technical_indicators"],
            [
                "symbol",
                "date",
                "rsi_14",
                "macd",
                "macd_signal",
                "macd_histogram",
                "sma_20",
                "sma_50",
                "sma_200",
                "ema_12",
                "ema_26",
            ],
        )

    def _build_risk_snapshot_query(
        self,
        symbols: Sequence[str],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        params = {"symbol_0": symbols[0]} if symbols else {}
        sql = (
            "SELECT rm.symbol, rm.calculation_date, rm.volatility_30d, rm.volatility_90d, "
            "rm.beta, rm.sharpe_ratio, rm.sortino_ratio, rm.value_at_risk_95 "
            "FROM risk_metrics rm "
        )
        if symbols:
            sql += (
                "WHERE rm.symbol = :symbol_0 "
                "AND rm.calculation_date = ("
                "  SELECT MAX(rm2.calculation_date) FROM risk_metrics rm2 WHERE rm2.symbol = rm.symbol"
                ") "
            )
        sql += "ORDER BY rm.symbol, rm.calculation_date DESC"
        return self._result(
            sql,
            params,
            "Latest risk snapshot per requested symbol.",
            complexity,
            ["risk_metrics"],
            [
                "symbol",
                "calculation_date",
                "volatility_30d",
                "volatility_90d",
                "beta",
                "sharpe_ratio",
                "sortino_ratio",
                "value_at_risk_95",
            ],
        )

    def _build_risk_comparison_query(
        self,
        symbols: Sequence[str],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        params: Dict[str, Any] = {}
        placeholders: List[str] = []
        for idx, symbol in enumerate(symbols[:5]):
            key = f"symbol_{idx}"
            params[key] = symbol
            placeholders.append(f":{key}")
        sql = (
            "SELECT rm.symbol, rm.calculation_date, rm.volatility_30d, rm.volatility_90d, "
            "rm.beta, rm.sharpe_ratio "
            "FROM risk_metrics rm "
        )
        if placeholders:
            sql += (
                f"WHERE rm.symbol IN ({', '.join(placeholders)}) "
                "AND rm.calculation_date = ("
                "  SELECT MAX(rm2.calculation_date) FROM risk_metrics rm2 WHERE rm2.symbol = rm.symbol"
                ") "
            )
        sql += "ORDER BY rm.symbol"
        return self._result(
            sql,
            params,
            "Risk comparison across the latest available per-symbol snapshot.",
            complexity,
            ["risk_metrics"],
            ["symbol", "calculation_date", "volatility_30d", "volatility_90d", "beta", "sharpe_ratio"],
        )

    def _build_sentiment_trend_query(
        self,
        symbols: Sequence[str],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        params = {"symbol_0": symbols[0]} if symbols else {}
        sql = (
            "SELECT ss.symbol, s.company_name, DATE(fn.published_at) AS sentiment_date, "
            "AVG(ss.sentiment_score) AS avg_sentiment, COUNT(*) AS news_count, "
            "AVG(ss.confidence_score) AS avg_confidence "
            "FROM sentiment_scores ss "
            "JOIN financial_news fn ON fn.id = ss.news_id "
            "LEFT JOIN stocks s ON s.symbol = ss.symbol "
        )
        if symbols:
            sql += (
                "WHERE ss.symbol = :symbol_0 "
                "AND DATE(fn.published_at) >= ("
                "  SELECT DATE_TRUNC('month', MAX(DATE(fn2.published_at)))::date "
                "  FROM sentiment_scores ss2 "
                "  JOIN financial_news fn2 ON fn2.id = ss2.news_id "
                "  WHERE ss2.symbol = ss.symbol"
                ") "
            )
        sql += (
            "GROUP BY ss.symbol, s.company_name, DATE(fn.published_at) "
            "ORDER BY sentiment_date ASC"
        )
        return self._result(
            sql,
            params,
            "Sentiment trend aggregated directly from sentiment scores and news timestamps.",
            complexity,
            ["sentiment_scores", "financial_news", "stocks"],
            ["symbol", "company_name", "sentiment_date", "avg_sentiment", "news_count", "avg_confidence"],
        )

    def _build_recent_sentiment_query(
        self,
        symbols: Sequence[str],
        entities: Dict[str, Any],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        params: Dict[str, Any] = {}
        sql = (
            "SELECT ss.symbol, fn.published_at, fn.headline, fn.source, "
            "ss.sentiment_label, ss.sentiment_score, ss.confidence_score "
            "FROM sentiment_scores ss "
            "JOIN financial_news fn ON fn.id = ss.news_id "
        )
        if symbols:
            params["symbol_0"] = symbols[0]
            sql += (
                "WHERE ss.symbol = :symbol_0 "
                "AND fn.published_at >= ("
                "  SELECT MAX(fn2.published_at) - INTERVAL '6 days' "
                "  FROM sentiment_scores ss2 "
                "  JOIN financial_news fn2 ON fn2.id = ss2.news_id "
                "  WHERE ss2.symbol = ss.symbol"
                ") "
            )
        elif entities.get("time_period") == "1w":
            sql += (
                "WHERE fn.published_at >= ("
                "  SELECT MAX(fn2.published_at) - INTERVAL '6 days' FROM financial_news fn2"
                ") "
            )
        elif entities.get("time_period") == "1m":
            sql += (
                "WHERE fn.published_at >= ("
                "  SELECT MAX(fn2.published_at) - INTERVAL '30 days' FROM financial_news fn2"
                ") "
            )
        if entities.get("sentiment_filter") == "negative":
            sql += ("AND " if "WHERE" in sql else "WHERE ") + "ss.sentiment_score < 0 "
        elif entities.get("sentiment_filter") == "positive":
            sql += ("AND " if "WHERE" in sql else "WHERE ") + "ss.sentiment_score > 0 "
        sql += "ORDER BY fn.published_at DESC LIMIT 20"
        return self._result(
            sql,
            params,
            "Article-level recent sentiment query joined to financial news.",
            complexity,
            ["sentiment_scores", "financial_news"],
            ["symbol", "published_at", "headline", "source", "sentiment_label", "sentiment_score", "confidence_score"],
        )

    def _build_volume_ranking_query(
        self,
        entities: Dict[str, Any],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        limit = int(entities.get("limit") or 5)
        params = {"limit": max(1, min(limit, 25))}
        sql = (
            "SELECT sp.symbol, sp.date, sp.volume, sp.close_price "
            "FROM stock_prices sp "
            "WHERE sp.date = (SELECT MAX(sp2.date) FROM stock_prices sp2) "
            "ORDER BY sp.volume DESC NULLS LAST "
            "LIMIT :limit"
        )
        return self._result(
            sql,
            params,
            "Volume ranking on the latest available trading date.",
            complexity,
            ["stock_prices"],
            ["symbol", "date", "volume", "close_price"],
        )

    def _build_low_volatility_sector_screen(
        self,
        entities: Dict[str, Any],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        sector = (entities.get("sectors") or ["Technology"])[0]
        limit = int(entities.get("limit") or 10)
        params = {"sector": sector, "limit": max(1, min(limit, 25))}
        sql = (
            "SELECT s.symbol, s.company_name, s.sector, rm.calculation_date, "
            "rm.volatility_30d, rm.beta, rm.sharpe_ratio "
            "FROM stocks s "
            "JOIN risk_metrics rm ON rm.stock_id = s.id "
            "WHERE s.sector = :sector "
            "AND rm.calculation_date = ("
            "  SELECT MAX(rm2.calculation_date) FROM risk_metrics rm2 WHERE rm2.stock_id = rm.stock_id"
            ") "
            "AND rm.volatility_30d <= (SELECT AVG(rm3.volatility_30d) FROM risk_metrics rm3) "
            "ORDER BY rm.volatility_30d ASC NULLS LAST, rm.sharpe_ratio DESC NULLS LAST "
            "LIMIT :limit"
        )
        return self._result(
            sql,
            params,
            "Sector screen using latest risk metrics and below-average 30-day volatility.",
            QueryComplexity.MODERATE,
            ["stocks", "risk_metrics"],
            ["symbol", "company_name", "sector", "calculation_date", "volatility_30d", "beta", "sharpe_ratio"],
        )

    def _build_portfolio_query(
        self,
        entities: Dict[str, Any],
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        portfolio_id = entities.get("portfolio_id")
        params: Dict[str, Any] = {}
        sql = (
            "SELECT p.id AS portfolio_id, p.name AS portfolio_name, ph.symbol, ph.shares, "
            "ph.avg_cost_basis, ph.current_price, ph.market_value, "
            "ph.unrealized_gain_loss, ph.weight_percentage "
            "FROM portfolios p "
            "LEFT JOIN portfolio_holdings ph ON ph.portfolio_id = p.id "
        )
        if portfolio_id:
            params["portfolio_id"] = portfolio_id
            sql += "WHERE p.id = :portfolio_id "
        sql += "ORDER BY p.name, ph.weight_percentage DESC NULLS LAST"
        return self._result(
            sql,
            params,
            "Portfolio holdings query using the verified public portfolio tables.",
            complexity,
            ["portfolios", "portfolio_holdings"],
            [
                "portfolio_id",
                "portfolio_name",
                "symbol",
                "shares",
                "avg_cost_basis",
                "current_price",
                "market_value",
                "unrealized_gain_loss",
                "weight_percentage",
            ],
        )

    async def _generate_with_llm(
        self,
        intent: str,
        entities: Dict[str, Any],
        user_query: str,
        complexity: QueryComplexity,
    ) -> SQLQueryResult:
        semantic_requirements = self._semantic_requirements(user_query, entities)
        prompt = (
            self.system_prompt
            + "\n\nSEMANTIC REQUIREMENTS:\n"
            + "\n".join(f"- {item}" for item in semantic_requirements)
            + "\n\nREQUEST JSON:\n"
            + json.dumps(
                {
                    "intent": intent,
                    "entities": entities,
                    "user_query": user_query,
                    "complexity": complexity.value,
                },
                indent=2,
            )
        )

        payload_text = await self._invoke_generation_llm(
            system_prompt=prompt,
            user_prompt="Generate the JSON response now.",
        )
        payload = self._parse_json_payload(payload_text)

        sql = str(payload.get("sql", "")).strip()
        tables = payload.get("tables") or self._extract_tables_from_sql(sql)
        expected_columns = payload.get("expected_columns") or []
        notes = payload.get("notes") or []

        return SQLQueryResult(
            sql=sql or "SELECT s.symbol, s.company_name FROM stocks s ORDER BY s.symbol LIMIT 10",
            parameters={},
            reasoning="LLM-generated SQL constrained by the verified schema prompt.",
            complexity=complexity,
            estimated_execution_time=0.35,
            tables_involved=list(tables),
            expected_output_columns=list(expected_columns),
            metadata={"notes": notes, "system_prompt": self.system_prompt},
        )

    async def _repair_with_llm(
        self,
        intent: str,
        entities: Dict[str, Any],
        user_query: str,
        complexity: QueryComplexity,
        sql: str,
        errors: List[str],
    ) -> Optional[SQLQueryResult]:
        semantic_requirements = self._semantic_requirements(user_query, entities)
        prompt = (
            self.system_prompt
            + "\n\nThe previous SQL was invalid.\n"
            + "Fix it using the verified schema and return JSON only.\n"
            + "Honor the semantic requirements exactly.\n"
            + "\nSEMANTIC REQUIREMENTS:\n"
            + "\n".join(f"- {item}" for item in semantic_requirements)
            + json.dumps(
                {
                    "intent": intent,
                    "entities": entities,
                    "user_query": user_query,
                    "complexity": complexity.value,
                    "invalid_sql": sql,
                    "validation_errors": errors,
                },
                indent=2,
            )
        )

        try:
            payload_text = await self._invoke_generation_llm(
                system_prompt=prompt,
                user_prompt="Rewrite the SQL so it is valid for the schema.",
            )
        except Exception:
            return None

        payload = self._parse_json_payload(payload_text)
        repaired_sql = str(payload.get("sql", "")).strip()
        tables = payload.get("tables") or self._extract_tables_from_sql(repaired_sql)
        expected_columns = payload.get("expected_columns") or []
        notes = payload.get("notes") or []

        return SQLQueryResult(
            sql=repaired_sql,
            parameters={},
            reasoning="LLM-repaired SQL after schema validation failure.",
            complexity=complexity,
            estimated_execution_time=0.35,
            tables_involved=list(tables),
            expected_output_columns=list(expected_columns),
            metadata={"notes": notes, "system_prompt": self.system_prompt},
        )

    async def _invoke_generation_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if self.llm is not None:
            try:
                response = await self.llm.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                return self._extract_content(response.content)
            except Exception:
                pass

        if not self.groq_client.enabled:
            raise RuntimeError("No SQL-generation LLM is available.")

        groq_text = await asyncio.to_thread(
            self.groq_client.generate,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.settings.GROQ_SQL_MODEL,
            temperature=0.0,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )
        return groq_text

    def _extract_content(self, response_content: Any) -> str:
        content = response_content
        if isinstance(content, list):
            content = " ".join(part if isinstance(part, str) else str(part) for part in content)
        return str(content)

    def _parse_json_payload(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    def _validate_sql(self, sql: str) -> List[str]:
        errors: List[str] = []
        lowered = sql.lower()

        if not sql.strip():
            return ["SQL is empty."]
        if "--" in sql or "/*" in sql or "*/" in sql:
            errors.append("SQL comments are not allowed.")
        if "sentiment_analysis" in lowered:
            errors.append("sentiment_analysis does not exist; use sentiment_scores or v_daily_sentiment.")
        if re.search(r"\bticker\b", lowered):
            errors.append("ticker does not exist; use symbol.")

        tables = self._extract_tables_from_sql(sql)
        allowed_relations = set(self.schema.get_allowed_relations())
        alias_map = self._extract_alias_map(sql)
        cte_names = self._extract_cte_names(sql)

        for table_name in tables:
            if table_name in cte_names:
                continue
            if table_name not in allowed_relations:
                errors.append(f"Unknown relation referenced: {table_name}")

        for alias, column_name in re.findall(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", lowered):
            table_name = alias_map.get(alias)
            if table_name is None:
                continue
            if table_name in cte_names:
                continue
            if column_name not in self.schema.VERIFIED_SCHEMA.get(table_name, {}):
                if table_name == "technical_indicators" and column_name == "volume":
                    errors.append(
                        f"Unknown column {alias}.volume for relation technical_indicators; use {alias}.volume_sma_20 or join stock_prices for raw volume"
                    )
                else:
                    errors.append(f"Unknown column {alias}.{column_name} for relation {table_name}")

        if not re.match(r"^\s*(select|with)\b", lowered):
            errors.append("Only SELECT or WITH queries are allowed.")
        if re.search(r"\bwhere\b[\s\S]*\b(?:lag|lead|row_number|rank|dense_rank)\s*\(", lowered):
            errors.append("Window functions cannot be used directly inside WHERE; compute them in a subquery or CTE and filter on the derived columns.")

        return sorted(set(errors))

    def _validate_semantics(
        self,
        user_query: str,
        entities: Dict[str, Any],
        sql: str,
    ) -> List[str]:
        errors: List[str] = []
        query_lower = user_query.lower()
        sql_lower = sql.lower()
        metric_names = [str(metric).lower() for metric in (entities.get("metrics", []) or [])]
        requested_symbols = self._get_symbols(entities)

        needs_time_window = any(
            phrase in query_lower
            for phrase in (
                "over the last",
                "last week",
                "last month",
                "last 2 weeks",
                "this month",
                "trend",
            )
        )
        has_time_window = any(
            token in sql_lower
            for token in (
                "interval",
                "date_trunc",
                "max(date)",
                "max(calculation_date)",
                "max(published_at)",
                "max(processed_at)",
                "sentiment_date",
            )
        )
        if needs_time_window and not has_time_window:
            errors.append("Query asks for a time window or trend, but SQL does not encode a clear date window.")
        if needs_time_window and any(token in sql_lower for token in ("current_date", "current_timestamp", "now()")) and not any(
            token in sql_lower
            for token in ("max(date)", "max(calculation_date)", "max(processed_at)", "max(published_at)", "max(sentiment_date)")
        ):
            errors.append("Relative date filters should be anchored to the latest available data in the tables, not only CURRENT_DATE/NOW().")

        rising_terms = ("rising", "increasing", "improving", "higher")
        falling_terms = ("falling", "decreasing", "declining", "lower")
        crossover_terms = ("crossover", "cross above", "cross below")

        has_series_comparison = any(
            token in sql_lower
            for token in (
                "lag(",
                "lead(",
                "row_number() over",
                "partition by",
                "start_",
                "end_",
                "over (",
            )
        )
        has_ranked_latest_logic = any(
            token in sql_lower
            for token in (
                "row_number() over",
                "rank() over",
                "dense_rank() over",
                "max(date)",
                "max(calculation_date)",
                "max(published_at)",
                "max(processed_at)",
                "max(sentiment_date)",
            )
        )
        if any(term in query_lower for term in ("latest", "current", "today", "most recent")) and not has_ranked_latest_logic:
            errors.append("Query asks for latest/current data, but SQL does not encode latest-available row selection.")

        if any(term in query_lower for term in rising_terms) and not (
            has_series_comparison and ">" in sql_lower
        ):
            errors.append("Query asks for a rising/increasing condition, but SQL does not encode a clear upward comparison.")

        if any(term in query_lower for term in falling_terms) and not (
            has_series_comparison and "<" in sql_lower
        ):
            errors.append("Query asks for a falling/decreasing condition, but SQL does not encode a clear downward comparison.")

        if any(term in query_lower for term in crossover_terms):
            has_crossover_logic = (
                "macd" in sql_lower
                and "macd_signal" in sql_lower
                and has_series_comparison
                and (">" in sql_lower or "<" in sql_lower)
            )
            if not has_crossover_logic:
                errors.append("Query asks for a crossover condition, but SQL does not compare current and prior MACD versus signal values.")

        if "volume" in metric_names and "volume" in query_lower:
            mentions_valid_volume_source = any(
                token in sql_lower
                for token in ("stock_prices", ".volume ", ".volume)", ".volume,", "volume_sma_20", "sp.volume")
            )
            if not mentions_valid_volume_source:
                errors.append("Volume was requested, but SQL does not reference stock_prices.volume or technical_indicators.volume_sma_20.")
            if any(term in query_lower for term in falling_terms) and "volume" in query_lower and not (
                "volume_sma_20" in sql_lower
                or re.search(r"\b[a-z_][a-z0-9_]*\.volume\b", sql_lower)
            ):
                errors.append("Query asks for a falling volume condition, but SQL does not compare a valid volume series.")

        if "rsi" in query_lower and "rsi_14" not in sql_lower:
            errors.append("RSI was requested, but SQL does not reference rsi_14.")
        if "rsi" in query_lower and any(term in query_lower for term in rising_terms) and not re.search(
            r"rsi_14\s*[<>]",
            sql_lower,
        ):
            errors.append("Query asks for rising RSI, but SQL does not compare rsi_14 values across time.")

        if "sentiment" in query_lower and not any(
            token in sql_lower for token in ("sentiment_scores", "avg_sentiment", "sentiment_score")
        ):
            errors.append("Sentiment was requested, but SQL does not reference sentiment data.")
        if "sentiment" in query_lower and any(term in query_lower for term in rising_terms + falling_terms) and not (
            has_series_comparison or "group by" in sql_lower
        ):
            errors.append("Query asks for sentiment trend direction, but SQL does not compare sentiment values over time.")

        if len(requested_symbols) > 1 and not any(
            token in sql_lower for token in (" in (", "= any", "or", "union")
        ):
            errors.append("Query references multiple symbols, but SQL does not clearly include more than one symbol.")

        if "screen" in query_lower or "which stocks" in query_lower or "find stocks" in query_lower:
            if not any(token in sql_lower for token in ("where", "having")):
                errors.append("Screening query is missing filtering conditions.")

        if any(term in query_lower for term in ("compared to average", "below-average", "above-average", "lower-than-average", "higher-than-average")):
            if not any(token in sql_lower for token in ("avg(", "window_avg", "overall_avg", "benchmark")):
                errors.append("Query asks for an average-relative comparison, but SQL does not compute or join an average benchmark.")

        if "technical_indicators" in sql_lower and "stock_prices" in sql_lower:
            aligned_dates = re.search(
                r"\b[a-z_][a-z0-9_]*\.date\s*=\s*[a-z_][a-z0-9_]*\.date\b",
                sql_lower,
            )
            if not aligned_dates:
                errors.append("SQL combines stock_prices and technical_indicators but does not align their date columns in the join or filter logic.")

        return sorted(set(errors))

    def _semantic_requirements(
        self,
        user_query: str,
        entities: Dict[str, Any],
    ) -> List[str]:
        query_lower = user_query.lower()
        requirements = [
            "Return one PostgreSQL query that matches the user intent semantically, not just syntactically.",
            "If the request asks for latest/current data, select the latest available row per symbol with MAX(...) or ROW_NUMBER().",
            "If the request asks for multiple symbols, keep the requested symbols explicit in the SQL.",
        ]

        if any(
            phrase in query_lower
            for phrase in ("over the last", "last week", "last month", "last 2 weeks", "this month", "trend")
        ):
            requirements.append(
                "Include an explicit date window using INTERVAL, DATE_TRUNC, or latest-available date logic anchored to data in the tables."
            )

        if any(term in query_lower for term in ("rising", "increasing", "improving", "higher")):
            requirements.append(
                "For rising/increasing conditions, compare a newer value against an older value using LAG/LEAD, ROW_NUMBER start/end snapshots, or equivalent."
            )

        if any(term in query_lower for term in ("falling", "decreasing", "declining", "lower")):
            requirements.append(
                "For falling/decreasing conditions, compare a newer value against an older value using LAG/LEAD, ROW_NUMBER start/end snapshots, or equivalent."
            )

        if any(term in query_lower for term in ("crossover", "cross above", "cross below")):
            requirements.append(
                "For MACD crossover logic, compare current and prior MACD versus MACD signal values, not just current MACD and signal."
            )

        if "volume" in query_lower:
            requirements.append(
                "Use stock_prices.volume for raw volume trends, or technical_indicators.volume_sma_20 only when the question allows indicator-based volume context."
            )
            requirements.append(
                "If the query compares RSI from technical_indicators with raw volume from stock_prices, align the joined rows on both stock_id and date."
            )

        if "rsi" in query_lower:
            requirements.append("Use technical_indicators.rsi_14 for RSI-related logic.")

        if "sentiment" in query_lower:
            requirements.append(
                "Use sentiment_scores joined to financial_news, or v_daily_sentiment, and preserve trend direction if the question asks for improving or declining sentiment."
            )

        if any(term in query_lower for term in ("above-average", "below-average", "lower-than-average", "higher-than-average")):
            requirements.append("Compute the relevant average benchmark inside the SQL and compare each symbol against it.")

        if any(term in query_lower for term in ("which stocks", "find stocks", "screen")):
            requirements.append("Return a filtered stock list, not a generic lookup query.")

        metrics = [str(metric).lower() for metric in (entities.get("metrics", []) or [])]
        if metrics:
            requirements.append(f"Metrics requested and expected in SQL logic: {', '.join(sorted(set(metrics)))}.")

        return requirements

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        patterns = re.findall(
            r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\b",
            sql.lower(),
        )
        return list(dict.fromkeys(patterns))

    def _extract_alias_map(self, sql: str) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        pattern = re.compile(
            r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\s+(?:as\s+)?([a-z_][a-z0-9_]*)\b",
            re.IGNORECASE,
        )
        for table_name, alias in pattern.findall(sql):
            alias_map[alias.lower()] = table_name.lower()
        return alias_map

    def _extract_cte_names(self, sql: str) -> List[str]:
        return list(
            dict.fromkeys(
                re.findall(r"\b([a-z_][a-z0-9_]*)\s+as\s*\(", sql.lower())
            )
        )

    def _get_symbols(self, entities: Dict[str, Any]) -> List[str]:
        raw_symbols = entities.get("stocks", []) or []
        result: List[str] = []
        for symbol in raw_symbols:
            normalized = str(symbol).upper()
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    def _get_unresolved_mentions(self, entities: Dict[str, Any]) -> List[str]:
        unresolved = entities.get("unknown_symbols") or entities.get("unresolved_mentions") or []
        return [str(item).strip() for item in unresolved if str(item).strip()]

    def _apply_known_repairs(self, sql: str) -> str:
        repaired = sql
        repaired = re.sub(
            r"\b([a-z_][a-z0-9_]*)\.volume\b",
            lambda match: (
                f"{match.group(1)}.volume_sma_20"
                if self._alias_points_to_relation(repaired, match.group(1), "technical_indicators")
                else match.group(0)
            ),
            repaired,
        )
        return repaired

    def _alias_points_to_relation(self, sql: str, alias: str, relation_name: str) -> bool:
        alias_map = self._extract_alias_map(sql)
        return alias_map.get(alias.lower()) == relation_name

    def _result(
        self,
        sql: str,
        parameters: Dict[str, Any],
        reasoning: str,
        complexity: QueryComplexity,
        tables: List[str],
        expected_output_columns: List[str],
    ) -> SQLQueryResult:
        return SQLQueryResult(
            sql=sql.strip(),
            parameters=parameters,
            reasoning=reasoning,
            complexity=complexity,
            estimated_execution_time=0.15 if complexity == QueryComplexity.SIMPLE else 0.35,
            tables_involved=tables,
            expected_output_columns=expected_output_columns,
            metadata={"system_prompt": self.system_prompt},
        )


async def generate_sql_query(
    intent: str,
    entities: Dict[str, Any],
    user_query: str,
) -> SQLQueryResult:
    """Main entry point for SQL generation."""

    generator = AdvancedSQLGenerator()
    return await generator.generate_sql(intent, entities, user_query)
