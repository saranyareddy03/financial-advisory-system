"""
Database Query Executor
Real-Time Financial Advisory System - Phase 5 Component 3

This module executes read-only SQL with schema validation, date-range checks,
result caching, and structured error handling.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import exc as sqlalchemy_exc
from sqlalchemy import text as sql_text

from src.database.connection import db_manager
from src.langgraph.sql_generator import DatabaseSchemaKnowledge, QueryComplexity, SQLQueryResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    CACHED = "cached"


class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QueryExecutionResult:
    status: ExecutionStatus
    data: Optional[List[Dict[str, Any]]] = None
    row_count: int = 0
    execution_time: float = 0.0
    query_hash: Optional[str] = None
    cached: bool = False
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Optional[Dict[str, Any]] = None
    column_info: Optional[List[Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "data": self.data or [],
            "row_count": self.row_count,
            "execution_time": self.execution_time,
            "query_hash": self.query_hash,
            "cached": self.cached,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "warnings": self.warnings,
            "performance_metrics": self.performance_metrics,
            "column_info": self.column_info or [],
        }


@dataclass
class QueryCacheEntry:
    result: QueryExecutionResult
    timestamp: datetime
    expiry: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None


class SecurityValidator:
    """SQL validator for read-only schema-constrained queries."""

    DANGEROUS_PATTERNS = [
        r";\s*(drop|delete|update|insert|alter|create|truncate|grant|revoke)\b",
        r"\b(union|intersect|except)\b",
        r"(--|/\*|\*/)",
        r"\b(exec|execute|copy|vacuum|analyze)\b",
        r"\b(load_file|into\s+outfile|pg_sleep)\b",
    ]

    ALLOWED_OPERATIONS = ["select", "with"]
    ALLOWED_RELATIONS = DatabaseSchemaKnowledge.get_allowed_relations()
    VERIFIED_SCHEMA = DatabaseSchemaKnowledge.VERIFIED_SCHEMA

    @classmethod
    def validate_query(
        cls,
        sql: str,
        params: Dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
    ) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        lowered = sql.strip().lower()

        if not lowered:
            return False, ["SQL is empty."]

        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, lowered, re.IGNORECASE):
                return False, [f"Dangerous SQL pattern detected: {pattern}"]

        first_word = lowered.split()[0] if lowered.split() else ""
        if first_word not in cls.ALLOWED_OPERATIONS:
            return False, [f"Operation not allowed: {first_word}"]

        if "sentiment_analysis" in lowered:
            return False, ["Relation sentiment_analysis does not exist."]
        if re.search(r"\bticker\b", lowered):
            return False, ["Column ticker does not exist; use symbol."]

        relation_names = cls._extract_relations(sql)
        cte_names = cls._extract_cte_names(sql)
        for relation_name in relation_names:
            if relation_name in cte_names:
                continue
            if relation_name not in cls.ALLOWED_RELATIONS:
                return False, [f"Relation not allowed or unknown: {relation_name}"]

        alias_map = cls._extract_alias_map(sql)
        for alias, column_name in re.findall(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", lowered):
            relation_name = alias_map.get(alias)
            if relation_name is None:
                continue
            if relation_name in cte_names:
                continue
            if column_name not in cls.VERIFIED_SCHEMA.get(relation_name, {}):
                return False, [f"Unknown column {alias}.{column_name} on relation {relation_name}"]

        date_error = cls._validate_date_parameters(params)
        if date_error:
            return False, [date_error]

        if security_level in {SecurityLevel.HIGH, SecurityLevel.CRITICAL} and len(sql) > 6000:
            warnings.append("Query text is unusually long.")

        return True, warnings

    @classmethod
    def sanitize_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}

        for key, value in params.items():
            if isinstance(value, str):
                sanitized[key] = value[:200]
            elif isinstance(value, (int, float, Decimal, date, datetime)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = [str(item)[:100] for item in value[:25]]
            else:
                sanitized[key] = str(value)[:200]

        return sanitized

    @classmethod
    def _extract_relations(cls, sql: str) -> List[str]:
        return list(
            dict.fromkeys(
                re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\b", sql.lower())
            )
        )

    @classmethod
    def _extract_alias_map(cls, sql: str) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        pattern = re.compile(
            r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\s+(?:as\s+)?([a-z_][a-z0-9_]*)\b",
            re.IGNORECASE,
        )
        for relation_name, alias in pattern.findall(sql):
            alias_map[alias.lower()] = relation_name.lower()
        return alias_map

    @classmethod
    def _extract_cte_names(cls, sql: str) -> List[str]:
        return list(
            dict.fromkeys(
                re.findall(r"\b([a-z_][a-z0-9_]*)\s+as\s*\(", sql.lower())
            )
        )

    @classmethod
    def _validate_date_parameters(cls, params: Dict[str, Any]) -> Optional[str]:
        start = params.get("start_date")
        end = params.get("end_date")

        if start and end and start > end:
            return "Invalid date range: start_date is after end_date."

        for key, value in params.items():
            if "date" not in key:
                continue
            if isinstance(value, str) and value:
                try:
                    datetime.fromisoformat(value)
                except ValueError:
                    return f"Invalid ISO date/datetime value for {key}."

        return None


class QueryCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, QueryCacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def _generate_hash(self, sql: str, params: Dict[str, Any]) -> str:
        content = f"{sql}:{json.dumps(params, sort_keys=True, default=str)}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, sql: str, params: Dict[str, Any]) -> Optional[QueryExecutionResult]:
        query_hash = self._generate_hash(sql, params)
        cache_entry = self.cache.get(query_hash)
        if cache_entry is None:
            return None

        if datetime.now() > cache_entry.expiry:
            del self.cache[query_hash]
            return None

        cache_entry.access_count += 1
        cache_entry.last_access = datetime.now()

        cached_result = self._clone_result(cache_entry.result)
        cached_result.status = ExecutionStatus.CACHED
        cached_result.cached = True
        cached_result.query_hash = query_hash
        return cached_result

    def set(
        self,
        sql: str,
        params: Dict[str, Any],
        result: QueryExecutionResult,
        ttl: Optional[int] = None,
    ) -> None:
        if result.status != ExecutionStatus.SUCCESS:
            return

        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        query_hash = self._generate_hash(sql, params)
        ttl = ttl or self.default_ttl
        self.cache[query_hash] = QueryCacheEntry(
            result=self._clone_result(result),
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(seconds=ttl),
            access_count=1,
            last_access=datetime.now(),
        )

    def _clone_result(self, result: QueryExecutionResult) -> QueryExecutionResult:
        return QueryExecutionResult(
            status=result.status,
            data=[dict(row) for row in (result.data or [])],
            row_count=result.row_count,
            execution_time=result.execution_time,
            query_hash=result.query_hash,
            cached=result.cached,
            error_message=result.error_message,
            error_code=result.error_code,
            warnings=list(result.warnings),
            performance_metrics=dict(result.performance_metrics) if result.performance_metrics else None,
            column_info=[dict(item) for item in (result.column_info or [])],
        )

    def _evict_oldest(self) -> None:
        if not self.cache:
            return

        sorted_entries = sorted(
            self.cache.items(),
            key=lambda item: item[1].last_access or item[1].timestamp,
        )
        for key, _ in sorted_entries[: max(1, len(sorted_entries) // 5)]:
            del self.cache[key]

    def clear_expired(self) -> int:
        now = datetime.now()
        expired_keys = [key for key, entry in self.cache.items() if now > entry.expiry]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)


class PerformanceMonitor:
    def __init__(self):
        self.query_stats: Dict[str, Dict[str, Any]] = {}

    def record_execution(
        self,
        sql: str,
        execution_time: float,
        row_count: int,
        complexity: QueryComplexity,
    ) -> Dict[str, Any]:
        signature = self._get_query_signature(sql)
        stats = self.query_stats.setdefault(
            signature,
            {
                "total_executions": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "total_rows": 0,
                "avg_rows": 0.0,
                "complexity": complexity.value,
            },
        )

        stats["total_executions"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["total_executions"]
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["total_rows"] += row_count
        stats["avg_rows"] = stats["total_rows"] / stats["total_executions"]

        return {
            "query_signature": signature,
            "current_execution_time": execution_time,
            "average_execution_time": stats["avg_time"],
            "performance_rating": self._get_performance_rating(execution_time, complexity),
        }

    def _get_query_signature(self, sql: str) -> str:
        normalized = re.sub(r"'[^']*'", "'?'", sql)
        normalized = re.sub(r"\b\d+\b", "?", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized[:200]

    def _get_performance_rating(self, execution_time: float, complexity: QueryComplexity) -> str:
        thresholds = {
            QueryComplexity.SIMPLE: (0.1, 0.3, 0.8),
            QueryComplexity.MODERATE: (0.3, 0.8, 2.0),
            QueryComplexity.COMPLEX: (0.8, 2.0, 5.0),
            QueryComplexity.ADVANCED: (2.0, 5.0, 10.0),
        }
        excellent, good, fair = thresholds.get(complexity, thresholds[QueryComplexity.SIMPLE])
        if execution_time <= excellent:
            return "excellent"
        if execution_time <= good:
            return "good"
        if execution_time <= fair:
            return "fair"
        return "slow"


class DatabaseQueryExecutor:
    """Main query executor with validation, caching, and structured errors."""

    def __init__(self):
        self.security_validator = SecurityValidator()
        self.cache = QueryCache(max_size=500, default_ttl=300)
        self.performance_monitor = PerformanceMonitor()
        self.max_execution_time = 30.0
        self.max_result_size = 10000

    async def execute_query(
        self,
        sql_result: SQLQueryResult,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        use_cache: bool = True,
        max_rows: Optional[int] = None,
    ) -> QueryExecutionResult:
        start_time = time.time()
        safe_params = self.security_validator.sanitize_parameters(sql_result.parameters)

        is_valid, warnings = self.security_validator.validate_query(
            sql_result.sql,
            safe_params,
            security_level,
        )
        if not is_valid:
            return QueryExecutionResult(
                status=ExecutionStatus.ERROR,
                execution_time=time.time() - start_time,
                error_message=warnings[0],
                error_code="VALIDATION_ERROR",
                warnings=warnings,
            )

        if use_cache:
            cached = self.cache.get(sql_result.sql, safe_params)
            if cached is not None:
                return cached

        try:
            result = await self._execute_with_timeout(
                sql_result.sql,
                safe_params,
                max_rows or self.max_result_size,
            )
        except Exception as exc:
            logger.error("Query execution error: %s", exc)
            return QueryExecutionResult(
                status=ExecutionStatus.ERROR,
                execution_time=time.time() - start_time,
                error_message=str(exc),
                error_code="EXECUTION_ERROR",
            )

        result.warnings.extend(warnings)
        result.performance_metrics = self.performance_monitor.record_execution(
            sql_result.sql,
            result.execution_time,
            result.row_count,
            sql_result.complexity,
        )

        if use_cache and result.status == ExecutionStatus.SUCCESS:
            ttl = self._calculate_cache_ttl(sql_result.complexity, result.execution_time)
            self.cache.set(sql_result.sql, safe_params, result, ttl)

        return result

    async def _execute_with_timeout(
        self,
        sql: str,
        parameters: Dict[str, Any],
        max_rows: int,
    ) -> QueryExecutionResult:
        start_time = time.time()
        try:
            async with asyncio.timeout(self.max_execution_time):
                result = await self._execute_query_internal(sql, parameters, max_rows)
        except asyncio.TimeoutError:
            return QueryExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                execution_time=time.time() - start_time,
                error_message=f"Query timed out after {self.max_execution_time} seconds.",
                error_code="TIMEOUT_ERROR",
            )

        result.execution_time = time.time() - start_time
        return result

    async def _execute_query_internal(
        self,
        sql: str,
        parameters: Dict[str, Any],
        max_rows: int,
    ) -> QueryExecutionResult:
        try:
            with db_manager.get_session() as session:
                proxy = session.execute(sql_text(sql), parameters)
                rows = proxy.mappings().fetchmany(max_rows)
                data = [dict(row) for row in rows]

                if not data:
                    return QueryExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        data=[],
                        row_count=0,
                        column_info=[],
                        warnings=["Query completed successfully but returned no rows."],
                    )

                column_info = [
                    {"name": key, "type": type(value).__name__}
                    for key, value in data[0].items()
                ]
                warnings: List[str] = []
                if len(data) >= max_rows:
                    warnings.append(f"Result limited to {max_rows} rows.")

                return QueryExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    data=data,
                    row_count=len(data),
                    column_info=column_info,
                    warnings=warnings,
                )
        except sqlalchemy_exc.SQLAlchemyError as exc:
            return QueryExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=str(exc),
                error_code=type(exc).__name__,
            )

    def _calculate_cache_ttl(self, complexity: QueryComplexity, execution_time: float) -> int:
        ttl = {
            QueryComplexity.SIMPLE: 300,
            QueryComplexity.MODERATE: 600,
            QueryComplexity.COMPLEX: 900,
            QueryComplexity.ADVANCED: 1200,
        }.get(complexity, 300)

        if execution_time > 5.0:
            ttl *= 2

        return min(ttl, 3600)

    def get_cache_stats(self) -> Dict[str, Any]:
        total_entries = len(self.cache.cache)
        if total_entries == 0:
            return {"total_entries": 0, "hit_rate": 0.0}

        total_access = sum(entry.access_count for entry in self.cache.cache.values())
        return {
            "total_entries": total_entries,
            "total_access_count": total_access,
            "cache_size_limit": self.cache.max_size,
            "cache_utilization": (total_entries / self.cache.max_size) * 100,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "total_query_types": len(self.performance_monitor.query_stats),
            "query_stats": self.performance_monitor.query_stats,
        }

    async def health_check(self) -> Dict[str, Any]:
        start_time = time.time()
        result = await self._execute_query_internal("SELECT 1 AS health_check", {}, 1)
        return {
            "status": "healthy" if result.status == ExecutionStatus.SUCCESS else "unhealthy",
            "database_response_time": time.time() - start_time,
            "cache_stats": self.get_cache_stats(),
            "expired_cache_entries_cleared": self.cache.clear_expired(),
            "timestamp": datetime.now().isoformat(),
        }
