"""
Portfolio advisory analytics service.

This module loads a seeded demo portfolio from the database, computes live
portfolio analytics from historical returns, and generates deterministic
portfolio answers for the NLP agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.analytics.portfolio_optimization import PortfolioOptimizer
from src.database.connection import db_manager
from src.langgraph.query_executor import ExecutionStatus, QueryExecutionResult
from src.langgraph.response_formatter import ConfidenceLevel, FormattedResponse
from src.langgraph.sql_generator import QueryComplexity, SQLQueryResult


@dataclass
class PortfolioSnapshot:
    portfolio_id: str
    portfolio_name: str
    description: Optional[str]
    risk_tolerance: Optional[str]
    cash_balance: float
    total_equity: float
    total_value: float
    holdings: List[Dict[str, Any]]
    current_weights: Dict[str, float]
    sector_exposure: Dict[str, float]
    metrics: Dict[str, Optional[float]]
    top_risk_contributors: List[Dict[str, float]]


class PortfolioAdvisoryService:
    """Analytics-backed question answering for portfolio optimization."""

    def __init__(self):
        self.optimizer = PortfolioOptimizer(lookback_days=252)

    def answer_query(
        self,
        user_query: str,
        portfolio_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        snapshot = self.load_portfolio_snapshot(portfolio_id)
        if snapshot is None:
            return None

        question_type = self._detect_question_type(user_query)
        optimized = self._optimize_snapshot(snapshot, user_query, question_type)

        if question_type == "allocation":
            formatted = self._format_allocation_answer(snapshot)
        elif question_type == "diversification":
            formatted = self._format_diversification_answer(snapshot)
        elif question_type == "risk":
            formatted = self._format_risk_answer(snapshot)
        elif question_type == "rebalance":
            formatted = self._format_rebalance_answer(snapshot, optimized)
        elif question_type == "optimize":
            formatted = self._format_optimized_answer(snapshot, optimized)
        elif question_type == "compare_optimized":
            formatted = self._format_compare_answer(snapshot, optimized)
        else:
            formatted = self._format_summary_answer(snapshot)

        sql_result = SQLQueryResult(
            sql="",
            parameters={"portfolio_id": snapshot.portfolio_id},
            reasoning="Portfolio analytics route uses database-backed holdings plus optimization analytics.",
            complexity=QueryComplexity.COMPLEX,
            estimated_execution_time=0.25,
            tables_involved=["portfolios", "portfolio_holdings", "stocks", "stock_prices"],
            expected_output_columns=["symbol", "shares", "current_price", "market_value", "weight_percentage"],
            metadata={"portfolio_id": snapshot.portfolio_id, "question_type": question_type},
        )
        execution_result = QueryExecutionResult(
            status=ExecutionStatus.SUCCESS,
            data=snapshot.holdings,
            row_count=len(snapshot.holdings),
            warnings=[
                f"Portfolio analytics route used for portfolio '{snapshot.portfolio_name}'."
            ],
            column_info=[
                {"name": key, "type": type(value).__name__}
                for key, value in snapshot.holdings[0].items()
            ] if snapshot.holdings else [],
        )

        return {
            "portfolio_id": snapshot.portfolio_id,
            "question_type": question_type,
            "formatted_response": formatted,
            "sql_result": sql_result,
            "execution_result": execution_result,
            "snapshot": snapshot,
            "optimized": optimized,
        }

    def load_portfolio_snapshot(self, portfolio_id: Optional[str] = None) -> Optional[PortfolioSnapshot]:
        query = """
            WITH latest_prices AS (
                SELECT sp.symbol, sp.date, sp.close_price
                FROM stock_prices sp
                JOIN (
                    SELECT symbol, MAX(date) AS max_date
                    FROM stock_prices
                    GROUP BY symbol
                ) mx ON mx.symbol = sp.symbol AND mx.max_date = sp.date
            )
            SELECT p.id AS portfolio_id, p.name AS portfolio_name, p.description, p.cash_balance,
                   u.risk_tolerance, ph.symbol, ph.shares, ph.avg_cost_basis, ph.purchase_date,
                   s.company_name, s.sector, lp.date AS latest_price_date,
                   lp.close_price AS latest_close_price
            FROM portfolios p
            LEFT JOIN users u ON u.id = p.user_id
            LEFT JOIN portfolio_holdings ph ON ph.portfolio_id = p.id
            LEFT JOIN stocks s ON s.id = ph.stock_id
            LEFT JOIN latest_prices lp ON lp.symbol = ph.symbol
        """
        params: Dict[str, Any] = {}
        if portfolio_id:
            query += " WHERE p.id = :portfolio_id ORDER BY ph.symbol"
            params["portfolio_id"] = portfolio_id
        else:
            query += " WHERE p.is_default = TRUE ORDER BY ph.symbol LIMIT 100"

        with db_manager.get_connection() as conn:
            rows = conn.execute(text(query), params).mappings().all()

        if not rows:
            return None

        first = rows[0]
        holdings: List[Dict[str, Any]] = []
        total_equity = 0.0
        for row in rows:
            if not row.get("symbol"):
                continue
            current_price = float(row["latest_close_price"]) if row.get("latest_close_price") is not None else 0.0
            shares = float(row["shares"] or 0.0)
            avg_cost_basis = float(row["avg_cost_basis"] or 0.0)
            market_value = shares * current_price
            unrealized_gain_loss = shares * (current_price - avg_cost_basis)
            total_equity += market_value
            holdings.append(
                {
                    "symbol": row["symbol"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "shares": round(shares, 4),
                    "avg_cost_basis": round(avg_cost_basis, 2),
                    "current_price": round(current_price, 2),
                    "latest_price_date": row["latest_price_date"].isoformat() if row.get("latest_price_date") else None,
                    "market_value": round(market_value, 2),
                    "unrealized_gain_loss": round(unrealized_gain_loss, 2),
                    "purchase_date": row["purchase_date"].isoformat() if row.get("purchase_date") else None,
                }
            )

        cash_balance = float(first["cash_balance"] or 0.0)
        total_value = total_equity + cash_balance

        for holding in holdings:
            overall_weight = (holding["market_value"] / total_value * 100) if total_value else 0.0
            invested_weight = (holding["market_value"] / total_equity * 100) if total_equity else 0.0
            holding["weight_percentage"] = round(overall_weight, 2)
            holding["invested_weight_percentage"] = round(invested_weight, 2)

        holdings.sort(key=lambda item: item["market_value"], reverse=True)
        current_weights = {
            holding["symbol"]: holding["market_value"] / total_equity
            for holding in holdings
            if total_equity
        }
        sector_exposure = self._compute_sector_exposure(holdings, total_equity)
        metrics, top_risk_contributors = self._compute_metrics(holdings, current_weights)

        return PortfolioSnapshot(
            portfolio_id=str(first["portfolio_id"]),
            portfolio_name=first["portfolio_name"],
            description=first["description"],
            risk_tolerance=first["risk_tolerance"],
            cash_balance=round(cash_balance, 2),
            total_equity=round(total_equity, 2),
            total_value=round(total_value, 2),
            holdings=holdings,
            current_weights=current_weights,
            sector_exposure=sector_exposure,
            metrics=metrics,
            top_risk_contributors=top_risk_contributors,
        )

    def _compute_sector_exposure(
        self,
        holdings: List[Dict[str, Any]],
        total_equity: float,
    ) -> Dict[str, float]:
        sector_totals: Dict[str, float] = {}
        for holding in holdings:
            sector = holding.get("sector") or "Unknown"
            sector_totals[sector] = sector_totals.get(sector, 0.0) + float(holding["market_value"])
        return {
            sector: round(value / total_equity * 100, 2) if total_equity else 0.0
            for sector, value in sorted(sector_totals.items(), key=lambda item: item[1], reverse=True)
        }

    def _compute_metrics(
        self,
        holdings: List[Dict[str, Any]],
        current_weights: Dict[str, float],
    ) -> tuple[Dict[str, Optional[float]], List[Dict[str, float]]]:
        symbols = [holding["symbol"] for holding in holdings]
        returns = self.optimizer.get_returns_data(symbols=symbols)
        if returns.empty:
            return (
                {
                    "annual_return": None,
                    "annual_volatility": None,
                    "sharpe_ratio": None,
                    "sortino_ratio": None,
                    "max_drawdown": None,
                    "var_95": None,
                    "calmar_ratio": None,
                    "effective_holdings": self._effective_holdings(current_weights),
                },
                [],
            )

        aligned_symbols = [symbol for symbol in returns.columns if symbol in current_weights]
        weight_vector = np.array([current_weights[symbol] for symbol in aligned_symbols], dtype=float)
        weight_vector = weight_vector / weight_vector.sum()
        metrics = self.optimizer.calculate_portfolio_metrics(weight_vector, returns[aligned_symbols])
        metrics["effective_holdings"] = self._effective_holdings(
            {symbol: weight_vector[idx] for idx, symbol in enumerate(aligned_symbols)}
        )
        top_risk_contributors = self._risk_contributors(returns[aligned_symbols], weight_vector)
        return metrics, top_risk_contributors

    def _risk_contributors(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> List[Dict[str, float]]:
        cov_matrix = returns.cov().values * 252
        portfolio_variance = float(weights.T @ cov_matrix @ weights)
        if portfolio_variance <= 0:
            return []
        portfolio_volatility = sqrt(portfolio_variance)
        marginal = cov_matrix @ weights / portfolio_volatility
        contribution = weights * marginal
        total = contribution.sum()
        rows: List[Dict[str, float]] = []
        for symbol, value in zip(returns.columns, contribution):
            pct = (float(value) / total * 100) if total else 0.0
            rows.append({"symbol": symbol, "risk_contribution_pct": round(pct, 2)})
        rows.sort(key=lambda item: item["risk_contribution_pct"], reverse=True)
        return rows[:3]

    def _optimize_snapshot(
        self,
        snapshot: PortfolioSnapshot,
        user_query: str,
        question_type: str,
    ) -> Dict[str, Any]:
        symbols = [holding["symbol"] for holding in snapshot.holdings]
        returns = self.optimizer.get_returns_data(symbols=symbols)
        if returns.empty or len(returns.columns) < 2:
            return {}

        lowered = user_query.lower()
        if question_type == "rebalance" or "minimum volatility" in lowered or "min volatility" in lowered or "lower risk" in lowered:
            result = self.optimizer.optimize_portfolio(
                returns=returns[sorted(snapshot.current_weights.keys())],
                objective="min_risk",
                max_weight=0.25,
                min_weight=0.0,
            )
            result["optimization_goal"] = "min_risk_rebalance"
            current_top_weight = max((holding["invested_weight_percentage"] for holding in snapshot.holdings), default=0.0) / 100
            optimized_top_weight = max((result.get("weights") or {}).values(), default=0.0)
            current_vol = snapshot.metrics.get("annual_volatility")
            optimized_vol = result.get("metrics", {}).get("annual_volatility")
            concentration_improved = optimized_top_weight <= current_top_weight + 1e-9
            volatility_improved = (
                current_vol is None
                or optimized_vol is None
                or optimized_vol <= current_vol + 1e-9
            )
            result["meets_criteria"] = concentration_improved and volatility_improved
            result["current_top_weight"] = current_top_weight
            result["optimized_top_weight"] = optimized_top_weight
        else:
            risk_profile = self._extract_risk_profile(user_query, snapshot.risk_tolerance)
            result = self.optimizer.recommend_portfolio_for_risk_profile(
                risk_profile=risk_profile,
                symbols=symbols,
            )
            result["optimization_goal"] = "risk_profile_recommendation"

        if not result:
            return {}

        allocation = result.get("allocation") or result.get("weights") or {}
        normalized_allocation = {
            symbol: round(weight * 100, 2)
            for symbol, weight in sorted(allocation.items(), key=lambda item: item[1], reverse=True)
            if weight >= 0.01
        }
        result["normalized_allocation"] = normalized_allocation
        return result

    def _detect_question_type(self, user_query: str) -> str:
        lowered = user_query.lower()
        if "compare" in lowered and "optimized" in lowered:
            return "compare_optimized"
        if any(term in lowered for term in ("rebalance", "reduce concentration", "overweight", "underweight")):
            return "rebalance"
        if any(term in lowered for term in ("minimum volatility", "min volatility", "optimize", "optimized")):
            return "optimize"
        if any(term in lowered for term in ("diversified", "diversification", "sector", "exposure")):
            return "diversification"
        if any(term in lowered for term in ("volatility", "sharpe", "beta", "drawdown", "risk")):
            return "risk"
        if any(term in lowered for term in ("top holdings", "largest holdings", "allocation", "weight")):
            return "allocation"
        return "summary"

    def _extract_risk_profile(self, user_query: str, default: Optional[str]) -> str:
        lowered = user_query.lower()
        if "conservative" in lowered:
            return "conservative"
        if "aggressive" in lowered or "growth" in lowered:
            return "aggressive"
        if "moderate" in lowered:
            return "moderate"
        return default or "moderate"

    def _effective_holdings(self, weights: Dict[str, float]) -> float:
        if not weights:
            return 0.0
        return round(1.0 / sum(weight ** 2 for weight in weights.values()), 2)

    def _format_summary_answer(self, snapshot: PortfolioSnapshot) -> FormattedResponse:
        top = snapshot.holdings[0]
        total_gain_loss = round(sum(item["unrealized_gain_loss"] for item in snapshot.holdings), 2)
        return FormattedResponse(
            natural_language=(
                f"{snapshot.portfolio_name} is currently valued at ${snapshot.total_value:,.2f}, "
                f"with ${snapshot.total_equity:,.2f} invested across {len(snapshot.holdings)} holdings "
                f"and ${snapshot.cash_balance:,.2f} in cash."
            ),
            confidence=ConfidenceLevel.HIGH,
            key_insights=[
                f"Top holding: {top['symbol']} at {top['weight_percentage']:.2f}% of total portfolio value.",
                f"Total unrealized gain/loss across holdings is ${total_gain_loss:,.2f}.",
                f"Effective holdings score is {snapshot.metrics.get('effective_holdings', 0):.2f}.",
            ],
            recommendations=[
                "Ask about diversification, risk, or rebalancing to go deeper.",
            ],
            risk_warnings=self._concentration_warnings(snapshot),
            data_summary={
                "portfolio_name": snapshot.portfolio_name,
                "total_value": snapshot.total_value,
                "cash_balance": snapshot.cash_balance,
                "holdings_count": len(snapshot.holdings),
            },
            explanation="Summary uses live holdings, latest available prices, and current portfolio composition.",
            follow_up_suggestions=[
                "How diversified is my portfolio?",
                "What is my portfolio volatility?",
                "How should I rebalance to reduce concentration?",
            ],
            technical_details={"holdings": snapshot.holdings},
            disclaimer="Portfolio analytics are based on available historical market data and should not replace personalized financial advice.",
        )

    def _format_allocation_answer(self, snapshot: PortfolioSnapshot) -> FormattedResponse:
        top_holdings = snapshot.holdings[:4]
        return FormattedResponse(
            natural_language=(
                f"{snapshot.portfolio_name} is led by {top_holdings[0]['symbol']}, "
                f"and the top four holdings account for "
                f"{sum(item['weight_percentage'] for item in top_holdings):.2f}% of total portfolio value."
            ),
            confidence=ConfidenceLevel.HIGH,
            key_insights=[
                f"{item['symbol']}: ${item['market_value']:,.2f} ({item['weight_percentage']:.2f}% total, {item['invested_weight_percentage']:.2f}% invested)"
                for item in top_holdings
            ],
            recommendations=[
                "Use this view to spot position-size concentration before optimizing.",
            ],
            risk_warnings=self._concentration_warnings(snapshot),
            data_summary={"allocation": top_holdings},
            explanation="Allocation answer ranks holdings by live market value and current weight.",
            follow_up_suggestions=[
                "Which sector has the highest exposure?",
                "Which holdings are causing the most risk?",
            ],
            technical_details={"holdings": snapshot.holdings},
            disclaimer="Allocation analysis reflects the latest available prices in the database.",
        )

    def _format_diversification_answer(self, snapshot: PortfolioSnapshot) -> FormattedResponse:
        top_sector, top_sector_pct = next(iter(snapshot.sector_exposure.items()))
        diversification_quality = "well diversified"
        if top_sector_pct >= 55 or snapshot.metrics.get("effective_holdings", 0) < 4:
            diversification_quality = "highly concentrated"
        elif top_sector_pct >= 40 or snapshot.metrics.get("effective_holdings", 0) < 5:
            diversification_quality = "moderately concentrated"

        return FormattedResponse(
            natural_language=(
                f"{snapshot.portfolio_name} is {diversification_quality}. "
                f"The largest sector exposure is {top_sector} at {top_sector_pct:.2f}% of invested capital."
            ),
            confidence=ConfidenceLevel.HIGH,
            key_insights=[
                f"Effective holdings score: {snapshot.metrics.get('effective_holdings', 0):.2f}.",
                *[
                    f"{sector}: {weight:.2f}% of invested capital."
                    for sector, weight in list(snapshot.sector_exposure.items())[:3]
                ],
            ],
            recommendations=self._diversification_recommendations(snapshot),
            risk_warnings=self._concentration_warnings(snapshot),
            data_summary={"sector_exposure": snapshot.sector_exposure},
            explanation="Diversification is measured using effective holdings and sector concentration.",
            follow_up_suggestions=[
                "How should I rebalance to reduce concentration?",
                "Compare my portfolio with an optimized one",
            ],
            technical_details={"sector_exposure": snapshot.sector_exposure},
            disclaimer="Diversification analysis does not account for off-platform assets or liabilities.",
        )

    def _format_risk_answer(self, snapshot: PortfolioSnapshot) -> FormattedResponse:
        metrics = snapshot.metrics
        return FormattedResponse(
            natural_language=(
                f"{snapshot.portfolio_name} currently shows annualized volatility of "
                f"{(metrics.get('annual_volatility') or 0) * 100:.2f}% and a Sharpe ratio of "
                f"{metrics.get('sharpe_ratio') or 0:.2f}."
            ),
            confidence=ConfidenceLevel.MEDIUM if metrics.get("annual_volatility") is None else ConfidenceLevel.HIGH,
            key_insights=[
                f"Annual return: {(metrics.get('annual_return') or 0) * 100:.2f}%",
                f"Annual volatility: {(metrics.get('annual_volatility') or 0) * 100:.2f}%",
                f"Max drawdown: {(metrics.get('max_drawdown') or 0) * 100:.2f}%",
                *[
                    f"{item['symbol']} contributes about {item['risk_contribution_pct']:.2f}% of portfolio risk."
                    for item in snapshot.top_risk_contributors
                ],
            ],
            recommendations=[
                "If you want smoother returns, ask for a lower-risk rebalance or minimum-volatility version.",
            ],
            risk_warnings=self._concentration_warnings(snapshot),
            data_summary=metrics,
            explanation="Risk metrics are computed from 1-year historical returns of the current holdings.",
            follow_up_suggestions=[
                "Which holdings are causing the most risk?",
                "Can you suggest a minimum-volatility version of my portfolio?",
            ],
            technical_details={"risk_metrics": metrics, "top_risk_contributors": snapshot.top_risk_contributors},
            disclaimer="Risk estimates come from historical returns and can change materially in new market regimes.",
        )

    def _format_rebalance_answer(self, snapshot: PortfolioSnapshot, optimized: Dict[str, Any]) -> FormattedResponse:
        trades = self._rebalance_trades(snapshot, optimized)
        if not trades:
            recommendations = ["Current allocation is already close to the optimized mix for the available holdings."]
        else:
            recommendations = trades[:5]
        confidence = self._optimization_confidence(optimized, prefer_high=True)
        risk_warnings = self._concentration_warnings(snapshot)
        optimization_gap = self._optimization_gap_warning(optimized)
        if optimization_gap:
            risk_warnings = [*risk_warnings, optimization_gap]
        return FormattedResponse(
            natural_language=(
                f"A lower-concentration version of {snapshot.portfolio_name} would trim oversized positions "
                "and redistribute weight toward a more balanced mix across the existing holdings."
            ),
            confidence=confidence,
            key_insights=[
                f"Current top position weight is {snapshot.holdings[0]['weight_percentage']:.2f}% of total value.",
                *[
                    f"{item['symbol']} contributes about {item['risk_contribution_pct']:.2f}% of total risk."
                    for item in snapshot.top_risk_contributors[:2]
                ],
            ],
            recommendations=recommendations,
            risk_warnings=risk_warnings,
            data_summary={"optimized_allocation": optimized.get("normalized_allocation", {})},
            explanation="Rebalancing compares the current live allocation with a minimum-risk target mix built from the existing holdings universe.",
            follow_up_suggestions=[
                "Compare my current portfolio with an optimized portfolio",
                "Can you suggest a minimum-volatility version of my portfolio?",
            ],
            technical_details={"optimized": optimized},
            disclaimer="Rebalancing suggestions are model-based and should be reviewed for taxes, liquidity, and personal constraints.",
        )

    def _format_optimized_answer(self, snapshot: PortfolioSnapshot, optimized: Dict[str, Any]) -> FormattedResponse:
        allocation = optimized.get("normalized_allocation", {})
        top_allocations = list(allocation.items())[:5]
        confidence = self._optimization_confidence(optimized, prefer_high=False)
        risk_warnings = self._concentration_warnings(snapshot)
        optimization_gap = self._optimization_gap_warning(optimized)
        if optimization_gap:
            risk_warnings = [*risk_warnings, optimization_gap]
        return FormattedResponse(
            natural_language=(
                f"I generated an optimized portfolio for the current holdings universe in {snapshot.portfolio_name}. "
                "The recommended mix reduces concentration and improves risk-adjusted balance based on historical returns."
            ),
            confidence=confidence,
            key_insights=[
                f"{symbol}: target weight {weight:.2f}%"
                for symbol, weight in top_allocations
            ] or ["Optimization could not improve materially on the current mix."],
            recommendations=self._rebalance_trades(snapshot, optimized)[:5] or [
                "Current allocation is already close to the optimizer's preferred mix."
            ],
            risk_warnings=risk_warnings,
            data_summary={"optimized_metrics": optimized.get("metrics", {}), "optimized_allocation": allocation},
            explanation="Optimization uses the existing holdings universe and 1-year historical returns.",
            follow_up_suggestions=[
                "Compare my current portfolio with an optimized portfolio",
                "How diversified is my portfolio?",
            ],
            technical_details={"optimized": optimized},
            disclaimer="Optimization outputs depend on historical return assumptions and do not guarantee future outcomes.",
        )

    def _format_compare_answer(self, snapshot: PortfolioSnapshot, optimized: Dict[str, Any]) -> FormattedResponse:
        current = snapshot.metrics
        target = optimized.get("metrics", {})
        confidence = self._optimization_confidence(optimized, prefer_high=False)
        risk_warnings = self._concentration_warnings(snapshot)
        optimization_gap = self._optimization_gap_warning(optimized)
        if optimization_gap:
            risk_warnings = [*risk_warnings, optimization_gap]
        return FormattedResponse(
            natural_language=(
                f"Compared with the current allocation, the optimized version of {snapshot.portfolio_name} "
                "aims to lower volatility and improve the portfolio's efficiency."
            ),
            confidence=confidence,
            key_insights=[
                f"Current volatility: {(current.get('annual_volatility') or 0) * 100:.2f}% vs optimized {(target.get('annual_volatility') or 0) * 100:.2f}%",
                f"Current Sharpe: {current.get('sharpe_ratio') or 0:.2f} vs optimized {target.get('sharpe_ratio') or 0:.2f}",
                f"Current annual return: {(current.get('annual_return') or 0) * 100:.2f}% vs optimized {(target.get('annual_return') or 0) * 100:.2f}%",
            ],
            recommendations=self._rebalance_trades(snapshot, optimized)[:5] or [
                "No large allocation changes are required based on the optimizer."
            ],
            risk_warnings=risk_warnings,
            data_summary={"current_metrics": current, "optimized_metrics": target},
            explanation="Comparison uses the live current allocation and a model-generated optimized allocation over the same holdings universe.",
            follow_up_suggestions=[
                "How should I rebalance to reduce concentration?",
                "Which holdings are causing the most risk?",
            ],
            technical_details={"current_metrics": current, "optimized": optimized},
            disclaimer="Optimized comparisons are scenario-based and should be reviewed before executing trades.",
        )

    def _diversification_recommendations(self, snapshot: PortfolioSnapshot) -> List[str]:
        top_sector, top_sector_pct = next(iter(snapshot.sector_exposure.items()))
        if top_sector_pct >= 50:
            return [
                f"Reduce {top_sector} exposure by trimming the top position and increasing lower-correlated holdings.",
                "Consider redistributing part of the portfolio into Healthcare, Energy, or Financials if that matches your mandate.",
            ]
        return ["Diversification is reasonable, but you can still improve balance by trimming oversized positions above 20%."]

    def _concentration_warnings(self, snapshot: PortfolioSnapshot) -> List[str]:
        warnings: List[str] = []
        top_holding = snapshot.holdings[0]
        if top_holding["weight_percentage"] >= 20:
            warnings.append(f"{top_holding['symbol']} represents {top_holding['weight_percentage']:.2f}% of total portfolio value.")
        top_sector, top_sector_pct = next(iter(snapshot.sector_exposure.items()))
        if top_sector_pct >= 45:
            warnings.append(f"{top_sector} exposure is concentrated at {top_sector_pct:.2f}% of invested capital.")
        if snapshot.metrics.get("max_drawdown") is not None and snapshot.metrics["max_drawdown"] < -0.2:
            warnings.append(f"Historical max drawdown is {(snapshot.metrics['max_drawdown']) * 100:.2f}%.")
        return warnings

    def _rebalance_trades(self, snapshot: PortfolioSnapshot, optimized: Dict[str, Any]) -> List[str]:
        allocation = optimized.get("normalized_allocation", {})
        if not allocation:
            return []
        current = {holding["symbol"]: holding["invested_weight_percentage"] for holding in snapshot.holdings}
        trades: List[str] = []
        for symbol, target_weight in allocation.items():
            current_weight = current.get(symbol, 0.0)
            delta = round(target_weight - current_weight, 2)
            if abs(delta) < 3:
                continue
            action = "Increase" if delta > 0 else "Trim"
            trades.append(f"{action} {symbol} by about {abs(delta):.2f} percentage points toward a {target_weight:.2f}% target weight.")
        return trades

    def _optimization_confidence(
        self,
        optimized: Dict[str, Any],
        *,
        prefer_high: bool,
    ) -> ConfidenceLevel:
        if not optimized:
            return ConfidenceLevel.MEDIUM
        meets_criteria = optimized.get("meets_criteria")
        if meets_criteria is False:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.HIGH if prefer_high else ConfidenceLevel.MEDIUM

    def _optimization_gap_warning(self, optimized: Dict[str, Any]) -> Optional[str]:
        if not optimized:
            return None
        target_vol = optimized.get("target_volatility")
        actual_vol = optimized.get("actual_volatility")
        if target_vol is not None and actual_vol is not None and actual_vol > target_vol * 1.2:
            return (
                f"The optimized mix still implies annualized volatility of {actual_vol * 100:.2f}%, "
                f"which is above the target of {target_vol * 100:.2f}%."
            )
        current_top = optimized.get("current_top_weight")
        optimized_top = optimized.get("optimized_top_weight")
        if current_top is not None and optimized_top is not None and optimized_top >= current_top:
            return (
                f"The optimized mix does not materially reduce top-position concentration "
                f"({current_top * 100:.2f}% vs {optimized_top * 100:.2f}%)."
            )
        return None
