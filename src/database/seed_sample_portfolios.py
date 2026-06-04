"""
Seed a demo user and demo portfolio for portfolio optimization workflows.
"""

from __future__ import annotations

from datetime import date, datetime
import uuid

from sqlalchemy import text

from src.database.connection import db_manager


DEMO_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "financial-advisory-demo-user"))
DEMO_PORTFOLIO_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "financial-advisory-demo-portfolio"))

DEMO_HOLDINGS = [
    {"symbol": "MSFT", "shares": 50.0, "avg_cost_basis": 390.00, "purchase_date": date(2024, 3, 15)},
    {"symbol": "NVDA", "shares": 120.0, "avg_cost_basis": 140.00, "purchase_date": date(2024, 6, 20)},
    {"symbol": "GOOG", "shares": 45.0, "avg_cost_basis": 180.00, "purchase_date": date(2024, 2, 12)},
    {"symbol": "TSLA", "shares": 18.0, "avg_cost_basis": 300.00, "purchase_date": date(2024, 1, 18)},
    {"symbol": "ABBV", "shares": 30.0, "avg_cost_basis": 185.00, "purchase_date": date(2024, 5, 10)},
    {"symbol": "CVX", "shares": 25.0, "avg_cost_basis": 155.00, "purchase_date": date(2024, 4, 8)},
    {"symbol": "V", "shares": 15.0, "avg_cost_basis": 290.00, "purchase_date": date(2024, 7, 1)},
    {"symbol": "AMZN", "shares": 25.0, "avg_cost_basis": 210.00, "purchase_date": date(2024, 8, 14)},
]


def seed_demo_portfolio() -> dict:
    with db_manager.get_session() as session:
        latest_price_rows = session.execute(
            text(
                """
                SELECT sp.symbol, st.id AS stock_id, sp.close_price, sp.date
                FROM stock_prices sp
                JOIN stocks st ON st.symbol = sp.symbol
                JOIN (
                    SELECT symbol, MAX(date) AS max_date
                    FROM stock_prices
                    GROUP BY symbol
                ) mx ON mx.symbol = sp.symbol AND mx.max_date = sp.date
                WHERE sp.symbol = ANY(:symbols)
                """
            ),
            {"symbols": [holding["symbol"] for holding in DEMO_HOLDINGS]},
        ).mappings().all()
        latest_by_symbol = {row["symbol"]: row for row in latest_price_rows}

        if len(latest_by_symbol) != len(DEMO_HOLDINGS):
            missing = sorted({holding["symbol"] for holding in DEMO_HOLDINGS} - set(latest_by_symbol))
            raise ValueError(f"Missing latest price data for symbols: {missing}")

        session.execute(
            text(
                """
                INSERT INTO users (
                    id, email, username, full_name, risk_tolerance, investment_horizon,
                    age_group, annual_income_range, investment_experience,
                    preferred_sectors, created_at, updated_at, last_login, is_active
                ) VALUES (
                    :id, :email, :username, :full_name, :risk_tolerance, :investment_horizon,
                    :age_group, :annual_income_range, :investment_experience,
                    :preferred_sectors, :created_at, :updated_at, :last_login, :is_active
                )
                ON CONFLICT (id) DO UPDATE SET
                    email = EXCLUDED.email,
                    username = EXCLUDED.username,
                    full_name = EXCLUDED.full_name,
                    risk_tolerance = EXCLUDED.risk_tolerance,
                    updated_at = EXCLUDED.updated_at,
                    last_login = EXCLUDED.last_login,
                    is_active = EXCLUDED.is_active
                """
            ),
            {
                "id": DEMO_USER_ID,
                "email": "demo.portfolio@advisor.local",
                "username": "demo_portfolio_user",
                "full_name": "Demo Portfolio User",
                "risk_tolerance": "moderate",
                "investment_horizon": "long_term",
                "age_group": "35_44",
                "annual_income_range": "100k_250k",
                "investment_experience": "intermediate",
                "preferred_sectors": ["Technology", "Healthcare", "Financial Services"],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "is_active": True,
            },
        )

        holding_rows = []
        total_equity = 0.0
        for index, holding in enumerate(DEMO_HOLDINGS, start=1):
            latest = latest_by_symbol[holding["symbol"]]
            current_price = float(latest["close_price"])
            market_value = holding["shares"] * current_price
            total_equity += market_value
            holding_rows.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"financial-advisory-demo-holding-{index}-{holding['symbol']}")),
                    "portfolio_id": DEMO_PORTFOLIO_ID,
                    "stock_id": str(latest["stock_id"]),
                    "symbol": holding["symbol"],
                    "shares": holding["shares"],
                    "avg_cost_basis": holding["avg_cost_basis"],
                    "current_price": round(current_price, 2),
                    "market_value": round(market_value, 2),
                    "unrealized_gain_loss": round((current_price - holding["avg_cost_basis"]) * holding["shares"], 2),
                    "purchase_date": holding["purchase_date"],
                }
            )

        cash_balance = 10000.00
        total_value = round(total_equity + cash_balance, 2)
        for row in holding_rows:
            row["weight_percentage"] = round(row["market_value"] / total_value * 100, 2)
            row["last_updated"] = datetime.utcnow()

        session.execute(
            text(
                """
                INSERT INTO portfolios (
                    id, user_id, name, description, total_value, cash_balance,
                    is_default, created_at, updated_at
                ) VALUES (
                    :id, :user_id, :name, :description, :total_value, :cash_balance,
                    :is_default, :created_at, :updated_at
                )
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    total_value = EXCLUDED.total_value,
                    cash_balance = EXCLUDED.cash_balance,
                    is_default = EXCLUDED.is_default,
                    updated_at = EXCLUDED.updated_at
                """
            ),
            {
                "id": DEMO_PORTFOLIO_ID,
                "user_id": DEMO_USER_ID,
                "name": "Demo Balanced Growth Portfolio",
                "description": "Seeded sample portfolio for portfolio optimization, diversification, and rebalancing demos.",
                "total_value": total_value,
                "cash_balance": cash_balance,
                "is_default": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        )

        for row in holding_rows:
            session.execute(
                text(
                    """
                    INSERT INTO portfolio_holdings (
                        id, portfolio_id, stock_id, symbol, shares, avg_cost_basis,
                        current_price, market_value, unrealized_gain_loss,
                        weight_percentage, purchase_date, last_updated
                    ) VALUES (
                        :id, :portfolio_id, :stock_id, :symbol, :shares, :avg_cost_basis,
                        :current_price, :market_value, :unrealized_gain_loss,
                        :weight_percentage, :purchase_date, :last_updated
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        shares = EXCLUDED.shares,
                        avg_cost_basis = EXCLUDED.avg_cost_basis,
                        current_price = EXCLUDED.current_price,
                        market_value = EXCLUDED.market_value,
                        unrealized_gain_loss = EXCLUDED.unrealized_gain_loss,
                        weight_percentage = EXCLUDED.weight_percentage,
                        purchase_date = EXCLUDED.purchase_date,
                        last_updated = EXCLUDED.last_updated
                    """
                ),
                row,
            )

    return {
        "user_id": DEMO_USER_ID,
        "portfolio_id": DEMO_PORTFOLIO_ID,
        "portfolio_name": "Demo Balanced Growth Portfolio",
        "holdings_count": len(holding_rows),
        "total_value": total_value,
        "cash_balance": cash_balance,
    }


if __name__ == "__main__":
    result = seed_demo_portfolio()
    print(result)
