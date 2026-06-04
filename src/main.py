import asyncio
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.langgraph.advisory_service import FinancialAdvisoryService
from src.langgraph.response_formatter import ResponseStyle


async def process_user_query(service: FinancialAdvisoryService, user_query: str) -> None:
    """Process one user query through the hybrid advisory pipeline."""
    result = await service.process_query(
        user_query=user_query,
        style=ResponseStyle.CONVERSATIONAL,
    )

    print(f"\nQuery: {user_query}")
    print(f"Intent: {result.intent}")
    print(f"Route: {result.route}")
    print(f"Source Path: {result.source_path}")

    if result.sql_result.sql:
        print(f"SQL: {result.sql_result.sql}")
    else:
        print("SQL: no SQL executed for the final answer")

    print("\nAnswer:")
    print(result.formatted_response.natural_language)

    if result.formatted_response.key_insights:
        print("\nKey insights:")
        for insight in result.formatted_response.key_insights:
            print(f"- {insight}")

    if result.execution_result.warnings:
        print("\nWarnings:")
        for warning in result.execution_result.warnings:
            print(f"- {warning}")


async def main() -> None:
    """Simple CLI loop for ad-hoc financial queries."""
    service = FinancialAdvisoryService()

    print("Financial Advisory System")
    print("Type a financial question, or 'quit' to exit.")

    while True:
        user_query = input("\n> ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"quit", "exit"}:
            break
        await process_user_query(service, user_query)


if __name__ == "__main__":
    asyncio.run(main())
