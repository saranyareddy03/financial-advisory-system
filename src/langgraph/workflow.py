"""
LangGraph Workflow Orchestrator
Real-Time Financial Advisory System - Phase 5

This module implements the core conversational workflow that processes
natural language financial queries and coordinates the NLP-SQL pipeline.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
import json

from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor # Not used and causing import error
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Import our custom modules (will be created in subsequent components)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.settings import config as settings
from src.database.connection import db_manager
from src.langgraph.advisory_service import FinancialAdvisoryService


class ConversationState(TypedDict):
    """State management for the conversational workflow"""
    messages: List[Any]
    user_query: str
    intent: Optional[str]
    parsed_entities: Dict[str, Any]
    sql_query: Optional[str]
    query_results: Optional[List[Dict]]
    response: Optional[str]
    error_message: Optional[str]
    user_id: Optional[str]
    session_id: str
    timestamp: datetime


@dataclass
class WorkflowResponse:
    """Standardized response format"""
    success: bool
    response: str
    data: Optional[Dict] = None
    query_executed: Optional[str] = None
    execution_time: Optional[float] = None
    error_details: Optional[str] = None


class FinancialAdvisoryWorkflow:
    """
    Main LangGraph workflow for financial advisory conversations.
    Orchestrates the flow from natural language query to structured response.
    """
    
    def __init__(self):
        self.settings = settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=self.settings.GEMINI_API_KEY,
            temperature=0.1,
            max_tokens=1000
        )
        self.service = FinancialAdvisoryService()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> Any:
        """Build the LangGraph state machine for conversational flow"""
        
        # Define workflow nodes
        workflow = StateGraph(ConversationState)
        
        # Add nodes for each step in the conversation pipeline
        workflow.add_node("intent_detector", self._detect_intent)
        workflow.add_node("entity_extractor", self._extract_entities)
        workflow.add_node("sql_generator", self._generate_sql)
        workflow.add_node("query_executor", self._execute_query)
        workflow.add_node("response_formatter", self._format_response)
        workflow.add_node("error_handler", self._handle_error)
        
        # Define the conversation flow
        workflow.set_entry_point("intent_detector")
        
        # Routing logic between nodes
        workflow.add_conditional_edges(
            "intent_detector",
            self._route_after_intent,
            {
                "extract_entities": "entity_extractor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "entity_extractor",
            self._route_after_extraction,
            {
                "generate_sql": "sql_generator",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "sql_generator",
            self._route_after_sql,
            {
                "execute_query": "query_executor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "query_executor",
            self._route_after_execution,
            {
                "format_response": "response_formatter",
                "error": "error_handler"
            }
        )
        
        # All paths lead to END
        workflow.add_edge("response_formatter", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def _detect_intent(self, state: ConversationState) -> ConversationState:
        """
        Node 1: Analyze user query and detect financial advisory intent
        """
        try:
            user_query = state["user_query"]
            
            intent_prompt = f"""
            You are a financial advisory intent classifier. Analyze this query and determine the user's intent.
            
            Query: "{user_query}"
            
            Possible intents:
            - stock_analysis: Asking about individual stock performance, prices, trends
            - portfolio_review: Questions about portfolio composition, performance, risk
            - sentiment_analysis: Questions about news sentiment, market sentiment
            - technical_analysis: Questions about technical indicators, charts, patterns
            - risk_assessment: Questions about risk metrics, volatility, drawdown
            - general_inquiry: General questions about markets, definitions
            - unsupported: Queries outside financial scope
            
            Respond with just the intent category.
            """
            
            response = await self.llm.ainvoke([
                SystemMessage(content=intent_prompt),
                HumanMessage(content=user_query)
            ])
            
            # Handle response content correctly (it might be a string or list of content parts)
            content = response.content
            if isinstance(content, list):
                # If it's a list, join the text parts or take the first one
                content = " ".join([part if isinstance(part, str) else str(part) for part in content])
            
            intent = content.strip().lower()
            
            state["intent"] = intent
            state["messages"].append(AIMessage(content=f"Intent detected: {intent}"))
            
            return state
            
        except Exception as e:
            state["error_message"] = f"Intent detection failed: {str(e)}"
            return state
    
    async def _extract_entities(self, state: ConversationState) -> ConversationState:
        """
        Node 2: Extract relevant financial entities from the query
        """
        try:
            user_query = state["user_query"]
            intent = state["intent"]
            
            entity_prompt = f"""
            Extract financial entities from this query based on the detected intent.
            
            Query: "{user_query}"
            Intent: {intent}
            
            Extract these entities as JSON:
            {{
                "stocks": ["SYMBOL1", "SYMBOL2"],  // Stock symbols mentioned
                "time_period": "1d/1w/1m/3m/6m/1y",  // Time frame if mentioned
                "metrics": ["price", "volume", "rsi", "sentiment"],  // Specific metrics
                "comparison_type": "absolute/relative/benchmark",  // Type of analysis
                "risk_level": "low/medium/high",  // Risk preference if mentioned
                "portfolio_id": null  // Portfolio identifier if applicable
            }}
            
            Only include entities that are explicitly mentioned or clearly implied.
            """
            
            response = await self.llm.ainvoke([
                SystemMessage(content=entity_prompt),
                HumanMessage(content=user_query)
            ])
            
            # Parse the JSON response
            try:
                # Need to handle potential markdown formatting from LLM (```json ... ```)
                content = response.content
                if isinstance(content, list):
                    content = " ".join([part if isinstance(part, str) else str(part) for part in content])
                
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                
                entities = json.loads(content.strip())
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                entities = {
                    "stocks": [],
                    "time_period": "1m",
                    "metrics": [],
                    "comparison_type": "absolute",
                    "risk_level": "medium",
                    "portfolio_id": None
                }
            
            state["parsed_entities"] = entities
            state["messages"].append(AIMessage(content=f"Entities extracted: {entities}"))
            
            return state
            
        except Exception as e:
            state["error_message"] = f"Entity extraction failed: {str(e)}"
            return state
    
    async def _generate_sql(self, state: ConversationState) -> ConversationState:
        """
        Node 3: Generate SQL query based on intent and entities
        Will be implemented in Component 2: SQL Query Generator
        """
        # Placeholder - will be implemented in next component
        state["sql_query"] = "-- SQL generation pending Component 2"
        return state
    
    async def _execute_query(self, state: ConversationState) -> ConversationState:
        """
        Node 4: Execute SQL query against database
        Will be implemented in Component 3: Query Executor
        """
        # Placeholder - will be implemented in next component
        state["query_results"] = []
        return state
    
    async def _format_response(self, state: ConversationState) -> ConversationState:
        """
        Node 5: Format query results into natural language response
        Will be implemented in Component 4: Response Formatter
        """
        # Placeholder - will be implemented in next component
        state["response"] = "Response formatting pending Component 4"
        return state
    
    async def _handle_error(self, state: ConversationState) -> ConversationState:
        """Error handling node"""
        error_msg = state.get("error_message", "Unknown error occurred")
        state["response"] = f"I encountered an issue: {error_msg}. Please try rephrasing your question."
        return state
    
    # Routing functions for conditional edges
    def _route_after_intent(self, state: ConversationState) -> str:
        """Route after intent detection"""
        if state.get("error_message"):
            return "error"
        if state.get("intent") == "unsupported":
            return "error"
        return "extract_entities"
    
    def _route_after_extraction(self, state: ConversationState) -> str:
        """Route after entity extraction"""
        if state.get("error_message"):
            return "error"
        return "generate_sql"
    
    def _route_after_sql(self, state: ConversationState) -> str:
        """Route after SQL generation"""
        if state.get("error_message") or not state.get("sql_query"):
            return "error"
        return "execute_query"
    
    def _route_after_execution(self, state: ConversationState) -> str:
        """Route after query execution"""
        if state.get("error_message"):
            return "error"
        return "format_response"
    
    async def process_query(
        self, 
        user_query: str, 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> WorkflowResponse:
        """
        Main entry point for processing user queries
        """
        start_time = datetime.now()
        
        try:
            result = await self.service.process_query(user_query)
            execution_time = (datetime.now() - start_time).total_seconds()

            final_state: ConversationState = {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "intent": result.intent,
                "parsed_entities": result.entities,
                "sql_query": result.sql_result.sql,
                "query_results": result.execution_result.data or [],
                "response": result.formatted_response.natural_language,
                "error_message": None,
                "user_id": user_id,
                "session_id": session_id or f"session_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now(),
            }
            await self._log_user_query(final_state, execution_time)

            return WorkflowResponse(
                success=True,
                response=result.formatted_response.natural_language,
                data={
                    "intent": result.intent,
                    "entities": result.entities,
                    "results": result.execution_result.data,
                    "route": result.route,
                    "source_path": result.source_path,
                },
                query_executed=result.sql_result.sql,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return WorkflowResponse(
                success=False,
                response="I apologize, but I encountered an unexpected error. Please try again.",
                error_details=str(e),
                execution_time=execution_time
            )
    
    async def _log_user_query(self, state: ConversationState, execution_time: float):
        """Log user queries for analytics and system improvement"""
        from sqlalchemy import text as sql_text
        try:
            with db_manager.get_session() as session:
                # Extract stocks for the array column
                entities = state.get("parsed_entities", {})
                stocks = entities.get("stocks", []) if isinstance(entities, dict) else []
                
                log_entry = {
                    "user_id": state.get("user_id"),
                    "session_id": state.get("session_id"),
                    "query_text": state.get("user_query"),
                    "intent": state.get("intent"),
                    "extracted_symbols": stocks,
                    "sql_generated": state.get("sql_query"),
                    "response_text": state.get("response"),
                    "response_time_ms": int(execution_time * 1000),
                    "timestamp": state.get("timestamp")
                }
                
                # Insert into user_queries table matching the actual schema:
                # id (uuid), user_id (uuid), response_time_ms (int), created_at (ts), 
                # intent (str), extracted_symbols (arr), sql_generated (text), 
                # response_text (text), session_id (str), query_text (text)
                insert_query = sql_text("""
                    INSERT INTO user_queries (
                        user_id, session_id, query_text, intent, extracted_symbols,
                        sql_generated, response_text, response_time_ms, created_at
                    ) VALUES (
                        :user_id, :session_id, :query_text, :intent, :extracted_symbols,
                        :sql_generated, :response_text, :response_time_ms, :timestamp
                    )
                """)
                
                session.execute(insert_query, log_entry)
                session.commit()
                
        except Exception as e:
            # Don't fail the main query if logging fails
            print(f"Warning: Query logging failed: {e}")


# Convenience function for easy workflow usage
async def process_financial_query(
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> WorkflowResponse:
    """
    Convenience function to process a financial query through the LangGraph workflow
    """
    workflow = FinancialAdvisoryWorkflow()
    return await workflow.process_query(query, user_id, session_id)


if __name__ == "__main__":
    # Example usage for testing
    async def main():
        response = await process_financial_query("What's the current price of AAPL?")
        print(f"Success: {response.success}")
        print(f"Response: {response.response}")
        print(f"Execution Time: {response.execution_time}s")
    
    # Run test
    asyncio.run(main())
