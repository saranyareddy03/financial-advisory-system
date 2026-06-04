"""
Response Formatter
Real-Time Financial Advisory System - Phase 5 Component 4

This module implements intelligent response formatting that converts structured database
results into natural, conversational financial advice with explainable reasoning.
"""

import asyncio
import json
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import re

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Custom JSON encoder to handle Decimal and Date objects
class FinancialJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super(FinancialJsonEncoder, self).default(obj)

def safe_json_dumps(data: Any, indent: Optional[int] = None) -> str:
    """Safely dump financial data to JSON string"""
    return json.dumps(data, cls=FinancialJsonEncoder, indent=indent)

# Import our modules
from src.config.settings import config as settings
from src.langgraph.sql_generator import SQLQueryResult, QueryComplexity
from src.langgraph.query_executor import QueryExecutionResult, ExecutionStatus


class ResponseStyle(Enum):
    """Response formatting styles"""
    CONVERSATIONAL = "conversational"     # Natural, friendly responses
    ANALYTICAL = "analytical"             # Data-focused, technical
    ADVISORY = "advisory"                 # Investment recommendations
    EDUCATIONAL = "educational"           # Learning-focused explanations
    EXECUTIVE = "executive"               # Brief, high-level summaries


class ConfidenceLevel(Enum):
    """Confidence levels for financial advice"""
    HIGH = "high"           # Strong data support, clear trends
    MEDIUM = "medium"       # Good data, some uncertainty
    LOW = "low"            # Limited data, high uncertainty
    INSUFFICIENT = "insufficient"  # Not enough data for advice


@dataclass
class FormattedResponse:
    """Comprehensive formatted response structure"""
    natural_language: str
    confidence: ConfidenceLevel
    key_insights: List[str]
    recommendations: List[str]
    risk_warnings: List[str]
    data_summary: Dict[str, Any]
    explanation: str
    follow_up_suggestions: List[str]
    technical_details: Optional[Dict[str, Any]] = None
    disclaimer: Optional[str] = None


class FinancialDomainKnowledge:
    """
    Financial domain knowledge and interpretation guidelines
    """
    
    @staticmethod
    def get_indicator_interpretations() -> Dict[str, Dict[str, Any]]:
        """Technical indicator interpretation guidelines"""
        return {
            "rsi_14": {
                "name": "RSI (14-day)",
                "ranges": {
                    "oversold": {"min": 0, "max": 30, "signal": "potential buy opportunity"},
                    "neutral": {"min": 30, "max": 70, "signal": "normal trading range"},
                    "overbought": {"min": 70, "max": 100, "signal": "potential sell signal"}
                },
                "description": "Measures momentum, indicates if stock is overbought or oversold"
            },
            "macd_line": {
                "name": "MACD Line",
                "interpretation": "crossing above signal line suggests uptrend, below suggests downtrend",
                "description": "Moving Average Convergence Divergence - trend and momentum indicator"
            },
            "macd_signal": {
                "name": "MACD Signal Line",
                "interpretation": "compare with MACD line for buy/sell signals",
                "description": "Smoothed version of MACD line"
            },
            "volatility_30d": {
                "name": "30-Day Volatility",
                "ranges": {
                    "low": {"min": 0, "max": 15, "signal": "stable, lower risk"},
                    "moderate": {"min": 15, "max": 30, "signal": "normal market volatility"},
                    "high": {"min": 30, "max": float('inf'), "signal": "higher risk, more unpredictable"}
                },
                "description": "Measures price fluctuation over 30 days"
            },
            "beta": {
                "name": "Beta",
                "ranges": {
                    "defensive": {"min": 0, "max": 1, "signal": "less volatile than market"},
                    "market": {"min": 0.9, "max": 1.1, "signal": "moves with market"},
                    "aggressive": {"min": 1.1, "max": float('inf'), "signal": "more volatile than market"}
                },
                "description": "Measures stock's volatility relative to overall market"
            },
            "sharpe_ratio": {
                "name": "Sharpe Ratio",
                "ranges": {
                    "poor": {"min": float('-inf'), "max": 1, "signal": "poor risk-adjusted returns"},
                    "good": {"min": 1, "max": 2, "signal": "good risk-adjusted returns"},
                    "excellent": {"min": 2, "max": float('inf'), "signal": "excellent risk-adjusted returns"}
                },
                "description": "Risk-adjusted return measure"
            },
            "sentiment_score": {
                "name": "News Sentiment",
                "ranges": {
                    "negative": {"min": -1, "max": -0.1, "signal": "negative market sentiment"},
                    "neutral": {"min": -0.1, "max": 0.1, "signal": "neutral market sentiment"},
                    "positive": {"min": 0.1, "max": 1, "signal": "positive market sentiment"}
                },
                "description": "AI-analyzed news sentiment from financial media"
            }
        }
    
    @staticmethod
    def get_risk_assessments() -> Dict[str, str]:
        """Risk level assessments and descriptions"""
        return {
            "low": "Conservative investment with lower volatility and stable returns",
            "medium": "Balanced risk-reward profile suitable for moderate investors",
            "high": "Higher volatility investment suitable for risk-tolerant investors",
            "very_high": "Speculative investment with significant risk of loss"
        }
    
    @staticmethod
    def get_investment_disclaimers() -> Dict[str, str]:
        """Standard investment disclaimers by advice type"""
        return {
            "general": "This analysis is for educational purposes only and not personalized financial advice. Consult a qualified financial advisor for investment decisions.",
            "stock_analysis": "Stock analysis based on historical data and current metrics. Past performance does not guarantee future results.",
            "risk_assessment": "Risk metrics are estimates based on historical data. Actual investment risk may vary significantly.",
            "sentiment_analysis": "Sentiment analysis reflects current market mood but may change rapidly with new information.",
            "portfolio_advice": "Portfolio recommendations require consideration of your personal financial situation, goals, and risk tolerance."
        }


class FinancialResponseFormatter:
    """
    Advanced financial response formatter with domain expertise
    """
    
    def __init__(self):
        self.settings = settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=self.settings.GEMINI_API_KEY,
            temperature=0.3,  # Slightly higher for more natural responses
            max_tokens=1500
        )
        self.domain_knowledge = FinancialDomainKnowledge()
    
    async def format_response(
        self,
        user_query: str,
        intent: str,
        entities: Dict[str, Any],
        sql_result: SQLQueryResult,
        execution_result: QueryExecutionResult,
        style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    ) -> FormattedResponse:
        """
        Main response formatting method using domain-aware reasoning
        """
        try:
            # Step 1: Analyze the data and determine confidence
            confidence, data_analysis = self._analyze_data_quality(execution_result)
            
            # Step 2: Generate insights based on intent and data
            insights = await self._generate_insights(
                intent, entities, execution_result, user_query
            )
            
            # Step 3: Create recommendations
            recommendations = await self._generate_recommendations(
                intent, entities, execution_result, confidence
            )
            
            # Step 4: Identify risks and warnings
            risk_warnings = self._identify_risks(execution_result, confidence)
            
            # Step 5: Format natural language response
            natural_response = await self._generate_natural_language(
                user_query, intent, execution_result, insights, 
                recommendations, risk_warnings, style
            )
            
            # Step 6: Generate explanations
            explanation = await self._generate_explanation(
                intent, sql_result, execution_result, insights
            )
            
            # Step 7: Suggest follow-ups
            follow_ups = self._generate_follow_ups(intent, entities, execution_result)
            
            return FormattedResponse(
                natural_language=natural_response,
                confidence=confidence,
                key_insights=insights,
                recommendations=recommendations,
                risk_warnings=risk_warnings,
                data_summary=data_analysis,
                explanation=explanation,
                follow_up_suggestions=follow_ups,
                technical_details=self._extract_technical_details(execution_result),
                disclaimer=self._get_appropriate_disclaimer(intent)
            )
            
        except Exception as e:
            # Error fallback response
            return FormattedResponse(
                natural_language=f"I encountered an issue analyzing the data: {str(e)}. Please try rephrasing your question.",
                confidence=ConfidenceLevel.INSUFFICIENT,
                key_insights=[],
                recommendations=[],
                risk_warnings=["Unable to provide reliable analysis due to data processing error"],
                data_summary={"error": str(e)},
                explanation="Data processing failed",
                follow_up_suggestions=["Please try rephrasing your question", "Check if the requested data is available"],
                disclaimer=self.domain_knowledge.get_investment_disclaimers()["general"]
            )
    
    def _analyze_data_quality(self, execution_result: QueryExecutionResult) -> Tuple[ConfidenceLevel, Dict[str, Any]]:
        """Analyze data quality and determine confidence level"""
        
        if execution_result.status != ExecutionStatus.SUCCESS:
            return ConfidenceLevel.INSUFFICIENT, {
                "status": "error",
                "message": execution_result.error_message,
                "row_count": 0
            }
        
        row_count = execution_result.row_count
        data = execution_result.data or []
        
        # Determine confidence based on data availability
        if row_count == 0:
            confidence = ConfidenceLevel.INSUFFICIENT
        elif row_count < 5:
            confidence = ConfidenceLevel.LOW
        elif row_count < 20:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.HIGH
        
        # Analyze data completeness
        if data:
            sample_row = data[0]
            null_count = sum(1 for value in sample_row.values() if value is None)
            completeness = 1 - (null_count / len(sample_row))
            
            # Adjust confidence based on data completeness
            if completeness < 0.5:
                confidence = ConfidenceLevel.LOW
            elif completeness < 0.8 and confidence == ConfidenceLevel.HIGH:
                confidence = ConfidenceLevel.MEDIUM
        
        analysis = {
            "row_count": row_count,
            "execution_time": execution_result.execution_time,
            "cached": execution_result.cached,
            "data_completeness": completeness if data else 0,
            "confidence_factors": {
                "data_availability": "good" if row_count > 10 else "limited",
                "data_freshness": "recent" if execution_result.execution_time < 1.0 else "acceptable",
                "data_completeness": "high" if completeness > 0.8 else "moderate"
            }
        }
        
        return confidence, analysis
    
    async def _generate_insights(
        self,
        intent: str,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        user_query: str
    ) -> List[str]:
        """Generate key insights from the data"""
        
        if not execution_result.data:
            return ["No data available to generate insights"]
        
        # Use domain knowledge to interpret financial metrics
        insights = []
        interpretations = self.domain_knowledge.get_indicator_interpretations()
        
        for row in execution_result.data[:3]:  # Analyze first 3 rows
            for metric, value in row.items():
                if metric in interpretations and value is not None:
                    interpretation = self._interpret_metric(metric, value, interpretations[metric])
                    if interpretation:
                        insights.append(interpretation)
        
        # Generate AI-powered insights if we have insufficient rule-based insights
        if len(insights) < 2:
            ai_insights = await self._generate_ai_insights(
                intent, entities, execution_result, user_query
            )
            insights.extend(ai_insights)
        
        return insights[:5]  # Limit to top 5 insights
    
    def _interpret_metric(self, metric: str, value: Any, interpretation_rules: Dict[str, Any]) -> Optional[str]:
        """Interpret a financial metric using domain rules"""
        try:
            if value is None:
                return None
                
            # Convert to float if it's a Decimal or string
            val_float = float(value)
            
            if "ranges" in interpretation_rules:
                for range_name, range_info in interpretation_rules["ranges"].items():
                    if range_info["min"] <= val_float <= range_info["max"]:
                        return f"{interpretation_rules['name']}: {val_float:.2f} - {range_info['signal']}"
            
            return f"{interpretation_rules['name']}: {val_float:.2f}"
            
        except (TypeError, KeyError, ValueError):
            return None
    
    async def _generate_ai_insights(
        self,
        intent: str,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        user_query: str
    ) -> List[str]:
        """Generate insights using LLM analysis"""
        
        data_summary = safe_json_dumps(execution_result.data[:3], indent=2) if execution_result.data else "No data"
        
        insight_prompt = f"""
        You are a financial analyst generating insights from market data.
        
        USER QUERY: "{user_query}"
        INTENT: {intent}
        ENTITIES: {safe_json_dumps(entities)}
        
        DATA RESULTS:
        {data_summary}
        
        TASK: Generate 2-3 concise, actionable financial insights from this data.
        
        REQUIREMENTS:
        1. Focus on the most important financial indicators
        2. Explain what the numbers mean for investors
        3. Be specific and data-driven
        4. Avoid generic statements
        5. Each insight should be 1-2 sentences maximum
        
        Respond with only the insights, one per line, starting with "-".
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=insight_prompt),
                HumanMessage(content=user_query)
            ])
            
            content = self._extract_content(response.content)
            
            # Parse insights from response
            insights = []
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    insight = line[1:].strip()
                    if insight and len(insight) > 10:
                        insights.append(insight)
            
            return insights[:3]
            
        except Exception as e:
            return [f"Unable to generate AI insights: {str(e)}"]
    
    async def _generate_recommendations(
        self,
        intent: str,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        confidence: ConfidenceLevel
    ) -> List[str]:
        """Generate investment recommendations based on analysis"""
        
        if confidence == ConfidenceLevel.INSUFFICIENT:
            return ["Insufficient data to provide reliable recommendations"]
        
        if not execution_result.data:
            return ["No data available for recommendations"]
        
        # Intent-based recommendation strategy
        recommendation_strategies = {
            "stock_analysis": self._generate_stock_recommendations,
            "sentiment_analysis": self._generate_sentiment_recommendations,
            "technical_analysis": self._generate_technical_recommendations,
            "risk_assessment": self._generate_risk_recommendations,
            "portfolio_review": self._generate_portfolio_recommendations
        }
        
        strategy = recommendation_strategies.get(intent, self._generate_general_recommendations)
        return await strategy(entities, execution_result, confidence)
    
    async def _generate_stock_recommendations(
        self,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        confidence: ConfidenceLevel
    ) -> List[str]:
        """Generate stock-specific recommendations"""
        
        recommendations = []
        data = execution_result.data[0] if execution_result.data else {}
        
        # Price and volume analysis
        if data.get('close_price') is not None and data.get('volume') is not None:
            price = float(data['close_price'])
            volume = int(data.get('volume', 0))
            
            if volume > 1000000:  # High volume
                recommendations.append(f"High trading volume ({volume:,}) suggests strong investor interest")
            
            # Technical indicator recommendations
            rsi = data.get('rsi_14')
            if rsi is not None:
                rsi_val = float(rsi)
                if rsi_val < 30:
                    recommendations.append("RSI indicates oversold condition - potential buying opportunity")
                elif rsi_val > 70:
                    recommendations.append("RSI indicates overbought condition - consider profit-taking")
        
        # Risk-based recommendations
        volatility = data.get('volatility_30d')
        beta = data.get('beta')
        
        if volatility is not None and beta is not None:
            vol_val = float(volatility)
            beta_val = float(beta)
            if vol_val > 30:
                recommendations.append("High volatility detected - suitable for risk-tolerant investors only")
            if beta_val > 1.5:
                recommendations.append("High beta indicates amplified market movements - monitor closely")
        
        return recommendations[:4]
    
    async def _generate_sentiment_recommendations(
        self,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        confidence: ConfidenceLevel
    ) -> List[str]:
        """Generate sentiment-based recommendations"""
        
        recommendations = []
        
        if execution_result.data:
            positive_count = sum(1 for row in execution_result.data 
                               if row.get('sentiment_score', 0) > 0.1)
            negative_count = sum(1 for row in execution_result.data 
                               if row.get('sentiment_score', 0) < -0.1)
            total_count = len(execution_result.data)
            
            if positive_count > negative_count * 2:
                recommendations.append("Predominantly positive news sentiment supports bullish outlook")
            elif negative_count > positive_count * 2:
                recommendations.append("Negative news sentiment suggests caution advised")
            else:
                recommendations.append("Mixed sentiment signals - monitor news developments closely")
            
            # Recent sentiment trend
            if total_count > 5:
                recent_sentiment = sum(row.get('sentiment_score', 0) 
                                     for row in execution_result.data[:5]) / 5
                if recent_sentiment > 0.2:
                    recommendations.append("Recent news flow is particularly positive")
                elif recent_sentiment < -0.2:
                    recommendations.append("Recent news flow shows concerning trends")
        
        return recommendations[:3]
    
    async def _generate_technical_recommendations(
        self,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        confidence: ConfidenceLevel
    ) -> List[str]:
        """Generate technical analysis recommendations"""
        
        recommendations = []
        
        if execution_result.data:
            data = execution_result.data[0]
            
            # MACD analysis
            macd_line = data.get('macd_line')
            macd_signal = data.get('macd_signal')
            
            if macd_line and macd_signal:
                if macd_line > macd_signal:
                    recommendations.append("MACD line above signal line indicates potential uptrend")
                else:
                    recommendations.append("MACD line below signal line suggests downward momentum")
            
            # Moving average analysis
            sma_20 = data.get('sma_20')
            sma_50 = data.get('sma_50')
            close_price = data.get('close_price')
            
            if sma_20 is not None and sma_50 is not None and close_price is not None:
                if sma_20 > sma_50:
                    recommendations.append("Short-term moving average above long-term suggests bullish trend")
                if close_price > sma_20:
                    recommendations.append("Price above 20-day moving average indicates short-term strength")
        
        return recommendations[:3]
    
    async def _generate_risk_recommendations(
        self,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        confidence: ConfidenceLevel
    ) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        if execution_result.data:
            for row in execution_result.data:
                volatility = row.get('volatility_30d')
                beta = row.get('beta')
                sharpe_ratio = row.get('sharpe_ratio')
                
                if volatility is not None:
                    vol_val = float(volatility)
                    if vol_val > 25:
                        recommendations.append(f"High volatility ({vol_val:.1f}%) requires strict position sizing")
                
                if sharpe_ratio is not None:
                    sharpe_val = float(sharpe_ratio)
                    if sharpe_val > 1.5:
                        recommendations.append("Strong risk-adjusted returns based on Sharpe ratio")
                    elif sharpe_val < 0.5:
                        recommendations.append("Poor risk-adjusted returns - consider alternatives")
                
                if beta is not None:
                    beta_val = float(beta)
                    if beta_val > 1.5:
                        recommendations.append("High beta indicates amplified market risk exposure")
        
        return recommendations[:4]
    
    async def _generate_portfolio_recommendations(
        self,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        confidence: ConfidenceLevel
    ) -> List[str]:
        """Generate portfolio management recommendations"""
        
        recommendations = []
        
        if execution_result.data:
            total_value = sum(row.get('current_value', 0) for row in execution_result.data)
            
            # Portfolio concentration analysis
            positions = len(execution_result.data)
            if positions < 5:
                recommendations.append("Portfolio appears concentrated - consider diversification")
            
            # Performance analysis
            positive_returns = sum(1 for row in execution_result.data 
                                 if row.get('return_pct', 0) > 0)
            
            if positive_returns > len(execution_result.data) * 0.7:
                recommendations.append("Strong portfolio performance with majority of positions profitable")
            elif positive_returns < len(execution_result.data) * 0.3:
                recommendations.append("Portfolio underperformance requires review and rebalancing")
        
        return recommendations[:3]
    
    async def _generate_general_recommendations(
        self,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult,
        confidence: ConfidenceLevel
    ) -> List[str]:
        """Generate general recommendations as fallback"""
        
        if confidence == ConfidenceLevel.LOW:
            return ["Limited data available - suggest monitoring for additional information"]
        
        return ["Consider consulting with a financial advisor for personalized recommendations"]
    
    def _identify_risks(
        self,
        execution_result: QueryExecutionResult,
        confidence: ConfidenceLevel
    ) -> List[str]:
        """Identify potential risks and warnings"""
        
        risks = []
        
        if confidence == ConfidenceLevel.INSUFFICIENT:
            risks.append("Insufficient data for reliable risk assessment")
            return risks
        
        if execution_result.data:
            for row in execution_result.data:
                volatility = row.get('volatility_30d')
                beta = row.get('beta')
                
                if volatility is not None:
                    vol_val = float(volatility)
                    if vol_val > 40:
                        risks.append("Extremely high volatility poses significant downside risk")
                    elif vol_val > 25:
                        risks.append("High volatility may result in substantial price swings")
                
                if beta is not None:
                    beta_val = float(beta)
                    if beta_val > 2.0:
                        risks.append("Very high beta amplifies market downturns significantly")
                
                # Sentiment risks
                sentiment = row.get('sentiment_score')
                if sentiment is not None:
                    sent_val = float(sentiment)
                    if sent_val < -0.5:
                        risks.append("Strongly negative sentiment may pressure stock price")
        
        # Data quality risks
        if execution_result.cached:
            risks.append("Analysis based on cached data - may not reflect latest market conditions")
        
        if execution_result.execution_time > 5.0:
            risks.append("Query execution delay may indicate data latency issues")
        
        return risks[:4]
    
    async def _generate_natural_language(
        self,
        user_query: str,
        intent: str,
        execution_result: QueryExecutionResult,
        insights: List[str],
        recommendations: List[str],
        risk_warnings: List[str],
        style: ResponseStyle
    ) -> str:
        """Generate natural language response using LLM"""
        
        # Prepare data summary
        data_context = ""
        if execution_result.data and len(execution_result.data) > 0:
            sample = execution_result.data[0]
            data_context = f"Key metrics: {safe_json_dumps(sample, indent=2)}"
        
        style_instructions = {
            ResponseStyle.CONVERSATIONAL: "Use a friendly, approachable tone as if speaking to a friend",
            ResponseStyle.ANALYTICAL: "Use a data-focused, professional analytical tone",
            ResponseStyle.ADVISORY: "Use a confident, advisory tone with clear recommendations",
            ResponseStyle.EDUCATIONAL: "Use an educational tone that explains concepts clearly",
            ResponseStyle.EXECUTIVE: "Use a brief, high-level executive summary style"
        }
        
        response_prompt = f"""
        You are a financial advisor providing personalized investment guidance.
        
        USER QUESTION: "{user_query}"
        INTENT: {intent}
        
        DATA RESULTS:
        Rows returned: {execution_result.row_count}
        Execution time: {execution_result.execution_time:.2f}s
        
        {data_context}
        
        KEY INSIGHTS:
        {chr(10).join(f"- {insight}" for insight in insights)}
        
        RECOMMENDATIONS:
        {chr(10).join(f"- {rec}" for rec in recommendations)}
        
        RISK WARNINGS:
        {chr(10).join(f"- {risk}" for risk in risk_warnings)}
        
        STYLE: {style_instructions[style]}
        
        TASK: Write a comprehensive response that:
        1. Directly answers the user's question
        2. Incorporates the key insights naturally
        3. Provides clear, actionable recommendations
        4. Mentions important risks appropriately
        5. Uses the specified communication style
        6. Keeps response to 2-3 paragraphs maximum
        
        Remember: This is financial analysis, not personal financial advice.
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=response_prompt),
                HumanMessage(content=user_query)
            ])
            
            return self._extract_content(response.content)
            
        except Exception as e:
            return f"Based on the analysis, here are the key findings: {'. '.join(insights[:2])}. {'. '.join(recommendations[:2])}. Please note: {risk_warnings[0] if risk_warnings else 'Consider consulting a financial advisor for personalized advice.'}."
    
    async def _generate_explanation(
        self,
        intent: str,
        sql_result: SQLQueryResult,
        execution_result: QueryExecutionResult,
        insights: List[str]
    ) -> str:
        """Generate explanation of the analysis process"""
        
        explanation_parts = []
        
        # Data source explanation
        explanation_parts.append(f"Analysis based on {execution_result.row_count} data points from our financial database.")
        
        # Methodology explanation
        if sql_result.complexity == QueryComplexity.SIMPLE:
            explanation_parts.append("Used direct metric lookup for rapid analysis.")
        elif sql_result.complexity == QueryComplexity.MODERATE:
            explanation_parts.append("Applied cross-table analysis combining price data with technical indicators.")
        elif sql_result.complexity == QueryComplexity.COMPLEX:
            explanation_parts.append("Performed comprehensive analysis across multiple data dimensions.")
        else:
            explanation_parts.append("Conducted advanced analytics with sophisticated data modeling.")
        
        # Performance note
        if execution_result.execution_time > 2.0:
            explanation_parts.append("Complex analysis required additional processing time for accuracy.")
        
        return " ".join(explanation_parts)
    
    def _generate_follow_ups(
        self,
        intent: str,
        entities: Dict[str, Any],
        execution_result: QueryExecutionResult
    ) -> List[str]:
        """Generate relevant follow-up question suggestions"""
        
        follow_ups = []
        symbols = entities.get("stocks", [])
        
        if intent == "stock_analysis":
            if symbols:
                follow_ups.extend([
                    f"How has {symbols[0]} performed compared to the market?",
                    f"What's the technical outlook for {symbols[0]}?",
                    f"Show me recent news sentiment for {symbols[0]}"
                ])
        
        elif intent == "sentiment_analysis":
            follow_ups.extend([
                "What specific news is driving the sentiment?",
                "How does current sentiment compare to historical levels?",
                "Which other stocks have similar sentiment patterns?"
            ])
        
        elif intent == "technical_analysis":
            follow_ups.extend([
                "What are the key support and resistance levels?",
                "How do these indicators compare to sector peers?",
                "What's the optimal entry point based on technical analysis?"
            ])
        
        elif intent == "portfolio_review":
            follow_ups.extend([
                "How can I optimize my portfolio allocation?",
                "What are the risk factors in my current holdings?",
                "Which underperforming positions should I consider selling?"
            ])
        
        # Generic follow-ups
        follow_ups.extend([
            "What are the current market trends affecting this analysis?",
            "Can you provide a risk assessment for this investment?"
        ])
        
        return follow_ups[:4]
    
    def _extract_technical_details(self, execution_result: QueryExecutionResult) -> Dict[str, Any]:
        """Extract technical details for advanced users"""
        
        return {
            "query_performance": {
                "execution_time": execution_result.execution_time,
                "row_count": execution_result.row_count,
                "cached": execution_result.cached,
                "query_hash": execution_result.query_hash
            },
            "data_quality": {
                "status": execution_result.status.value,
                "warnings": execution_result.warnings,
                "column_info": execution_result.column_info
            },
            "performance_metrics": execution_result.performance_metrics
        }
    
    def _get_appropriate_disclaimer(self, intent: str) -> str:
        """Get appropriate disclaimer based on intent"""
        
        disclaimers = self.domain_knowledge.get_investment_disclaimers()
        
        disclaimer_map = {
            "stock_analysis": disclaimers["stock_analysis"],
            "portfolio_review": disclaimers["portfolio_advice"],
            "risk_assessment": disclaimers["risk_assessment"],
            "sentiment_analysis": disclaimers["sentiment_analysis"]
        }
        
        return disclaimer_map.get(intent, disclaimers["general"])
    
    def _extract_content(self, response_content: Any) -> str:
        """Extract text content from LLM response"""
        content = response_content
        if isinstance(content, list):
            content = " ".join([part if isinstance(part, str) else str(part) for part in content])
        return str(content)


# Main formatting function for integration with workflow
async def format_financial_response(
    user_query: str,
    intent: str,
    entities: Dict[str, Any],
    sql_result: SQLQueryResult,
    execution_result: QueryExecutionResult,
    style: ResponseStyle = ResponseStyle.CONVERSATIONAL
) -> FormattedResponse:
    """
    Main entry point for formatting financial responses
    """
    formatter = FinancialResponseFormatter()
    return await formatter.format_response(
        user_query, intent, entities, sql_result, execution_result, style
    )


if __name__ == "__main__":
    # Test the response formatter
    async def test_response_formatter():
        # Mock test data
        from src.langgraph.sql_generator import SQLQueryResult, QueryComplexity
        from src.langgraph.query_executor import QueryExecutionResult, ExecutionStatus
        
        # Sample execution result
        mock_execution_result = QueryExecutionResult(
            status=ExecutionStatus.SUCCESS,
            data=[{
                "symbol": "AAPL",
                "company_name": "Apple Inc.",
                "close_price": 150.25,
                "volume": 45000000,
                "rsi_14": 65.5,
                "macd_line": 1.25,
                "volatility_30d": 22.3,
                "beta": 1.15
            }],
            row_count=1,
            execution_time=0.45,
            cached=False
        )
        
        # Sample SQL result
        mock_sql_result = SQLQueryResult(
            sql="SELECT * FROM stocks WHERE symbol = 'AAPL'",
            parameters={"symbol": "AAPL"},
            reasoning="Simple stock lookup",
            complexity=QueryComplexity.SIMPLE,
            estimated_execution_time=0.1,
            tables_involved=["stocks", "stock_prices"],
            validation_errors=[],
            optimization_suggestions=[]
        )
        
        formatter = FinancialResponseFormatter()
        
        result = await formatter.format_response(
            user_query="What's the current price and volume of Apple stock?",
            intent="stock_analysis",
            entities={"stocks": ["AAPL"], "metrics": ["price", "volume"]},
            sql_result=mock_sql_result,
            execution_result=mock_execution_result,
            style=ResponseStyle.CONVERSATIONAL
        )
        
        print("=== FORMATTED RESPONSE ===")
        print(f"Natural Language:\n{result.natural_language}\n")
        print(f"Confidence: {result.confidence.value}")
        print(f"Key Insights: {result.key_insights}")
        print(f"Recommendations: {result.recommendations}")
        print(f"Risk Warnings: {result.risk_warnings}")
        print(f"Follow-ups: {result.follow_up_suggestions}")
        print(f"Disclaimer: {result.disclaimer}")
    
    asyncio.run(test_response_formatter())