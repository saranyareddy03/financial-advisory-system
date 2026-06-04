"""
Streamlit Frontend Application
Real-Time Financial Advisory System - Phase 6

This module provides a sophisticated web interface for the AI-powered financial advisory system,
integrating all backend components into an intuitive user experience.
"""

import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import asyncio
import html
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import your backend components
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.langgraph.advisory_service import FinancialAdvisoryService
from src.langgraph.query_executor import ExecutionStatus
from src.langgraph.response_formatter import ResponseStyle
from src.sentiment.finbert_setup import FinBERTProcessor

@st.cache_resource
def _load_finbert():
    """Download and cache FinBERT model at startup."""
    processor = FinBERTProcessor()
    processor.download_and_setup_model()
    return processor

class FinancialAdvisoryUI:
    """
    Streamlit-based user interface for the Financial Advisory System
    """
    
    def __init__(self):
        self.setup_page_config()
        self.setup_session_state()
        self.total_queries_placeholder = None
        self.session_started_placeholder = None
        self.system_status_placeholder = None
        self.service = FinancialAdvisoryService()
        self.executor = self.service.executor
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AI Financial Advisory System",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .response-container {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #dde6f2;
            font-size: 1rem;
            line-height: 1.6;
            color: #223554;
        }
        .insight-box {
            background-color: #e8f5e8;
            padding: 1rem;
            border-left: 4px solid #4CAF50;
            margin: 0.5rem 0;
            font-size: 0.98rem;
            line-height: 1.55;
            color: #1f3b2c;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-left: 4px solid #ffc107;
            margin: 0.5rem 0;
            font-size: 0.98rem;
            line-height: 1.55;
            color: #5b4700;
        }
        .recommendation-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-left: 4px solid #2196F3;
            margin: 0.5rem 0;
            font-size: 0.98rem;
            line-height: 1.55;
            color: #173b61;
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.3rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            margin: 0.5rem;
        }
        .header-card {
            background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
            border: 1px solid #e3e8ef;
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            min-height: 88px;
        }
        .header-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #5b6b7a;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .header-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #1c355e;
            line-height: 1.15;
        }
        .header-subtle {
            font-size: 0.9rem;
            font-weight: 600;
            color: #2c7a4b;
        }
        .response-title {
            margin: 0 0 0.75rem 0;
            color: #1c355e;
            font-size: 1.45rem;
            font-weight: 700;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def setup_session_state(self):
        """Initialize session state variables"""
        st.session_state.setdefault("conversation_history", [])
        st.session_state.setdefault("query_count", 0)
        st.session_state.setdefault("last_response", None)
        st.session_state.setdefault("session_started_at", datetime.now())
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<div class="main-header">🤖 AI Financial Advisory System</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.total_queries_placeholder = st.empty()
            self.total_queries_placeholder.markdown(
                self._header_card_html("Total Queries", str(st.session_state.get("query_count", 0))),
                unsafe_allow_html=True,
            )
        with col2:
            session_started = st.session_state.get("session_started_at", datetime.now())
            self.session_started_placeholder = st.empty()
            self.session_started_placeholder.markdown(
                self._header_card_html("Session Started", session_started.strftime("%H:%M")),
                unsafe_allow_html=True,
            )
        with col3:
            self.system_status_placeholder = st.empty()
            self.system_status_placeholder.markdown(
                self._header_card_html("System Status", "Online", "Live services healthy"),
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                self._header_card_html("Session Actions", "", "Reset local query history"),
                unsafe_allow_html=True,
            )
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.conversation_history = []
                st.session_state.query_count = 0
                st.session_state.last_response = None
                st.session_state.session_started_at = datetime.now()
                self._refresh_header_metrics()
                st.rerun()
    
    def render_sidebar(self):
        """Render the sidebar with examples and settings"""
        with st.sidebar:
            st.header("📋 Quick Examples")
            
            example_queries = [
                "What's the current price and volume of Apple stock?",
                "Show me the risk metrics for Tesla (TSLA)",
                "What's the recent sentiment for Microsoft?",
                "Compare Google and Amazon trading volumes",
                "What are the RSI and MACD indicators for Netflix?",
                "How volatile is Amazon stock?",
                "What's the beta for Apple compared to the market?",
                "What is my current portfolio value?",
                "How diversified is my portfolio?",
                "Can you suggest a minimum-volatility version of my portfolio?"
            ]
            
            for i, example in enumerate(example_queries):
                if st.button(f"📊 {example}", key=f"example_{i}"):
                    st.session_state.selected_query = example
                    st.rerun()
            
            st.header("⚙️ Settings")
            
            response_style = st.selectbox(
                "Response Style",
                ["conversational", "analytical", "advisory", "educational", "executive"],
                index=0
            )
            st.session_state.response_style = response_style
            
            show_technical_details = st.checkbox("Show Technical Details", False)
            st.session_state.show_technical_details = show_technical_details
            
            show_query_performance = st.checkbox("Show Query Performance", False)
            st.session_state.show_query_performance = show_query_performance
            
            st.header("📈 System Stats")
            if st.button("🔍 Health Check"):
                with st.spinner("Checking system health..."):
                    health_status = asyncio.run(self.executor.health_check())
                    st.json(health_status)
    
    def render_query_input(self):
        """Render the main query input interface"""
        st.header("💬 Ask Your Financial Question")
        
        # Check if there's a selected query from sidebar
        default_query = ""
        if hasattr(st.session_state, 'selected_query'):
            default_query = st.session_state.selected_query
            del st.session_state.selected_query
        
        user_query = st.text_input(
            "Enter your financial question:",
            value=default_query,
            placeholder="e.g., What's the current price of Apple stock?",
            key="user_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Analyze", type="primary", use_container_width=True):
                if user_query.strip():
                    self.process_user_query(user_query.strip())
                else:
                    st.warning("Please enter a financial question.")
        
        with col2:
            if st.button("🔄 Try Random", use_container_width=True):
                import random
                random_examples = [
                    "What's the current price of AAPL?",
                    "Show me TSLA risk metrics",
                    "MSFT sentiment analysis",
                    "Compare GOOGL and AMZN",
                    "Netflix technical indicators",
                    "What are my top holdings by weight?",
                    "Compare my current portfolio with an optimized portfolio"
                ]
                random_query = random.choice(random_examples)
                st.session_state.selected_query = random_query
                st.rerun()
        
        with col3:
            if st.button("📊 Portfolio", use_container_width=True):
                st.session_state.selected_query = "Show me my portfolio performance"
                st.rerun()
    
    def process_user_query(self, user_query: str):
        """Process the user query through the financial advisory pipeline"""
        
        start_time = time.time()
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("Processing your financial query..."):
                
                response_style_map = {
                    "conversational": ResponseStyle.CONVERSATIONAL,
                    "analytical": ResponseStyle.ANALYTICAL,
                    "advisory": ResponseStyle.ADVISORY,
                    "educational": ResponseStyle.EDUCATIONAL,
                    "executive": ResponseStyle.EXECUTIVE
                }
                
                style = response_style_map.get(
                    st.session_state.get('response_style', 'conversational'),
                    ResponseStyle.CONVERSATIONAL
                )

                status_text.text("🧠 Routing through SQL and fallback providers...")
                progress_bar.progress(35)

                service_result = asyncio.run(
                    self.service.process_query(user_query=user_query, style=style)
                )

                status_text.text("🎯 Finalizing answer...")
                progress_bar.progress(80)
                
                # Step 5: Display Results
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                processing_time = time.time() - start_time
                
                # Store in session state
                st.session_state.last_response = {
                    'query': user_query,
                    'intent': service_result.intent,
                    'entities': service_result.entities,
                    'sql_result': service_result.sql_result,
                    'execution_result': service_result.execution_result,
                    'formatted_response': service_result.formatted_response,
                    'route': service_result.route,
                    'source_path': service_result.source_path,
                    'metadata': service_result.metadata,
                    'processing_time': processing_time,
                    'timestamp': datetime.now()
                }
                
                st.session_state.conversation_history.append(st.session_state.last_response)
                st.session_state.query_count += 1
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                self._refresh_header_metrics()
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ System Error: {str(e)}")
            st.exception(e)
    
    def display_response(self):
        """Display the formatted response with all components"""
        
        if not st.session_state.last_response:
            return
        
        response_data = st.session_state.last_response
        formatted_response = response_data['formatted_response']
        execution_result = response_data['execution_result']
        
        st.markdown("## 🎯 Analysis Results")
        
        # Main Response
        response_html = (
            '<div class="response-container">'
            '<div class="response-title">💬 AI Financial Advisor Response</div>'
            f'{self._html_block(formatted_response.natural_language)}'
            '</div>'
        )
        st.markdown(response_html, unsafe_allow_html=True)
        
        # Create columns for organized display
        col1, col2 = st.columns(2)
        
        with col1:
            # Key Insights
            if formatted_response.key_insights:
                st.markdown("### 💡 Key Insights")
                for insight in formatted_response.key_insights:
                    st.markdown(self._html_block(insight, "insight-box", "📊"), unsafe_allow_html=True)
            
            # Recommendations
            if formatted_response.recommendations:
                st.markdown("### 🎯 Recommendations")
                for rec in formatted_response.recommendations:
                    st.markdown(self._html_block(rec, "recommendation-box", "💡"), unsafe_allow_html=True)
        
        with col2:
            # Risk Warnings
            if formatted_response.risk_warnings:
                st.markdown("### ⚠️ Risk Considerations")
                for warning in formatted_response.risk_warnings:
                    st.markdown(self._html_block(warning, "warning-box", "⚠️"), unsafe_allow_html=True)
            
            # Confidence and Metadata
            st.markdown("### 📊 Analysis Metadata")
            confidence_color = {
                "high": "🟢",
                "medium": "🟡", 
                "low": "🟠",
                "insufficient": "🔴"
            }
            
            conf_icon = confidence_color.get(formatted_response.confidence.value, "⚪")
            st.write(f"**Confidence Level:** {conf_icon} {formatted_response.confidence.value.title()}")
            st.write(f"**Processing Time:** {response_data['processing_time']:.2f} seconds")
            st.write(f"**Data Points:** {execution_result.row_count}")
            st.write(f"**Route:** {response_data.get('route', 'sql')}")
            st.write(f"**Source Path:** {response_data.get('source_path', 'database')}")
        
        # Data Visualization
        if execution_result.status == ExecutionStatus.SUCCESS and execution_result.data:
            self.render_data_visualization(execution_result.data, response_data['intent'])
        
        # Follow-up Suggestions
        if formatted_response.follow_up_suggestions:
            st.markdown("### 🔄 Suggested Follow-up Questions")
            cols = st.columns(2)
            response_key = response_data['timestamp'].strftime("%Y%m%d%H%M%S%f")
            for i, suggestion in enumerate(formatted_response.follow_up_suggestions[:4]):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(f"❓ {suggestion}", key=f"followup_{response_key}_{i}"):
                        st.session_state.selected_query = suggestion
                        st.rerun()
        
        # Technical Details (if enabled)
        if st.session_state.get('show_technical_details', False):
            self.render_technical_details(response_data)
        
        # Query Performance (if enabled)
        if st.session_state.get('show_query_performance', False):
            self.render_query_performance(response_data)
        
        # Disclaimer
        if formatted_response.disclaimer:
            st.markdown("---")
            st.caption(f"📝 **Disclaimer:** {formatted_response.disclaimer}")

    def _html_block(self, text: str, css_class: Optional[str] = None, prefix: str = "") -> str:
        escaped = html.escape(text or "").replace("\n", "<br>")
        class_attr = f' class="{css_class}"' if css_class else ""
        content = f"{prefix} {escaped}".strip()
        return f"<div{class_attr}>{content}</div>"

    def _header_card_html(self, label: str, value: str, subtitle: str = "") -> str:
        subtitle_html = f'<div class="header-subtle">{html.escape(subtitle)}</div>' if subtitle else ""
        value_html = f'<div class="header-value">{html.escape(value)}</div>' if value else ""
        return (
            '<div class="header-card">'
            f'<div class="header-label">{html.escape(label)}</div>'
            f"{value_html}"
            f"{subtitle_html}"
            '</div>'
        )

    def _refresh_header_metrics(self):
        session_started = st.session_state.get("session_started_at", datetime.now())
        if self.total_queries_placeholder is not None:
            self.total_queries_placeholder.markdown(
                self._header_card_html("Total Queries", str(st.session_state.get("query_count", 0))),
                unsafe_allow_html=True,
            )
        if self.session_started_placeholder is not None:
            self.session_started_placeholder.markdown(
                self._header_card_html("Session Started", session_started.strftime("%H:%M")),
                unsafe_allow_html=True,
            )
        if self.system_status_placeholder is not None:
            self.system_status_placeholder.markdown(
                self._header_card_html("System Status", "Online", "Live services healthy"),
                unsafe_allow_html=True,
            )
    
    def render_data_visualization(self, data: List[Dict], intent: str):
        """Render data visualizations based on the query intent and data"""
        
        if not data:
            return
        
        st.markdown("### 📈 Data Visualization")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Stock Analysis Visualization
        if intent == "stock_analysis" and 'close_price' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'volume' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df.get('symbol', ['Stock']),
                        y=df['volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ))
                    fig.update_layout(title="Trading Volume", xaxis_title="Stock", yaxis_title="Volume")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'close_price' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df.get('symbol', ['Stock']),
                        y=df['close_price'],
                        name='Price',
                        marker_color='lightgreen'
                    ))
                    fig.update_layout(title="Stock Price", xaxis_title="Stock", yaxis_title="Price ($)")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Technical Indicators Visualization
        if intent == "technical_analysis" or "rsi_14" in df.columns:
            if 'rsi_14' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.get('symbol', range(len(df))),
                    y=df['rsi_14'],
                    mode='lines+markers',
                    name='RSI',
                    line=dict(color='purple')
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig.update_layout(title="RSI Indicator", xaxis_title="Stock", yaxis_title="RSI Value")
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk Metrics Visualization
        if intent == "risk_assessment" or "volatility_30d" in df.columns:
            risk_cols = ['volatility_30d', 'beta', 'sharpe_ratio']
            available_risk_cols = [col for col in risk_cols if col in df.columns]
            
            if available_risk_cols:
                risk_df = df[['symbol'] + available_risk_cols] if 'symbol' in df.columns else df[available_risk_cols]
                
                fig = go.Figure()
                for col in available_risk_cols:
                    fig.add_trace(go.Bar(
                        x=df.get('symbol', [f'Stock {i}' for i in range(len(df))]),
                        y=df[col],
                        name=col.replace('_', ' ').title()
                    ))
                
                fig.update_layout(title="Risk Metrics Comparison", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        
        # Raw Data Table
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, use_container_width=True)
    
    def render_technical_details(self, response_data):
        """Render technical details about the query execution"""
        
        with st.expander("🔧 Technical Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Query Analysis:**")
                query_analysis = {
                    "Intent": response_data['intent'],
                    "Entities": response_data['entities'],
                    "Route": response_data.get('route', 'sql'),
                    "Source Path": response_data.get('source_path', 'database'),
                }
                if response_data.get('sql_result'):
                    query_analysis["SQL Complexity"] = response_data['sql_result'].complexity.value
                    query_analysis["Tables Involved"] = response_data['sql_result'].tables_involved
                st.json(query_analysis)
            
            with col2:
                st.markdown("**Execution Metrics:**")
                exec_result = response_data['execution_result']
                st.json({
                    "Status": exec_result.status.value,
                    "Execution Time": f"{exec_result.execution_time:.3f}s",
                    "Row Count": exec_result.row_count,
                    "Cached": exec_result.cached,
                    "Warnings": exec_result.warnings
                })
            
            st.markdown("**Generated SQL Query:**")
            sql_text = response_data['sql_result'].sql if response_data.get('sql_result') else ""
            if sql_text:
                st.code(sql_text, language="sql")
            else:
                st.caption("No final SQL was used for this answer; external fallback handled the response.")
    
    def render_query_performance(self, response_data):
        """Render query performance analysis"""
        
        with st.expander("⚡ Performance Analysis"):
            exec_result = response_data['execution_result']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Query Time", f"{exec_result.execution_time:.3f}s")
            with col2:
                st.metric("Rows Returned", exec_result.row_count)
            with col3:
                st.metric("Cache Status", "Hit" if exec_result.cached else "Miss")
            with col4:
                processing_time = response_data['processing_time']
                st.metric("Total Time", f"{processing_time:.2f}s")
            
            if exec_result.performance_metrics:
                st.json(exec_result.performance_metrics)
    
    def render_conversation_history(self):
        """Render conversation history"""
        
        if st.session_state.conversation_history:
            st.markdown("## 💭 Conversation History")
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Query {len(st.session_state.conversation_history) - i}: {conv['query'][:50]}..."):
                    st.write(f"**Time:** {conv['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**Intent:** {conv['intent']}")
                    st.write(f"**Response:** {conv['formatted_response'].natural_language[:200]}...")
                    
                    if st.button(f"🔄 Ask Again", key=f"repeat_{i}"):
                        st.session_state.selected_query = conv['query']
                        st.rerun()
    
    def run(self):
        """Main application runner"""
        
        # Header
        self.render_header()
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        self.render_query_input()
        
        # Display last response if available
        if st.session_state.last_response:
            self.display_response()
        
        # Conversation history
        self.render_conversation_history()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "🤖 **AI Financial Advisory System** | "
            "Built with advanced LLM reasoning and real-time data analysis | "
            f"Session: {st.session_state.query_count} queries processed"
        )


if __name__ == "__main__":
    app = FinancialAdvisoryUI()
    app.run()
