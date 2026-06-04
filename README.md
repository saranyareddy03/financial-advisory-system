# 🤖 AI Financial Advisory System

**A Real-Time, Intelligent Financial Analytics Platform powered by Large Language Models and LangGraph.**

This project implements a sophisticated **Text-to-SQL** system specialized for the financial domain. It transforms natural language queries into executable SQL, runs them against a structured financial database, and synthesizes the results into actionable investment insights, risk assessments, and portfolio recommendations.

---

## 🌟 Key Features

### 🧠 Advanced Natural Language Understanding
*   **Intent Detection**: Automatically classifies user queries into categories like `Stock Analysis`, `Risk Assessment`, `Sentiment Analysis`, `Portfolio Review`, or `Technical Analysis`.
*   **Entity Extraction**: Identifies key financial parameters (Symbols like AAPL/TSLA, metrics like RSI/MACD, time periods, and comparison logic) using **Google Gemini 2.0 Flash**.
*   **Context Awareness**: Maintains conversation history for follow-up questions (e.g., "How about its volatility?").

### ⚙️ Intelligent SQL Generation
*   **Schema-Aware Reasoning**: The system understands the complex relationships between `stocks`, `prices`, `technical_indicators`, and `news`.
*   **Complexity Handling**: Adaptive reasoning strategies based on query difficulty:
    *   *Simple*: Direct pattern matching.
    *   *Moderate*: Few-shot prompting with examples.
    *   *Complex*: Step-by-step chain-of-thought construction.
    *   *Advanced*: Multi-agent collaboration simulation.
*   **Self-Correction**: Automatically validates generated SQL syntax and logic before execution.

### 🛡️ Enterprise-Grade Query Execution
*   **Security First**: Built-in `SecurityValidator` prevents SQL injection and restricts dangerous operations (DROP, DELETE).
*   **Performance Optimization**:
    *   Intelligent **Caching System** (TTL based on query complexity).
    *   **Performance Monitoring** tracks execution time and row counts.
*   **Robust Error Handling**: Graceful degradation with user-friendly error messages.

### 📊 Rich Visualization & Reporting
*   **Interactive Dashboard**: A **Streamlit** frontend featuring dynamic charts (Plotly) for price history, volume, and technical indicators.
*   **Smart Formatting**: 
    *   **Key Insights**: AI-generated bullet points highlighting critical trends.
    *   **Risk Warnings**: Automatic flagging of high volatility or negative sentiment.
    *   **Confidence Levels**: Data-driven confidence scoring (High/Medium/Low) based on data coverage.
*   **Multi-Persona Responses**: Switch between styles like *Conversational*, *Analytical*, *Executive*, or *Educational*.

---

## 🏗️ Architecture Overview

The system is orchestrated using **LangGraph**, creating a reliable state machine that guides the request through the following pipeline:

1.  **User Input**: "How is Apple performing compared to Microsoft?"
2.  **Intent/Entity Extraction**: `Compare(AAPL, MSFT)`, Intent: `Comparative Analysis`.
3.  **SQL Generation**: Constructs a query joining `stock_prices`, `technical_indicators`, and `sentiment_scores`.
4.  **Query Execution**: Runs securely against the **PostgreSQL** database.
5.  **Response Synthesis**: Formats the raw data into a natural language comparison with charts.

---

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) & [LangChain](https://www.langchain.com/)
*   **LLM**: Google Gemini (`gemini-2.5-pro`)
*   **Database**: PostgreSQL (via Supabase)
*   **ORM**: SQLAlchemy
*   **Frontend**: Streamlit
*   **Visualization**: Plotly
*   **Data Analysis**: Pandas, NumPy

---

## 📂 Project Structure

```bash
financial_advisory_system/
├── src/
│   ├── analytics/           # Financial calculation modules (Risk, Technicals)
│   ├── config/              # Environment and app settings
│   ├── database/            # Database connections and seeders
│   ├── langgraph/           # Core Logic
│   │   ├── intent_entity_extractor.py  # NLP Parsing
│   │   ├── sql_generator.py            # Text-to-SQL Logic
│   │   ├── query_executor.py           # Secure DB Execution
│   │   ├── response_formatter.py       # Insight Generation
│   │   └── workflow.py                 # LangGraph State Machine
│   ├── main.py              # CLI Entry point
│   └── streamlit_app.py     # Web Interface
├── tests/                   # Unit and Integration tests
└── requirements.txt         # Dependencies
```

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10 or higher
*   PostgreSQL database (or Supabase account)
*   Google Cloud API Key (for Gemini)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/financial-advisory-system.git
    cd financial-advisory-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration:**
    Create a `.env` file in the root directory:
    ```env
    # Database
    host=your-db-host
    port=5432
    dbname=postgres
    user=your-db-user
    password=your-db-password

    # API Keys
    GEMINI_API_KEY=your_google_gemini_api_key

    # App Settings
    DEBUG=true
    DB_POOL_SIZE=5
    ```

4.  **Database Setup:**
    Ensure your PostgreSQL database has the required schema. You can run the initialization scripts found in `src/database/` (if available) or ensure tables like `stocks`, `stock_prices`, `financial_news` exist.

---

## 🖥️ Usage

### Web Interface (Streamlit)
The recommended way to interact with the system.
```bash
streamlit run src/streamlit_app.py
```
*   Navigate to `http://localhost:8501`.
*   Use the sidebar to try example queries or type your own.
*   View the "Technical Details" expander to see the raw SQL generated.

### Command Line Interface (CLI)
For quick testing or debugging:
```bash
python src/main.py
```

---

## 💾 Database Schema Summary

The system relies on a relational schema designed for financial analytics:

*   **`stocks`**: Master table for company info (Symbol, Sector, Market Cap).
*   **`stock_prices`**: Historical OHLCV data.
*   **`technical_indicators`**: Pre-calculated metrics (RSI, MACD, SMA).
*   **`risk_metrics`**: Advanced risk data (Beta, Sharpe Ratio, Volatility).
*   **`financial_news`**: News headlines and content.
*   **`sentiment_scores`**: LLM-derived sentiment scores linked to news and stocks.
*   **`user_queries`**: Logging table for system improvement and history.

---

## 🔮 Future Roadmap

*   [ ] **Portfolio Optimization**: Add "What-if" analysis for portfolio rebalancing.
*   [ ] **Real-time Data Stream**: Integrate live WebSocket feeds for second-by-second updates.
*   [ ] **Multi-Modal Analysis**: Allow users to upload PDF earnings reports for RAG-based Q&A.
*   [ ] **Personalized Alerts**: User-defined triggers for price or sentiment shifts.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
