# Real-Time Financial Advisory System

## Overview

The Real-Time Financial Advisory System is a Python-based application developed to assist users in analyzing stock market data and obtaining financial insights through natural language queries. The system integrates financial analytics, sentiment analysis, risk assessment, and technical indicators to provide meaningful information for investment-related decision making.

The project uses LangGraph for workflow management, PostgreSQL for data storage, and Streamlit for the user interface.

---

## Features

* Natural language query processing
* Stock performance analysis
* Portfolio risk assessment
* Technical indicator analysis
* Financial sentiment analysis
* Interactive Streamlit dashboard
* Automated SQL query generation and execution
* Database-driven financial data management

---

## Technologies Used

* Python
* Streamlit
* LangGraph
* PostgreSQL
* SQLAlchemy
* Pandas
* NumPy
* Plotly

---

## Project Structure

```text
financial_advisory_system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   ├── analytics/
│   ├── config/
│   ├── database/
│   ├── langgraph/
│   ├── sentiment/
│   ├── utils/
│   ├── main.py
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Screenshots

### Home Page


<img width="1919" height="805" alt="home-page" src="https://github.com/user-attachments/assets/e64dd439-7c37-470a-b6f5-32980c9f980a" />


### Analysis Results


<img width="1536" height="855" alt="analysis-results" src="https://github.com/user-attachments/assets/6f662936-a047-42d1-b29d-9aed32263eed" />

### Dashboard View


<img width="1475" height="855" alt="dashboard" src="https://github.com/user-attachments/assets/662932f7-bfb2-431a-98ab-69ef31172da9" />

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/saranyareddy03/financial-advisory-system.git
cd financial-advisory-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root and add the required API keys and database credentials.

Example:

```env
GEMINI_API_KEY=your_api_key
DB_HOST=your_host
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

---

## Running the Application

### Streamlit Interface

```bash
streamlit run src/streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

### Command Line Interface

```bash
python src/main.py
```

---

## Future Enhancements

* Real-time stock market data integration
* Portfolio optimization features
* Advanced forecasting models
* Personalized investment recommendations
* Additional visualization dashboards

---

---

## Research Publication

This project has been published as a research paper in the International Journal of Engineering Research and Science & Technology (IJERST).

### Paper Details

**Title:**  
Real-Time Financial Advisory System: A GenAI-Driven Conversational and Explainable Framework

**Authors:**  
- Nallimilli Saranya Reddy
- K. Siva Ganesh
- K. Sravani
- K. Jaya Ram
- K. Komali

**Journal:** International Journal of Engineering Research and Science & Technology (IJERST)

**Volume:** 22  
**Issue:** 1(2)  
**Year:** 2026  
**Pages:** 202–208

**DOI:**  
https://doi.org/10.62643/ijerst.2026.v22.n1(2).pp202-208

### Publication Summary

The research presents a GenAI-driven financial advisory platform that combines natural language processing, sentiment analysis, portfolio optimization, and explainable AI techniques to assist investors in making informed financial decisions. The system integrates LangGraph-based workflow orchestration, FinBERT sentiment analysis, PostgreSQL-based financial data management, and a Streamlit interface for conversational financial analytics. The proposed framework processes large-scale financial datasets and provides explainable investment insights, risk assessment, and portfolio recommendations through natural language interactions. :contentReference[oaicite:0]{index=0}

