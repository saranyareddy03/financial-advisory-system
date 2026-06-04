# Financial Advisory System

## Overview

The Financial Advisory System is a Python-based application developed to assist users in analyzing stock market data and obtaining financial insights through natural language queries. The system integrates financial analytics, sentiment analysis, risk assessment, and technical indicators to provide meaningful information for investment-related decision making.

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

![Home Page](<img width="1919" height="805" alt="home-page" src="https://github.com/user-attachments/assets/e64dd439-7c37-470a-b6f5-32980c9f980a" />)


### Analysis Results

![Analysis Results](<img width="1536" height="855" alt="analysis-results" src="https://github.com/user-attachments/assets/6f662936-a047-42d1-b29d-9aed32263eed" />)

### Dashboard View

![Dashboard](<img width="1475" height="810" alt="dashboard" src="https://github.com/user-attachments/assets/662932f7-bfb2-431a-98ab-69ef31172da9" />)

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

## Author

**Nallimilli Saranya Reddy**

Final Year Project
