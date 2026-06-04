# ===============================
# DATABASE CONNECTION MANAGER
# Financial Advisory System
# ===============================

import logging
import pandas as pd
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, date
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import psycopg2
from psycopg2.extras import RealDictCursor

# Import configuration
import sys
import os
# Dynamically add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Centralized database connection and operations manager
    for Real-Time Financial Advisory System
    """
    
    def __init__(self):
        """Initialize database manager with connection pooling"""
        self.engine = None
        self.SessionLocal = None
        self.metadata = None
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Initialize SQLAlchemy engine and session factory"""
        try:
            # Create SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                config.DATABASE_URL,
                poolclass=QueuePool,
                pool_size=config.DB_POOL_SIZE,
                max_overflow=config.DB_MAX_OVERFLOW,
                pool_timeout=config.DB_POOL_TIMEOUT,
                pool_recycle=config.DB_POOL_RECYCLE,
                echo=config.DEBUG,  # Log SQL queries in debug mode
                future=True
            )
            if not config.DEBUG:
                logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Initialize metadata
            self.metadata = MetaData()
            
            # Test connection
            self._test_connection()
            
            logger.info("Database connection initialized successfully")
            logger.info(f"Pool size: {config.DB_POOL_SIZE}, Max overflow: {config.DB_MAX_OVERFLOW}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    def _test_connection(self):
        """Test database connection"""
        if self.engine is None:
            raise RuntimeError("Database engine is not initialized")

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions
        Automatically handles commit/rollback and session cleanup
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database session factory is not initialized")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for raw database connections
        """
        if self.engine is None:
            raise RuntimeError("Database engine is not initialized")

        conn = self.engine.connect()
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            conn.close()
    
    def get_engine(self):
        """Get the SQLAlchemy engine."""
        return self.engine

    # ===============================
    # STOCK DATA OPERATIONS
    # ===============================
    
    def insert_stock(self, symbol: str, company_name: str, 
                    sector: Optional[str] = None, industry: Optional[str] = None,
                    market_cap: Optional[int] = None, exchange: Optional[str] = None,
                    country: str = 'US', currency: str = 'USD') -> str:
        """
        Insert a new stock record
        
        Returns:
            Stock UUID
        """
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO stocks (symbol, company_name, sector, industry, market_cap, currency, exchange, country)
                    VALUES (:symbol, :company_name, :sector, :industry, :market_cap, :currency, :exchange, :country)
                    ON CONFLICT (symbol) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        market_cap = EXCLUDED.market_cap,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """)
                
                result = session.execute(query, {
                    'symbol': symbol.upper(),
                    'company_name': company_name,
                    'sector': sector,
                    'industry': industry,
                    'market_cap': market_cap,
                    'currency': currency,
                    'exchange': exchange,
                    'country': country
                })
                
                row = result.fetchone()
                if row is None:
                    raise ValueError(f"Failed to insert stock {symbol}")
                stock_id = row[0]
                logger.info(f"Inserted/updated stock: {symbol}")
                return str(stock_id)
                
        except Exception as e:
            logger.error(f"Error inserting stock {symbol}: {e}")
            raise
    
    def bulk_insert_stock_prices(self, price_data: pd.DataFrame) -> int:
        """
        Bulk insert stock price data from DataFrame
        
        Args:
            price_data: DataFrame with columns [symbol, date, open, high, low, close, volume]
        
        Returns:
            Number of records inserted
        """
        try:
            # Ensure required columns exist
            required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in price_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Prepare data for insertion
            price_data = price_data.copy()
            price_data['symbol'] = price_data['symbol'].str.upper()
            price_data['volume'] = price_data.get('volume', 0)
            price_data['adjusted_close'] = price_data.get('adjusted_close', price_data['close'])
            
            # Convert DataFrame to list of dictionaries
            records = price_data.rename(columns={
                'open': 'open_price',
                'high': 'high_price', 
                'low': 'low_price',
                'close': 'close_price'
            }).to_dict('records')
            
            with self.get_session() as session:
                # Get stock IDs for symbols
                symbols = price_data['symbol'].unique()
                stock_query = text("""
                    SELECT id, symbol FROM stocks WHERE symbol = ANY(:symbols)
                """)
                stock_result = session.execute(stock_query, {'symbols': list(symbols)})
                symbol_to_id = {row[1]: str(row[0]) for row in stock_result}
                
                # Add stock_id to records
                for record in records:
                    if record['symbol'] in symbol_to_id:
                        record['stock_id'] = symbol_to_id[record['symbol']]
                    else:
                        logger.warning(f"Stock {record['symbol']} not found in database")
                        continue
                
                # Filter records with valid stock_id
                valid_records = [r for r in records if 'stock_id' in r]
                
                if not valid_records:
                    logger.warning("No valid records to insert")
                    return 0
                
                # Bulk insert using SQLAlchemy core
                insert_query = text("""
                    INSERT INTO stock_prices (stock_id, symbol, date, open_price, high_price, low_price, close_price, volume, adjusted_close)
                    VALUES (:stock_id, :symbol, :date, :open_price, :high_price, :low_price, :close_price, :volume, :adjusted_close)
                    ON CONFLICT (stock_id, date) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        adjusted_close = EXCLUDED.adjusted_close
                """)
                
                session.execute(insert_query, valid_records)
                
                logger.info(f"Bulk inserted {len(valid_records)} price records")
                return len(valid_records)
                
        except Exception as e:
            logger.error(f"Error bulk inserting stock prices: {e}")
            raise
    
    def get_stock_prices(self, symbol: str, start_date: Optional[date] = None, 
                        end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get stock price data for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with price data
        """
        try:
            query = """
                SELECT 
                    sp.date,
                    sp.open_price,
                    sp.high_price,
                    sp.low_price,
                    sp.close_price,
                    sp.volume,
                    sp.adjusted_close,
                    s.company_name,
                    s.sector
                FROM stock_prices sp
                JOIN stocks s ON sp.stock_id = s.id
                WHERE sp.symbol = :symbol
            """
            
            params: Dict[str, Any] = {'symbol': symbol.upper()}
            
            if start_date:
                query += " AND sp.date >= :start_date"
                params['start_date'] = start_date
                
            if end_date:
                query += " AND sp.date <= :end_date"
                params['end_date'] = end_date
                
            query += " ORDER BY sp.date DESC"
            
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
                
            logger.info(f"Retrieved {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock prices for {symbol}: {e}")
            raise
    
    # ===============================
    # NEWS & SENTIMENT OPERATIONS
    # ===============================
    
    def insert_financial_news(self, headline: str, content: Optional[str] = None,
                             publisher: Optional[str] = None, published_at: Optional[datetime] = None,
                             url: Optional[str] = None, source: Optional[str] = None,
                             category: Optional[str] = None) -> str:
        """Insert financial news record"""
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO financial_news (headline, content, publisher, published_at, url, source, category)
                    VALUES (:headline, :content, :publisher, :published_at, :url, :source, :category)
                    RETURNING id
                """)
                
                result = session.execute(query, {
                    'headline': headline,
                    'content': content,
                    'publisher': publisher,
                    'published_at': published_at or datetime.now(),
                    'url': url,
                    'source': source,
                    'category': category
                })
                
                row = result.fetchone()
                if row is None:
                    raise ValueError("Failed to insert news")
                news_id = row[0]
                logger.info(f"Inserted news: {headline[:50]}...")
                return str(news_id)
                
        except Exception as e:
            logger.error(f"Error inserting news: {e}")
            raise
    
    def insert_sentiment_score(self, news_id: str, stock_id: Optional[str] = None,
                              symbol: Optional[str] = None, sentiment_label: str = 'neutral',
                              sentiment_score: float = 0.0, confidence_score: float = 0.0,
                              model_version: str = 'finbert-1.0') -> str:
        """Insert sentiment analysis result"""
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO sentiment_scores (news_id, stock_id, symbol, sentiment_label, 
                                                sentiment_score, confidence_score, model_version)
                    VALUES (:news_id, :stock_id, :symbol, :sentiment_label, 
                           :sentiment_score, :confidence_score, :model_version)
                    RETURNING id
                """)
                
                result = session.execute(query, {
                    'news_id': news_id,
                    'stock_id': stock_id,
                    'symbol': symbol.upper() if symbol else None,
                    'sentiment_label': sentiment_label,
                    'sentiment_score': sentiment_score,
                    'confidence_score': confidence_score,
                    'model_version': model_version
                })
                
                row = result.fetchone()
                if row is None:
                    raise ValueError("Failed to insert sentiment score")
                sentiment_id = row[0]
                logger.info(f"Inserted sentiment for {symbol}: {sentiment_label}")
                return str(sentiment_id)
                
        except Exception as e:
            logger.error(f"Error inserting sentiment: {e}")
            raise

    def bulk_insert_financial_news(self, news_data: List[Dict]) -> List[Dict]:
        """
        Bulk insert financial news records and return mapping of headlines to IDs
        
        Args:
            news_data: List of dictionaries with news data
            
        Returns:
            List of dicts with 'headline' and 'id'
        """
        try:
            if not news_data:
                return []
                
            with self.get_session() as session:
                query = text("""
                    INSERT INTO financial_news (headline, content, publisher, published_at, url, source, category)
                    VALUES (:headline, :content, :publisher, :published_at, :url, :source, :category)
                    ON CONFLICT (headline, published_at) DO NOTHING
                    RETURNING id, headline
                """)
                
                # Ensure all fields are present
                records = []
                for item in news_data:
                    records.append({
                        'headline': item.get('headline'),
                        'content': item.get('content'),
                        'publisher': item.get('publisher'),
                        'published_at': item.get('published_at', datetime.now()),
                        'url': item.get('url'),
                        'source': item.get('source'),
                        'category': item.get('category')
                    })
                
                result = session.execute(query, records)
                inserted_records = [{'id': str(row[0]), 'headline': row[1]} for row in result.fetchall()]
                
                logger.info(f"Bulk inserted {len(inserted_records)} news items")
                return inserted_records
                
        except Exception as e:
            logger.error(f"Error bulk inserting news: {e}")
            raise

    def bulk_insert_sentiment_scores(self, sentiment_data: List[Dict]) -> int:
        """
        Bulk insert sentiment scores
        
        Args:
            sentiment_data: List of dictionaries with sentiment data
            
        Returns:
            Number of records inserted
        """
        try:
            if not sentiment_data:
                return 0
                
            with self.get_session() as session:
                query = text("""
                    INSERT INTO sentiment_scores (news_id, stock_id, symbol, sentiment_label, 
                                                sentiment_score, confidence_score, model_version)
                    VALUES (:news_id, :stock_id, :symbol, :sentiment_label, 
                           :sentiment_score, :confidence_score, :model_version)
                """)
                
                # Prepare records
                records = []
                for item in sentiment_data:
                    records.append({
                        'news_id': item['news_id'],
                        'stock_id': item.get('stock_id'),
                        'symbol': item.get('symbol'),
                        'sentiment_label': item['sentiment_label'],
                        'sentiment_score': item['sentiment_score'],
                        'confidence_score': item['confidence_score'],
                        'model_version': item.get('model_version', 'finbert-1.0')
                    })
                
                session.execute(query, records)
                logger.info(f"Bulk inserted {len(records)} sentiment scores")
                return len(records)
                
        except Exception as e:
            logger.error(f"Error bulk inserting sentiment scores: {e}")
            raise
    
    # ===============================
    # UTILITY METHODS
    # ===============================
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute custom SQL query and return DataFrame"""
        try:
            with self.get_connection() as conn:
                if params:
                    df = pd.read_sql(query, conn, params=params)
                else:
                    df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get record counts for all tables"""
        try:
            with self.get_connection() as conn:
                tables = [
                    'stocks', 'stock_prices', 'financial_news', 'news_stock_mentions',
                    'sentiment_scores', 'users', 'portfolios', 'portfolio_holdings',
                    'technical_indicators', 'risk_metrics', 'user_queries'
                ]
                
                stats = {}
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    row = result.fetchone()
                    stats[table] = row[0] if row else 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            raise
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for common operations
def get_db_session():
    """Get database session context manager"""
    return db_manager.get_session()

def get_db_connection():
    """Get database connection context manager"""
    return db_manager.get_connection()

def execute_sql(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """Execute SQL query and return DataFrame"""
    return db_manager.execute_query(query, params)
