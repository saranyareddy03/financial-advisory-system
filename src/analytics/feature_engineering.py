"""
Feature Engineering Engine for Financial Advisory System
Phase 3: Feature Engineering & Dataset Construction

Combines data from multiple sources to create training datasets:
- Price & Volume data (stock_prices)
- Technical Indicators (technical_indicators)
- Risk Metrics (risk_metrics)
- Sentiment Scores (sentiment_scores)

Creates lag features, target variables, and cleans data for ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
from sqlalchemy import text

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import DatabaseManager
from src.config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering Engine for building ML datasets.
    
    Aggregates data from disparate sources, aligns timestamps,
    generates lag features, and creates target variables.
    """
    
    def __init__(self):
        """Initialize with database connection."""
        self.db_manager = DatabaseManager()
        
    def get_price_volume_data(self, symbol: str, start_date: datetime) -> pd.DataFrame:
        """Fetch raw price and volume data."""
        query = """
        SELECT sp.date, sp.open_price, sp.high_price, sp.low_price, 
               sp.close_price, sp.adjusted_close, sp.volume
        FROM stock_prices sp
        JOIN stocks s ON sp.stock_id = s.id
        WHERE s.symbol = %s AND sp.date >= %s
        ORDER BY sp.date ASC
        """
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=(symbol, start_date))
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return pd.DataFrame()

    def get_technical_indicators(self, symbol: str, start_date: datetime) -> pd.DataFrame:
        """Fetch pre-calculated technical indicators."""
        query = """
        SELECT ti.date, ti.sma_20, ti.sma_50, ti.sma_200, ti.rsi_14, 
               ti.macd, ti.macd_signal, ti.macd_histogram, 
               ti.bollinger_upper, ti.bollinger_lower, ti.volume_sma_20
        FROM technical_indicators ti
        JOIN stocks s ON ti.stock_id = s.id
        WHERE s.symbol = %s AND ti.date >= %s
        ORDER BY ti.date ASC
        """
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=(symbol, start_date))
                
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching indicators for {symbol}: {e}")
            return pd.DataFrame()

    def get_risk_metrics(self, symbol: str, start_date: datetime) -> pd.DataFrame:
        """Fetch risk metrics."""
        query = """
        SELECT rm.calculation_date as date, rm.volatility_30d, rm.beta, 
               rm.sharpe_ratio, rm.value_at_risk_95, rm.max_drawdown
        FROM risk_metrics rm
        JOIN stocks s ON rm.stock_id = s.id
        WHERE s.symbol = %s AND rm.calculation_date >= %s
        ORDER BY rm.calculation_date ASC
        """
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=(symbol, start_date))
                
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching risk metrics for {symbol}: {e}")
            return pd.DataFrame()

    def get_sentiment_data(self, symbol: str, start_date: datetime) -> pd.DataFrame:
        """
        Fetch and aggregate sentiment data.
        """
        query = """
        SELECT ss.processed_at as date, ss.sentiment_score, ss.sentiment_label
        FROM sentiment_scores ss
        JOIN stocks s ON ss.stock_id = s.id
        WHERE s.symbol = %s AND ss.processed_at >= %s
        ORDER BY ss.processed_at ASC
        """
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=(symbol, start_date))
                
            if df.empty:
                return pd.DataFrame()
                
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
            
            # Add missing relevance_score
            if 'relevance_score' not in df.columns:
                 df['relevance_score'] = 1.0

            # Aggregate to daily level
            agg_funcs = {
                'sentiment_score': 'mean',
                'relevance_score': 'mean',
                'sentiment_label': 'count' # Proxy for news volume
            }
            
            daily_sentiment = df.groupby('date').agg(agg_funcs)
            daily_sentiment = daily_sentiment.rename(columns={'sentiment_label': 'news_volume'})
            
            return daily_sentiment
        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}")
            return pd.DataFrame()

    def create_feature_set(self, symbol: str, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Master method to create the complete feature set for a stock.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            
        Returns:
            DataFrame with all features and targets
        """
        if start_date is None:
            # Default to sufficient history
            start_date = datetime(2022, 1, 1)
            
        logger.info(f"Building feature set for {symbol} starting from {start_date.date()}")
        
        try:
            # 1. Fetch data
            prices = self.get_price_volume_data(symbol, start_date)
            
            if prices.empty:
                logger.warning(f"No price data found for {symbol}")
                return pd.DataFrame()
                
            tech_ind = self.get_technical_indicators(symbol, start_date)
            risk = self.get_risk_metrics(symbol, start_date)
            sentiment = self.get_sentiment_data(symbol, start_date)
            
            # 2. Merge data (Left join on prices to keep trading days)
            # Use 'date' index for joining
            df = prices.copy()
            
            if not tech_ind.empty:
                df = df.join(tech_ind, rsuffix='_ti')
            else:
                logger.warning(f"No technical indicators found for {symbol}")
                
            if not risk.empty:
                df = df.join(risk, rsuffix='_risk')
            else:
                logger.warning(f"No risk metrics found for {symbol}")
                
            if not sentiment.empty:
                df = df.join(sentiment, rsuffix='_sent')
            else:
                logger.info(f"No sentiment data found for {symbol}")
            
            # 3. Handle missing values
            # Fill sentiment with neutral/zeros
            sentiment_cols = ['sentiment_score', 'relevance_score', 'news_volume']
            for col in sentiment_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            # Forward fill risk metrics (they might be calculated less frequently)
            df.fillna(method='ffill', inplace=True)
            
            # 4. Feature Engineering
            
            # Returns
            df['return_1d'] = df['adjusted_close'].pct_change()
            df['log_return'] = np.log(df['adjusted_close'] / df['adjusted_close'].shift(1))
            
            # Rolling features
            df['volatility_5d'] = df['log_return'].rolling(window=5).std()
            df['volatility_20d'] = df['log_return'].rolling(window=20).std()
            
            # Momentum
            df['momentum_5d'] = df['adjusted_close'] / df['adjusted_close'].shift(5) - 1
            
            # Interaction features
            if 'sma_50' in df.columns:
                df['price_vs_sma50'] = df['adjusted_close'] / df['sma_50'] - 1
            
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['adjusted_close']
            
            # Drop remaining NaNs (e.g., calculation warmup periods) AFTER creating features
            # The previous dropna was too aggressive and removed data needed for rolling windows
            # We also need to be careful not to drop everything if risk metrics are sparse
            original_len = len(df)
            
            # Drop only if essential features are missing
            essential_cols = ['return_1d', 'sma_200', 'rsi_14']
            available_essential = [col for col in essential_cols if col in df.columns]
            
            if available_essential:
                df.dropna(subset=available_essential, inplace=True)
            
            logger.info(f"Dropped {original_len - len(df)} rows due to missing data")
                
            # 5. Create Targets
            # Target: Next day return
            df['target_return_1d'] = df['return_1d'].shift(-1)
            
            # Target: Next 5 day return
            df['target_return_5d'] = df['adjusted_close'].shift(-5) / df['adjusted_close'] - 1
            
            # Target: Binary direction (1 for up, 0 for down)
            df['target_direction'] = (df['target_return_1d'] > 0).astype(int)
            
            # Remove last rows with NaN targets
            df.dropna(subset=['target_return_1d'], inplace=True)
            
            logger.info(f"✅ Successfully created feature set for {symbol}: {len(df)} samples, {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error creating feature set for {symbol}: {e}")
            return pd.DataFrame()
    
    def store_features_to_db(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Store features to ml_features table.
        Creates table if it doesn't exist.
        """
        try:
            # 1. Ensure table exists
            # We use raw SQL for table creation to ensure it exists
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS ml_features (
                id SERIAL PRIMARY KEY,
                stock_id UUID REFERENCES stocks(id),
                date DATE NOT NULL,
                
                -- Price & Volume
                adjusted_close DECIMAL(10, 2),
                volume BIGINT,
                return_1d DECIMAL(10, 4),
                log_return DECIMAL(10, 4),
                volatility_5d DECIMAL(10, 4),
                volatility_20d DECIMAL(10, 4),
                momentum_5d DECIMAL(10, 4),
                price_vs_sma50 DECIMAL(10, 4),
                
                -- Technical Indicators
                rsi_14 DECIMAL(10, 4),
                macd DECIMAL(10, 4),
                macd_signal DECIMAL(10, 4),
                macd_histogram DECIMAL(10, 4),
                bb_upper DECIMAL(10, 4),
                bb_lower DECIMAL(10, 4),
                bb_width DECIMAL(10, 4),
                volume_sma_20 DECIMAL(15, 2),
                
                -- Risk Metrics
                volatility_30d DECIMAL(10, 4),
                beta DECIMAL(10, 4),
                sharpe_ratio DECIMAL(10, 4),
                value_at_risk_95 DECIMAL(10, 4),
                max_drawdown DECIMAL(10, 4),
                
                -- Sentiment
                sentiment_score DECIMAL(10, 4),
                relevance_score DECIMAL(10, 4),
                news_volume INTEGER,
                
                -- Targets
                target_return_1d DECIMAL(10, 4),
                target_return_5d DECIMAL(10, 4),
                target_direction INTEGER,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stock_id, date)
            );
            """
            
            with self.db_manager.get_connection() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            
            # 2. Get stock_id
            stock_query = "SELECT id FROM stocks WHERE symbol = %s"
            with self.db_manager.get_connection() as conn:
                stock_result = pd.read_sql(stock_query, conn, params=(symbol,))
                if stock_result.empty:
                    logger.error(f"Stock {symbol} not found")
                    return False
                stock_id = stock_result.iloc[0]['id']
            
            # 3. Prepare data for insertion
            records = []
            for date, row in df.iterrows():
                record = {
                    'stock_id': stock_id,
                    'date': date,
                    'adjusted_close': row.get('adjusted_close'),
                    'volume': row.get('volume'),
                    'return_1d': row.get('return_1d'),
                    'log_return': row.get('log_return'),
                    'volatility_5d': row.get('volatility_5d'),
                    'volatility_20d': row.get('volatility_20d'),
                    'momentum_5d': row.get('momentum_5d'),
                    'price_vs_sma50': row.get('price_vs_sma50'),
                    'rsi_14': row.get('rsi_14'),
                    'macd': row.get('macd'), # Note: df might have macd_line or macd
                    'macd_signal': row.get('macd_signal'),
                    'macd_histogram': row.get('macd_histogram'),
                    'bb_upper': row.get('bb_upper'),
                    'bb_lower': row.get('bb_lower'),
                    'bb_width': row.get('bb_width'),
                    'volume_sma_20': row.get('volume_sma_20'),
                    'volatility_30d': row.get('volatility_30d'),
                    'beta': row.get('beta'),
                    'sharpe_ratio': row.get('sharpe_ratio'),
                    'value_at_risk_95': row.get('value_at_risk_95'),
                    'max_drawdown': row.get('max_drawdown'),
                    'sentiment_score': row.get('sentiment_score'),
                    'relevance_score': row.get('relevance_score'),
                    'news_volume': row.get('news_volume'),
                    'target_return_1d': row.get('target_return_1d'),
                    'target_return_5d': row.get('target_return_5d'),
                    'target_direction': row.get('target_direction')
                }
                
                # Handle NaNs (replace with None for SQL)
                for k, v in record.items():
                    if pd.isna(v):
                        record[k] = None
                        
                records.append(record)
            
            if not records:
                return True
                
            # 4. Insert data
            insert_sql = text("""
            INSERT INTO ml_features (
                stock_id, date, adjusted_close, volume, return_1d, log_return,
                volatility_5d, volatility_20d, momentum_5d, price_vs_sma50,
                rsi_14, macd, macd_signal, macd_histogram, bb_upper, bb_lower, bb_width,
                volume_sma_20, volatility_30d, beta, sharpe_ratio, value_at_risk_95,
                max_drawdown, sentiment_score, relevance_score, news_volume,
                target_return_1d, target_return_5d, target_direction
            ) VALUES (
                :stock_id, :date, :adjusted_close, :volume, :return_1d, :log_return,
                :volatility_5d, :volatility_20d, :momentum_5d, :price_vs_sma50,
                :rsi_14, :macd, :macd_signal, :macd_histogram, :bb_upper, :bb_lower, :bb_width,
                :volume_sma_20, :volatility_30d, :beta, :sharpe_ratio, :value_at_risk_95,
                :max_drawdown, :sentiment_score, :relevance_score, :news_volume,
                :target_return_1d, :target_return_5d, :target_direction
            )
            ON CONFLICT (stock_id, date) DO UPDATE SET
                adjusted_close = EXCLUDED.adjusted_close,
                return_1d = EXCLUDED.return_1d,
                target_return_1d = EXCLUDED.target_return_1d,
                target_direction = EXCLUDED.target_direction
            """)
            
            with self.db_manager.get_connection() as conn:
                conn.execute(insert_sql, records)
                conn.commit()
                
            logger.info(f"Stored {len(records)} feature records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing features for {symbol}: {e}")
            return False

    def process_stock_features(self, symbol: str) -> bool:
        """
        Complete pipeline: create features and store them.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            bool: Success status
        """
        logger.info(f"Processing features for {symbol}")
        
        # 1. Create features
        df = self.create_feature_set(symbol)
        
        if df.empty:
            logger.warning(f"No features created for {symbol}")
            return False
            
        # 2. Store features
        return self.store_features_to_db(df, symbol)

def main():
    """Test the Feature Engineer."""
    print("🏗️ Feature Engineering Test")
    print("=" * 50)
    
    fe = FeatureEngineer()
    test_symbols = ['MSFT', 'GOOG']
    
    for symbol in test_symbols:
        print(f"\nProcessing {symbol}...")
        success = fe.process_stock_features(symbol)
        
        if success:
            print(f"✅ Features processed and stored for {symbol}")
        else:
            print(f"❌ Failed to process features for {symbol}")

if __name__ == "__main__":
    main()
