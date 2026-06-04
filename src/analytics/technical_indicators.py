
"""
Technical Indicators Calculator for Financial Advisory System
Phase 3: Technical Analysis & Feature Engineering

Calculates essential trading indicators:
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index) 
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators

Stores results in technical_indicators table for ML and advisory features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
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

class TechnicalIndicators:
    """
    Technical Indicators Calculator for stock market analysis.
    
    Calculates major technical indicators and stores them in database
    for use in portfolio optimization and ML models.
    """
    
    def __init__(self):
        """Initialize with database connection and settings."""
        self.settings = config
        self.db_manager = DatabaseManager()
        
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=window, min_periods=window).mean()
    
    def calculate_ema(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss over specified period
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Returns:
            dict: Contains 'macd_line', 'signal_line', 'histogram'
        """
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line, 
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, 
                                 window: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Returns:
            dict: Contains 'middle_band', 'upper_band', 'lower_band'
        """
        middle_band = self.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band
        }
    
    def calculate_volume_indicators(self, prices: pd.Series, 
                                  volumes: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators.
        
        Returns:
            dict: Contains 'volume_sma', 'price_volume', 'volume_ratio'
        """
        volume_sma = volumes.rolling(window=20).mean()
        price_volume = prices * volumes  # Price-weighted volume
        volume_ratio = volumes / volume_sma  # Current vs average volume
        
        return {
            'volume_sma_20': volume_sma,
            'price_volume': price_volume,
            'volume_ratio': volume_ratio
        }
    
    def get_stock_price_data(self, symbol: str, 
                           start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch stock price data from database.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Optional start date (defaults to 1 year ago)
        
        Returns:
            DataFrame with price data sorted by date
        """
        if start_date is None:
            # Use 2022-01-01 as default start date to capture available historical data
            # Data range in DB is typically 2022-2024
            start_date = datetime(2022, 1, 1)
        
        query = """
        SELECT sp.date, sp.open_price, sp.high_price, sp.low_price, 
               sp.close_price, sp.volume, sp.adjusted_close
        FROM stock_prices sp
        JOIN stocks s ON sp.stock_id = s.id
        WHERE s.symbol = %s AND sp.date >= %s
        ORDER BY sp.date ASC
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=(symbol, start_date))
                
            if df.empty:
                logger.warning(f"No price data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_all_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Calculate all technical indicators for a given stock symbol.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            DataFrame with all indicators or None if error
        """
        try:
            # Get price data
            price_data = self.get_stock_price_data(symbol)
            
            if price_data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Use adjusted close for calculations
            prices = price_data['adjusted_close']
            volumes = price_data['volume']
            
            # Initialize results DataFrame
            indicators = pd.DataFrame(index=price_data.index)
            indicators['symbol'] = symbol
            indicators['close_price'] = prices
            
            # Moving Averages
            indicators['sma_20'] = self.calculate_sma(prices, 20)
            indicators['sma_50'] = self.calculate_sma(prices, 50)
            indicators['sma_200'] = self.calculate_sma(prices, 200)
            indicators['ema_12'] = self.calculate_ema(prices, 12)
            indicators['ema_26'] = self.calculate_ema(prices, 26)
            
            # RSI
            indicators['rsi_14'] = self.calculate_rsi(prices, 14)
            
            # MACD
            macd = self.calculate_macd(prices)
            indicators['macd_line'] = macd['macd_line']
            indicators['macd_signal'] = macd['signal_line']
            indicators['macd_histogram'] = macd['histogram']
            
            # Bollinger Bands
            bb = self.calculate_bollinger_bands(prices)
            indicators['bb_middle'] = bb['middle_band']
            indicators['bb_upper'] = bb['upper_band']
            indicators['bb_lower'] = bb['lower_band']
            
            # Volume Indicators
            volume_indicators = self.calculate_volume_indicators(prices, volumes)
            indicators['volume_sma_20'] = volume_indicators['volume_sma_20']
            indicators['volume_ratio'] = volume_indicators['volume_ratio']
            
            # Additional derived indicators
            indicators['price_sma20_ratio'] = prices / indicators['sma_20']
            indicators['bb_position'] = ((prices - indicators['bb_lower']) / 
                                       (indicators['bb_upper'] - indicators['bb_lower']))
            
            # Remove rows with insufficient data (NaN values in key indicators)
            indicators = indicators.dropna(subset=['sma_20', 'rsi_14', 'macd_line'])
            
            logger.info(f"Calculated indicators for {symbol}: {len(indicators)} records")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    def store_indicators_to_db(self, indicators_df: pd.DataFrame, symbol: str) -> bool:
        """
        Store calculated indicators to technical_indicators table.
        
        Args:
            indicators_df: DataFrame with calculated indicators
            symbol: Stock symbol
            
        Returns:
            bool: Success status
        """
        try:
            # Get stock_id for the symbol
            stock_query = "SELECT id FROM stocks WHERE symbol = %s"
            
            with self.db_manager.get_connection() as conn:
                stock_result = pd.read_sql(stock_query, conn, params=(symbol,))
                
                if stock_result.empty:
                    logger.error(f"Stock symbol {symbol} not found in stocks table")
                    return False
                
                stock_id = stock_result.iloc[0]['id']
                
                # Prepare data for insertion
                records = []
                for date, row in indicators_df.iterrows():
                    record = {
                        'stock_id': stock_id,
                        'calculation_date': date,
                        'sma_20': row.get('sma_20'),
                        'sma_50': row.get('sma_50'),
                        'sma_200': row.get('sma_200'),
                        'ema_12': row.get('ema_12'),
                        'ema_26': row.get('ema_26'),
                        'rsi_14': row.get('rsi_14'),
                        'macd_line': row.get('macd_line'),
                        'macd_signal': row.get('macd_signal'),
                        'macd_histogram': row.get('macd_histogram'),
                        'bb_upper': row.get('bb_upper'),
                        'bb_middle': row.get('bb_middle'),
                        'bb_lower': row.get('bb_lower'),
                        'volume_sma_20': row.get('volume_sma_20'),
                        'symbol': symbol,
                        'created_at': datetime.now()
                    }
                    records.append(record)
                
                # Insert records (using ON CONFLICT to handle duplicates)
                insert_query = text("""
                INSERT INTO technical_indicators 
                (stock_id, date, sma_20, sma_50, sma_200, ema_12, ema_26,
                 rsi_14, macd, macd_signal, macd_histogram, bollinger_upper, bollinger_lower,
                 volume_sma_20, symbol, created_at)
                VALUES (:stock_id, :calculation_date, :sma_20, :sma_50, :sma_200,
                        :ema_12, :ema_26, :rsi_14, :macd_line, :macd_signal,
                        :macd_histogram, :bb_upper, :bb_lower,
                        :volume_sma_20, :symbol, :created_at)
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    sma_20 = EXCLUDED.sma_20,
                    sma_50 = EXCLUDED.sma_50,
                    sma_200 = EXCLUDED.sma_200,
                    ema_12 = EXCLUDED.ema_12,
                    ema_26 = EXCLUDED.ema_26,
                    rsi_14 = EXCLUDED.rsi_14,
                    macd = EXCLUDED.macd,
                    macd_signal = EXCLUDED.macd_signal,
                    macd_histogram = EXCLUDED.macd_histogram,
                    bollinger_upper = EXCLUDED.bollinger_upper,
                    bollinger_lower = EXCLUDED.bollinger_lower,
                    volume_sma_20 = EXCLUDED.volume_sma_20
                """)
                
                conn.execute(insert_query, records)
                conn.commit()
                
                logger.info(f"Stored {len(records)} indicator records for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing indicators for {symbol}: {e}")
            return False
    
    def process_stock_indicators(self, symbol: str) -> bool:
        """
        Complete pipeline: calculate and store indicators for a stock.
        
        Args:
            symbol: Stock symbol to process
            
        Returns:
            bool: Success status
        """
        logger.info(f"Processing technical indicators for {symbol}")
        
        # Calculate indicators
        indicators = self.calculate_all_indicators(symbol)
        
        if indicators is None or indicators.empty:
            logger.warning(f"No indicators calculated for {symbol}")
            return False
        
        # Store to database
        success = self.store_indicators_to_db(indicators, symbol)
        
        if success:
            logger.info(f"✅ Successfully processed indicators for {symbol}")
        else:
            logger.error(f"❌ Failed to store indicators for {symbol}")
        
        return success
    
    def process_all_stocks(self, symbols: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Process technical indicators for multiple stocks.
        
        Args:
            symbols: List of symbols to process (if None, processes all stocks in DB)
            
        Returns:
            dict: Symbol -> success status mapping
        """
        if symbols is None:
            # Get all symbols from database
            try:
                with self.db_manager.get_connection() as conn:
                    symbols_df = pd.read_sql("SELECT symbol FROM stocks ORDER BY symbol", conn)
                    symbols = symbols_df['symbol'].tolist()
            except Exception as e:
                logger.error(f"Error fetching stock symbols: {e}")
                return {}
        
        logger.info(f"Processing indicators for {len(symbols)} stocks")
        
        results = {}
        successful = 0
        
        for symbol in symbols:
            try:
                success = self.process_stock_indicators(symbol)
                results[symbol] = success
                
                if success:
                    successful += 1
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = False
        
        logger.info(f"✅ Processing complete: {successful}/{len(symbols)} successful")
        return results


def main():
    """
    Main execution function for testing technical indicators calculator.
    """
    print("🚀 PHASE 3: Technical Indicators Calculator")
    print("=" * 50)
    
    try:
        # Initialize calculator
        calculator = TechnicalIndicators()
        
        # Test with a few major stocks first
        test_symbols = ['MSFT', 'GOOG', 'TSLA', 'NVDA', 'ABBV', 'CVX']
        
        print(f"📊 Testing indicators calculation for: {', '.join(test_symbols)}")
        
        results = {}
        for symbol in test_symbols:
            print(f"\n🔍 Processing {symbol}...")
            success = calculator.process_stock_indicators(symbol)
            results[symbol] = success
            
            if success:
                print(f"   ✅ {symbol}: SUCCESS")
            else:
                print(f"   ❌ {symbol}: FAILED")
        
        # Summary
        successful = sum(results.values())
        print(f"\n📈 RESULTS SUMMARY:")
        print(f"   Processed: {len(test_symbols)} stocks")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(test_symbols) - successful}")
        
        if successful > 0:
            print(f"\n✅ Technical indicators are working!")
            print(f"📊 Indicators stored in 'technical_indicators' table")
            print(f"🎯 Ready for next Phase 3 component")
        else:
            print(f"\n❌ Issues found. Check database connection and stock data.")
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()