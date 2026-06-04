#!/usr/bin/env python3
# ===============================
# DATA CLEANING & PREPROCESSING
# Financial Advisory System - Clean all stock data
# ===============================

import pandas as pd
import numpy as np
from datetime import datetime, date
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append('/teamspace/studios/this_studio/financial_advisory_system')

from src.database.connection import db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Cleans and preprocesses all stock market data for the Financial Advisory System
    Handles US S&P 500 stocks + INFY data quality issues
    """
    
    def __init__(self):
        """Initialize data cleaner"""
        self.raw_data_dir = "/teamspace/studios/this_studio/financial_advisory_system/data/raw"
        self.processed_data_dir = "/teamspace/studios/this_studio/financial_advisory_system/data/processed"
        
        # Ensure processed directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
    def load_raw_stock_data(self) -> pd.DataFrame:
        """
        Load raw S&P 500 stock data from CSV files
        
        Returns:
            Combined DataFrame with all US stock data
        """
        try:
            logger.info("Loading raw S&P 500 stock data")
            
            # Load the main stock price data
            stock_file = os.path.join(self.raw_data_dir, "sp500_stocks.csv")
            
            if not os.path.exists(stock_file):
                raise FileNotFoundError(f"Stock data file not found: {stock_file}")
            
            # Read the stock data
            df = pd.read_csv(stock_file)
            
            logger.info(f"Loaded {len(df)} raw stock price records")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading raw stock data: {e}")
            raise
    
    def load_stock_universe(self) -> pd.DataFrame:
        """
        Load the selected 51-stock universe
        
        Returns:
            DataFrame with stock universe metadata
        """
        try:
            universe_file = os.path.join(self.processed_data_dir, "stock_universe.csv")
            
            if not os.path.exists(universe_file):
                raise FileNotFoundError(f"Stock universe file not found: {universe_file}")
            
            df = pd.read_csv(universe_file)
            logger.info(f"Loaded {len(df)} stocks in universe")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading stock universe: {e}")
            raise
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate stock price data
        
        Args:
            df: Raw price data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info("Cleaning stock price data")
            
            # Make a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Step 1: Standardize column names
            # Lowercase all columns first to handle case variations
            cleaned_df.columns = cleaned_df.columns.str.lower()
            
            column_mapping = {
                'date': 'date',
                'symbol': 'symbol', 
                'ticker': 'symbol', # Handle potential ticker column name
                'open': 'open',
                'high': 'high',
                'low': 'low', 
                'close': 'close',
                'adj close': 'adjusted_close',
                'adjusted_close': 'adjusted_close',
                'volume': 'volume'
            }
            
            # Rename columns if they exist
            cleaned_df = cleaned_df.rename(columns=column_mapping)
            
            # Verify required columns exist
            required_columns = ['symbol', 'date', 'close']
            missing = [col for col in required_columns if col not in cleaned_df.columns]
            if missing:
                raise ValueError(f"Missing required columns in price data: {missing}. Available: {list(cleaned_df.columns)}")
            
            # Step 2: Convert date column to datetime
            if 'date' in cleaned_df.columns:
                cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
                # Convert to date only (remove time component)
                cleaned_df['date'] = cleaned_df['date'].dt.date
            
            # Step 3: Ensure symbol is uppercase
            if 'symbol' in cleaned_df.columns:
                cleaned_df['symbol'] = cleaned_df['symbol'].str.upper()
            
            # Step 4: Convert price columns to numeric
            price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
            for col in price_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            # Step 5: Convert volume to integer
            if 'volume' in cleaned_df.columns:
                cleaned_df['volume'] = pd.to_numeric(cleaned_df['volume'], errors='coerce')
                cleaned_df['volume'] = cleaned_df['volume'].fillna(0).astype('int64')
            
            # Step 6: Filter target date range (2022-2024)
            if 'date' in cleaned_df.columns:
                start_date = date(2022, 1, 1)
                end_date = date(2024, 12, 31)
                
                mask = (cleaned_df['date'] >= start_date) & (cleaned_df['date'] <= end_date)
                cleaned_df = cleaned_df[mask].copy()
            
            # Step 7: Remove rows with invalid prices (<=0 or null)
            price_cols = [col for col in price_columns if col in cleaned_df.columns]
            for col in price_cols:
                cleaned_df = cleaned_df[cleaned_df[col] > 0].copy()
            
            # Step 8: Validate price relationships (high >= low, etc.)
            if all(col in cleaned_df.columns for col in ['high', 'low', 'open', 'close']):
                # Remove records where high < low (invalid data)
                invalid_mask = cleaned_df['high'] < cleaned_df['low']
                if invalid_mask.sum() > 0:
                    logger.warning(f"Removing {invalid_mask.sum()} records with invalid high/low prices")
                    cleaned_df = cleaned_df[~invalid_mask].copy()
                
                # Ensure high >= open, close and low <= open, close
                cleaned_df = cleaned_df[
                    (cleaned_df['high'] >= cleaned_df['open']) &
                    (cleaned_df['high'] >= cleaned_df['close']) &
                    (cleaned_df['low'] <= cleaned_df['open']) &
                    (cleaned_df['low'] <= cleaned_df['close'])
                ].copy()
            
            # Step 9: Sort by symbol and date
            if all(col in cleaned_df.columns for col in ['symbol', 'date']):
                cleaned_df = cleaned_df.sort_values(['symbol', 'date']).reset_index(drop=True)
            
            logger.info(f"Cleaned data: {len(cleaned_df)} records remaining")
            logger.info(f"Date range: {cleaned_df['date'].min()} to {cleaned_df['date'].max()}")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning price data: {e}")
            raise
    
    def filter_universe_stocks(self, df: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to include only stocks in our 51-stock universe
        
        Args:
            df: All stock data
            universe_df: Stock universe metadata
            
        Returns:
            Filtered DataFrame with universe stocks only
        """
        try:
            logger.info("Filtering data for stock universe")
            
            # Ensure proper columns exist in universe_df
            if 'Symbol' in universe_df.columns:
                universe_symbols = set(universe_df['Symbol'].str.upper())
            elif 'symbol' in universe_df.columns:
                universe_symbols = set(universe_df['symbol'].str.upper())
            else:
                # Try to find symbol column
                symbol_col = next((col for col in universe_df.columns if col.lower() == 'symbol'), None)
                if symbol_col:
                    universe_symbols = set(universe_df[symbol_col].str.upper())
                else:
                    raise KeyError("Could not find 'symbol' column in universe data")
            
            # Filter main data
            filtered_df = df[df['symbol'].isin(universe_symbols)].copy()
            
            logger.info(f"Filtered to {len(filtered_df)} records for universe stocks")
            logger.info(f"Unique symbols: {filtered_df['symbol'].nunique()}")
            
            # Check coverage
            found_symbols = set(filtered_df['symbol'].unique())
            missing_symbols = universe_symbols - found_symbols
            
            if missing_symbols:
                logger.warning(f"Missing data for symbols: {missing_symbols}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering universe stocks: {e}")
            raise
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic calculated features to the dataset
        
        Args:
            df: Clean price data
            
        Returns:
            DataFrame with additional features
        """
        try:
            logger.info("Adding basic calculated features")
            
            # Sort by symbol and date to ensure proper calculation
            df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
            
            # Calculate daily returns
            df['daily_return'] = df.groupby('symbol')['close'].pct_change()
            
            # Calculate price change
            df['price_change'] = df.groupby('symbol')['close'].diff()
            
            # Calculate intraday range
            df['intraday_range'] = df['high'] - df['low']
            df['intraday_range_pct'] = (df['intraday_range'] / df['close']) * 100
            
            # Calculate typical price (HLC/3)
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Add trading day of week
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)
            
            logger.info("Added basic features: daily_return, price_change, intraday_range, typical_price")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding basic features: {e}")
            raise
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        try:
            logger.info("Generating data quality report")
            
            report = {
                'total_records': len(df),
                'unique_symbols': df['symbol'].nunique(),
                'date_range': {
                    'start': str(df['date'].min()),
                    'end': str(df['date'].max()),
                    'days': (df['date'].max() - df['date'].min()).days
                },
                'missing_values': df.isnull().sum().to_dict(),
                'data_coverage': {},
                'price_statistics': {},
                'volume_statistics': {}
            }
            
            # Data coverage by symbol
            symbol_counts = df['symbol'].value_counts()
            report['data_coverage'] = {
                'records_per_symbol': symbol_counts.describe().to_dict(),
                'symbols_with_full_data': int((symbol_counts >= 700).sum()),  # ~3 years of trading days
                'symbols_with_partial_data': int((symbol_counts < 700).sum())
            }
            
            # Price statistics
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    report['price_statistics'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std())
                    }
            
            # Volume statistics
            if 'volume' in df.columns:
                report['volume_statistics'] = {
                    'min': int(df['volume'].min()),
                    'max': int(df['volume'].max()),
                    'mean': float(df['volume'].mean()),
                    'zero_volume_days': int((df['volume'] == 0).sum())
                }
            
            # Daily return statistics
            if 'daily_return' in df.columns:
                returns = df['daily_return'].dropna()
                report['return_statistics'] = {
                    'mean_daily_return': float(returns.mean()),
                    'volatility': float(returns.std()),
                    'min_return': float(returns.min()),
                    'max_return': float(returns.max()),
                    'skewness': float(returns.skew()),
                    'kurtosis': float(returns.kurtosis())
                }
            
            logger.info("Data quality report generated")
            return report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            raise
    
    def save_cleaned_data(self, df: pd.DataFrame, filename: str = "cleaned_stock_data.csv") -> str:
        """
        Save cleaned data to CSV file
        
        Args:
            df: Cleaned DataFrame
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        try:
            filepath = os.path.join(self.processed_data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Saved cleaned data to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            raise
    
    def run_complete_cleaning_pipeline(self) -> Dict:
        """
        Run the complete data cleaning pipeline
        
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING DATA CLEANING PIPELINE")
            logger.info("=" * 60)
            
            results = {}
            
            # Step 1: Load raw data
            logger.info("Step 1: Loading raw stock data")
            raw_data = self.load_raw_stock_data()
            results['raw_records'] = len(raw_data)
            
            # Step 2: Load stock universe
            logger.info("Step 2: Loading stock universe")
            universe_data = self.load_stock_universe()
            results['universe_stocks'] = len(universe_data)
            
            # Step 3: Clean price data
            logger.info("Step 3: Cleaning price data")
            cleaned_data = self.clean_price_data(raw_data)
            results['cleaned_records'] = len(cleaned_data)
            
            # Step 4: Filter for universe stocks
            logger.info("Step 4: Filtering for universe stocks")
            universe_data_clean = self.filter_universe_stocks(cleaned_data, universe_data)
            results['universe_records'] = len(universe_data_clean)
            
            # Step 5: Add calculated features
            logger.info("Step 5: Adding calculated features")
            final_data = self.add_basic_features(universe_data_clean)
            results['final_records'] = len(final_data)
            
            # Step 6: Generate quality report
            logger.info("Step 6: Generating data quality report")
            quality_report = self.generate_data_quality_report(final_data)
            results['quality_report'] = quality_report
            
            # Step 7: Save cleaned data
            logger.info("Step 7: Saving cleaned data")
            csv_path = self.save_cleaned_data(final_data)
            results['output_file'] = csv_path
            
            # Step 8: Save quality report
            import json
            report_path = os.path.join(self.processed_data_dir, "data_quality_report.json")
            with open(report_path, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            results['quality_report_file'] = report_path
            
            logger.info("=" * 60)
            logger.info("DATA CLEANING PIPELINE COMPLETED!")
            logger.info(f"✅ Raw records: {results['raw_records']:,}")
            logger.info(f"✅ Cleaned records: {results['cleaned_records']:,}")
            logger.info(f"✅ Universe records: {results['universe_records']:,}")
            logger.info(f"✅ Final records: {results['final_records']:,}")
            logger.info(f"✅ Unique symbols: {quality_report['unique_symbols']}")
            logger.info(f"✅ Output file: {csv_path}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Data cleaning pipeline failed: {e}")
            raise

def main():
    """Main function to run data cleaning"""
    try:
        # Create cleaner instance
        cleaner = DataCleaner()
        
        # Run complete cleaning pipeline
        results = cleaner.run_complete_cleaning_pipeline()
        
        # Print summary
        print("\n🎯 DATA CLEANING SUMMARY:")
        print("=" * 50)
        print(f"📊 Raw Records: {results['raw_records']:,}")
        print(f"📊 Final Records: {results['final_records']:,}")
        print(f"📊 Universe Stocks: {results['universe_stocks']}")
        print(f"📊 Unique Symbols: {results['quality_report']['unique_symbols']}")
        print(f"📊 Date Range: {results['quality_report']['date_range']['start']} to {results['quality_report']['date_range']['end']}")
        print(f"📁 Output File: {results['output_file']}")
        print("=" * 50)
        print("✅ Data cleaning completed successfully!")
        print("🚀 Ready for FinBERT setup!")
        
        return True
        
    except Exception as e:
        print(f"❌ Data cleaning failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ All stock data is now clean and ready!")
    else:
        print("\n❌ Fix errors and try again")