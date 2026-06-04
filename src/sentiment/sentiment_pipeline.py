#!/usr/bin/env python3
# ===============================
# SENTIMENT ANALYSIS PIPELINE
# Financial Advisory System - Complete News Sentiment Processing
# ===============================

import pandas as pd
import numpy as np
from datetime import datetime, date
import sys
import os
from typing import Dict, List, Tuple, Optional
import json
import re

# Add project root to path
sys.path.append('/teamspace/studios/this_studio/financial_advisory_system')

from src.sentiment.finbert_setup import FinBERTProcessor
from src.database.connection import db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPipeline:
    """
    Complete sentiment analysis pipeline for financial news
    Processes news data, extracts stock mentions, runs FinBERT analysis,
    and stores results in database with stock associations
    """
    
    def __init__(self):
        """Initialize sentiment analysis pipeline"""
        self.finbert_processor = FinBERTProcessor()
        self.raw_data_dir = "/teamspace/studios/this_studio/financial_advisory_system/data/raw"
        self.processed_data_dir = "/teamspace/studios/this_studio/financial_advisory_system/data/processed"
        
        # Load stock universe for symbol matching
        self.stock_symbols = self._load_stock_universe()
        
    def _load_stock_universe(self) -> set:
        """
        Load stock symbols from universe file
        
        Returns:
            Set of stock symbols in uppercase
        """
        try:
            universe_file = os.path.join(self.processed_data_dir, "stock_universe.csv")
            df = pd.read_csv(universe_file)
            
            # Handle different possible column names
            if 'Symbol' in df.columns:
                symbols = set(df['Symbol'].str.upper())
            elif 'symbol' in df.columns:
                symbols = set(df['symbol'].str.upper())
            else:
                # Try to find symbol column
                symbol_col = next((col for col in df.columns if col.lower() == 'symbol'), None)
                if symbol_col:
                    symbols = set(df[symbol_col].str.upper())
                else:
                    raise KeyError("Could not find 'symbol' column in universe data")
            
            logger.info(f"Loaded {len(symbols)} stock symbols for mention detection")
            return symbols
            
        except Exception as e:
            logger.error(f"Error loading stock universe: {e}")
            # Return default set if loading fails
            return {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'INFY'}
    
    def load_news_data(self) -> pd.DataFrame:
        """
        Load financial news data from CSV files
        
        Returns:
            Combined DataFrame with news data
        """
        try:
            logger.info("Loading financial news data")
            
            news_dir = os.path.join(self.raw_data_dir, "news")
            news_files = []
            
            # Look for all CSV files in news directory
            if os.path.exists(news_dir):
                for file in os.listdir(news_dir):
                    if file.endswith('.csv'):
                        news_files.append(os.path.join(news_dir, file))
            
            if not news_files:
                raise FileNotFoundError("No news CSV files found in data/raw/news/")
            
            # Load and combine all news files
            all_news = []
            for file_path in news_files:
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = os.path.basename(file_path)
                    all_news.append(df)
                    logger.info(f"Loaded {len(df)} records from {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
            
            if not all_news:
                raise ValueError("No news data could be loaded")
            
            # Combine all news data
            combined_df = pd.concat(all_news, ignore_index=True)
            
            logger.info(f"Total news records loaded: {len(combined_df)}")
            logger.info(f"Columns available: {list(combined_df.columns)}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            raise
    
    def preprocess_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize news data
        
        Args:
            df: Raw news DataFrame
            
        Returns:
            Cleaned news DataFrame
        """
        try:
            logger.info("Preprocessing news data")
            
            # Make a copy
            news_df = df.copy()
            
            # Standardize column names (handle different possible column names)
            column_mapping = {}
            cols_lower = {col.lower(): col for col in news_df.columns}
            
            # Map common column variations
            if 'headline' in cols_lower:
                column_mapping[cols_lower['headline']] = 'headline'
            elif 'title' in cols_lower:
                column_mapping[cols_lower['title']] = 'headline'
            elif 'text' in cols_lower:
                column_mapping[cols_lower['text']] = 'headline'
            
            if 'date' in cols_lower:
                column_mapping[cols_lower['date']] = 'published_at'
            elif 'published_at' in cols_lower:
                column_mapping[cols_lower['published_at']] = 'published_at'
            elif 'timestamp' in cols_lower:
                column_mapping[cols_lower['timestamp']] = 'published_at'
            
            if 'publisher' in cols_lower:
                column_mapping[cols_lower['publisher']] = 'publisher'
            elif 'source' in cols_lower:
                column_mapping[cols_lower['source']] = 'publisher'
            
            # Apply column mapping
            news_df = news_df.rename(columns=column_mapping)
            
            # Ensure we have required columns
            if 'headline' not in news_df.columns:
                # Try to create headline from available text columns
                text_cols = [col for col in news_df.columns if 'text' in col.lower() or 'title' in col.lower()]
                if text_cols:
                    news_df['headline'] = news_df[text_cols[0]]
                else:
                    raise ValueError("No headline/title column found in news data")
            
            # Clean headlines
            news_df['headline'] = news_df['headline'].astype(str)
            news_df['headline'] = news_df['headline'].str.strip()
            
            # Remove empty headlines
            news_df = news_df[news_df['headline'].str.len() > 10].copy()
            
            # Handle dates if available
            if 'published_at' in news_df.columns:
                try:
                    news_df['published_at'] = pd.to_datetime(news_df['published_at'])
                except:
                    logger.warning("Could not parse published_at dates")
                    news_df['published_at'] = datetime.now()
            else:
                news_df['published_at'] = datetime.now()
            
            # Add default publisher if missing
            if 'publisher' not in news_df.columns:
                news_df['publisher'] = 'Unknown'
            
            # Add unique ID for tracking
            news_df['news_id'] = range(1, len(news_df) + 1)
            
            # Sort by date (newest first)
            news_df = news_df.sort_values('published_at', ascending=False).reset_index(drop=True)
            
            logger.info(f"Preprocessed {len(news_df)} news records")
            return news_df
            
        except Exception as e:
            logger.error(f"Error preprocessing news data: {e}")
            raise
    
    def extract_stock_mentions(self, headline: str) -> List[str]:
        """
        Extract stock symbol mentions from news headline
        
        Args:
            headline: News headline text
            
        Returns:
            List of mentioned stock symbols
        """
        try:
            if not headline or pd.isna(headline):
                return []
            
            headline_upper = str(headline).upper()
            mentioned_symbols = []
            
            # Look for exact symbol matches
            for symbol in self.stock_symbols:
                # Match whole words only to avoid false positives
                pattern = r'\b' + re.escape(symbol) + r'\b'
                if re.search(pattern, headline_upper):
                    mentioned_symbols.append(symbol)
            
            # Look for common company name patterns
            company_patterns = {
                'AAPL': ['APPLE', 'APPLE INC'],
                'MSFT': ['MICROSOFT', 'MICROSOFT CORP'],
                'GOOGL': ['GOOGLE', 'ALPHABET', 'ALPHABET INC'],
                'AMZN': ['AMAZON', 'AMAZON.COM'],
                'TSLA': ['TESLA', 'TESLA INC'],
                'INFY': ['INFOSYS', 'INFOSYS LIMITED'],
                'META': ['META', 'FACEBOOK'],
                'NFLX': ['NETFLIX'],
                'NVDA': ['NVIDIA'],
                'JPM': ['JPMORGAN', 'JP MORGAN']
            }
            
            for symbol, names in company_patterns.items():
                if symbol in self.stock_symbols:
                    for name in names:
                        if name in headline_upper:
                            if symbol not in mentioned_symbols:
                                mentioned_symbols.append(symbol)
            
            return mentioned_symbols
            
        except Exception as e:
            logger.warning(f"Error extracting stock mentions from '{headline}': {e}")
            return []
    
    def process_news_with_sentiment(self, news_df: pd.DataFrame, 
                                   batch_size: int = 16) -> pd.DataFrame:
        """
        Process news data with FinBERT sentiment analysis
        
        Args:
            news_df: Preprocessed news DataFrame
            batch_size: Batch size for FinBERT processing
            
        Returns:
            DataFrame with sentiment analysis results
        """
        try:
            logger.info("Processing news with FinBERT sentiment analysis")
            
            # Setup FinBERT if not already done
            if not self.finbert_processor.sentiment_pipeline:
                logger.info("Setting up FinBERT model...")
                success = self.finbert_processor.download_and_setup_model()
                if not success:
                    raise RuntimeError("Failed to setup FinBERT model")
            
            # Run sentiment analysis on headlines
            sentiment_df = self.finbert_processor.analyze_news_dataframe(
                news_df, 
                text_column='headline',
                batch_size=batch_size
            )
            
            logger.info("✅ Sentiment analysis completed")
            return sentiment_df
            
        except Exception as e:
            logger.error(f"Error processing news with sentiment: {e}")
            raise
    
    def create_stock_news_associations(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create associations between news items and mentioned stocks
        
        Args:
            sentiment_df: News data with sentiment analysis
            
        Returns:
            DataFrame with news-stock associations
        """
        try:
            logger.info("Creating news-stock associations")
            
            associations = []
            
            for idx, row in sentiment_df.iterrows():
                # Extract stock mentions from headline
                mentioned_stocks = self.extract_stock_mentions(row['headline'])
                
                if mentioned_stocks:
                    # Create association record for each mentioned stock
                    for symbol in mentioned_stocks:
                        association = {
                            'news_id': row['news_id'],
                            'headline': row['headline'],
                            'published_at': row['published_at'],
                            'publisher': row['publisher'],
                            'symbol': symbol,
                            'sentiment_label': row['sentiment_label'],
                            'sentiment_score': row['sentiment_score'],
                            'confidence_score': row['confidence_score'],
                            'positive_score': row['positive_score'],
                            'negative_score': row['negative_score'],
                            'neutral_score': row['neutral_score'],
                            'mention_context': row['headline'][:200],  # First 200 chars
                            'relevance_score': 1.0  # Default relevance
                        }
                        associations.append(association)
                else:
                    # Create general market sentiment record
                    association = {
                        'news_id': row['news_id'],
                        'headline': row['headline'],
                        'published_at': row['published_at'],
                        'publisher': row['publisher'],
                        'symbol': None,  # General market news
                        'sentiment_label': row['sentiment_label'],
                        'sentiment_score': row['sentiment_score'],
                        'confidence_score': row['confidence_score'],
                        'positive_score': row['positive_score'],
                        'negative_score': row['negative_score'],
                        'neutral_score': row['neutral_score'],
                        'mention_context': row['headline'][:200],
                        'relevance_score': 0.5  # Lower relevance for general news
                    }
                    associations.append(association)
            
            associations_df = pd.DataFrame(associations)
            
            logger.info(f"Created {len(associations_df)} news-stock associations")
            
            # Summary statistics
            stock_mentions = associations_df[associations_df['symbol'].notna()]
            if len(stock_mentions) > 0:
                mention_counts = stock_mentions['symbol'].value_counts()
                logger.info(f"Top mentioned stocks: {mention_counts.head().to_dict()}")
            
            return associations_df
            
        except Exception as e:
            logger.error(f"Error creating stock-news associations: {e}")
            raise
    
    def store_sentiment_data(self, associations_df: pd.DataFrame) -> Dict:
        """
        Store sentiment analysis results in database
        
        Args:
            associations_df: News-stock associations with sentiment
            
        Returns:
            Dictionary with storage results
        """
        try:
            logger.info("Storing sentiment data in database")
            
            results = {
                'news_inserted': 0,
                'sentiments_inserted': 0,
                'associations_inserted': 0,
                'errors': []
            }
            
            # Process each association
            for idx, row in associations_df.iterrows():
                try:
                    # Insert news record
                    news_id = db_manager.insert_financial_news(
                        headline=row['headline'],
                        publisher=row['publisher'],
                        published_at=row['published_at'],
                        source='financial_news_pipeline'
                    )
                    results['news_inserted'] += 1
                    
                    # Insert sentiment record
                    if row['symbol'] is not None:
                        # Get stock_id for the symbol
                        stock_query = "SELECT id FROM stocks WHERE symbol = %s"
                        with db_manager.get_connection() as conn:
                            stock_result = conn.execute(stock_query, (row['symbol'],))
                            stock_row = stock_result.fetchone()
                            stock_id = str(stock_row[0]) if stock_row else None
                    else:
                        stock_id = None
                    
                    sentiment_id = db_manager.insert_sentiment_score(
                        news_id=news_id,
                        stock_id=stock_id,
                        symbol=row['symbol'],
                        sentiment_label=row['sentiment_label'],
                        sentiment_score=row['sentiment_score'],
                        confidence_score=row['confidence_score'],
                        model_version='finbert-1.0'
                    )
                    results['sentiments_inserted'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing row {idx}: {e}"
                    results['errors'].append(error_msg)
                    logger.warning(error_msg)
                    continue
            
            logger.info(f"Stored {results['news_inserted']} news items and {results['sentiments_inserted']} sentiment records")
            
            if results['errors']:
                logger.warning(f"Encountered {len(results['errors'])} errors during storage")
            
            return results
            
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")
            raise
    
    def run_complete_pipeline(self, batch_size: int = 16) -> Dict:
        """
        Run the complete sentiment analysis pipeline
        
        Args:
            batch_size: Batch size for FinBERT processing
            
        Returns:
            Dictionary with pipeline results
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING SENTIMENT ANALYSIS PIPELINE")
            logger.info("=" * 60)
            
            results = {}
            
            # Step 1: Load news data
            logger.info("Step 1: Loading financial news data")
            news_data = self.load_news_data()
            results['raw_news_count'] = len(news_data)
            
            # Step 2: Preprocess news data
            logger.info("Step 2: Preprocessing news data")
            clean_news = self.preprocess_news_data(news_data)
            results['processed_news_count'] = len(clean_news)
            
            # Step 3: Run sentiment analysis
            logger.info("Step 3: Running FinBERT sentiment analysis")
            sentiment_news = self.process_news_with_sentiment(clean_news, batch_size)
            results['sentiment_analysis_count'] = len(sentiment_news)
            
            # Step 4: Create stock associations
            logger.info("Step 4: Creating news-stock associations")
            associations = self.create_stock_news_associations(sentiment_news)
            results['associations_count'] = len(associations)
            
            # Step 5: Store in database
            logger.info("Step 5: Storing sentiment data in database")
            storage_results = self.store_sentiment_data(associations)
            results.update(storage_results)
            
            # Step 6: Save processed data
            logger.info("Step 6: Saving processed data")
            processed_file = os.path.join(self.processed_data_dir, "sentiment_analysis_results.csv")
            associations.to_csv(processed_file, index=False)
            results['output_file'] = processed_file
            
            # Step 7: Generate summary
            logger.info("Step 7: Generating pipeline summary")
            sentiment_summary = associations['sentiment_label'].value_counts().to_dict()
            stock_summary = associations[associations['symbol'].notna()]['symbol'].value_counts().head(10).to_dict()
            
            results['sentiment_distribution'] = sentiment_summary
            results['top_mentioned_stocks'] = stock_summary
            
            logger.info("=" * 60)
            logger.info("SENTIMENT ANALYSIS PIPELINE COMPLETED!")
            logger.info(f"✅ Raw news: {results['raw_news_count']}")
            logger.info(f"✅ Processed news: {results['processed_news_count']}")
            logger.info(f"✅ Sentiment analyses: {results['sentiment_analysis_count']}")
            logger.info(f"✅ Stock associations: {results['associations_count']}")
            logger.info(f"✅ Database records: {results['news_inserted']} news, {results['sentiments_inserted']} sentiments")
            logger.info(f"✅ Output file: {processed_file}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Sentiment analysis pipeline failed: {e}")
            raise

def main():
    """Main function to run sentiment pipeline"""
    try:
        # Create pipeline instance
        pipeline = SentimentPipeline()
        
        # Determine optimal batch size based on device
        if pipeline.finbert_processor.device == 'cuda':
            batch_size = 64
            print(f"🚀 GPU detected! Using increased batch size: {batch_size} for fast processing")
        else:
            batch_size = 16
            print(f"💻 CPU detected. Using standard batch size: {batch_size}")
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(batch_size=batch_size)
        
        # Print summary
        print("\n🎯 SENTIMENT ANALYSIS PIPELINE SUMMARY:")
        print("=" * 50)
        print(f"📰 Raw News: {results['raw_news_count']:,}")
        print(f"🧹 Processed News: {results['processed_news_count']:,}")
        print(f"🤖 Sentiment Analyses: {results['sentiment_analysis_count']:,}")
        print(f"🔗 Stock Associations: {results['associations_count']:,}")
        print(f"💾 Database Records: {results['news_inserted']} news, {results['sentiments_inserted']} sentiments")
        
        print(f"\n📊 Sentiment Distribution:")
        for sentiment, count in results['sentiment_distribution'].items():
            print(f"   {sentiment}: {count:,}")
        
        print(f"\n📈 Top Mentioned Stocks:")
        for stock, count in list(results['top_mentioned_stocks'].items())[:5]:
            print(f"   {stock}: {count:,} mentions")
        
        print("=" * 50)
        print("✅ Sentiment analysis pipeline completed!")
        print("🎉 Phase 2 completed! Ready for Phase 3!")
        
        return True
        
    except Exception as e:
        print(f"❌ Sentiment pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Phase 2 Complete! All data processing and sentiment analysis ready!")
    else:
        print("\n❌ Fix errors and try again")