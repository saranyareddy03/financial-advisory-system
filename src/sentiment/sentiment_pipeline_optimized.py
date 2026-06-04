#!/usr/bin/env python3
# ===============================
# SENTIMENT ANALYSIS PIPELINE (OPTIMIZED)
# Financial Advisory System - Batch Processing for Robustness
# ===============================

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from typing import Dict, List, Tuple, Optional
import re
import logging

# Add project root to path
sys.path.append('/teamspace/studios/this_studio/financial_advisory_system')

from src.sentiment.finbert_setup import FinBERTProcessor
from src.database.connection import db_manager
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentPipelineOptimized:
    """
    Optimized sentiment analysis pipeline for financial news
    Processes data in chunks to handle large datasets and prevent memory issues
    Includes resume capability to skip already processed items
    """
    
    def __init__(self):
        """Initialize optimized sentiment analysis pipeline"""
        self.finbert_processor = FinBERTProcessor()
        self.raw_data_dir = "/teamspace/studios/this_studio/financial_advisory_system/data/raw"
        self.processed_data_dir = "/teamspace/studios/this_studio/financial_advisory_system/data/processed"
        
        # Load stock universe for symbol matching
        self.stock_symbols = self._load_stock_universe()
        
        # Cache for stock IDs to avoid repeated DB lookups
        self.stock_id_cache = {}
        self._load_stock_id_cache()
        
    def _load_stock_universe(self) -> set:
        """Load stock symbols from universe file"""
        try:
            universe_file = os.path.join(self.processed_data_dir, "stock_universe.csv")
            if os.path.exists(universe_file):
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
                        return {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'INFY'}
                
                logger.info(f"Loaded {len(symbols)} stock symbols")
                return symbols
            else:
                logger.warning(f"Universe file not found: {universe_file}")
                return {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'INFY'}
            
        except Exception as e:
            logger.error(f"Error loading stock universe: {e}")
            return {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'INFY'}
    
    def _load_stock_id_cache(self):
        """Pre-load stock symbol-to-ID mapping from database"""
        try:
            with db_manager.get_connection() as conn:
                result = conn.execute(text("SELECT symbol, id FROM stocks"))
                for row in result:
                    self.stock_id_cache[row[0]] = str(row[1])
            logger.info(f"Cached IDs for {len(self.stock_id_cache)} stocks")
        except Exception as e:
            logger.error(f"Error loading stock ID cache: {e}")

    def load_news_files(self) -> List[str]:
        """Get list of news CSV files"""
        news_dir = os.path.join(self.raw_data_dir, "news")
        if not os.path.exists(news_dir):
            raise FileNotFoundError(f"News directory not found: {news_dir}")
            
        files = [os.path.join(news_dir, f) for f in os.listdir(news_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(files)} news files")
        return files

    def get_processed_headlines(self) -> set:
        """Get set of headlines that have already been processed"""
        try:
            with db_manager.get_connection() as conn:
                # We only need headlines that have associated sentiment scores
                # But to keep it simple and safe, we'll just check what's in financial_news
                # for now, as that's the first step of insertion.
                # A more robust check would be checking sentiment_scores, but let's assume
                # transaction integrity handled insertion of both or neither.
                query = text("SELECT headline FROM financial_news")
                result = conn.execute(query)
                processed = {row[0] for row in result}
                logger.info(f"Found {len(processed)} already processed headlines")
                return processed
        except Exception as e:
            logger.error(f"Error getting processed headlines: {e}")
            return set()

    def preprocess_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize a chunk of news data"""
        try:
            # Make a copy
            news_df = df.copy()
            
            # Standardize column names
            cols_lower = {col.lower(): col for col in news_df.columns}
            column_mapping = {}
            
            # Map headline
            for possible in ['headline', 'title', 'text']:
                if possible in cols_lower:
                    column_mapping[cols_lower[possible]] = 'headline'
                    break
            
            # Map date
            for possible in ['date', 'published_at', 'timestamp']:
                if possible in cols_lower:
                    column_mapping[cols_lower[possible]] = 'published_at'
                    break
            
            # Map publisher
            for possible in ['publisher', 'source']:
                if possible in cols_lower:
                    column_mapping[cols_lower[possible]] = 'publisher'
                    break
            
            news_df = news_df.rename(columns=column_mapping)
            
            # Ensure required columns
            if 'headline' not in news_df.columns:
                return pd.DataFrame() # Skip chunk if no headline
            
            # Clean headlines
            news_df['headline'] = news_df['headline'].astype(str).str.strip()
            
            # Filter empty/short headlines
            news_df = news_df[news_df['headline'].str.len() > 10].copy()
            
            # Handle dates
            if 'published_at' in news_df.columns:
                news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce')
                news_df['published_at'] = news_df['published_at'].fillna(datetime.now())
            else:
                news_df['published_at'] = datetime.now()
            
            # Handle publisher
            if 'publisher' not in news_df.columns:
                news_df['publisher'] = 'Unknown'
            else:
                news_df['publisher'] = news_df['publisher'].fillna('Unknown')
            
            # Add other fields if missing
            if 'url' not in news_df.columns:
                news_df['url'] = None
            if 'content' not in news_df.columns:
                news_df['content'] = None
            if 'category' not in news_df.columns:
                news_df['category'] = 'financial'
                
            return news_df
            
        except Exception as e:
            logger.error(f"Error preprocessing chunk: {e}")
            return pd.DataFrame()

    def extract_stock_mentions(self, headline: str) -> List[str]:
        """Extract stock symbol mentions from news headline"""
        if not headline or pd.isna(headline):
            return []
        
        headline_upper = str(headline).upper()
        mentioned_symbols = []
        
        # Common company name patterns (simplified for speed)
        company_patterns = {
            'AAPL': ['APPLE'], 'MSFT': ['MICROSOFT'], 'GOOGL': ['GOOGLE', 'ALPHABET'],
            'AMZN': ['AMAZON'], 'TSLA': ['TESLA'], 'META': ['FACEBOOK'],
            'NFLX': ['NETFLIX'], 'NVDA': ['NVIDIA'], 'JPM': ['JPMORGAN']
        }
        
        # Check exact symbol matches
        for symbol in self.stock_symbols:
            if re.search(r'\b' + re.escape(symbol) + r'\b', headline_upper):
                mentioned_symbols.append(symbol)
        
        # Check company names
        for symbol, names in company_patterns.items():
            if symbol in self.stock_symbols and symbol not in mentioned_symbols:
                for name in names:
                    if name in headline_upper:
                        mentioned_symbols.append(symbol)
                        break
                        
        return mentioned_symbols

    def process_chunk(self, chunk: pd.DataFrame, batch_size: int = 32) -> Dict:
        """
        Process a single chunk of data:
        1. Preprocess
        2. Run Sentiment Analysis
        3. Extract Mentions
        4. Store in DB
        """
        results = {'processed': 0, 'inserted': 0, 'errors': 0}
        
        try:
            # 1. Preprocess
            clean_chunk = self.preprocess_chunk(chunk)
            if clean_chunk.empty:
                return results
                
            # 2. Run Sentiment Analysis (batched)
            # Setup model if needed
            if not self.finbert_processor.sentiment_pipeline:
                self.finbert_processor.download_and_setup_model()
                
            sentiment_df = self.finbert_processor.analyze_news_dataframe(
                clean_chunk, 
                text_column='headline',
                batch_size=batch_size
            )
            
            # 3. Prepare data for DB insertion
            news_records = []
            sentiment_records = []
            
            # First, identify relevant news (ONLY those mentioning stocks in our universe)
            relevant_indices = []
            
            for idx, row in sentiment_df.iterrows():
                mentions = self.extract_stock_mentions(row['headline'])
                
                # CRITICAL: Only proceed if stock mentions are found
                # This filters out ~90% of general news to save DB space
                if mentions:
                    relevant_indices.append((idx, mentions))
            
            if not relevant_indices:
                return results # No relevant news in this batch
            
            # Prepare news records for ONLY the relevant items
            for idx, mentions in relevant_indices:
                row = sentiment_df.loc[idx]
                news_records.append({
                    'headline': row['headline'],
                    'content': row.get('content'),
                    'publisher': row['publisher'],
                    'published_at': row['published_at'],
                    'url': row.get('url'),
                    'source': 'financial_news_pipeline',
                    'category': row.get('category')
                })
            
            # Bulk insert news and get back mappings (headline -> id)
            if news_records:
                inserted_news = db_manager.bulk_insert_financial_news(news_records)
                headline_to_id = {item['headline']: item['id'] for item in inserted_news}
                
                # Create sentiment records linked to news IDs
                for idx, mentions in relevant_indices:
                    row = sentiment_df.loc[idx]
                    news_id = headline_to_id.get(row['headline'])
                    if not news_id:
                        continue
                        
                    for symbol in mentions:
                        stock_id = self.stock_id_cache.get(symbol)
                        sentiment_records.append({
                            'news_id': news_id,
                            'stock_id': stock_id,
                            'symbol': symbol,
                            'sentiment_label': row['sentiment_label'],
                            'sentiment_score': row['sentiment_score'],
                            'confidence_score': row['confidence_score'],
                            'model_version': 'finbert-1.0'
                        })
                
                # Bulk insert sentiment scores
                if sentiment_records:
                    inserted_count = db_manager.bulk_insert_sentiment_scores(sentiment_records)
                    results['inserted'] = inserted_count
                    
            results['processed'] = len(clean_chunk)
            return results
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            results['errors'] = 1
            return results

    def run(self, batch_size: int = 32, chunk_size: int = 100):
        """Run the optimized pipeline"""
        logger.info("Starting Optimized Sentiment Pipeline")
        
        # Setup FinBERT
        if self.finbert_processor.device == 'cuda':
            batch_size = 64
            logger.info(f"Using GPU with batch size {batch_size}")
        else:
            logger.info(f"Using CPU with batch size {batch_size}")

        # Get processed headlines to skip
        processed_headlines = self.get_processed_headlines()
        
        # Load all files
        news_files = self.load_news_files()
        total_stats = {'processed': 0, 'inserted': 0, 'skipped': 0}
        
        for file_path in news_files:
            logger.info(f"Processing file: {os.path.basename(file_path)}")
            
            try:
                # Read CSV in chunks
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    # Filter out already processed items
                    # We need to normalize column names first to check 'headline'
                    temp_chunk = self.preprocess_chunk(chunk)
                    if temp_chunk.empty:
                        continue
                        
                    # Filter duplicates
                    original_len = len(temp_chunk)
                    temp_chunk = temp_chunk[~temp_chunk['headline'].isin(processed_headlines)]
                    skipped_count = original_len - len(temp_chunk)
                    total_stats['skipped'] += skipped_count
                    
                    if temp_chunk.empty:
                        logger.info(f"Skipped chunk (all {original_len} items already processed)")
                        continue
                        
                    logger.info(f"Processing chunk of {len(temp_chunk)} items ({skipped_count} skipped)")
                    
                    # Process the chunk
                    results = self.process_chunk(temp_chunk, batch_size)
                    
                    # Update stats
                    total_stats['processed'] += results['processed']
                    total_stats['inserted'] += results['inserted']
                    
                    # Update local cache of processed headlines to avoid dupes within same run
                    processed_headlines.update(temp_chunk['headline'].tolist())
                    
                    logger.info(f"Chunk progress: Processed {results['processed']}, Inserted {results['inserted']} sentiment records")
                    
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                continue
                
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED")
        logger.info(f"Total News Processed: {total_stats['processed']}")
        logger.info(f"Total Records Skipped: {total_stats['skipped']}")
        logger.info(f"Total Sentiment Records Inserted: {total_stats['inserted']}")
        logger.info("=" * 50)

if __name__ == "__main__":
    pipeline = SentimentPipelineOptimized()
    pipeline.run()
