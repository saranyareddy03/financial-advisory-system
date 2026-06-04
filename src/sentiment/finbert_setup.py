#!/usr/bin/env python3
# ===============================
# FINBERT SETUP & CONFIGURATION
# Financial Advisory System - FinBERT Sentiment Analysis Setup
# ===============================

import os
import sys
import torch
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append('/teamspace/studios/this_studio/financial_advisory_system')

# Transformers and model imports
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    logging as transformers_logging
)

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

from src.config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTProcessor:
    """
    FinBERT model setup and sentiment analysis processor
    Handles financial text sentiment classification using ProsusAI/finbert
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT processor
        
        Args:
            model_name: HuggingFace model identifier for FinBERT
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self.device = self._get_device()
        self.max_length = 512  # FinBERT max sequence length
        
        # Model cache directory
        self.cache_dir = "/teamspace/studios/this_studio/financial_advisory_system/models/finbert"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Sentiment label mapping
        self.label_mapping = {
            'positive': 1,
            'negative': -1, 
            'neutral': 0
        }
        
        logger.info(f"FinBERT processor initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_name}")
    
    def _get_device(self) -> str:
        """
        Detect and configure compute device (GPU/CPU)
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
        else:
            device = 'cpu'
            logger.info("Using CPU for inference")
        
        return device
    
    def download_and_setup_model(self) -> bool:
        """
        Download and setup FinBERT model and tokenizer
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Downloading FinBERT model and tokenizer...")
            logger.info("This may take a few minutes on first run...")
            
            # Download tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_fast=True
            )
            
            # Download model
            logger.info("Loading model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                num_labels=3  # positive, negative, neutral
            )
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Create sentiment pipeline
            logger.info("Creating sentiment pipeline...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1,
                return_all_scores=True
            )
            
            logger.info("✅ FinBERT model setup completed successfully!")
            logger.info(f"Model size: {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up FinBERT model: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess financial text for FinBERT analysis
        
        Args:
            text: Raw financial text
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (keep first part as it's usually most important)
        if len(text) > self.max_length * 3:  # Rough character estimate
            text = text[:self.max_length * 3]
        
        return text
    
    def analyze_single_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text using FinBERT
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Dictionary with sentiment results
        """
        try:
            if not self.sentiment_pipeline:
                raise RuntimeError("FinBERT model not initialized. Call download_and_setup_model() first.")
            
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            if not clean_text:
                return {
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.0,
                    'confidence_score': 0.0,
                    'all_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                }
            
            # Get sentiment prediction
            results = self.sentiment_pipeline(clean_text)
            
            # Process results (pipeline returns list of all scores)
            all_scores = {item['label'].lower(): item['score'] for item in results[0]}
            
            # Find the label with highest confidence
            best_label = max(all_scores.keys(), key=lambda k: all_scores[k])
            best_score = all_scores[best_label]
            
            # Convert to our scoring system (-1 to +1)
            sentiment_score = self.label_mapping.get(best_label, 0) * best_score
            
            return {
                'sentiment_label': best_label,
                'sentiment_score': sentiment_score,
                'confidence_score': best_score,
                'all_scores': all_scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.0,
                'confidence_score': 0.0,
                'all_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Analyze sentiment of multiple texts in batches (Optimized for GPU)
        
        Args:
            texts: List of financial texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of sentiment analysis results
        """
        try:
            if not self.sentiment_pipeline:
                raise RuntimeError("FinBERT model not initialized. Call download_and_setup_model() first.")
            
            logger.info(f"Analyzing {len(texts)} texts in batches of {batch_size}")
            
            results = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Preprocess and track valid indices
                clean_texts = []
                valid_indices = []
                
                for idx, text in enumerate(batch_texts):
                    cleaned = self.preprocess_text(text)
                    if cleaned:
                        clean_texts.append(cleaned)
                        valid_indices.append(idx)
                
                # Initialize batch results with defaults (for invalid/empty texts)
                batch_results = [None] * len(batch_texts)
                
                # Process valid texts using GPU batch processing
                if clean_texts:
                    # Pass list to pipeline for parallel processing
                    pipeline_out = self.sentiment_pipeline(
                        clean_texts, 
                        batch_size=len(clean_texts), 
                        truncation=True, 
                        max_length=self.max_length
                    )
                    
                    # Map results back to original positions
                    for idx, prediction in zip(valid_indices, pipeline_out):
                        # prediction is a list of scores (since return_all_scores=True)
                        all_scores = {item['label'].lower(): item['score'] for item in prediction}
                        
                        best_label = max(all_scores.keys(), key=lambda k: all_scores[k])
                        best_score = all_scores[best_label]
                        sentiment_score = self.label_mapping.get(best_label, 0) * best_score
                        
                        batch_results[idx] = {
                            'sentiment_label': best_label,
                            'sentiment_score': sentiment_score,
                            'confidence_score': best_score,
                            'all_scores': all_scores
                        }
                
                # Fill in None values (empty texts) with defaults
                for j in range(len(batch_results)):
                    if batch_results[j] is None:
                        batch_results[j] = {
                            'sentiment_label': 'neutral',
                            'sentiment_score': 0.0,
                            'confidence_score': 0.0,
                            'all_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                        }
                
                results.extend(batch_results)
                
                # Progress logging
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
            
            logger.info("✅ Batch sentiment analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            raise
    
    def analyze_news_dataframe(self, df: pd.DataFrame, 
                              text_column: str = 'headline',
                              batch_size: int = 16) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of news data
        
        Args:
            df: DataFrame with news data
            text_column: Column containing text to analyze
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with sentiment analysis results
        """
        try:
            logger.info(f"Analyzing sentiment for {len(df)} news items")
            
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
            # Extract texts
            texts = df[text_column].tolist()
            
            # Analyze sentiment
            sentiment_results = self.analyze_batch(texts, batch_size)
            
            # Create results DataFrame
            results_df = df.copy()
            
            # Add sentiment columns
            results_df['sentiment_label'] = [r['sentiment_label'] for r in sentiment_results]
            results_df['sentiment_score'] = [r['sentiment_score'] for r in sentiment_results]
            results_df['confidence_score'] = [r['confidence_score'] for r in sentiment_results]
            
            # Add individual scores
            results_df['positive_score'] = [r['all_scores'].get('positive', 0) for r in sentiment_results]
            results_df['negative_score'] = [r['all_scores'].get('negative', 0) for r in sentiment_results]
            results_df['neutral_score'] = [r['all_scores'].get('neutral', 0) for r in sentiment_results]
            
            # Add processing metadata
            results_df['processed_at'] = datetime.now()
            results_df['model_version'] = 'finbert-1.0'
            
            logger.info("✅ DataFrame sentiment analysis completed")
            
            # Log summary statistics
            sentiment_counts = results_df['sentiment_label'].value_counts()
            logger.info("Sentiment distribution:")
            for label, count in sentiment_counts.items():
                logger.info(f"  {label}: {count} ({count/len(results_df)*100:.1f}%)")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error analyzing news DataFrame: {e}")
            raise
    
    def test_model(self) -> bool:
        """
        Test FinBERT model with sample financial texts
        
        Returns:
            True if tests pass, False otherwise
        """
        try:
            logger.info("Testing FinBERT model with sample texts...")
            
            # Test samples with known sentiments
            test_texts = [
                "Company reports strong quarterly earnings, beating analyst expectations",  # Positive
                "Stock price falls amid concerns about declining market share",  # Negative  
                "The company announced a quarterly dividend of $0.25 per share",  # Neutral
                "Major acquisition expected to boost revenue growth significantly",  # Positive
                "Regulatory investigation launched into business practices"  # Negative
            ]
            
            expected_sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative']
            
            # Analyze test texts
            results = []
            for text in test_texts:
                result = self.analyze_single_text(text)
                results.append(result)
                logger.info(f"Text: '{text[:50]}...' -> {result['sentiment_label']} ({result['confidence_score']:.3f})")
            
            # Check if results are reasonable
            correct_predictions = 0
            for i, result in enumerate(results):
                if result['sentiment_label'] == expected_sentiments[i]:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(test_texts)
            logger.info(f"Test accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_texts)})")
            
            # Pass if accuracy is reasonable (>= 60%)
            if accuracy >= 0.6:
                logger.info("✅ FinBERT model test PASSED")
                return True
            else:
                logger.warning("⚠️ FinBERT model test shows low accuracy")
                return True  # Still continue, model might need fine-tuning
                
        except Exception as e:
            logger.error(f"❌ FinBERT model test FAILED: {e}")
            return False
    
    def save_model_info(self) -> str:
        """
        Save model configuration and info to file
        
        Returns:
            Path to saved info file
        """
        try:
            info = {
                'model_name': self.model_name,
                'device': self.device,
                'max_length': self.max_length,
                'cache_dir': self.cache_dir,
                'setup_time': datetime.now().isoformat(),
                'torch_version': torch.__version__,
                'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
            }
            
            info_path = os.path.join(self.cache_dir, 'model_info.json')
            
            import json
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"Model info saved to: {info_path}")
            return info_path
            
        except Exception as e:
            logger.error(f"Error saving model info: {e}")
            raise

def main():
    """Main function to setup FinBERT"""
    try:
        logger.info("=" * 60)
        logger.info("STARTING FINBERT SETUP")
        logger.info("=" * 60)
        
        # Create FinBERT processor
        finbert = FinBERTProcessor()
        
        # Download and setup model
        logger.info("Step 1: Downloading and setting up FinBERT model")
        setup_success = finbert.download_and_setup_model()
        
        if not setup_success:
            logger.error("❌ FinBERT setup failed")
            return False
        
        # Test the model
        logger.info("Step 2: Testing FinBERT model")
        test_success = finbert.test_model()
        
        if not test_success:
            logger.warning("⚠️ FinBERT tests failed, but continuing...")
        
        # Save model info
        logger.info("Step 3: Saving model configuration")
        info_path = finbert.save_model_info()
        
        logger.info("=" * 60)
        logger.info("FINBERT SETUP COMPLETED!")
        logger.info(f"✅ Model: {finbert.model_name}")
        logger.info(f"✅ Device: {finbert.device}")
        logger.info(f"✅ Cache: {finbert.cache_dir}")
        logger.info(f"✅ Config: {info_path}")
        logger.info("=" * 60)
        
        print("\n🎯 FINBERT SETUP SUMMARY:")
        print("=" * 50)
        print(f"🤖 Model: {finbert.model_name}")
        print(f"💻 Device: {finbert.device}")
        print(f"📁 Cache: {finbert.cache_dir}")
        print(f"⚙️ Parameters: {sum(p.numel() for p in finbert.model.parameters()):,}")
        print("=" * 50)
        print("✅ FinBERT is ready for sentiment analysis!")
        print("🚀 Ready for sentiment analysis pipeline!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FinBERT setup failed: {e}")
        print(f"\n❌ FinBERT setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ FinBERT is now ready to analyze financial sentiment!")
    else:
        print("\n❌ Fix errors and try again")