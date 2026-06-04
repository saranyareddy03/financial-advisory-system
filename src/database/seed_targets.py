import pandas as pd
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_target_prices(db: DatabaseManager, targets: list):
    """Seed stock_prices table for specific target stocks"""
    try:
        prices_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'cleaned_stock_data.csv')
        logger.info(f"Loading prices from {prices_path}")
        
        df = pd.read_csv(prices_path)
        
        # Filter for targets
        df = df[df['symbol'].isin(targets)]
        
        if df.empty:
            logger.warning("No data found for target stocks")
            return False
            
        logger.info(f"Found {len(df)} records for {targets}")
        
        # Ensure date is properly formatted
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Batch insert
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            inserted = db.bulk_insert_stock_prices(batch)
            total_inserted += inserted
            logger.info(f"Inserted batch {i//batch_size + 1}: {inserted} records")
            
        logger.info(f"Successfully seeded {total_inserted} price records for targets")
        return True
        
    except Exception as e:
        logger.error(f"Error seeding prices: {e}")
        return False

def main():
    db = DatabaseManager()
    
    # Stocks we want to ensure are in DB
    targets = ['MSFT', 'TSLA', 'NVDA']
    
    logger.info(f"Starting targeted seeding for: {targets}")
    
    if seed_target_prices(db, targets):
        logger.info("✅ Targeted seeding completed successfully")
    else:
        logger.error("❌ Failed to seed targets")

if __name__ == "__main__":
    main()
