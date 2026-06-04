import pandas as pd
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_stocks(db: DatabaseManager):
    """Seed stocks table from stock_universe.csv"""
    try:
        universe_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'stock_universe.csv')
        logger.info(f"Loading stocks from {universe_path}")
        
        df = pd.read_csv(universe_path)
        
        count = 0
        for _, row in df.iterrows():
            try:
                db.insert_stock(
                    symbol=row['Symbol'],
                    company_name=row['Longname'],
                    sector=row['Sector'],
                    industry=row['Industry'],
                    market_cap=int(row['Marketcap']) if pd.notna(row['Marketcap']) else None,
                    exchange=row['Exchange'],
                    country=row['Country']
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to insert stock {row['Symbol']}: {e}")
        
        logger.info(f"Successfully seeded {count} stocks")
        return True
        
    except Exception as e:
        logger.error(f"Error seeding stocks: {e}")
        return False

def seed_prices(db: DatabaseManager):
    """Seed stock_prices table from cleaned_stock_data.csv"""
    try:
        prices_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'cleaned_stock_data.csv')
        logger.info(f"Loading prices from {prices_path}")
        
        df = pd.read_csv(prices_path)
        
        # Ensure date is properly formatted
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Rename columns to match what DatabaseManager expects if necessary
        # DatabaseManager expects: symbol, date, open, high, low, close
        # CSV has: symbol, date, open, high, low, close, adjusted_close, volume
        
        # Batch insert to avoid memory issues if file is huge
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            inserted = db.bulk_insert_stock_prices(batch)
            total_inserted += inserted
            logger.info(f"Inserted batch {i//batch_size + 1}: {inserted} records")
            
        logger.info(f"Successfully seeded {total_inserted} price records")
        return True
        
    except Exception as e:
        logger.error(f"Error seeding prices: {e}")
        return False

def main():
    db = DatabaseManager()
    
    logger.info("Starting database seeding...")
    
    if seed_stocks(db):
        if seed_prices(db):
            logger.info("✅ Database seeding completed successfully")
        else:
            logger.error("❌ Failed to seed prices")
    else:
        logger.error("❌ Failed to seed stocks")

if __name__ == "__main__":
    main()
