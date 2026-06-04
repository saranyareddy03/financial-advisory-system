# ===============================
# ENVIRONMENT CONFIGURATION
# Financial Advisory System
# ===============================

import os
from typing import Optional
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Environment configuration management for Financial Advisory System"""
    
    def __init__(self, env_file: Optional[str] = None):
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from default .env file
            
        self._load_config()
    
    def _load_config(self):
        """Load all configuration from environment variables"""
        
        # DATABASE CONFIGURATION
        self.DB_USER = os.getenv('user', 'postgres.vyjydroffsorchebruiv')
        self.DB_PASSWORD = os.getenv('password', 'Realtime@2025')
        self.DB_HOST = os.getenv('host', 'aws-1-ap-south-1.pooler.supabase.com')
        self.DB_PORT = int(os.getenv('port', '6543'))
        self.DB_NAME = os.getenv('dbname', 'postgres')
        
        # Full database URL
        # URL encode user and password to handle special characters like '@'
        import urllib.parse
        encoded_user = urllib.parse.quote_plus(self.DB_USER)
        encoded_password = urllib.parse.quote_plus(self.DB_PASSWORD)

        self.DATABASE_URL = (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
        
        # API CONFIGURATION
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
        self.GROQ_SQL_MODEL = os.getenv('GROQ_SQL_MODEL', 'llama-3.3-70b-versatile')
        self.GROQ_REASONING_MODEL = os.getenv('GROQ_REASONING_MODEL', 'llama-3.3-70b-versatile')
        self.FMP_API_KEY = os.getenv('FMP_API_KEY', '')
        self.TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')
        
        # APPLICATION SETTINGS
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
        self.DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
        self.EXTERNAL_PRICE_MAX_AGE_DAYS = int(os.getenv('EXTERNAL_PRICE_MAX_AGE_DAYS', '5'))
        self.EXTERNAL_NEWS_LIMIT = int(os.getenv('EXTERNAL_NEWS_LIMIT', '5'))
        self.EXTERNAL_TAVILY_RESULT_LIMIT = int(os.getenv('EXTERNAL_TAVILY_RESULT_LIMIT', '5'))
        
        # CONNECTION POOL SETTINGS
        self.DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '5'))
        self.DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '10'))
        self.DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        self.DB_POOL_RECYCLE = int(os.getenv('DB_POOL_RECYCLE', '1800'))
        
        logger.info("Configuration loaded successfully")
    
    def get_database_url(self) -> str:
        return self.DATABASE_URL
    
    def get_gemini_api_key(self) -> str:
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        return self.GEMINI_API_KEY

# Global configuration instance
config = Config()

# Export commonly used values
DATABASE_URL = config.DATABASE_URL
GEMINI_API_KEY = config.GEMINI_API_KEY
DEBUG = config.DEBUG
