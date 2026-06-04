"""
Risk Metrics Engine for Financial Advisory System
Phase 3: Risk Assessment & Portfolio Analytics
Calculates comprehensive risk metrics:
- Volatility Analysis (daily, weekly, monthly)
- Value at Risk (VaR) calculations
- Maximum Drawdown analysis
- Sharpe Ratio and risk-adjusted returns
- Correlation Analysis between stocks
- Beta calculations (market correlation)

Integrates with technical indicators for complete risk assessment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
from sqlalchemy import text
from scipy import stats

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import DatabaseManager
from src.config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskMetricsEngine:
    """
    Risk Metrics Engine for comprehensive portfolio and stock risk analysis.
    
    Calculates various risk metrics and stores them in database
    for use in portfolio optimization and advisory recommendations.
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize with database connection and risk-free rate.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.settings = config
        self.db_manager = DatabaseManager()
        self.risk_free_rate = risk_free_rate
        
    def get_stock_returns(self, symbol: str, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.Series:
        """
        Fetch stock price data and calculate returns.
        
        Args:
            symbol: Stock symbol
            start_date: Optional start date (defaults to 2022-01-01)
            end_date: Optional end date (defaults to today)
        
        Returns:
            Series of daily returns
        """
        if start_date is None:
            start_date = datetime(2022, 1, 1)
        if end_date is None:
            end_date = datetime.now()
        
        query = """
        SELECT sp.date, sp.adjusted_close
        FROM stock_prices sp
        JOIN stocks s ON sp.stock_id = s.id
        WHERE s.symbol = %s AND sp.date >= %s AND sp.date <= %s
        ORDER BY sp.date ASC
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
                
            if df.empty:
                logger.warning(f"No price data found for symbol: {symbol}")
                return pd.Series(dtype=float)
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Calculate daily returns
            returns = df['adjusted_close'].pct_change().dropna()
            
            logger.info(f"Retrieved {len(returns)} return periods for {symbol}")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching returns for {symbol}: {e}")
            return pd.Series(dtype=float)
    
    def get_market_returns(self, market_symbol: str = 'SPY',
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.Series:
        """
        Get market returns for beta calculations.
        Uses SPY as market proxy by default.
        """
        return self.get_stock_returns(market_symbol, start_date, end_date)
    
    def calculate_volatility(self, returns: pd.Series, 
                           periods: int = 252) -> Dict[str, float]:
        """
        Calculate volatility metrics.
        
        Args:
            returns: Series of daily returns
            periods: Number of periods for annualization (252 = trading days)
        
        Returns:
            dict: Various volatility metrics
        """
        if returns.empty:
            return {}
        
        # Daily volatility
        daily_vol = returns.std()
        
        # Annualized volatility
        annual_vol = daily_vol * np.sqrt(periods)
        
        # Rolling volatilities
        rolling_30d = returns.rolling(window=30).std() * np.sqrt(periods)
        rolling_90d = returns.rolling(window=90).std() * np.sqrt(periods)
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'rolling_30d_volatility': rolling_30d.iloc[-1] if len(rolling_30d) > 0 else None,
            'rolling_90d_volatility': rolling_90d.iloc[-1] if len(rolling_90d) > 0 else None,
            'volatility_percentile': stats.percentileofscore(rolling_30d.dropna(), daily_vol) if len(rolling_30d.dropna()) > 0 else None
        }
    
    def calculate_var(self, returns: pd.Series, 
                     confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using historical method.
        
        Args:
            returns: Series of daily returns
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
        
        Returns:
            dict: VaR values for different confidence levels
        """
        if returns.empty:
            return {}
        
        var_results = {}
        
        for confidence in confidence_levels:
            # Historical VaR
            var_value = returns.quantile(1 - confidence)
            var_results[f'var_{int(confidence*100)}'] = var_value
            
            # Conditional VaR (Expected Shortfall)
            cvar_value = returns[returns <= var_value].mean()
            var_results[f'cvar_{int(confidence*100)}'] = cvar_value
        
        return var_results
    
    def calculate_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            prices: Series of price data
        
        Returns:
            dict: Drawdown metrics
        """
        if prices.empty:
            return {}
        
        # Calculate cumulative returns
        cum_returns = (1 + prices.pct_change()).cumprod()
        
        # Calculate running maximum
        rolling_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - rolling_max) / rolling_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Duration of maximum drawdown
        max_dd_start = rolling_max.idxmax()
        max_dd_end = drawdown.idxmin()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Average drawdown
        negative_dd = drawdown[drawdown < 0]
        avg_drawdown = negative_dd.mean() if not negative_dd.empty else 0
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_dd_start_date': max_dd_start,
            'max_dd_end_date': max_dd_end,
            'drawdown_duration_days': (max_dd_end - max_dd_start).days if max_dd_start and max_dd_end else None
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                              periods: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of daily returns
            periods: Number of periods for annualization
        
        Returns:
            float: Sharpe ratio
        """
        if returns.empty or len(returns) < 2:
            return 0.0
        
        # Annualized return
        annual_return = (1 + returns.mean()) ** periods - 1
        
        # Annualized volatility
        annual_vol = returns.std() * np.sqrt(periods)
        
        # Sharpe ratio
        if annual_vol == 0:
            return 0.0
        
        sharpe = (annual_return - self.risk_free_rate) / annual_vol
        return sharpe
    
    def calculate_beta(self, stock_returns: pd.Series, 
                      market_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate beta and correlation with market.
        
        Args:
            stock_returns: Stock daily returns
            market_returns: Market daily returns
        
        Returns:
            dict: Beta and correlation metrics
        """
        if stock_returns.empty or market_returns.empty:
            return {}
        
        # Align dates
        aligned_data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 30:  # Minimum 30 observations
            return {}
        
        stock_aligned = aligned_data['stock']
        market_aligned = aligned_data['market']
        
        # Calculate beta using covariance method
        covariance = np.cov(stock_aligned, market_aligned)[0][1]
        market_variance = np.var(market_aligned)
        
        if market_variance == 0:
            beta = 0.0
        else:
            beta = covariance / market_variance
        
        # Calculate correlation
        correlation = np.corrcoef(stock_aligned, market_aligned)[0][1]
        
        # Calculate alpha (intercept)
        stock_mean = stock_aligned.mean() * 252  # Annualized
        market_mean = market_aligned.mean() * 252  # Annualized
        alpha = stock_mean - beta * market_mean
        
        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'r_squared': correlation ** 2,
            'tracking_error': (stock_aligned - beta * market_aligned).std() * np.sqrt(252)
        }
    
    def calculate_information_ratio(self, stock_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            stock_returns: Stock returns
            benchmark_returns: Benchmark returns
        
        Returns:
            float: Information ratio
        """
        # Align returns
        aligned_data = pd.DataFrame({
            'stock': stock_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        excess_returns = aligned_data['stock'] - aligned_data['benchmark']
        
        if excess_returns.std() == 0:
            return 0.0
        
        # Annualized excess return / tracking error
        annual_excess = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        return annual_excess / tracking_error
    
    def calculate_comprehensive_risk_metrics(self, symbol: str,
                                           market_symbol: str = 'SPY') -> Dict[str, Union[float, str, None]]:
        """
        Calculate all risk metrics for a stock.
        
        Args:
            symbol: Stock symbol to analyze
            market_symbol: Market benchmark symbol
        
        Returns:
            dict: Comprehensive risk metrics
        """
        try:
            # Get stock and market returns
            stock_returns = self.get_stock_returns(symbol)
            market_returns = self.get_market_returns(market_symbol)
            
            if stock_returns.empty:
                logger.warning(f"No returns data for {symbol}")
                return {}
            
            # Get price data for drawdown calculation
            stock_prices = self.get_stock_price_data(symbol)
            
            # Calculate all metrics
            risk_metrics = {
                'symbol': symbol,
                'calculation_date': datetime.now(),
                'data_points': len(stock_returns)
            }
            
            # Volatility metrics
            vol_metrics = self.calculate_volatility(stock_returns)
            risk_metrics.update(vol_metrics)
            
            # VaR metrics
            var_metrics = self.calculate_var(stock_returns)
            risk_metrics.update(var_metrics)
            
            # Drawdown metrics
            if not stock_prices.empty:
                dd_metrics = self.calculate_drawdown(stock_prices['adjusted_close'])
                risk_metrics.update(dd_metrics)
            
            # Sharpe ratio
            risk_metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(stock_returns)
            
            # Beta and market metrics
            if not market_returns.empty:
                beta_metrics = self.calculate_beta(stock_returns, market_returns)
                risk_metrics.update(beta_metrics)
                
                # Information ratio
                risk_metrics['information_ratio'] = self.calculate_information_ratio(
                    stock_returns, market_returns
                )
            
            # Additional risk metrics
            risk_metrics['skewness'] = stock_returns.skew()
            risk_metrics['kurtosis'] = stock_returns.kurtosis()
            risk_metrics['downside_deviation'] = stock_returns[stock_returns < 0].std() * np.sqrt(252)
            
            # Sortino ratio (using downside deviation)
            if risk_metrics.get('downside_deviation', 0) > 0:
                annual_return = (1 + stock_returns.mean()) ** 252 - 1
                risk_metrics['sortino_ratio'] = (annual_return - self.risk_free_rate) / risk_metrics['downside_deviation']
            else:
                risk_metrics['sortino_ratio'] = 0.0
            
            logger.info(f"Calculated comprehensive risk metrics for {symbol}")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {e}")
            return {}
    
    def get_stock_price_data(self, symbol: str) -> pd.DataFrame:
        """Helper method to get stock price data."""
        query = """
        SELECT sp.date, sp.adjusted_close
        FROM stock_prices sp
        JOIN stocks s ON sp.stock_id = s.id
        WHERE s.symbol = %s AND sp.date >= %s
        ORDER BY sp.date ASC
        """
        
        start_date = datetime(2022, 1, 1)
        
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
    
    def store_risk_metrics_to_db(self, risk_metrics: Dict, symbol: str) -> bool:
        """
        Store calculated risk metrics to risk_metrics table.
        
        Args:
            risk_metrics: Dictionary of risk metrics
            symbol: Stock symbol
            
        Returns:
            bool: Success status
        """
        try:
            # Get stock_id
            stock_query = "SELECT id FROM stocks WHERE symbol = %s"
            
            with self.db_manager.get_connection() as conn:
                stock_result = pd.read_sql(stock_query, conn, params=(symbol,))
                
                if stock_result.empty:
                    logger.error(f"Stock symbol {symbol} not found in stocks table")
                    return False
                
                stock_id = stock_result.iloc[0]['id']
                
                # Prepare record for insertion
                record = {
                    'stock_id': stock_id,
                    'calculation_date': datetime.now().date(),
                    'volatility': risk_metrics.get('annual_volatility'),
                    'beta': risk_metrics.get('beta'),
                    'sharpe_ratio': risk_metrics.get('sharpe_ratio'),
                    'max_drawdown': risk_metrics.get('max_drawdown'),
                    'var_95': risk_metrics.get('var_95'),
                    'var_99': risk_metrics.get('var_99'),
                    'correlation_market': risk_metrics.get('correlation'),
                    'information_ratio': risk_metrics.get('information_ratio'),
                    'sortino_ratio': risk_metrics.get('sortino_ratio'),
                    'alpha': risk_metrics.get('alpha'),
                    'tracking_error': risk_metrics.get('tracking_error'),
                    'downside_deviation': risk_metrics.get('downside_deviation'),
                    'skewness': risk_metrics.get('skewness'),
                    'kurtosis': risk_metrics.get('kurtosis'),
                    'symbol': symbol,
                    'created_at': datetime.now()
                }
                
                # Insert with conflict handling
                insert_query = text("""
                INSERT INTO risk_metrics 
                (stock_id, calculation_date, volatility_30d, beta, sharpe_ratio, max_drawdown, value_at_risk_95, 
                 sortino_ratio, return_1d, return_7d, return_30d, return_90d, return_1y, symbol, created_at)
                VALUES (:stock_id, :calculation_date, :volatility_30d, :beta, :sharpe_ratio, 
                        :max_drawdown, :value_at_risk_95, :sortino_ratio, :return_1d, 
                        :return_7d, :return_30d, :return_90d, :return_1y, :symbol, :created_at)
                ON CONFLICT (stock_id, calculation_date) DO UPDATE SET
                    volatility_30d = EXCLUDED.volatility_30d,
                    beta = EXCLUDED.beta,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    max_drawdown = EXCLUDED.max_drawdown,
                    value_at_risk_95 = EXCLUDED.value_at_risk_95,
                    sortino_ratio = EXCLUDED.sortino_ratio,
                    return_1d = EXCLUDED.return_1d,
                    return_7d = EXCLUDED.return_7d,
                    return_30d = EXCLUDED.return_30d,
                    return_90d = EXCLUDED.return_90d,
                    return_1y = EXCLUDED.return_1y
                """)
                
                # Map metrics to table columns
                record_mapped = {
                    'stock_id': stock_id,
                    'calculation_date': datetime.now().date(),
                    'volatility_30d': risk_metrics.get('annual_volatility'), # Mapping annual vol to volatility_30d as placeholder or primary vol metric
                    'beta': risk_metrics.get('beta'),
                    'sharpe_ratio': risk_metrics.get('sharpe_ratio'),
                    'max_drawdown': risk_metrics.get('max_drawdown'),
                    'value_at_risk_95': risk_metrics.get('var_95'),
                    'sortino_ratio': risk_metrics.get('sortino_ratio'),
                    'return_1d': 0.0, # Placeholders for return metrics not calculated in this script yet
                    'return_7d': 0.0,
                    'return_30d': 0.0,
                    'return_90d': 0.0,
                    'return_1y': 0.0,
                    'symbol': symbol,
                    'created_at': datetime.now()
                }

                conn.execute(insert_query, record_mapped)
                conn.commit()
                
                logger.info(f"Stored risk metrics for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing risk metrics for {symbol}: {e}")
            return False
    
    def process_stock_risk_metrics(self, symbol: str) -> bool:
        """
        Complete pipeline: calculate and store risk metrics for a stock.
        
        Args:
            symbol: Stock symbol to process
            
        Returns:
            bool: Success status
        """
        logger.info(f"Processing risk metrics for {symbol}")
        
        # Calculate risk metrics
        risk_metrics = self.calculate_comprehensive_risk_metrics(symbol)
        
        if not risk_metrics:
            logger.warning(f"No risk metrics calculated for {symbol}")
            return False
        
        # Store to database
        success = self.store_risk_metrics_to_db(risk_metrics, symbol)
        
        if success:
            logger.info(f"✅ Successfully processed risk metrics for {symbol}")
        else:
            logger.error(f"❌ Failed to store risk metrics for {symbol}")
        
        return success
    
    def process_multiple_stocks(self, symbols: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Process risk metrics for multiple stocks.
        
        Args:
            symbols: List of symbols to process
            
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
        
        logger.info(f"Processing risk metrics for {len(symbols)} stocks")
        
        results = {}
        successful = 0
        
        for symbol in symbols:
            try:
                success = self.process_stock_risk_metrics(symbol)
                results[symbol] = success
                
                if success:
                    successful += 1
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = False
        
        logger.info(f"✅ Risk metrics processing complete: {successful}/{len(symbols)} successful")
        return results


def main():
    """
    Main execution function for testing risk metrics engine.
    """
    print("📊 PHASE 3: Risk Metrics Engine")
    print("=" * 50)
    
    try:
        # Initialize risk engine
        risk_engine = RiskMetricsEngine()
        
        # Test with major stocks (same as technical indicators)
        test_symbols = ['MSFT', 'GOOG', 'TSLA', 'NVDA', 'ABBV', 'CVX']
        
        print(f"📈 Testing risk metrics calculation for: {', '.join(test_symbols)}")
        print(f"📊 Risk metrics include: Volatility, VaR, Sharpe Ratio, Beta, Drawdown")
        
        results = {}
        for symbol in test_symbols:
            print(f"\n🔍 Processing {symbol}...")
            success = risk_engine.process_stock_risk_metrics(symbol)
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
            print(f"\n✅ Risk metrics engine is working!")
            print(f"📊 Risk metrics stored in 'risk_metrics' table")
            print(f"🎯 Ready for next Phase 3 component")
            print(f"\n💡 Your AI system can now analyze:")
            print(f"   • Portfolio volatility and risk levels")
            print(f"   • Market correlation and beta analysis") 
            print(f"   • Value-at-Risk calculations")
            print(f"   • Risk-adjusted performance (Sharpe, Sortino)")
        else:
            print(f"\n❌ Issues found. Check database connection and price data.")
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()