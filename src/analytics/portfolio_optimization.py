"""
Portfolio Optimization Engine for Financial Advisory System
Phase 4: Portfolio Optimization & Risk Models

Implements Markowitz Modern Portfolio Theory for optimal asset allocation:
- Mean-Variance Optimization
- Risk-Return Efficient Frontier
- Multiple Optimization Objectives (Max Sharpe, Min Risk, Target Return)
- Integration with Technical Indicators and Risk Metrics
- User Risk Profile Alignment

Integrates with existing database schema:
- stocks, stock_prices, risk_metrics, technical_indicators
- Supports 51-stock universe (Top 50 US + INFY)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union, Literal
import sys
import os
from sqlalchemy import text
import warnings
warnings.filterwarnings('ignore')

# Scientific Computing & Optimization
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import DatabaseManager
from src.config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization Engine using Modern Portfolio Theory.
    
    Features:
    - Markowitz Mean-Variance Optimization
    - Multiple optimization objectives
    - Risk profile alignment  
    - Efficient frontier generation
    - Portfolio performance analytics
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize Portfolio Optimizer.
        
        Args:
            lookback_days (int): Days of historical data for optimization (default: 1 year)
        """
        self.settings = config
        self.db_manager = DatabaseManager()
        self.lookback_days = lookback_days
        
        # Risk tolerance mapping
        self.risk_profiles = {
            'conservative': {'target_volatility': 0.10, 'min_sharpe': 0.5},
            'moderate': {'target_volatility': 0.15, 'min_sharpe': 0.7}, 
            'aggressive': {'target_volatility': 0.25, 'min_sharpe': 0.8}
        }
        
    def get_returns_data(self, symbols: List[str] = None, 
                        end_date: datetime = None) -> pd.DataFrame:
        """
        Fetch historical returns data for portfolio optimization.
        
        Args:
            symbols (List[str]): Stock symbols to include (None = all available)
            end_date (datetime): End date for data (None = latest available)
            
        Returns:
            pd.DataFrame: Returns matrix with symbols as columns, dates as index
        """
        logger.info(f"Fetching returns data for {len(symbols) if symbols else 'all'} symbols")
        
        try:
            with self.db_manager.get_connection() as conn:
                # Get date range for data fetch
                if end_date is None:
                    end_date = datetime.now()
                start_date = end_date - timedelta(days=self.lookback_days + 30)  # Buffer for weekends
                
                # Build symbol filter clause
                symbol_filter = ""
                if symbols:
                    symbol_list = "', '".join(symbols)
                    symbol_filter = f"AND symbol IN ('{symbol_list}')"
                
                # Fetch price data with proper ordering
                # Ensure start_date and end_date are string formatted for the query
                # Use a much wider date range to catch any available data
                start_date_str = '2020-01-01'
                end_date_str = datetime.now().strftime('%Y-%m-%d')
                
                query = text(f"""
                SELECT s.symbol, sp.date, sp.close_price
                FROM stock_prices sp
                JOIN stocks s ON sp.stock_id = s.id
                WHERE sp.date >= '{start_date_str}'
                  AND sp.date <= '{end_date_str}'
                  {symbol_filter.replace('symbol', 's.symbol')}
                ORDER BY s.symbol, sp.date
                """)
                
                df_prices = pd.read_sql(query, conn)
                
                if df_prices.empty:
                    logger.warning("No price data found for specified criteria")
                    return pd.DataFrame()
                
                # Pivot to get symbols as columns
                price_matrix = df_prices.pivot(index='date', columns='symbol', values='close_price')
                
                # Fill forward missing values and drop rows with any NaN
                price_matrix = price_matrix.fillna(method='ffill').dropna()
                
                # Calculate daily returns
                returns_matrix = price_matrix.pct_change().dropna()
                
                # Ensure minimum data requirements
                min_observations = 30  # At least 30 trading days
                if len(returns_matrix) < min_observations:
                    logger.error(f"Insufficient data: {len(returns_matrix)} days < {min_observations} required")
                    return pd.DataFrame()
                
                # Filter out stocks with insufficient data or extreme volatility
                valid_stocks = []
                for symbol in returns_matrix.columns:
                    stock_returns = returns_matrix[symbol]
                    
                    # Check data quality criteria
                    if (len(stock_returns.dropna()) >= min_observations and 
                        stock_returns.std() < 0.5 and  # Max 50% daily volatility
                        stock_returns.std() > 0.001):  # Min 0.1% daily volatility
                        valid_stocks.append(symbol)
                
                final_returns = returns_matrix[valid_stocks]
                
                logger.info(f"Successfully processed returns for {len(final_returns.columns)} stocks "
                           f"over {len(final_returns)} trading days")
                
                return final_returns
                
        except Exception as e:
            logger.error(f"Error fetching returns data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                   returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights (np.ndarray): Portfolio weights
            returns (pd.DataFrame): Historical returns matrix
            
        Returns:
            Dict[str, float]: Portfolio metrics
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = returns.dot(weights)
            
            # Annualized metrics (252 trading days)
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Risk-free rate (assume 2% annually)
            risk_free_rate = 0.02
            
            # Sharpe ratio
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            
            # Sortino ratio (downside volatility)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (95% confidence)
            var_95 = portfolio_returns.quantile(0.05)
            
            # Calmar ratio (annual return / max drawdown)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def optimize_portfolio(self, returns: pd.DataFrame,
                          objective: Literal['max_sharpe', 'min_risk', 'target_return'] = 'max_sharpe',
                          target_return: float = None,
                          risk_free_rate: float = 0.02,
                          max_weight: float = 0.3,
                          min_weight: float = 0.0) -> Dict:
        """
        Optimize portfolio using specified objective.
        
        Args:
            returns (pd.DataFrame): Historical returns matrix
            objective (str): Optimization objective ('max_sharpe', 'min_risk', 'target_return')
            target_return (float): Target annual return (for 'target_return' objective)
            risk_free_rate (float): Risk-free rate for Sharpe calculation
            max_weight (float): Maximum weight per asset
            min_weight (float): Minimum weight per asset
            
        Returns:
            Dict: Optimization results with weights and metrics
        """
        try:
            n_assets = len(returns.columns)
            
            if n_assets < 2:
                logger.error("Need at least 2 assets for optimization")
                return {}
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Initial guess (equal weights)
            initial_guess = np.array([1.0 / n_assets] * n_assets)
            
            # Constraints: weights sum to 1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Bounds: min_weight <= weight <= max_weight
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            # Define objective functions
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_volatility == 0:
                    return -1000  # Avoid division by zero
                sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
                return -sharpe  # Negative for minimization
            
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def negative_return(weights):
                return -np.dot(weights, expected_returns)
            
            # Select objective function
            if objective == 'max_sharpe':
                objective_func = negative_sharpe
            elif objective == 'min_risk':
                objective_func = portfolio_volatility
            elif objective == 'target_return':
                if target_return is None:
                    logger.error("target_return must be specified for 'target_return' objective")
                    return {}
                # Add constraint for target return
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: np.dot(x, expected_returns) - target_return
                })
                objective_func = portfolio_volatility
            else:
                logger.error(f"Unknown objective: {objective}")
                return {}
            
            # Optimize portfolio
            result = minimize(
                objective_func,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                # Fallback to equal weights
                optimal_weights = np.array([1.0 / n_assets] * n_assets)
            else:
                optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(optimal_weights, returns)
            
            # Create results dictionary
            results = {
                'success': result.success if 'result' in locals() else False,
                'objective': objective,
                'weights': dict(zip(returns.columns, optimal_weights)),
                'metrics': portfolio_metrics,
                'optimization_message': result.message if 'result' in locals() else "Fallback to equal weights"
            }
            
            logger.info(f"Portfolio optimization completed: {objective}")
            logger.info(f"Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 'N/A'):.3f}")
            logger.info(f"Annual Return: {portfolio_metrics.get('annual_return', 'N/A'):.2%}")
            logger.info(f"Annual Volatility: {portfolio_metrics.get('annual_volatility', 'N/A'):.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {}
    
    def generate_efficient_frontier(self, returns: pd.DataFrame, 
                                  n_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.
        
        Args:
            returns (pd.DataFrame): Historical returns matrix
            n_portfolios (int): Number of portfolios to generate
            
        Returns:
            pd.DataFrame: Efficient frontier with returns, volatilities, and weights
        """
        try:
            logger.info(f"Generating efficient frontier with {n_portfolios} portfolios")
            
            # Calculate expected returns range
            expected_returns = returns.mean() * 252
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            
            # Generate target returns
            target_returns = np.linspace(min_return, max_return, n_portfolios)
            
            frontier_results = []
            
            for target_ret in target_returns:
                # Optimize for minimum risk at target return
                result = self.optimize_portfolio(
                    returns=returns,
                    objective='target_return',
                    target_return=target_ret,
                    max_weight=0.5  # More concentrated for frontier
                )
                
                if result and result.get('metrics'):
                    frontier_point = {
                        'target_return': target_ret,
                        'annual_return': result['metrics']['annual_return'],
                        'annual_volatility': result['metrics']['annual_volatility'],
                        'sharpe_ratio': result['metrics']['sharpe_ratio'],
                        'weights': result['weights']
                    }
                    frontier_results.append(frontier_point)
            
            if not frontier_results:
                logger.warning("No valid frontier portfolios generated")
                return pd.DataFrame()
            
            frontier_df = pd.DataFrame(frontier_results)
            
            logger.info(f"Generated efficient frontier with {len(frontier_df)} valid portfolios")
            
            return frontier_df
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
            return pd.DataFrame()
    
    def recommend_portfolio_for_risk_profile(self, risk_profile: str,
                                           symbols: List[str] = None) -> Dict:
        """
        Generate portfolio recommendation based on user risk profile.
        
        Args:
            risk_profile (str): 'conservative', 'moderate', or 'aggressive'
            symbols (List[str]): Symbols to include (None = all available)
            
        Returns:
            Dict: Portfolio recommendation with allocation and metrics
        """
        try:
            logger.info(f"Generating portfolio recommendation for {risk_profile} risk profile")
            
            # Validate risk profile
            if risk_profile not in self.risk_profiles:
                logger.error(f"Unknown risk profile: {risk_profile}")
                return {}
            
            # Get returns data
            returns = self.get_returns_data(symbols)
            
            if returns.empty:
                logger.error("No returns data available for optimization")
                return {}
            
            # Get risk profile parameters
            profile_params = self.risk_profiles[risk_profile]
            target_volatility = profile_params['target_volatility']
            
            # Set optimization constraints based on risk profile
            if risk_profile == 'conservative':
                max_weight = 0.2  # More diversified
                objective = 'min_risk'
            elif risk_profile == 'moderate':
                max_weight = 0.25
                objective = 'max_sharpe'
            else:  # aggressive
                max_weight = 0.35
                objective = 'max_sharpe'
            
            # Optimize portfolio
            optimization_result = self.optimize_portfolio(
                returns=returns,
                objective=objective,
                max_weight=max_weight,
                min_weight=0.01  # Small minimum allocation
            )
            
            if not optimization_result:
                logger.error("Portfolio optimization failed")
                return {}
            
            # Check if portfolio meets risk profile requirements
            portfolio_volatility = optimization_result['metrics'].get('annual_volatility', 0)
            portfolio_sharpe = optimization_result['metrics'].get('sharpe_ratio', 0)
            
            # Create recommendation
            recommendation = {
                'risk_profile': risk_profile,
                'meets_criteria': (
                    portfolio_volatility <= target_volatility * 1.2 and  # Allow 20% tolerance
                    portfolio_sharpe >= profile_params['min_sharpe'] * 0.8  # Allow 20% tolerance
                ),
                'allocation': optimization_result['weights'],
                'metrics': optimization_result['metrics'],
                'target_volatility': target_volatility,
                'actual_volatility': portfolio_volatility,
                'recommendation_strength': 'Strong' if optimization_result['success'] else 'Moderate'
            }
            
            # Filter out very small allocations for cleaner presentation
            filtered_allocation = {
                symbol: weight for symbol, weight in recommendation['allocation'].items()
                if weight >= 0.01  # 1% minimum
            }
            recommendation['allocation'] = filtered_allocation
            
            logger.info(f"Portfolio recommendation generated:")
            logger.info(f"- Assets: {len(filtered_allocation)}")
            logger.info(f"- Target Vol: {target_volatility:.1%}, Actual Vol: {portfolio_volatility:.1%}")
            logger.info(f"- Sharpe Ratio: {portfolio_sharpe:.3f}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating portfolio recommendation: {str(e)}")
            return {}

def main():
    """
    Test the Portfolio Optimization Engine with sample data.
    """
    logger.info("=== PORTFOLIO OPTIMIZATION ENGINE TEST ===")
    
    try:
        # Initialize optimizer
        optimizer = PortfolioOptimizer(lookback_days=252)  # 1 year of data
        
        # Test with a few major stocks
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        logger.info(f"Testing with symbols: {test_symbols}")
        
        # Test 1: Get returns data
        logger.info("\n1. Testing returns data fetch...")
        returns = optimizer.get_returns_data(symbols=test_symbols)
        
        if returns.empty:
            logger.error("No returns data available - check database connection")
            return
        
        logger.info(f"✅ Returns data shape: {returns.shape}")
        
        # Test 2: Maximum Sharpe ratio optimization
        logger.info("\n2. Testing Max Sharpe optimization...")
        max_sharpe_result = optimizer.optimize_portfolio(returns, objective='max_sharpe')
        
        if max_sharpe_result:
            logger.info("✅ Max Sharpe optimization successful")
            logger.info(f"Sharpe Ratio: {max_sharpe_result['metrics']['sharpe_ratio']:.3f}")
        else:
            logger.error("❌ Max Sharpe optimization failed")
        
        # Test 3: Risk profile recommendations
        logger.info("\n3. Testing risk profile recommendations...")
        
        for profile in ['conservative', 'moderate', 'aggressive']:
            recommendation = optimizer.recommend_portfolio_for_risk_profile(
                risk_profile=profile,
                symbols=test_symbols
            )
            
            if recommendation:
                logger.info(f"✅ {profile.capitalize()} portfolio generated")
                logger.info(f"Assets: {len(recommendation['allocation'])}")
                logger.info(f"Volatility: {recommendation['actual_volatility']:.2%}")
            else:
                logger.error(f"❌ {profile.capitalize()} portfolio failed")
        
        logger.info("\n=== PORTFOLIO OPTIMIZATION TEST COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()