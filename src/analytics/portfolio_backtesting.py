"""
Portfolio Backtesting Engine for Financial Advisory System
Phase 4: Portfolio Optimization & Risk Models - Component 2

Validates portfolio strategies through historical simulation:
- Out-of-sample performance testing
- Rolling window optimization and rebalancing
- Performance attribution analysis
- Risk-adjusted metrics tracking
- Drawdown analysis and stress testing
- Integration with Portfolio Optimizer

Prevents overfitting by testing strategies on unseen data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
from sqlalchemy import text
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import DatabaseManager
from src.config.settings import config
from src.analytics.portfolio_optimization import PortfolioOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """
    Portfolio Backtesting Engine for strategy validation.
    
    Features:
    - Historical performance simulation
    - Rolling window rebalancing
    - Out-of-sample testing
    - Performance attribution
    - Risk metrics tracking
    - Benchmark comparison
    """
    
    def __init__(self, rebalance_frequency: str = 'monthly', 
                 transaction_cost: float = 0.001):
        """
        Initialize Portfolio Backtester.
        
        Args:
            rebalance_frequency (str): 'daily', 'weekly', 'monthly', 'quarterly'
            transaction_cost (float): Transaction cost as percentage (0.001 = 0.1%)
        """
        self.settings = config
        self.db_manager = DatabaseManager()
        self.optimizer = PortfolioOptimizer()
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        
        # Rebalancing periods mapping
        self.rebalance_periods = {
            'daily': 1,
            'weekly': 7, 
            'monthly': 30,
            'quarterly': 90
        }
        
    def get_historical_data(self, symbols: List[str], 
                           start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical price data for backtesting.
        
        Args:
            symbols (List[str]): Stock symbols to include
            start_date (datetime): Start date for backtesting
            end_date (datetime): End date for backtesting
            
        Returns:
            pd.DataFrame: Historical price matrix
        """
        logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        
        try:
            with self.db_manager.get_connection() as conn:
                # Build symbol filter
                symbol_list = "', '".join(symbols)
                
                # Fetch price data
                # Ensure start_date and end_date are formatted as strings
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                query = text(f"""
                SELECT s.symbol, sp.date, sp.close_price, sp.open_price, sp.volume
                FROM stock_prices sp
                JOIN stocks s ON sp.stock_id = s.id
                WHERE sp.date >= '{start_date_str}'
                  AND sp.date <= '{end_date_str}'
                  AND s.symbol IN ('{symbol_list}')
                ORDER BY s.symbol, sp.date
                """)
                
                df_prices = pd.read_sql(query, conn)
                
                if df_prices.empty:
                    logger.warning("No historical data found for specified criteria")
                    return pd.DataFrame()
                
                # Pivot to get symbols as columns
                price_matrix = df_prices.pivot(index='date', columns='symbol', values='close_price')
                
                # Fill forward missing values
                price_matrix = price_matrix.fillna(method='ffill').dropna()
                
                logger.info(f"Successfully fetched {len(price_matrix)} days of data for {len(price_matrix.columns)} symbols")
                
                return price_matrix
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_rebalancing_dates(self, start_date: datetime, 
                                  end_date: datetime) -> List[datetime]:
        """
        Calculate rebalancing dates based on frequency.
        
        Args:
            start_date (datetime): Backtest start date
            end_date (datetime): Backtest end date
            
        Returns:
            List[datetime]: List of rebalancing dates
        """
        rebalance_days = self.rebalance_periods[self.rebalance_frequency]
        dates = []
        
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=rebalance_days)
        
        # Always include the end date
        if dates[-1] != end_date:
            dates.append(end_date)
        
        logger.info(f"Generated {len(dates)} rebalancing dates with {self.rebalance_frequency} frequency")
        
        return dates
    
    def optimize_at_date(self, symbols: List[str], 
                        optimization_date: datetime,
                        risk_profile: str = 'moderate',
                        lookback_days: int = 252) -> Dict:
        """
        Run portfolio optimization at specific date using only past data.
        
        Args:
            symbols (List[str]): Symbols to optimize
            optimization_date (datetime): Date to run optimization
            risk_profile (str): Risk profile for optimization
            lookback_days (int): Days of data to use for optimization
            
        Returns:
            Dict: Optimization results with weights
        """
        try:
            # Create temporary optimizer with specific end date
            temp_optimizer = PortfolioOptimizer(lookback_days=lookback_days)
            
            # Get returns data up to optimization date (no look-ahead bias)
            returns = temp_optimizer.get_returns_data(
                symbols=symbols, 
                end_date=optimization_date
            )
            
            if returns.empty:
                logger.warning(f"No returns data available for optimization on {optimization_date.date()}")
                return {}
            
            # Filter returns to only include data before optimization date
            # Ensure index is datetime for comparison
            if not isinstance(returns.index, pd.DatetimeIndex):
                returns.index = pd.to_datetime(returns.index)
                
            returns = returns[returns.index < optimization_date]
            
            if len(returns) < 30:  # Minimum data requirement
                logger.warning(f"Insufficient data for optimization on {optimization_date.date()}")
                return {}
            
            # Run portfolio optimization
            result = temp_optimizer.recommend_portfolio_for_risk_profile(
                risk_profile=risk_profile,
                symbols=list(returns.columns)  # Use only available symbols
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio at {optimization_date.date()}: {str(e)}")
            return {}
    
    def simulate_portfolio_performance(self, price_data: pd.DataFrame,
                                     weights_timeline: Dict[datetime, Dict[str, float]],
                                     initial_capital: float = 100000) -> pd.DataFrame:
        """
        Simulate portfolio performance with rebalancing.
        
        Args:
            price_data (pd.DataFrame): Historical price matrix
            weights_timeline (Dict): Portfolio weights at each rebalancing date
            initial_capital (float): Starting portfolio value
            
        Returns:
            pd.DataFrame: Daily portfolio performance data
        """
        logger.info("Simulating portfolio performance...")
        
        try:
            # Initialize performance tracking
            portfolio_data = []
            current_capital = initial_capital
            current_shares = {}
            
            # Sort dates for chronological processing
            dates = sorted(price_data.index)
            rebalance_dates = sorted(weights_timeline.keys())
            
            for i, current_date in enumerate(dates):
                # Check if rebalancing is needed
                should_rebalance = False
                target_weights = {}
                
                # Find applicable weights for this date
                for rebal_date in reversed(rebalance_dates):
                    # Ensure datetime comparison works by checking type
                    current_date_ts = current_date
                    rebal_date_ts = rebal_date
                    
                    if isinstance(current_date, date) and not isinstance(current_date, datetime):
                        current_date_ts = datetime.combine(current_date, datetime.min.time())
                        
                    if isinstance(rebal_date, date) and not isinstance(rebal_date, datetime):
                        rebal_date_ts = datetime.combine(rebal_date, datetime.min.time())
                    
                    if rebal_date_ts <= current_date_ts:
                        target_weights = weights_timeline[rebal_date]
                        should_rebalance = (rebal_date_ts == current_date_ts)
                        break
                
                if not target_weights:
                    continue
                
                # Get current prices
                current_prices = price_data.loc[current_date]
                available_symbols = [s for s in target_weights.keys() if s in current_prices.index]
                
                if not available_symbols:
                    continue
                
                # Calculate current portfolio value
                portfolio_value = 0
                for symbol in available_symbols:
                    shares = current_shares.get(symbol, 0)
                    portfolio_value += shares * current_prices[symbol]
                
                # Add cash (if any)
                cash = current_capital - sum(
                    current_shares.get(symbol, 0) * current_prices[symbol] 
                    for symbol in available_symbols
                )
                portfolio_value += cash
                
                # Rebalancing logic
                transaction_costs = 0
                if should_rebalance and portfolio_value > 0:
                    # Calculate target allocations
                    new_shares = {}
                    total_target_weight = sum(target_weights.get(s, 0) for s in available_symbols)
                    
                    if total_target_weight > 0:
                        for symbol in available_symbols:
                            target_weight = target_weights.get(symbol, 0) / total_target_weight
                            target_value = portfolio_value * target_weight
                            target_shares = target_value / current_prices[symbol]
                            
                            # Calculate transaction costs
                            old_shares = current_shares.get(symbol, 0)
                            shares_traded = abs(target_shares - old_shares)
                            transaction_costs += shares_traded * current_prices[symbol] * self.transaction_cost
                            
                            new_shares[symbol] = target_shares
                        
                        current_shares = new_shares
                        current_capital = portfolio_value - transaction_costs
                
                # Calculate portfolio metrics for this day
                if i > 0:
                    prev_value = portfolio_data[-1]['portfolio_value']
                    daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
                else:
                    daily_return = 0
                
                # Store daily data
                portfolio_data.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'daily_return': daily_return,
                    'transaction_costs': transaction_costs if should_rebalance else 0,
                    'rebalanced': should_rebalance,
                    'cash': cash,
                    'shares': dict(current_shares)
                })
            
            # Convert to DataFrame
            performance_df = pd.DataFrame(portfolio_data)
            performance_df.set_index('date', inplace=True)
            
            logger.info(f"Simulation completed: {len(performance_df)} trading days")
            
            return performance_df
            
        except Exception as e:
            logger.error(f"Error simulating portfolio performance: {str(e)}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, performance_data: pd.DataFrame,
                                    benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            performance_data (pd.DataFrame): Portfolio performance data
            benchmark_returns (pd.Series): Benchmark returns for comparison
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        try:
            if performance_data.empty:
                return {}
            
            # Calculate returns
            returns = performance_data['daily_return']
            portfolio_values = performance_data['portfolio_value']
            
            # Basic metrics
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            annualized_volatility = returns.std() * np.sqrt(252)
            
            # Risk-free rate
            risk_free_rate = 0.02
            
            # Risk-adjusted metrics
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            # Downside metrics
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Value at Risk
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # Win rate
            win_rate = (returns > 0).sum() / len(returns)
            
            # Transaction costs impact
            total_transaction_costs = performance_data['transaction_costs'].sum()
            transaction_cost_impact = total_transaction_costs / portfolio_values.iloc[0]
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'win_rate': win_rate,
                'transaction_cost_impact': transaction_cost_impact,
                'total_transaction_costs': total_transaction_costs
            }
            
            # Benchmark comparison if provided
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align dates
                common_dates = benchmark_returns.index.intersection(returns.index)
                if len(common_dates) > 0:
                    aligned_returns = returns.loc[common_dates]
                    aligned_benchmark = benchmark_returns.loc[common_dates]
                    
                    # Calculate alpha and beta
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = aligned_benchmark.var()
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    
                    benchmark_annualized = (1 + aligned_benchmark.mean()) ** 252 - 1
                    alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized - risk_free_rate))
                    
                    # Information ratio
                    excess_returns = aligned_returns - aligned_benchmark
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
                    
                    metrics.update({
                        'alpha': alpha,
                        'beta': beta,
                        'information_ratio': information_ratio,
                        'tracking_error': tracking_error
                    })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def run_backtest(self, symbols: List[str],
                    start_date: datetime,
                    end_date: datetime,
                    risk_profile: str = 'moderate',
                    initial_capital: float = 100000,
                    optimization_lookback: int = 252) -> Dict:
        """
        Run complete portfolio backtest.
        
        Args:
            symbols (List[str]): Symbols to include in portfolio
            start_date (datetime): Backtest start date
            end_date (datetime): Backtest end date
            risk_profile (str): Risk profile for optimization
            initial_capital (float): Starting capital
            optimization_lookback (int): Days of data for optimization
            
        Returns:
            Dict: Complete backtest results
        """
        logger.info(f"Starting backtest: {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Risk Profile: {risk_profile}")
        logger.info(f"Rebalancing: {self.rebalance_frequency}")
        
        try:
            # Get historical data
            price_data = self.get_historical_data(symbols, start_date, end_date)
            
            if price_data.empty:
                logger.error("No historical data available for backtesting")
                return {}
            
            # Calculate rebalancing dates
            rebalance_dates = self.calculate_rebalancing_dates(start_date, end_date)
            
            # Generate portfolio weights for each rebalancing date
            weights_timeline = {}
            failed_optimizations = 0
            
            for rebal_date in rebalance_dates:
                optimization_result = self.optimize_at_date(
                    symbols=symbols,
                    optimization_date=rebal_date,
                    risk_profile=risk_profile,
                    lookback_days=optimization_lookback
                )
                
                if optimization_result and 'allocation' in optimization_result:
                    weights_timeline[rebal_date] = optimization_result['allocation']
                else:
                    failed_optimizations += 1
                    # Use equal weights as fallback
                    available_symbols = [s for s in symbols if s in price_data.columns]
                    equal_weight = 1.0 / len(available_symbols) if available_symbols else 0
                    weights_timeline[rebal_date] = {s: equal_weight for s in available_symbols}
            
            if not weights_timeline:
                logger.error("No valid portfolio weights generated")
                return {}
            
            # Simulate portfolio performance
            performance_data = self.simulate_portfolio_performance(
                price_data=price_data,
                weights_timeline=weights_timeline,
                initial_capital=initial_capital
            )
            
            if performance_data.empty:
                logger.error("Portfolio simulation failed")
                return {}
            
            # Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(performance_data)
            
            # Create backtest results
            backtest_results = {
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols,
                'risk_profile': risk_profile,
                'rebalance_frequency': self.rebalance_frequency,
                'initial_capital': initial_capital,
                'final_value': performance_data['portfolio_value'].iloc[-1],
                'total_rebalances': len(rebalance_dates),
                'failed_optimizations': failed_optimizations,
                'performance_metrics': performance_metrics,
                'weights_timeline': weights_timeline,
                'performance_data': performance_data
            }
            
            # Log summary
            logger.info("=== BACKTEST RESULTS ===")
            logger.info(f"Total Return: {performance_metrics.get('total_return', 0):.2%}")
            logger.info(f"Annualized Return: {performance_metrics.get('annualized_return', 0):.2%}")
            logger.info(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
            logger.info(f"Win Rate: {performance_metrics.get('win_rate', 0):.2%}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return {}
    
    def compare_strategies(self, strategies: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple strategy backtests.
        
        Args:
            strategies (Dict): Dictionary of strategy results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        try:
            comparison_data = []
            
            for strategy_name, results in strategies.items():
                if 'performance_metrics' in results:
                    metrics = results['performance_metrics']
                    
                    comparison_data.append({
                        'Strategy': strategy_name,
                        'Total Return': metrics.get('total_return', 0),
                        'Annualized Return': metrics.get('annualized_return', 0),
                        'Volatility': metrics.get('annualized_volatility', 0),
                        'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                        'Max Drawdown': metrics.get('max_drawdown', 0),
                        'Win Rate': metrics.get('win_rate', 0),
                        'Final Value': results.get('final_value', 0)
                    })
            
            if not comparison_data:
                return pd.DataFrame()
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.set_index('Strategy')
            
            logger.info("Strategy comparison completed")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")
            return pd.DataFrame()

def main():
    """
    Test the Portfolio Backtesting Engine.
    """
    logger.info("=== PORTFOLIO BACKTESTING ENGINE TEST ===")
    
    try:
        # Initialize backtester
        backtester = PortfolioBacktester(
            rebalance_frequency='monthly',
            transaction_cost=0.001  # 0.1% transaction cost
        )
        
        # Test parameters
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        logger.info(f"Testing backtest from {start_date.date()} to {end_date.date()}")
        logger.info(f"Test symbols: {test_symbols}")
        
        # Test 1: Run single strategy backtest
        logger.info("\n1. Testing single strategy backtest...")
        
        backtest_result = backtester.run_backtest(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            risk_profile='moderate',
            initial_capital=100000
        )
        
        if backtest_result and 'performance_metrics' in backtest_result:
            logger.info("✅ Backtest successful")
            logger.info(f"Total Return: {backtest_result['performance_metrics']['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {backtest_result['performance_metrics']['sharpe_ratio']:.3f}")
        else:
            logger.error("❌ Backtest failed")
            return
        
        # Test 2: Compare different risk profiles
        logger.info("\n2. Testing strategy comparison...")
        
        strategies = {}
        
        for risk_profile in ['conservative', 'moderate', 'aggressive']:
            logger.info(f"Testing {risk_profile} strategy...")
            
            result = backtester.run_backtest(
                symbols=test_symbols,
                start_date=start_date,
                end_date=end_date,
                risk_profile=risk_profile,
                initial_capital=100000
            )
            
            if result:
                strategies[risk_profile] = result
                logger.info(f"✅ {risk_profile.capitalize()} strategy completed")
            else:
                logger.error(f"❌ {risk_profile.capitalize()} strategy failed")
        
        # Compare strategies
        if len(strategies) > 1:
            comparison_df = backtester.compare_strategies(strategies)
            
            if not comparison_df.empty:
                logger.info("\n✅ Strategy comparison completed")
                logger.info(f"Best performing strategy: {comparison_df['Total Return'].idxmax()}")
                logger.info(f"Highest Sharpe ratio: {comparison_df['Sharpe Ratio'].idxmax()}")
            else:
                logger.error("❌ Strategy comparison failed")
        
        logger.info("\n=== PORTFOLIO BACKTESTING TEST COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()