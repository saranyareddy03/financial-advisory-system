"""
Risk Stress Testing Engine for Financial Advisory System
Phase 4: Portfolio Optimization & Risk Models - Component 3

Advanced risk analysis and stress testing capabilities:
- Monte Carlo Simulations (Portfolio path generation)
- Scenario Analysis (Market crashes, bull markets, inflation)
- Value at Risk (VaR) Modeling (Historical, Parametric, Monte Carlo)
- Stress Testing Framework (Black swan events, correlation breakdown)
- Portfolio Resilience Analysis (Drawdown recovery, tail risk)
- Integration with Portfolio Optimizer & Backtester

Ensures portfolios are robust under extreme market conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
from sqlalchemy import text
import warnings
warnings.filterwarnings('ignore')

# Statistical and scientific computing
from scipy import stats
from scipy.stats import norm, t, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import DatabaseManager
from src.config.settings import config
from src.analytics.portfolio_optimization import PortfolioOptimizer
from src.analytics.portfolio_backtesting import PortfolioBacktester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskStressTester:
    """
    Advanced Risk Stress Testing Engine for portfolio analysis.
    
    Features:
    - Monte Carlo portfolio simulations
    - Scenario-based stress testing
    - Multiple VaR methodologies
    - Correlation breakdown analysis
    - Black swan event modeling
    - Portfolio resilience metrics
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize Risk Stress Tester.
        
        Args:
            confidence_levels (List[float]): VaR confidence levels
        """
        self.settings = config
        self.db_manager = DatabaseManager()
        self.optimizer = PortfolioOptimizer()
        self.backtester = PortfolioBacktester()
        self.confidence_levels = confidence_levels
        
        # Predefined stress scenarios
        self.stress_scenarios = {
            'market_crash_2008': {
                'description': '2008 Financial Crisis Scenario',
                'market_shock': -0.40,  # 40% market decline
                'volatility_multiplier': 2.5,
                'correlation_increase': 0.3,
                'duration_days': 180
            },
            'covid_crash_2020': {
                'description': 'COVID-19 Market Crash Scenario',
                'market_shock': -0.35,  # 35% market decline
                'volatility_multiplier': 3.0,
                'correlation_increase': 0.4,
                'duration_days': 45
            },
            'dot_com_bubble_2000': {
                'description': 'Dot-com Bubble Burst Scenario',
                'market_shock': -0.50,  # 50% tech decline
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.2,
                'duration_days': 365
            },
            'inflation_spike': {
                'description': 'High Inflation Environment',
                'market_shock': -0.15,  # 15% market decline
                'volatility_multiplier': 1.8,
                'correlation_increase': 0.1,
                'duration_days': 252  # 1 year
            },
            'black_swan': {
                'description': 'Black Swan Event (6-sigma)',
                'market_shock': -0.60,  # 60% market decline
                'volatility_multiplier': 4.0,
                'correlation_increase': 0.8,
                'duration_days': 30
            }
        }
    
    def get_returns_data_for_stress_testing(self, symbols: List[str],
                                          years: int = 5) -> pd.DataFrame:
        """
        Fetch historical returns data for stress testing analysis.
        
        Args:
            symbols (List[str]): Stock symbols to analyze
            years (int): Years of historical data to fetch
            
        Returns:
            pd.DataFrame: Returns matrix for stress testing
        """
        logger.info(f"Fetching {years} years of returns data for stress testing")
        
        try:
            # Use portfolio optimizer's data fetching with longer lookback
            end_date = datetime.now()
            returns_data = self.optimizer.get_returns_data(
                symbols=symbols,
                end_date=end_date
            )
            
            if returns_data.empty:
                logger.warning("No returns data available for stress testing")
                return pd.DataFrame()
            
            # Filter to specified time period
            # Ensure index is datetime for comparison
            if not isinstance(returns_data.index, pd.DatetimeIndex):
                returns_data.index = pd.to_datetime(returns_data.index)
                
            cutoff_date = end_date - timedelta(days=years * 365)
            filtered_returns = returns_data[returns_data.index >= cutoff_date]
            
            logger.info(f"Successfully fetched {len(filtered_returns)} days of returns data for {len(filtered_returns.columns)} symbols")
            
            return filtered_returns
            
        except Exception as e:
            logger.error(f"Error fetching returns data for stress testing: {str(e)}")
            return pd.DataFrame()
    
    def calculate_var_methods(self, returns: pd.Series,
                            confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Value at Risk using multiple methodologies.
        
        Args:
            returns (pd.Series): Portfolio returns series
            confidence_level (float): Confidence level for VaR calculation
            
        Returns:
            Dict[str, float]: VaR estimates using different methods
        """
        try:
            if returns.empty or len(returns) < 30:
                logger.warning("Insufficient data for VaR calculation")
                return {}
            
            alpha = 1 - confidence_level
            
            # 1. Historical VaR (non-parametric)
            historical_var = returns.quantile(alpha)
            
            # 2. Parametric VaR (assuming normal distribution)
            mean_return = returns.mean()
            std_return = returns.std()
            parametric_var = mean_return + norm.ppf(alpha) * std_return
            
            # 3. Modified VaR (Cornish-Fisher expansion for skewness and kurtosis)
            skewness = skew(returns)
            kurt = kurtosis(returns)
            
            # Cornish-Fisher adjustment
            z = norm.ppf(alpha)
            z_adjusted = (z + 
                         (z**2 - 1) * skewness / 6 + 
                         (z**3 - 3*z) * kurt / 24 - 
                         (2*z**3 - 5*z) * skewness**2 / 36)
            
            modified_var = mean_return + z_adjusted * std_return
            
            # 4. Expected Shortfall (Conditional VaR)
            historical_es = returns[returns <= historical_var].mean()
            
            # 5. Student-t VaR (for heavy tails)
            try:
                # Fit t-distribution
                params = stats.t.fit(returns)
                df, loc, scale = params
                t_var = stats.t.ppf(alpha, df, loc, scale)
            except:
                t_var = parametric_var  # Fallback to parametric
            
            var_results = {
                f'historical_var_{int(confidence_level*100)}': historical_var,
                f'parametric_var_{int(confidence_level*100)}': parametric_var,
                f'modified_var_{int(confidence_level*100)}': modified_var,
                f'expected_shortfall_{int(confidence_level*100)}': historical_es,
                f't_distribution_var_{int(confidence_level*100)}': t_var
            }
            
            return var_results
            
        except Exception as e:
            logger.error(f"Error calculating VaR methods: {str(e)}")
            return {}
    
    def monte_carlo_simulation(self, portfolio_weights: Dict[str, float],
                             returns_data: pd.DataFrame,
                             num_simulations: int = 10000,
                             time_horizon: int = 252,
                             initial_value: float = 100000) -> Dict:
        """
        Run Monte Carlo simulation for portfolio value paths.
        
        Args:
            portfolio_weights (Dict): Portfolio allocation weights
            returns_data (pd.DataFrame): Historical returns data
            num_simulations (int): Number of simulation paths
            time_horizon (int): Days to simulate forward
            initial_value (float): Starting portfolio value
            
        Returns:
            Dict: Monte Carlo simulation results
        """
        logger.info(f"Running Monte Carlo simulation: {num_simulations} paths, {time_horizon} days")
        
        try:
            # Filter returns to portfolio symbols
            available_symbols = [s for s in portfolio_weights.keys() if s in returns_data.columns]
            if not available_symbols:
                logger.error("No matching symbols found in returns data")
                return {}
            
            portfolio_returns = returns_data[available_symbols]
            weights_array = np.array([portfolio_weights.get(s, 0) for s in available_symbols])
            weights_array = weights_array / weights_array.sum()  # Normalize weights
            
            # Calculate portfolio returns
            historical_portfolio_returns = portfolio_returns.dot(weights_array)
            
            # Calculate statistics for simulation
            mean_return = historical_portfolio_returns.mean()
            std_return = historical_portfolio_returns.std()
            
            # Calculate correlation matrix
            correlation_matrix = portfolio_returns.corr().values
            
            # Monte Carlo simulation
            np.random.seed(42)  # For reproducibility
            simulation_results = np.zeros((num_simulations, time_horizon + 1))
            simulation_results[:, 0] = initial_value
            
            for sim in range(num_simulations):
                portfolio_value = initial_value
                
                for day in range(1, time_horizon + 1):
                    # Generate correlated random returns
                    random_returns = np.random.multivariate_normal(
                        mean=[mean_return] * len(available_symbols),
                        cov=np.outer(std_return, std_return) * correlation_matrix,
                        size=1
                    ).flatten()
                    
                    # Calculate portfolio return
                    portfolio_return = np.dot(weights_array, random_returns)
                    
                    # Update portfolio value
                    portfolio_value *= (1 + portfolio_return)
                    simulation_results[sim, day] = portfolio_value
            
            # Calculate simulation statistics
            final_values = simulation_results[:, -1]
            returns_dist = (final_values / initial_value) - 1
            
            # VaR and Expected Shortfall from simulation
            var_95 = np.percentile(returns_dist, 5)
            var_99 = np.percentile(returns_dist, 1)
            es_95 = returns_dist[returns_dist <= var_95].mean()
            es_99 = returns_dist[returns_dist <= var_99].mean()
            
            # Probability of loss
            prob_loss = (returns_dist < 0).sum() / num_simulations
            prob_large_loss = (returns_dist < -0.20).sum() / num_simulations  # >20% loss
            
            # Path statistics
            max_values = simulation_results.max(axis=1)
            min_values = simulation_results.min(axis=1)
            max_drawdowns = (min_values / max_values) - 1
            
            simulation_summary = {
                'num_simulations': num_simulations,
                'time_horizon_days': time_horizon,
                'initial_value': initial_value,
                'mean_final_value': final_values.mean(),
                'median_final_value': np.median(final_values),
                'std_final_value': final_values.std(),
                'mean_return': returns_dist.mean(),
                'median_return': np.median(returns_dist),
                'std_return': returns_dist.std(),
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99,
                'probability_of_loss': prob_loss,
                'probability_large_loss': prob_large_loss,
                'mean_max_drawdown': max_drawdowns.mean(),
                'worst_max_drawdown': max_drawdowns.min(),
                'best_case_return': returns_dist.max(),
                'worst_case_return': returns_dist.min(),
                'simulation_paths': simulation_results,
                'final_values_distribution': final_values,
                'returns_distribution': returns_dist
            }
            
            logger.info(f"Monte Carlo completed: Mean return {returns_dist.mean():.2%}, VaR 95%: {var_95:.2%}")
            
            return simulation_summary
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {}
    
    def scenario_stress_test(self, portfolio_weights: Dict[str, float],
                           returns_data: pd.DataFrame,
                           scenario_name: str = 'market_crash_2008') -> Dict:
        """
        Apply predefined stress scenario to portfolio.
        
        Args:
            portfolio_weights (Dict): Portfolio allocation weights
            returns_data (pd.DataFrame): Historical returns data
            scenario_name (str): Stress scenario to apply
            
        Returns:
            Dict: Stress test results
        """
        logger.info(f"Running stress test scenario: {scenario_name}")
        
        try:
            if scenario_name not in self.stress_scenarios:
                logger.error(f"Unknown stress scenario: {scenario_name}")
                return {}
            
            scenario = self.stress_scenarios[scenario_name]
            
            # Filter returns to portfolio symbols
            available_symbols = [s for s in portfolio_weights.keys() if s in returns_data.columns]
            portfolio_returns = returns_data[available_symbols]
            weights_array = np.array([portfolio_weights.get(s, 0) for s in available_symbols])
            weights_array = weights_array / weights_array.sum()
            
            # Calculate baseline portfolio statistics
            historical_portfolio_returns = portfolio_returns.dot(weights_array)
            baseline_mean = historical_portfolio_returns.mean()
            baseline_std = historical_portfolio_returns.std()
            
            # Apply stress scenario
            stressed_mean = baseline_mean + scenario['market_shock'] / scenario['duration_days']
            stressed_std = baseline_std * scenario['volatility_multiplier']
            
            # Calculate stressed correlation matrix
            correlation_matrix = portfolio_returns.corr().values
            # Increase correlations during stress
            stress_correlation_matrix = correlation_matrix * (1 + scenario['correlation_increase'])
            np.fill_diagonal(stress_correlation_matrix, 1.0)  # Ensure diagonal is 1
            
            # Ensure positive semi-definite (adjust if necessary)
            eigenvals = np.linalg.eigvals(stress_correlation_matrix)
            if np.min(eigenvals) < 0:
                stress_correlation_matrix = correlation_matrix  # Fallback to original
            
            # Simulate stressed period
            np.random.seed(42)
            num_days = scenario['duration_days']
            stressed_returns = np.random.multivariate_normal(
                mean=[stressed_mean] * len(available_symbols),
                cov=np.outer(stressed_std, stressed_std) * stress_correlation_matrix,
                size=num_days
            )
            
            # Calculate portfolio returns during stress
            stressed_portfolio_returns = stressed_returns.dot(weights_array)
            
            # Calculate stress test metrics
            total_return = (1 + stressed_portfolio_returns).prod() - 1
            worst_day = stressed_portfolio_returns.min()
            volatility = stressed_portfolio_returns.std() * np.sqrt(252)
            
            # Value at Risk during stress
            stress_var_95 = np.percentile(stressed_portfolio_returns, 5)
            stress_var_99 = np.percentile(stressed_portfolio_returns, 1)
            
            # Maximum drawdown calculation
            cumulative_returns = (1 + stressed_portfolio_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Recovery analysis
            min_index = np.argmin(cumulative_returns)
            if min_index < len(cumulative_returns) - 1:
                recovery_period = len(cumulative_returns) - min_index - 1
                recovered = cumulative_returns[-1] >= cumulative_returns[min_index] * 1.1  # 10% recovery
            else:
                recovery_period = None
                recovered = False
            
            stress_results = {
                'scenario_name': scenario_name,
                'scenario_description': scenario['description'],
                'duration_days': num_days,
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252 / num_days) - 1,
                'worst_day_return': worst_day,
                'stressed_volatility': volatility,
                'max_drawdown': max_drawdown,
                'var_95_stressed': stress_var_95,
                'var_99_stressed': stress_var_99,
                'recovery_period_days': recovery_period,
                'portfolio_recovered': recovered,
                'stress_factor_applied': {
                    'market_shock': scenario['market_shock'],
                    'volatility_multiplier': scenario['volatility_multiplier'],
                    'correlation_increase': scenario['correlation_increase']
                }
            }
            
            logger.info(f"Stress test completed: {total_return:.2%} total return, {max_drawdown:.2%} max drawdown")
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error in scenario stress test: {str(e)}")
            return {}
    
    def comprehensive_risk_analysis(self, portfolio_weights: Dict[str, float],
                                  symbols: List[str]) -> Dict:
        """
        Comprehensive risk analysis combining all stress testing methods.
        
        Args:
            portfolio_weights (Dict): Portfolio allocation weights
            symbols (List[str]): Stock symbols in portfolio
            
        Returns:
            Dict: Complete risk analysis results
        """
        logger.info("Starting comprehensive risk analysis...")
        
        try:
            # Get historical returns data
            returns_data = self.get_returns_data_for_stress_testing(symbols, years=5)
            
            if returns_data.empty:
                logger.error("No returns data available for risk analysis")
                return {}
            
            # Filter to portfolio symbols and calculate portfolio returns
            available_symbols = [s for s in portfolio_weights.keys() if s in returns_data.columns]
            portfolio_returns_data = returns_data[available_symbols]
            weights_array = np.array([portfolio_weights.get(s, 0) for s in available_symbols])
            weights_array = weights_array / weights_array.sum()
            
            portfolio_returns = portfolio_returns_data.dot(weights_array)
            
            # 1. Multiple VaR calculations
            var_analysis = {}
            for confidence_level in self.confidence_levels:
                var_methods = self.calculate_var_methods(portfolio_returns, confidence_level)
                var_analysis.update(var_methods)
            
            # 2. Monte Carlo simulation
            monte_carlo_results = self.monte_carlo_simulation(
                portfolio_weights=portfolio_weights,
                returns_data=returns_data,
                num_simulations=10000,
                time_horizon=252  # 1 year
            )
            
            # 3. Stress testing scenarios
            stress_test_results = {}
            for scenario_name in self.stress_scenarios.keys():
                stress_result = self.scenario_stress_test(
                    portfolio_weights=portfolio_weights,
                    returns_data=returns_data,
                    scenario_name=scenario_name
                )
                if stress_result:
                    stress_test_results[scenario_name] = stress_result
            
            # 4. Portfolio statistics
            portfolio_stats = {
                'mean_daily_return': portfolio_returns.mean(),
                'daily_volatility': portfolio_returns.std(),
                'annualized_return': portfolio_returns.mean() * 252,
                'annualized_volatility': portfolio_returns.std() * np.sqrt(252),
                'skewness': skew(portfolio_returns),
                'kurtosis': kurtosis(portfolio_returns),
                'sharpe_ratio': (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252)),
                'max_historical_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'positive_days_ratio': (portfolio_returns > 0).sum() / len(portfolio_returns),
                'extreme_loss_days': (portfolio_returns < -0.05).sum(),  # Days with >5% loss
                'correlation_to_market': self._calculate_market_correlation(portfolio_returns, returns_data)
            }
            
            # 5. Risk summary and recommendations
            risk_score = self._calculate_risk_score(portfolio_stats, var_analysis, stress_test_results)
            
            comprehensive_analysis = {
                'analysis_date': datetime.now(),
                'portfolio_symbols': available_symbols,
                'portfolio_weights': {s: w for s, w in zip(available_symbols, weights_array)},
                'portfolio_statistics': portfolio_stats,
                'var_analysis': var_analysis,
                'monte_carlo_results': monte_carlo_results,
                'stress_test_results': stress_test_results,
                'risk_score': risk_score,
                'risk_recommendations': self._generate_risk_recommendations(risk_score, stress_test_results)
            }
            
            logger.info(f"Comprehensive risk analysis completed. Risk Score: {risk_score}/10")
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk analysis: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def _calculate_market_correlation(self, portfolio_returns: pd.Series, 
                                    returns_data: pd.DataFrame) -> float:
        """Calculate correlation with market (equal-weighted portfolio)."""
        try:
            market_returns = returns_data.mean(axis=1)
            common_dates = portfolio_returns.index.intersection(market_returns.index)
            if len(common_dates) > 30:
                return portfolio_returns.loc[common_dates].corr(market_returns.loc[common_dates])
            return 0.0
        except:
            return 0.0
    
    def _calculate_risk_score(self, portfolio_stats: Dict, 
                            var_analysis: Dict, 
                            stress_results: Dict) -> float:
        """Calculate overall risk score (1-10, where 10 is highest risk)."""
        try:
            risk_score = 0.0
            
            # Volatility component (0-3 points)
            volatility = portfolio_stats.get('annualized_volatility', 0)
            if volatility > 0.30:
                risk_score += 3.0
            elif volatility > 0.20:
                risk_score += 2.0
            elif volatility > 0.15:
                risk_score += 1.0
            
            # VaR component (0-2 points)
            var_95 = abs(var_analysis.get('historical_var_95', 0))
            if var_95 > 0.05:  # >5% daily VaR
                risk_score += 2.0
            elif var_95 > 0.03:
                risk_score += 1.0
            
            # Stress test component (0-3 points)
            avg_stress_drawdown = 0
            stress_count = 0
            for scenario_results in stress_results.values():
                if 'max_drawdown' in scenario_results:
                    avg_stress_drawdown += abs(scenario_results['max_drawdown'])
                    stress_count += 1
            
            if stress_count > 0:
                avg_stress_drawdown /= stress_count
                if avg_stress_drawdown > 0.40:
                    risk_score += 3.0
                elif avg_stress_drawdown > 0.25:
                    risk_score += 2.0
                elif avg_stress_drawdown > 0.15:
                    risk_score += 1.0
            
            # Skewness/tail risk component (0-2 points)
            skewness = portfolio_stats.get('skewness', 0)
            kurtosis_val = portfolio_stats.get('kurtosis', 0)
            if skewness < -1 or kurtosis_val > 5:  # Heavy negative tail
                risk_score += 2.0
            elif skewness < -0.5 or kurtosis_val > 3:
                risk_score += 1.0
            
            return min(risk_score, 10.0)  # Cap at 10
            
        except:
            return 5.0  # Default moderate risk
    
    def _generate_risk_recommendations(self, risk_score: float, 
                                     stress_results: Dict) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if risk_score >= 7:
            recommendations.append("HIGH RISK: Consider reducing portfolio volatility through diversification")
            recommendations.append("Consider adding defensive assets (bonds, utilities, consumer staples)")
            recommendations.append("Implement stop-loss or hedging strategies")
        elif risk_score >= 5:
            recommendations.append("MODERATE RISK: Portfolio shows acceptable risk levels")
            recommendations.append("Monitor correlation during market stress periods")
            recommendations.append("Consider periodic rebalancing to maintain target allocations")
        else:
            recommendations.append("LOW RISK: Portfolio is well-diversified and conservative")
            recommendations.append("May consider slightly higher allocations to growth assets")
        
        # Stress-specific recommendations
        worst_scenario = None
        worst_drawdown = 0
        for scenario_name, results in stress_results.items():
            drawdown = abs(results.get('max_drawdown', 0))
            if drawdown > worst_drawdown:
                worst_drawdown = drawdown
                worst_scenario = scenario_name
        
        if worst_scenario and worst_drawdown > 0.30:
            recommendations.append(f"Portfolio vulnerable to {worst_scenario.replace('_', ' ')} scenarios")
            recommendations.append("Consider stress testing allocation changes before implementation")
        
        return recommendations

def main():
    """
    Test the Risk Stress Testing Engine.
    """
    logger.info("=== RISK STRESS TESTING ENGINE TEST ===")
    
    try:
        # Initialize stress tester
        stress_tester = RiskStressTester()
        
        # Test portfolio (similar to previous tests)
        test_portfolio = {
            'AAPL': 0.3,
            'MSFT': 0.25,
            'GOOGL': 0.2,
            'AMZN': 0.15,
            'TSLA': 0.1
        }
        test_symbols = list(test_portfolio.keys())
        
        logger.info(f"Testing portfolio: {test_portfolio}")
        
        # Test 1: Monte Carlo Simulation
        logger.info("\n1. Testing Monte Carlo simulation...")
        
        returns_data = stress_tester.get_returns_data_for_stress_testing(test_symbols, years=3)
        
        if not returns_data.empty:
            monte_carlo_results = stress_tester.monte_carlo_simulation(
                portfolio_weights=test_portfolio,
                returns_data=returns_data,
                num_simulations=1000,  # Reduced for testing
                time_horizon=252
            )
            
            if monte_carlo_results:
                logger.info("✅ Monte Carlo simulation successful")
                logger.info(f"Mean Return: {monte_carlo_results['mean_return']:.2%}")
                logger.info(f"VaR 95%: {monte_carlo_results['var_95']:.2%}")
                logger.info(f"Probability of Loss: {monte_carlo_results['probability_of_loss']:.2%}")
            else:
                logger.error("❌ Monte Carlo simulation failed")
        
        # Test 2: Stress Testing
        logger.info("\n2. Testing stress scenarios...")
        
        stress_scenarios = ['market_crash_2008', 'covid_crash_2020', 'black_swan']
        
        for scenario in stress_scenarios:
            stress_result = stress_tester.scenario_stress_test(
                portfolio_weights=test_portfolio,
                returns_data=returns_data,
                scenario_name=scenario
            )
            
            if stress_result:
                logger.info(f"✅ {scenario} stress test successful")
                logger.info(f"Total Return: {stress_result['total_return']:.2%}")
                logger.info(f"Max Drawdown: {stress_result['max_drawdown']:.2%}")
            else:
                logger.error(f"❌ {scenario} stress test failed")
        
        # Test 3: Comprehensive Risk Analysis
        logger.info("\n3. Testing comprehensive risk analysis...")
        
        comprehensive_analysis = stress_tester.comprehensive_risk_analysis(
            portfolio_weights=test_portfolio,
            symbols=test_symbols
        )
        
        if comprehensive_analysis:
            logger.info("✅ Comprehensive risk analysis successful")
            logger.info(f"Risk Score: {comprehensive_analysis['risk_score']}/10")
            logger.info(f"Annualized Volatility: {comprehensive_analysis['portfolio_statistics']['annualized_volatility']:.2%}")
            logger.info(f"Sharpe Ratio: {comprehensive_analysis['portfolio_statistics']['sharpe_ratio']:.3f}")
            logger.info(f"Recommendations: {len(comprehensive_analysis['risk_recommendations'])}")
        else:
            logger.error("❌ Comprehensive risk analysis failed")
        
        logger.info("\n=== RISK STRESS TESTING TEST COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()