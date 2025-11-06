"""
Performance analysis tools for measuring trading strategy results.
"""

import pandas as pd
import numpy as np


class PerformanceAnalyzer:
    """
    Analyzes the performance of trading strategies.

    Provides methods to calculate returns, risk metrics, and performance statistics.
    """

    def __init__(self):
        """Initialize the PerformanceAnalyzer."""
        pass

    def calculate_daily_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate daily percentage returns from price data.

        Args:
            prices: Series of prices (close prices)

        Returns:
            Series of daily returns as percentages (e.g., 0.05 = 5% gain)
        """
        if prices.empty:
            return pd.Series(dtype=float)

        returns = prices.pct_change()
        return returns.dropna()

    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns from daily returns.

        Args:
            returns: Series of daily returns (as decimals)

        Returns:
            Series of cumulative returns showing portfolio growth
        """
        if returns.empty:
            return pd.Series(dtype=float)

        cumulative = (1 + returns).cumprod() - 1
        return cumulative

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate the Sharpe ratio (risk-adjusted return metric).

        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate (default: 0.0)
            periods_per_year: Number of periods per year (default: 365 for daily)

        Returns:
            Sharpe ratio value (higher is better, >1 is good, >2 is very good)
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        # Annualize returns and volatility
        mean_return = returns.mean() * periods_per_year
        std_return = returns.std() * np.sqrt(periods_per_year)

        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe

    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).

        Args:
            prices: Series of prices or cumulative returns

        Returns:
            Maximum drawdown as a decimal (e.g., -0.25 = -25% drawdown)
        """
        if prices.empty:
            return 0.0

        # Calculate running maximum
        running_max = prices.expanding().max()

        # Calculate drawdown at each point
        drawdown = (prices - running_max) / running_max

        # Return the maximum (most negative) drawdown
        return drawdown.min()

    def calculate_volatility(self, returns: pd.Series, periods_per_year: int = 365) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).

        Args:
            returns: Series of daily returns
            periods_per_year: Number of periods per year (default: 365 for daily)

        Returns:
            Annualized volatility as a decimal
        """
        if returns.empty:
            return 0.0

        return returns.std() * np.sqrt(periods_per_year)

    def calculate_total_return(self, prices: pd.Series) -> float:
        """
        Calculate total return from start to end.

        Args:
            prices: Series of prices

        Returns:
            Total return as a decimal (e.g., 0.50 = 50% gain)
        """
        if prices.empty or len(prices) < 2:
            return 0.0

        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        Calculate the percentage of profitable periods.

        Args:
            returns: Series of daily returns

        Returns:
            Win rate as a decimal (e.g., 0.55 = 55% of days were profitable)
        """
        if returns.empty:
            return 0.0

        profitable_periods = (returns > 0).sum()
        total_periods = len(returns)

        return profitable_periods / total_periods

    def generate_performance_summary(
        self,
        prices: pd.Series,
        returns: pd.Series = None,
        risk_free_rate: float = 0.0
    ) -> dict:
        """
        Generate a comprehensive performance summary.

        Args:
            prices: Series of prices or portfolio values
            returns: Optional pre-calculated returns series
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary containing various performance metrics
        """
        if returns is None:
            returns = self.calculate_daily_returns(prices)

        summary = {
            'total_return': self.calculate_total_return(prices),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns, risk_free_rate),
            'max_drawdown': self.calculate_max_drawdown(prices),
            'volatility': self.calculate_volatility(returns),
            'win_rate': self.calculate_win_rate(returns),
            'avg_daily_return': returns.mean(),
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'total_trades': len(returns)
        }

        return summary
