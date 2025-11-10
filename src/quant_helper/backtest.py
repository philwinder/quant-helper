"""
Backtesting engine for testing trading strategies on historical data.
"""

from typing import Callable, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .market_data import MarketData
from .performance import PerformanceAnalyzer
from .costs import TransactionCostModel


class Backtester:
    """
    Backtests trading strategies on historical cryptocurrency data.

    This class simulates trading based on a strategy function and calculates
    performance metrics.
    """

    def __init__(
        self,
        market_data: MarketData,
        performance_analyzer: PerformanceAnalyzer,
        initial_capital: float = 10000.0,
        cost_model: Optional[TransactionCostModel] = None
    ):
        """
        Initialize the Backtester.

        Args:
            market_data: MarketData instance for fetching price data
            performance_analyzer: PerformanceAnalyzer instance for metrics
            initial_capital: Starting capital in USD (default: $10,000)
        """
        self._market_data = market_data
        self._performance = performance_analyzer
        self._initial_capital = initial_capital
        self._cost_model = cost_model
        self._results = None

    def run_strategy(
        self,
        strategy_func: Callable[[pd.DataFrame], pd.Series],
        coin_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Execute a backtest of a trading strategy.

        Args:
            strategy_func: Function that takes OHLCV DataFrame and returns positions Series
                          (1.0 = fully long, 0.0 = flat/cash, -1.0 = fully short)
            coin_id: CoinGecko coin identifier
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dictionary containing backtest results and performance metrics

        The strategy function should have this signature:
            def strategy(prices: pd.DataFrame) -> pd.Series:
                # prices contains columns: open, high, low, close, volume
                # return a Series with same index, values are positions (-1, 0, or 1)
                pass
        """
        # Fetch historical price data
        prices_df = self._market_data.fetch_prices(coin_id, start_date, end_date)

        if prices_df.empty:
            raise ValueError(f"No price data available for {coin_id} in the specified date range")

        # Generate trading signals from strategy
        positions = strategy_func(prices_df)

        # Align positions with price data
        positions = positions.reindex(prices_df.index, method='ffill').fillna(0.0)

        # Calculate strategy returns
        portfolio_values, cumulative_costs = self._calculate_portfolio_values(
            prices_df['close'],
            positions
        )

        # Calculate buy-and-hold benchmark returns
        benchmark_values = self._calculate_buy_and_hold(prices_df['close'])

        # Generate performance metrics
        strategy_returns = self._performance.calculate_daily_returns(portfolio_values)
        benchmark_returns = self._performance.calculate_daily_returns(benchmark_values)

        self._results = {
            'coin_id': coin_id,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self._initial_capital,
            'final_value': portfolio_values.iloc[-1],
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'positions': positions,
            'prices': prices_df,
            'transaction_costs': cumulative_costs,
            'strategy_metrics': self._performance.generate_performance_summary(
                portfolio_values,
                strategy_returns
            ),
            'benchmark_metrics': self._performance.generate_performance_summary(
                benchmark_values,
                benchmark_returns
            )
        }

        return self._results

    def _calculate_portfolio_values(
        self,
        prices: pd.Series,
        positions: pd.Series
    ) -> (pd.Series, pd.Series):
        """
        Calculate portfolio value over time based on positions.

        Args:
            prices: Close prices
            positions: Position sizes (-1 to 1)

        Returns:
            Series of portfolio values over time
        """
        # Calculate returns for each period
        price_returns = prices.pct_change().fillna(0.0)

        # Strategy returns are position * price returns (shifted by 1 to avoid lookahead)
        # Position taken at close of day t affects returns from t to t+1
        shifted_positions = positions.shift(1).fillna(0.0)
        strategy_returns = shifted_positions * price_returns

        costs = pd.Series(0.0, index=prices.index)
        if self._cost_model:
            costs = self._cost_model.cost_series(positions, prices)
            strategy_returns -= costs / self._initial_capital

        # Calculate cumulative portfolio value
        portfolio_values = self._initial_capital * (1 + strategy_returns).cumprod()

        return portfolio_values, costs.cumsum()

    def _calculate_buy_and_hold(self, prices: pd.Series) -> pd.Series:
        """
        Calculate buy-and-hold benchmark portfolio values.

        Args:
            prices: Close prices

        Returns:
            Series of benchmark portfolio values
        """
        # Normalize prices to initial capital
        normalized_prices = prices / prices.iloc[0]
        benchmark_values = self._initial_capital * normalized_prices

        return benchmark_values

    def get_results(self) -> Dict:
        """
        Get the results of the most recent backtest.

        Returns:
            Dictionary containing backtest results

        Raises:
            RuntimeError: If no backtest has been run yet
        """
        if self._results is None:
            raise RuntimeError("No backtest has been run yet. Call run_strategy() first.")

        return self._results

    def print_summary(self):
        """
        Print a formatted summary of the backtest results.

        Raises:
            RuntimeError: If no backtest has been run yet
        """
        if self._results is None:
            raise RuntimeError("No backtest has been run yet. Call run_strategy() first.")

        results = self._results
        strategy = results['strategy_metrics']
        benchmark = results['benchmark_metrics']

        print("\n" + "=" * 60)
        print(f"BACKTEST SUMMARY: {results['coin_id'].upper()}")
        print("=" * 60)
        print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        if self._cost_model:
            total_cost = results['transaction_costs'].iloc[-1]
            print(f"Total Costs: ${total_cost:,.2f}")
        print("-" * 60)

        print("\nSTRATEGY PERFORMANCE:")
        print(f"  Total Return:     {strategy['total_return']:>10.2%}")
        print(f"  Sharpe Ratio:     {strategy['sharpe_ratio']:>10.2f}")
        print(f"  Max Drawdown:     {strategy['max_drawdown']:>10.2%}")
        print(f"  Volatility:       {strategy['volatility']:>10.2%}")
        print(f"  Win Rate:         {strategy['win_rate']:>10.2%}")

        print("\nBUY & HOLD BENCHMARK:")
        print(f"  Total Return:     {benchmark['total_return']:>10.2%}")
        print(f"  Sharpe Ratio:     {benchmark['sharpe_ratio']:>10.2f}")
        print(f"  Max Drawdown:     {benchmark['max_drawdown']:>10.2%}")
        print(f"  Volatility:       {benchmark['volatility']:>10.2%}")

        print("\nOUTPERFORMANCE:")
        outperformance = strategy['total_return'] - benchmark['total_return']
        print(f"  vs Buy & Hold:    {outperformance:>10.2%}")

        print("=" * 60 + "\n")
