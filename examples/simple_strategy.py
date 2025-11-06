"""
Example: Simple Moving Average Crossover Strategy

This example demonstrates how to use the quant-helper library to:
1. Fetch historical cryptocurrency data
2. Implement a simple trading strategy
3. Backtest the strategy
4. Analyze performance metrics
"""

from datetime import datetime, timedelta
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quant_helper import MarketData, PerformanceAnalyzer, Backtester
import pandas as pd


def moving_average_crossover_strategy(prices_df: pd.DataFrame) -> pd.Series:
    """
    Simple Moving Average (SMA) Crossover Strategy.

    Rules:
    - Buy (position = 1) when fast MA crosses above slow MA
    - Sell (position = 0) when fast MA crosses below slow MA

    Args:
        prices_df: DataFrame with OHLCV data

    Returns:
        Series of positions (1 = long, 0 = flat)
    """
    close_prices = prices_df['close']

    # Calculate moving averages
    fast_window = 7   # 7-day MA
    slow_window = 25  # 25-day MA

    fast_ma = close_prices.rolling(window=fast_window).mean()
    slow_ma = close_prices.rolling(window=slow_window).mean()

    # Generate signals
    positions = pd.Series(0.0, index=close_prices.index)

    # When fast MA > slow MA, go long (position = 1)
    positions[fast_ma > slow_ma] = 1.0

    # When fast MA < slow MA, stay flat (position = 0)
    positions[fast_ma <= slow_ma] = 0.0

    return positions


def main():
    """Run the example backtest."""
    print("\n" + "=" * 60)
    print("QUANT HELPER - MOVING AVERAGE CROSSOVER EXAMPLE")
    print("=" * 60 + "\n")

    # Initialize components
    market_data = MarketData()
    performance_analyzer = PerformanceAnalyzer()
    backtester = Backtester(
        market_data=market_data,
        performance_analyzer=performance_analyzer,
        initial_capital=10000.0  # Start with $10,000
    )

    # Set backtest parameters
    coin_id = 'bitcoin'  # Can also try: 'ethereum', 'solana', 'cardano'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months of data

    print(f"Fetching {coin_id} data from {start_date.date()} to {end_date.date()}...")
    print("This may take a few seconds...\n")

    try:
        # Run the backtest
        results = backtester.run_strategy(
            strategy_func=moving_average_crossover_strategy,
            coin_id=coin_id,
            start_date=start_date,
            end_date=end_date
        )

        # Print summary
        backtester.print_summary()

        # Additional insights
        print("STRATEGY DETAILS:")
        print(f"  Fast MA Window:   7 days")
        print(f"  Slow MA Window:   25 days")
        print(f"  Total Days:       {len(results['prices'])}")

        # Calculate how many days we were in the market
        positions = results['positions']
        days_in_market = (positions > 0).sum()
        days_out_market = (positions == 0).sum()

        print(f"  Days Long:        {days_in_market}")
        print(f"  Days Flat:        {days_out_market}")
        print(f"  Exposure:         {days_in_market / len(positions):.2%}")
        print("\n" + "=" * 60 + "\n")

        print("TIP: Try modifying the strategy parameters or testing different coins!")
        print("     Available coins: bitcoin, ethereum, solana, cardano, polkadot, etc.\n")

    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Verify the coin_id is valid (try 'bitcoin' or 'ethereum')")
        print("  3. Try a different date range")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
