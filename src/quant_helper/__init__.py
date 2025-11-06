"""
Quant Helper - A simple library for quantitative finance analysis.

This library provides tools for:
- Fetching cryptocurrency market data
- Calculating performance metrics
- Backtesting trading strategies
"""

from .market_data import MarketData
from .performance import PerformanceAnalyzer
from .backtest import Backtester

__all__ = ["MarketData", "PerformanceAnalyzer", "Backtester"]
__version__ = "0.1.0"
