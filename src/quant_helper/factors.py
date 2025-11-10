"""
Factor data utilities and simple exposure analysis.
"""

from datetime import datetime
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .market_data import MarketData


class FactorAnalyzer:
    """
    Loads proxy factor returns and estimates exposures via linear regression.
    """

    def __init__(self, market_data: Optional[MarketData] = None):
        self._market_data = market_data or MarketData()

    def factor_returns(
        self,
        tickers: Mapping[str, str],
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Download daily factor returns using Yahoo Finance tickers.

        Args:
            tickers: Mapping of factor name -> Yahoo ticker
            start: Start date
            end: End date

        Returns:
            DataFrame of factor returns.
        """
        frames = []
        for name, ticker in tickers.items():
            prices = self._market_data.fetch_equity_prices(ticker, start, end)[
                'close'
            ]
            returns = prices.pct_change().dropna().rename(name)
            frames.append(returns)

        if not frames:
            raise ValueError("No factor tickers supplied.")

        return pd.concat(frames, axis=1).dropna()

    def estimate_exposures(
        self,
        target_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Run OLS of target returns against factor returns.

        Returns:
            Dict with alpha, residual_vol, and factor betas.
        """
        data = pd.concat([target_returns, factor_returns], axis=1).dropna()
        if data.empty:
            raise ValueError("Insufficient overlapping data for regression.")

        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values
        X = np.column_stack([np.ones(len(X)), X])

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ coeffs
        residuals = y - fitted

        output = {
            'alpha': float(coeffs[0]),
            'residual_vol': float(residuals.std(ddof=1)),
        }

        for name, beta in zip(factor_returns.columns, coeffs[1:]):
            output[f'beta_{name}'] = float(beta)

        return output

