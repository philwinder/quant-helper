"""
Scenario and stress testing utilities.
"""

from datetime import datetime
from typing import Dict, Tuple

import pandas as pd


class ScenarioTester:
    """
    Provides simple shock and historical scenario analytics.
    """

    @staticmethod
    def apply_return_shock(
        equity_curve: pd.Series,
        shock_pct: float
    ) -> pd.Series:
        """
        Apply an instantaneous return shock to the series.
        """
        if equity_curve.empty:
            raise ValueError("equity_curve is empty.")
        shocked = equity_curve.astype(float).copy()
        shocked.iloc[-1] *= 1 + shock_pct
        return shocked

    @staticmethod
    def historical_scenario(
        prices: pd.Series,
        start: datetime,
        end: datetime
    ) -> pd.Series:
        """
        Extract and normalize a historical price window.
        """
        window = prices.loc[start:end]
        if window.empty:
            raise ValueError("No data in requested historical window.")
        return window / window.iloc[0]

    @staticmethod
    def scenario_summary(
        base_curve: pd.Series,
        stressed_curve: pd.Series
    ) -> Dict[str, float]:
        """
        Compare base vs stressed results.
        """
        aligned = pd.concat([base_curve, stressed_curve], axis=1).dropna()
        if aligned.empty:
            raise ValueError("Series do not overlap.")
        base = aligned.iloc[:, 0]
        stressed = aligned.iloc[:, 1]
        return {
            'base_return': float((base.iloc[-1] / base.iloc[0]) - 1),
            'stressed_return': float((stressed.iloc[-1] / stressed.iloc[0]) - 1),
            'difference': float((stressed.iloc[-1] / stressed.iloc[0]) - (base.iloc[-1] / base.iloc[0])),
        }

