"""
Transaction cost modeling utilities.
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class TransactionCostModel:
    """
    Applies simple proportional trading costs.

    Costs scale with notional traded (commission + slippage in decimal form).
    """

    commission: float = 0.0005
    slippage: float = 0.0005

    @property
    def rate(self) -> float:
        """Combined proportional cost rate."""
        return max(self.commission + self.slippage, 0.0)

    def cost_series(self, positions: pd.Series, prices: pd.Series) -> pd.Series:
        """
        Calculate period cost in currency units.

        Args:
            positions: Position sizes (-1 to 1)
            prices: Matching price series

        Returns:
            Series of costs aligned with positions index.
        """
        trades = positions.diff().fillna(positions)
        return trades.abs() * prices * self.rate

