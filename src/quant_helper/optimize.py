"""
Basic portfolio optimization helpers.
"""

from typing import Optional

import numpy as np
import pandas as pd


class PortfolioOptimizer:
    """
    Provides mean-variance and risk-parity weight calculations.
    """

    @staticmethod
    def mean_variance_weights(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_aversion: float = 1.0
    ) -> pd.Series:
        """
        Compute weights proportional to Σ^{-1} μ, normalized to sum to 1.
        """
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive.")

        inv_cov = np.linalg.pinv(cov_matrix.values)
        raw = inv_cov @ expected_returns.values
        weights = raw / raw.sum()
        return pd.Series(weights, index=expected_returns.index)

    @staticmethod
    def risk_parity_weights(
        cov_matrix: pd.DataFrame,
        tol: float = 1e-6,
        max_iter: int = 500
    ) -> pd.Series:
        """
        Solve for equal risk contribution weights via simple fixed-point iteration.
        """
        n = cov_matrix.shape[0]
        weights = np.ones(n) / n
        sigma = cov_matrix.values

        for _ in range(max_iter):
            portfolio_vol = np.sqrt(weights @ sigma @ weights)
            if portfolio_vol == 0:
                break
            marginal = sigma @ weights
            contributions = weights * marginal
            target = portfolio_vol ** 2 / n
            gradient = contributions - target
            if np.linalg.norm(gradient) < tol:
                break
            weights -= gradient / (marginal + 1e-12)
            weights = np.clip(weights, 1e-6, None)
            weights /= weights.sum()

        return pd.Series(weights, index=cov_matrix.index)

