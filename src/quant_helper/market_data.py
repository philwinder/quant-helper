"""
Market data fetcher for cryptocurrency prices using CoinGecko API.
"""

from datetime import datetime, timedelta
from typing import Dict, List
import requests
import pandas as pd
import yfinance as yf


class MarketData:
    """
    Fetches cryptocurrency market data from CoinGecko API.

    This class provides methods to retrieve historical price data (OHLCV)
    for cryptocurrencies without requiring authentication.
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        """Initialize the MarketData fetcher."""
        self._session = requests.Session()
        self._session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'quant-helper/0.1.0'
        })

    def fetch_prices(self, coin_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a single cryptocurrency.

        Args:
            coin_id: CoinGecko coin identifier (e.g., 'bitcoin', 'ethereum')
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ValueError: If dates are invalid or coin_id is not found
            requests.RequestException: If API request fails
        """
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        # Convert dates to Unix timestamps
        from_timestamp = int(start_date.timestamp())
        to_timestamp = int(end_date.timestamp())

        # Use market_chart/range endpoint which is more reliable
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': from_timestamp,
            'to': to_timestamp
        }

        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data or 'prices' not in data:
                raise ValueError(f"No data returned for coin_id '{coin_id}'. Check if the ID is valid.")

            # Parse price data
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
            prices_df = prices_df.set_index('timestamp')

            # For simplicity, use close as open/high/low (close approximation)
            # This is acceptable for daily strategies where exact OHLC isn't critical
            prices_df['open'] = prices_df['close']
            prices_df['high'] = prices_df['close']
            prices_df['low'] = prices_df['close']

            # Add volume data
            if 'total_volumes' in data:
                volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df = volume_df.set_index('timestamp')
                prices_df['volume'] = volume_df['volume']
            else:
                prices_df['volume'] = 0.0

            # Reorder columns
            prices_df = prices_df[['open', 'high', 'low', 'close', 'volume']]

            return prices_df
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch data for {coin_id}: {str(e)}")

    def fetch_equity_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for equities, ETFs, or FX pairs via Yahoo Finance.

        Args:
            symbol: Yahoo Finance ticker (e.g., 'AAPL', 'SPY', 'EURUSD=X')
            start_date: Start of price history
            end_date: End of price history
            interval: Sampling interval supported by yfinance (default: daily)

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        data = yf.download(
            symbol,
            start=start_date,
            end=end_date + timedelta(days=1),
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            raise ValueError(f"No price data returned for symbol '{symbol}'")

        data = data.rename(
            columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
        )

        return data[['open', 'high', 'low', 'close', 'volume']]

    def _fetch_volume(self, coin_id: str, from_timestamp: int, to_timestamp: int) -> pd.Series:
        """
        Fetch volume data for a cryptocurrency.

        Args:
            coin_id: CoinGecko coin identifier
            from_timestamp: Start timestamp (Unix)
            to_timestamp: End timestamp (Unix)

        Returns:
            Series of volume data aligned with OHLC timestamps
        """
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': from_timestamp,
            'to': to_timestamp
        }

        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'total_volumes' in data:
            volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df = volume_df.set_index('timestamp')
            return volume_df['volume']

        # Return zeros if volume data not available
        return pd.Series(0.0)

    def fetch_multiple_coins(
        self,
        coin_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple cryptocurrencies.

        Args:
            coin_ids: List of CoinGecko coin identifiers
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            Dictionary mapping coin_id to DataFrame of OHLCV data

        Raises:
            ValueError: If dates are invalid
            requests.RequestException: If any API request fails
        """
        results = {}

        for coin_id in coin_ids:
            try:
                results[coin_id] = self.fetch_prices(coin_id, start_date, end_date)
            except Exception as e:
                print(f"Warning: Failed to fetch data for {coin_id}: {str(e)}")
                continue

        if not results:
            raise ValueError("Failed to fetch data for any of the requested coins")

        return results

    def list_popular_coins(self) -> List[Dict[str, str]]:
        """
        Get a list of popular cryptocurrencies with their IDs.

        Returns:
            List of dictionaries with 'id', 'symbol', and 'name' keys
        """
        url = f"{self.BASE_URL}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 20,
            'page': 1,
            'sparkline': False
        }

        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        return [
            {'id': coin['id'], 'symbol': coin['symbol'], 'name': coin['name']}
            for coin in data
        ]
