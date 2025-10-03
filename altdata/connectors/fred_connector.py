"""
FRED (Federal Reserve Economic Data) Connector
Provides free economic indicators from the Federal Reserve Bank of St. Louis

API Documentation: https://fred.stlouisfed.org/docs/api/
"""
import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_connector import BaseAltDataConnector, AltDataPoint, ALTDATA_REQUESTS, ALTDATA_LATENCY, ALTDATA_ERRORS

logger = logging.getLogger(__name__)


class FREDConnector(BaseAltDataConnector):
    """
    FRED (Federal Reserve Economic Data) connector

    Provides access to 800,000+ US and international economic time series.

    Common series:
    - UNRATE: Unemployment rate
    - GDP: Gross Domestic Product
    - CPIAUCSL: Consumer Price Index (CPI)
    - FEDFUNDS: Federal Funds Rate
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - UMCSENT: University of Michigan Consumer Sentiment
    - PAYEMS: All Employees: Total Nonfarm
    - HOUST: Housing Starts
    - RETAILSALES: Advance Retail Sales
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Common series for equity trading
    COMMON_SERIES = {
        "UNRATE": "Unemployment Rate",
        "GDP": "Gross Domestic Product",
        "CPIAUCSL": "Consumer Price Index",
        "FEDFUNDS": "Federal Funds Rate",
        "DGS10": "10-Year Treasury Rate",
        "DGS2": "2-Year Treasury Rate",
        "UMCSENT": "Consumer Sentiment",
        "PAYEMS": "Total Nonfarm Payroll",
        "HOUST": "Housing Starts",
        "RETAILSALES": "Retail Sales",
        "INDPRO": "Industrial Production Index",
        "VIXCLS": "VIX (CBOE Volatility Index)"
    }

    def __init__(
        self,
        api_key: str,
        rate_limit_per_minute: int = 120,  # FRED allows 120 req/min
        timeout_seconds: int = 30
    ):
        super().__init__(
            source_name="FRED",
            api_key=api_key,
            rate_limit_per_minute=rate_limit_per_minute,
            timeout_seconds=timeout_seconds
        )

        self.client = httpx.Client(timeout=timeout_seconds)

        logger.info("FRED connector initialized with API key")

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request to FRED

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response

        Raises:
            Exception on API errors
        """
        # Check rate limit
        if not self.check_rate_limit():
            raise Exception(f"Rate limit exceeded for FRED")

        # Add API key
        params['api_key'] = self.api_key
        params['file_type'] = 'json'

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            import time
            start = time.time()

            response = self.client.get(url, params=params)
            response.raise_for_status()

            elapsed = time.time() - start
            ALTDATA_LATENCY.labels(source=self.source_name).observe(elapsed)

            self.record_request()
            ALTDATA_REQUESTS.labels(source=self.source_name, status='success').inc()

            return response.json()

        except httpx.HTTPStatusError as e:
            ALTDATA_REQUESTS.labels(source=self.source_name, status='error').inc()
            ALTDATA_ERRORS.labels(source=self.source_name, error_type='http_error').inc()
            logger.error(f"FRED API error: {e}")
            raise

        except Exception as e:
            ALTDATA_REQUESTS.labels(source=self.source_name, status='error').inc()
            ALTDATA_ERRORS.labels(source=self.source_name, error_type='unknown').inc()
            logger.error(f"FRED request failed: {e}")
            raise

    def fetch_latest(self, symbol: str) -> Optional[AltDataPoint]:
        """
        Fetch latest observation for a FRED series

        Args:
            symbol: FRED series ID (e.g., 'UNRATE', 'GDP')

        Returns:
            Latest data point
        """
        try:
            # Get latest observation
            response = self._make_request(
                'series/observations',
                {
                    'series_id': symbol,
                    'sort_order': 'desc',
                    'limit': 1
                }
            )

            observations = response.get('observations', [])

            if not observations:
                logger.warning(f"No observations found for {symbol}")
                return None

            obs = observations[0]

            # Parse observation
            data_point = AltDataPoint(
                source=self.source_name,
                symbol=symbol,
                timestamp=datetime.strptime(obs['date'], '%Y-%m-%d'),
                data={
                    'value': float(obs['value']) if obs['value'] != '.' else None,
                    'series_id': symbol,
                    'series_name': self.COMMON_SERIES.get(symbol, symbol),
                    'units': obs.get('units', 'unknown'),
                    'frequency': obs.get('frequency', 'unknown')
                },
                metadata={
                    'realtime_start': obs.get('realtime_start'),
                    'realtime_end': obs.get('realtime_end')
                },
                quality_score=1.0 if obs['value'] != '.' else 0.0
            )

            return data_point

        except Exception as e:
            logger.error(f"Error fetching latest for {symbol}: {e}")
            return None

    def fetch_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[AltDataPoint]:
        """
        Fetch historical observations for a FRED series

        Args:
            symbol: FRED series ID
            start_date: Start date
            end_date: End date

        Returns:
            List of historical data points
        """
        try:
            response = self._make_request(
                'series/observations',
                {
                    'series_id': symbol,
                    'observation_start': start_date.strftime('%Y-%m-%d'),
                    'observation_end': end_date.strftime('%Y-%m-%d'),
                    'sort_order': 'asc'
                }
            )

            observations = response.get('observations', [])

            data_points = []

            for obs in observations:
                # Skip missing values
                if obs['value'] == '.':
                    continue

                data_point = AltDataPoint(
                    source=self.source_name,
                    symbol=symbol,
                    timestamp=datetime.strptime(obs['date'], '%Y-%m-%d'),
                    data={
                        'value': float(obs['value']),
                        'series_id': symbol,
                        'series_name': self.COMMON_SERIES.get(symbol, symbol)
                    },
                    metadata={
                        'realtime_start': obs.get('realtime_start'),
                        'realtime_end': obs.get('realtime_end')
                    },
                    quality_score=1.0
                )

                data_points.append(data_point)

            logger.info(f"Fetched {len(data_points)} historical observations for {symbol}")

            return data_points

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []

    def validate_data(self, data_point: AltDataPoint) -> bool:
        """
        Validate a FRED data point

        Args:
            data_point: Data point to validate

        Returns:
            True if valid
        """
        # Check required fields
        if not data_point.data.get('value'):
            return False

        # Check timestamp is not in future
        if data_point.timestamp > datetime.utcnow():
            return False

        # Check value is numeric
        try:
            float(data_point.data['value'])
        except (ValueError, TypeError):
            return False

        return True

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available FRED series

        Returns:
            List of common series IDs
        """
        # Return commonly used series (FRED has 800K+ series)
        return list(self.COMMON_SERIES.keys())

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Get metadata about a FRED series

        Args:
            series_id: FRED series ID

        Returns:
            Series metadata
        """
        try:
            response = self._make_request(
                'series',
                {'series_id': series_id}
            )

            series_data = response.get('seriess', [])

            if not series_data:
                return {}

            series = series_data[0]

            return {
                'id': series.get('id'),
                'title': series.get('title'),
                'units': series.get('units'),
                'frequency': series.get('frequency'),
                'seasonal_adjustment': series.get('seasonal_adjustment'),
                'last_updated': series.get('last_updated'),
                'observation_start': series.get('observation_start'),
                'observation_end': series.get('observation_end'),
                'popularity': series.get('popularity', 0)
            }

        except Exception as e:
            logger.error(f"Error getting series info for {series_id}: {e}")
            return {}

    def close(self):
        """Close HTTP client"""
        self.client.close()


# Example usage
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    # Get API key from environment
    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        print("Error: FRED_API_KEY environment variable not set")
        print("Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        exit(1)

    # Initialize connector
    fred = FREDConnector(api_key=api_key)

    try:
        # Fetch unemployment rate
        print("\n=== Unemployment Rate (UNRATE) ===")
        unrate = fred.fetch_latest("UNRATE")
        if unrate:
            print(f"Date: {unrate.timestamp.date()}")
            print(f"Value: {unrate.data['value']}%")
            print(f"Quality: {unrate.quality_score}")

        # Fetch GDP
        print("\n=== GDP ===")
        gdp = fred.fetch_latest("GDP")
        if gdp:
            print(f"Date: {gdp.timestamp.date()}")
            print(f"Value: ${gdp.data['value']}B")

        # Fetch VIX
        print("\n=== VIX (Volatility Index) ===")
        vix = fred.fetch_latest("VIXCLS")
        if vix:
            print(f"Date: {vix.timestamp.date()}")
            print(f"Value: {vix.data['value']}")

        # Get historical data
        print("\n=== Historical Unemployment (Last 30 Days) ===")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        historical = fred.fetch_historical("UNRATE", start_date, end_date)
        print(f"Fetched {len(historical)} observations")

        # Health check
        print("\n=== Health Check ===")
        health = fred.health_check()
        print(f"Status: {health['status']}")
        print(f"Available series: {health['symbols_available']}")

        # Series info
        print("\n=== Series Info: UNRATE ===")
        info = fred.get_series_info("UNRATE")
        print(f"Title: {info.get('title')}")
        print(f"Units: {info.get('units')}")
        print(f"Frequency: {info.get('frequency')}")
        print(f"Last Updated: {info.get('last_updated')}")

    finally:
        fred.close()
