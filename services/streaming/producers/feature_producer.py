"""
Feature Producer for Redis Streams
Produces feature updates to streams for real-time consumption
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import yaml

# Add infrastructure to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'infrastructure' / 'streaming'))

from stream_client import RedisStreamClient

logger = logging.getLogger(__name__)


class FeatureProducer:
    """
    Produces feature updates to Redis Streams
    Handles both raw features and PIT-validated features
    """

    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path is None:
            config_path = project_root / 'infrastructure' / 'streaming' / 'redis_streams' / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize stream client
        redis_url = self.config['redis']['url']
        self.client = RedisStreamClient(redis_url)

        # Stream configurations
        self.streams = self.config['streams']

    async def connect(self):
        """Connect to Redis"""
        await self.client.connect()
        logger.info("Feature producer connected")

    async def close(self):
        """Close connection"""
        await self.client.close()
        logger.info("Feature producer closed")

    async def produce_raw_feature(
        self,
        symbol: str,
        features: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Produce raw feature data to the features.raw stream

        Args:
            symbol: Stock symbol
            features: Feature values
            metadata: Additional metadata

        Returns:
            Message ID
        """
        stream_config = self.streams['features_raw']

        message = {
            "symbol": symbol,
            "features": features,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        message_id = await self.client.produce(
            stream_name=stream_config['name'],
            data=message,
            maxlen=stream_config['maxlen'],
            approximate=stream_config['approximate']
        )

        logger.debug(f"Produced raw features for {symbol}: {message_id}")
        return message_id

    async def produce_pit_feature(
        self,
        symbol: str,
        features: Dict[str, Any],
        pit_validated: bool,
        validation_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Produce PIT-validated feature data to the features.pit stream

        Args:
            symbol: Stock symbol
            features: Feature values
            pit_validated: Whether features passed PIT validation
            validation_metadata: Validation results and metadata

        Returns:
            Message ID
        """
        stream_config = self.streams['features_pit']

        message = {
            "symbol": symbol,
            "features": features,
            "pit_validated": pit_validated,
            "validation_metadata": validation_metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        message_id = await self.client.produce(
            stream_name=stream_config['name'],
            data=message,
            maxlen=stream_config['maxlen'],
            approximate=stream_config['approximate']
        )

        logger.debug(f"Produced PIT features for {symbol}: {message_id}")
        return message_id

    async def produce_batch_features(
        self,
        features_batch: List[Dict[str, Any]],
        stream_type: str = "pit"
    ) -> List[str]:
        """
        Produce a batch of features

        Args:
            features_batch: List of feature dictionaries with 'symbol' and 'features'
            stream_type: Either 'raw' or 'pit'

        Returns:
            List of message IDs
        """
        message_ids = []

        for feature_data in features_batch:
            symbol = feature_data['symbol']
            features = feature_data['features']

            if stream_type == "raw":
                msg_id = await self.produce_raw_feature(symbol, features)
            else:
                pit_validated = feature_data.get('pit_validated', True)
                validation_metadata = feature_data.get('validation_metadata', {})
                msg_id = await self.produce_pit_feature(
                    symbol,
                    features,
                    pit_validated,
                    validation_metadata
                )

            message_ids.append(msg_id)

        logger.info(f"Produced batch of {len(message_ids)} features to {stream_type} stream")
        return message_ids


async def main():
    """Example usage of FeatureProducer"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    producer = FeatureProducer()
    await producer.connect()

    try:
        # Example: Produce raw features
        await producer.produce_raw_feature(
            symbol="AAPL",
            features={
                "sma_20": 185.45,
                "rsi_14": 67.8,
                "volume_sma_20": 50000000,
                "momentum_5d": 0.023
            },
            metadata={
                "source": "market_data_service",
                "calculation_time_ms": 5.2
            }
        )

        # Example: Produce PIT-validated features
        await producer.produce_pit_feature(
            symbol="AAPL",
            features={
                "sma_20": 185.45,
                "rsi_14": 67.8,
                "volume_sma_20": 50000000,
                "momentum_5d": 0.023,
                "earnings_surprise": 0.05
            },
            pit_validated=True,
            validation_metadata={
                "validator": "pit_validator_v1",
                "validation_time_ms": 2.3,
                "checks_passed": ["no_future_data", "proper_lag", "data_availability"]
            }
        )

        # Example: Batch production
        batch = [
            {
                "symbol": "MSFT",
                "features": {"sma_20": 374.23, "rsi_14": 58.3},
                "pit_validated": True
            },
            {
                "symbol": "GOOGL",
                "features": {"sma_20": 142.67, "rsi_14": 62.1},
                "pit_validated": True
            }
        ]
        await producer.produce_batch_features(batch, stream_type="pit")

        logger.info("Feature production examples completed")

    finally:
        await producer.close()


if __name__ == "__main__":
    asyncio.run(main())
