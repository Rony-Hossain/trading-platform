"""
Signal Consumer for Redis Streams
Consumes trading signals from streams and processes them
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Callable
from datetime import datetime
import yaml

# Add infrastructure to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'infrastructure' / 'streaming'))

from stream_client import RedisStreamClient, StreamMessage

logger = logging.getLogger(__name__)


class SignalConsumer:
    """
    Consumes trading signals from Redis Streams
    Processes signals and forwards to execution service
    """

    def __init__(self, config_path: str = None, consumer_name: str = None):
        # Load configuration
        if config_path is None:
            config_path = project_root / 'infrastructure' / 'streaming' / 'redis_streams' / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize stream client
        redis_url = self.config['redis']['url']
        self.client = RedisStreamClient(redis_url)

        # Consumer configuration
        self.consumer_name = consumer_name or f"signal_consumer_{id(self)}"
        self.consumer_config = self.config['consumer']

        # Signal handlers
        self.signal_handlers: Dict[str, Callable] = {}

    async def connect(self):
        """Connect to Redis"""
        await self.client.connect()
        logger.info(f"Signal consumer '{self.consumer_name}' connected")

    async def close(self):
        """Close connection"""
        await self.client.stop_consumer_loop()
        await self.client.close()
        logger.info(f"Signal consumer '{self.consumer_name}' closed")

    def register_handler(self, strategy: str, handler: Callable[[StreamMessage], Any]):
        """
        Register a handler for a specific strategy's signals

        Args:
            strategy: Strategy name
            handler: Async function to process signals
        """
        self.signal_handlers[strategy] = handler
        logger.info(f"Registered handler for strategy: {strategy}")

    async def process_signal(self, message: StreamMessage):
        """
        Process a signal message

        Args:
            message: StreamMessage containing signal data
        """
        try:
            data = message.data
            strategy = data.get('strategy')
            symbol = data.get('symbol')
            signal_type = data.get('signal_type')  # BUY, SELL, HOLD
            confidence = data.get('confidence', 0.0)

            logger.info(
                f"Processing signal: {strategy} - {symbol} - {signal_type} "
                f"(confidence: {confidence:.3f})"
            )

            # Call strategy-specific handler if registered
            if strategy in self.signal_handlers:
                await self.signal_handlers[strategy](message)
            else:
                # Default processing
                await self.default_signal_handler(message)

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            raise

    async def default_signal_handler(self, message: StreamMessage):
        """Default signal handler"""
        data = message.data
        logger.info(f"Default handler: {data.get('symbol')} - {data.get('signal_type')}")

        # In production, this would forward to execution service
        # For now, just log
        signal_data = {
            "symbol": data.get('symbol'),
            "signal_type": data.get('signal_type'),
            "strategy": data.get('strategy'),
            "confidence": data.get('confidence'),
            "features": data.get('features', {}),
            "timestamp": data.get('timestamp')
        }

        logger.debug(f"Signal data: {signal_data}")

    async def start_consuming(
        self,
        strategy: str = "*",
        group_name: str = None,
        count: int = None,
        block_ms: int = None
    ):
        """
        Start consuming signals for a strategy

        Args:
            strategy: Strategy name or '*' for all strategies
            group_name: Consumer group name
            count: Messages per batch
            block_ms: Block time in milliseconds
        """
        # Get stream name
        if strategy == "*":
            stream_name = "signals.all"
        else:
            stream_template = self.config['streams']['signals']['name_template']
            stream_name = stream_template.format(strategy=strategy)

        # Use defaults from config if not specified
        group_name = group_name or self.consumer_config['group_name']
        count = count or self.consumer_config['count']
        block_ms = block_ms or self.consumer_config['block_ms']

        logger.info(
            f"Starting signal consumption from '{stream_name}' "
            f"(group: {group_name}, consumer: {self.consumer_name})"
        )

        # Start consumer loop
        await self.client.start_consumer_loop(
            stream_name=stream_name,
            group_name=group_name,
            consumer_name=self.consumer_name,
            handler=self.process_signal,
            count=count,
            block_ms=block_ms
        )


class FeatureConsumer:
    """
    Consumes PIT-validated features from Redis Streams
    """

    def __init__(self, config_path: str = None, consumer_name: str = None):
        # Load configuration
        if config_path is None:
            config_path = project_root / 'infrastructure' / 'streaming' / 'redis_streams' / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize stream client
        redis_url = self.config['redis']['url']
        self.client = RedisStreamClient(redis_url)

        # Consumer configuration
        self.consumer_name = consumer_name or f"feature_consumer_{id(self)}"
        self.consumer_config = self.config['consumer']

    async def connect(self):
        """Connect to Redis"""
        await self.client.connect()
        logger.info(f"Feature consumer '{self.consumer_name}' connected")

    async def close(self):
        """Close connection"""
        await self.client.stop_consumer_loop()
        await self.client.close()
        logger.info(f"Feature consumer '{self.consumer_name}' closed")

    async def process_feature(self, message: StreamMessage):
        """Process a feature message"""
        try:
            data = message.data
            symbol = data.get('symbol')
            features = data.get('features', {})
            pit_validated = data.get('pit_validated', False)

            if not pit_validated:
                logger.warning(f"Received non-PIT-validated features for {symbol}")
                return

            logger.info(f"Processing features for {symbol}: {len(features)} features")

            # In production, this would:
            # 1. Store features in feature store
            # 2. Trigger model inference
            # 3. Generate signals

        except Exception as e:
            logger.error(f"Error processing features: {e}")
            raise

    async def start_consuming(
        self,
        group_name: str = None,
        count: int = None,
        block_ms: int = None
    ):
        """Start consuming PIT-validated features"""
        stream_config = self.config['streams']['features_pit']
        stream_name = stream_config['name']

        # Use defaults from config if not specified
        group_name = group_name or stream_config['consumer_group']
        count = count or self.consumer_config['count']
        block_ms = block_ms or self.consumer_config['block_ms']

        logger.info(
            f"Starting feature consumption from '{stream_name}' "
            f"(group: {group_name}, consumer: {self.consumer_name})"
        )

        # Start consumer loop
        await self.client.start_consumer_loop(
            stream_name=stream_name,
            group_name=group_name,
            consumer_name=self.consumer_name,
            handler=self.process_feature,
            count=count,
            block_ms=block_ms
        )


async def main():
    """Example usage of SignalConsumer"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example: Signal consumer
    signal_consumer = SignalConsumer(consumer_name="example_consumer")
    await signal_consumer.connect()

    # Register custom handler for a strategy
    async def custom_momentum_handler(message: StreamMessage):
        logger.info(f"Custom momentum handler: {message.data}")

    signal_consumer.register_handler("momentum", custom_momentum_handler)

    try:
        # Start consuming signals (this would run indefinitely)
        # await signal_consumer.start_consuming(strategy="momentum")

        # For demo, just show it's ready
        logger.info("Signal consumer ready (not starting loop in example)")

    finally:
        await signal_consumer.close()


if __name__ == "__main__":
    asyncio.run(main())
