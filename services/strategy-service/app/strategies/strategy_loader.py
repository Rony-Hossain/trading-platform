"""
Strategy Loader - Loads and manages trading strategies
"""

import logging
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod
import importlib
import inspect
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.initialized = False
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize strategy with market context"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get strategy description"""
        pass
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        return True
    
    def get_required_data(self) -> List[str]:
        """Get list of required data fields"""
        return ['close', 'volume']


class SimpleMovingAverageStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'short_window': 10,
            'long_window': 30,
            'symbols': ['AAPL']
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Simple Moving Average", default_params)
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize strategy"""
        self.short_window = self.parameters.get('short_window', 10)
        self.long_window = self.parameters.get('long_window', 30)
        self.symbols = self.parameters.get('symbols', ['AAPL'])
        self.initialized = True
        logger.info(f"Initialized SMA strategy: {self.short_window}/{self.long_window}")
    
    def generate_signals(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals"""
        if not self.initialized:
            raise ValueError("Strategy not initialized")
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in data:
                continue
                
            price_data = data[symbol]
            if len(price_data) < self.long_window:
                continue
            
            # Calculate moving averages
            short_ma = sum(price_data['close'][-self.short_window:]) / self.short_window
            long_ma = sum(price_data['close'][-self.long_window:]) / self.long_window
            
            # Generate signal
            if short_ma > long_ma:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': 100,
                    'reason': f'SMA crossover: {short_ma:.2f} > {long_ma:.2f}'
                })
            elif short_ma < long_ma:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 100,
                    'reason': f'SMA crossover: {short_ma:.2f} < {long_ma:.2f}'
                })
        
        return signals
    
    def get_description(self) -> str:
        return f"Simple Moving Average crossover strategy ({self.short_window}/{self.long_window})"
    
    def get_required_data(self) -> List[str]:
        return ['close']


class BuyAndHoldStrategy(BaseStrategy):
    """Buy and Hold Strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'symbols': ['SPY'],
            'allocation': 1.0
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Buy and Hold", default_params)
        self.initial_purchase_done = False
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize strategy"""
        self.symbols = self.parameters.get('symbols', ['SPY'])
        self.allocation = self.parameters.get('allocation', 1.0)
        self.initial_purchase_done = False
        self.initialized = True
        logger.info(f"Initialized Buy and Hold strategy for {self.symbols}")
    
    def generate_signals(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate buy signal at start, then hold"""
        if not self.initialized:
            raise ValueError("Strategy not initialized")
        
        signals = []
        
        if not self.initial_purchase_done:
            for symbol in self.symbols:
                if symbol in data:
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': 100,
                        'reason': 'Initial buy and hold purchase'
                    })
            self.initial_purchase_done = True
        
        return signals
    
    def get_description(self) -> str:
        return f"Buy and Hold strategy for {self.symbols}"


class StrategyLoader:
    """Loads and manages trading strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self._load_built_in_strategies()
    
    def _load_built_in_strategies(self) -> None:
        """Load built-in strategies"""
        self.strategies['sma'] = SimpleMovingAverageStrategy
        self.strategies['buy_and_hold'] = BuyAndHoldStrategy
        logger.info(f"Loaded {len(self.strategies)} built-in strategies")
    
    def load_strategy_from_file(self, file_path: str) -> None:
        """Load strategy from Python file"""
        try:
            spec = importlib.util.spec_from_file_location("custom_strategy", file_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load strategy from {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find strategy classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseStrategy) and 
                    obj != BaseStrategy):
                    strategy_key = name.lower().replace('strategy', '')
                    self.strategies[strategy_key] = obj
                    logger.info(f"Loaded custom strategy: {strategy_key}")
                    
        except Exception as e:
            logger.error(f"Failed to load strategy from {file_path}: {e}")
            raise
    
    def load_strategies_from_directory(self, directory: str) -> None:
        """Load all strategies from directory"""
        strategy_dir = Path(directory)
        if not strategy_dir.exists():
            logger.warning(f"Strategy directory does not exist: {directory}")
            return
        
        for file_path in strategy_dir.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            try:
                self.load_strategy_from_file(str(file_path))
            except Exception as e:
                logger.error(f"Failed to load strategy from {file_path}: {e}")
    
    def get_strategy(self, strategy_name: str, parameters: Dict[str, Any] = None) -> BaseStrategy:
        """Get strategy instance by name"""
        strategy_key = strategy_name.lower().replace(' ', '_').replace('-', '_')
        
        if strategy_key not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {list(self.strategies.keys())}")
        
        strategy_class = self.strategies[strategy_key]
        return strategy_class(parameters)
    
    def list_strategies(self) -> List[Dict[str, str]]:
        """List all available strategies"""
        strategies = []
        for key, strategy_class in self.strategies.items():
            # Create temporary instance to get description
            temp_instance = strategy_class()
            strategies.append({
                'name': key,
                'class_name': strategy_class.__name__,
                'description': temp_instance.get_description()
            })
        return strategies
    
    def validate_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for a strategy"""
        try:
            strategy = self.get_strategy(strategy_name, parameters)
            return strategy.validate_parameters()
        except Exception as e:
            logger.error(f"Parameter validation failed for {strategy_name}: {e}")
            return False


# Global strategy loader instance
strategy_loader = StrategyLoader()