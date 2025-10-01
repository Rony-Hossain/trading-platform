"""
Dataset Builder for Machine Learning in Trading Strategies

Integrates triple barrier labeling with feature engineering to create
high-quality datasets for machine learning models.

Features:
- Triple barrier labeling integration
- Feature engineering pipeline
- Data persistence with metadata
- Cross-validation ready datasets
- Sample weight handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import os
import pickle
from pathlib import Path

# Import our labeling components
from ..labels.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabel,
    TripleBarrierLabeler,
    MetaLabeler,
    create_meta_labels
)
from ..labels.vol_scaled_targets import (
    VolScaledConfig,
    VolScaledLabel,
    VolScaledLabeler,
    VolatilityEstimator
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    # Triple barrier settings
    tb_horizon_days: int = 5  # TB_HORIZON_DAYS
    tb_upper_sigma: float = 2.0  # TB_UPPER_SIGMA
    tb_lower_sigma: float = 1.5  # TB_LOWER_SIGMA
    
    # Vol-scaled targets settings
    use_vol_scaled_targets: bool = False
    vol_scaling_method: str = "realized"  # "realized", "ewm", "garch", "iv30"
    vol_window_days: int = 20
    vol_min_periods: int = 10
    iv30_fallback: bool = True  # Use realized vol if IV30 unavailable
    store_both_returns: bool = True  # Store both raw and vol-scaled returns
    
    # Feature engineering settings
    feature_lookback_days: int = 20
    include_technical_indicators: bool = True
    include_sentiment_features: bool = True
    include_macro_features: bool = True
    
    # Data splitting
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Persistence settings
    save_intermediate: bool = True
    output_dir: str = "datasets"


@dataclass
class DatasetMetadata:
    """Metadata for created datasets."""
    created_at: datetime
    config: DatasetConfig
    symbols: List[str]
    date_range: Tuple[datetime, datetime]
    n_samples: int
    n_features: int
    label_distribution: Dict[str, int]
    feature_names: List[str]
    performance_metrics: Dict[str, float]


class FeatureEngineer:
    """Feature engineering for trading datasets."""
    
    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        
    def create_price_features(self, prices: pd.Series) -> pd.DataFrame:
        """Create price-based technical features."""
        features = pd.DataFrame(index=prices.index)
        
        # Returns
        returns = prices.pct_change()
        features['returns_1d'] = returns
        features['returns_5d'] = prices.pct_change(5)
        features['returns_10d'] = prices.pct_change(10)
        features['returns_20d'] = prices.pct_change(20)
        
        # Volatility
        features['volatility_5d'] = returns.rolling(5).std() * np.sqrt(252)
        features['volatility_10d'] = returns.rolling(10).std() * np.sqrt(252)
        features['volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Moving averages
        features['sma_5'] = prices.rolling(5).mean()
        features['sma_10'] = prices.rolling(10).mean()
        features['sma_20'] = prices.rolling(20).mean()
        features['ema_5'] = prices.ewm(span=5).mean()
        features['ema_10'] = prices.ewm(span=10).mean()
        features['ema_20'] = prices.ewm(span=20).mean()
        
        # Price ratios
        features['price_to_sma5'] = prices / features['sma_5']
        features['price_to_sma20'] = prices / features['sma_20']
        features['sma5_to_sma20'] = features['sma_5'] / features['sma_20']
        
        # Momentum indicators
        features['rsi_14'] = self._calculate_rsi(prices, 14)
        features['rsi_5'] = self._calculate_rsi(prices, 5)
        
        # Bollinger Bands
        bb_mid = features['sma_20']
        bb_std = prices.rolling(20).std()
        features['bb_upper'] = bb_mid + (bb_std * 2)
        features['bb_lower'] = bb_mid - (bb_std * 2)
        features['bb_position'] = (prices - bb_mid) / (bb_std * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_mid
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_sentiment_features(self, 
                                sentiment_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Create sentiment-based features."""
        if sentiment_data is None:
            # Return empty DataFrame with consistent index
            return pd.DataFrame()
            
        features = pd.DataFrame(index=sentiment_data.index)
        
        if 'sentiment_score' in sentiment_data.columns:
            features['sentiment_score'] = sentiment_data['sentiment_score']
            features['sentiment_ma5'] = sentiment_data['sentiment_score'].rolling(5).mean()
            features['sentiment_std5'] = sentiment_data['sentiment_score'].rolling(5).std()
            
        if 'news_volume' in sentiment_data.columns:
            features['news_volume'] = sentiment_data['news_volume']
            features['news_volume_ma5'] = sentiment_data['news_volume'].rolling(5).mean()
            
        return features
    
    def create_macro_features(self, 
                            macro_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Create macro-economic features."""
        if macro_data is None:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=macro_data.index)
        
        # VIX-related features
        if 'vix' in macro_data.columns:
            features['vix'] = macro_data['vix']
            features['vix_ma10'] = macro_data['vix'].rolling(10).mean()
            features['vix_zscore'] = (macro_data['vix'] - macro_data['vix'].rolling(60).mean()) / macro_data['vix'].rolling(60).std()
            
        # Interest rates
        if 'treasury_10y' in macro_data.columns:
            features['treasury_10y'] = macro_data['treasury_10y']
            features['treasury_slope'] = macro_data.get('treasury_10y', 0) - macro_data.get('treasury_2y', 0)
            
        return features


class DatasetBuilder:
    """
    Main dataset builder that integrates triple barrier labeling
    with feature engineering and data persistence.
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config.feature_lookback_days)
        self.tb_config = TripleBarrierConfig(
            horizon_days=config.tb_horizon_days,
            upper_sigma=config.tb_upper_sigma,
            lower_sigma=config.tb_lower_sigma
        )
        self.labeler = TripleBarrierLabeler(self.tb_config)
        
        # Initialize vol-scaled components if enabled
        self.vol_scaled_labeler: Optional[VolScaledLabeler] = None
        if config.use_vol_scaled_targets:
            vol_config = VolScaledConfig(
                vol_method=config.vol_scaling_method,
                vol_lookback=config.vol_window_days
            )
            self.vol_scaled_labeler = VolScaledLabeler(self.tb_config, vol_config)
        
        self.metadata: Optional[DatasetMetadata] = None
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build_dataset(self,
                     prices: pd.Series,
                     events: pd.DataFrame,
                     symbol: str,
                     sentiment_data: Optional[pd.DataFrame] = None,
                     macro_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Build complete dataset with features and labels.
        
        Args:
            prices: Price series indexed by datetime
            events: Trading events/signals DataFrame
            symbol: Symbol identifier
            sentiment_data: Optional sentiment data
            macro_data: Optional macro-economic data
            
        Returns:
            Dictionary containing dataset components
        """
        
        logger.info(f"Building dataset for {symbol}")
        
        # Create labels (triple barrier or vol-scaled)
        if self.config.use_vol_scaled_targets and self.vol_scaled_labeler:
            logger.info("Creating vol-scaled target labels")
            iv30_data = macro_data.get('iv30') if macro_data is not None else None
            vol_labels = self.vol_scaled_labeler.create_vol_scaled_labels(prices, events, symbol, iv30_data)
            
            if not vol_labels:
                raise ValueError("No vol-scaled labels created")
            
            # Convert to compatible format for downstream processing
            tb_labels = []
            for vol_label in vol_labels:
                # Create a pseudo triple barrier label for compatibility
                tb_label = TripleBarrierLabel(
                    timestamp=vol_label.timestamp,
                    symbol=vol_label.symbol,
                    entry_price=vol_label.entry_price,
                    target_price=vol_label.target_price,
                    stop_loss_price=vol_label.stop_loss_price,
                    expiry_time=vol_label.expiry_time,
                    label=vol_label.label,
                    hit_barrier=vol_label.hit_barrier,
                    exit_price=vol_label.exit_price,
                    exit_time=vol_label.exit_time,
                    holding_time=vol_label.holding_time,
                    return_pct=vol_label.vol_scaled_return if self.config.store_both_returns else vol_label.return_pct,
                    upper_hit_ts=vol_label.upper_hit_ts,
                    lower_hit_ts=vol_label.lower_hit_ts,
                    time_expiry=vol_label.time_expiry,
                    sample_weight=vol_label.sample_weight
                )
                tb_labels.append(tb_label)
        else:
            logger.info("Creating standard triple barrier labels")
            tb_labels = self.labeler.create_labels(prices, events, symbol)
            
            if not tb_labels:
                raise ValueError("No triple barrier labels created")
            
        # Create features
        price_features = self.feature_engineer.create_price_features(prices)
        
        sentiment_features = pd.DataFrame()
        if self.config.include_sentiment_features and sentiment_data is not None:
            sentiment_features = self.feature_engineer.create_sentiment_features(sentiment_data)
            
        macro_features = pd.DataFrame()
        if self.config.include_macro_features and macro_data is not None:
            macro_features = self.feature_engineer.create_macro_features(macro_data)
        
        # Combine all features
        all_features = [price_features]
        if not sentiment_features.empty:
            all_features.append(sentiment_features)
        if not macro_features.empty:
            all_features.append(macro_features)
            
        features_df = pd.concat(all_features, axis=1) if len(all_features) > 1 else price_features
        
        # Align features with labels
        label_timestamps = [label.timestamp for label in tb_labels]
        aligned_features = features_df.loc[features_df.index.isin(label_timestamps)]
        
        # Create labels DataFrame
        labels_data = []
        for i, label in enumerate(tb_labels):
            if label.timestamp in aligned_features.index:
                label_dict = {
                    'timestamp': label.timestamp,
                    'symbol': label.symbol,
                    'label': label.label,
                    'return_pct': label.return_pct,
                    'holding_time': label.holding_time,
                    'hit_barrier': label.hit_barrier,
                    'upper_hit_ts': label.upper_hit_ts,
                    'lower_hit_ts': label.lower_hit_ts,
                    'time_expiry': label.time_expiry,
                    'sample_weight': label.sample_weight,
                    'entry_price': label.entry_price,
                    'exit_price': label.exit_price,
                    'target_price': label.target_price,
                    'stop_loss_price': label.stop_loss_price
                }
                
                # Add vol-scaled specific fields if using vol-scaled targets
                if self.config.use_vol_scaled_targets and self.vol_scaled_labeler and i < len(vol_labels):
                    vol_label = vol_labels[i]
                    if self.config.store_both_returns:
                        label_dict['raw_return'] = vol_label.return_pct
                        label_dict['vol_scaled_return'] = vol_label.vol_scaled_return
                        label_dict['entry_volatility'] = vol_label.entry_volatility
                        label_dict['realized_volatility'] = vol_label.realized_volatility
                        label_dict['vol_scaling_factor'] = vol_label.vol_scaling_factor
                        if vol_label.iv30 is not None:
                            label_dict['iv30'] = vol_label.iv30
                
                labels_data.append(label_dict)
                
        labels_df = pd.DataFrame(labels_data)
        labels_df.set_index('timestamp', inplace=True)
        
        # Align timestamps
        common_timestamps = aligned_features.index.intersection(labels_df.index)
        final_features = aligned_features.loc[common_timestamps]
        final_labels = labels_df.loc[common_timestamps]
        
        # Remove rows with NaN features
        valid_mask = ~final_features.isnull().any(axis=1)
        final_features = final_features[valid_mask]
        final_labels = final_labels[valid_mask]
        
        logger.info(f"Created dataset with {len(final_features)} samples and {len(final_features.columns)} features")
        
        # Create dataset dictionary
        dataset = {
            'features': final_features,
            'labels': final_labels,
            'symbol': symbol,
            'tb_labels': tb_labels,
            'config': self.config,
            'tb_config': self.tb_config
        }
        
        # Split dataset
        dataset.update(self._split_dataset(final_features, final_labels))
        
        # Create metadata
        self._create_metadata(dataset, symbol)
        
        # Save if requested
        if self.config.save_intermediate:
            self._save_dataset(dataset, symbol)
            
        return dataset
    
    def build_meta_labeling_dataset(self,
                                  prices: pd.Series,
                                  base_signals: pd.DataFrame,
                                  symbol: str,
                                  sentiment_data: Optional[pd.DataFrame] = None,
                                  macro_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Build dataset specifically for meta-labeling.
        
        Args:
            prices: Price series
            base_signals: Base strategy signals
            symbol: Symbol identifier
            sentiment_data: Optional sentiment data
            macro_data: Optional macro data
            
        Returns:
            Meta-labeling dataset with performance metrics
        """
        
        logger.info(f"Building meta-labeling dataset for {symbol}")
        
        # Create meta-labels using triple barrier method
        additional_features = None
        if sentiment_data is not None or macro_data is not None:
            sentiment_features = self.feature_engineer.create_sentiment_features(sentiment_data)
            macro_features = self.feature_engineer.create_macro_features(macro_data)
            
            additional_features = pd.concat([sentiment_features, macro_features], axis=1)
            additional_features = additional_features.dropna()
        
        meta_results, performance_metrics = create_meta_labels(
            prices, base_signals, self.tb_config, additional_features
        )
        
        if not meta_results:
            raise ValueError("No meta-labels created")
        
        # Convert results to DataFrame
        meta_data = []
        for result in meta_results:
            meta_data.append({
                'timestamp': result.timestamp,
                'symbol': result.symbol,
                'base_signal': result.base_signal,
                'meta_probability': result.meta_probability,
                'meta_prediction': result.meta_prediction,
                'confidence': result.confidence,
                **result.features
            })
            
        meta_df = pd.DataFrame(meta_data)
        meta_df.set_index('timestamp', inplace=True)
        
        dataset = {
            'meta_labels': meta_df,
            'meta_results': meta_results,
            'performance_metrics': performance_metrics,
            'symbol': symbol,
            'config': self.config
        }
        
        logger.info(f"Created meta-labeling dataset with {len(meta_df)} samples")
        logger.info(f"Performance metrics: {performance_metrics}")
        
        return dataset
    
    def _split_dataset(self, 
                      features: pd.DataFrame,
                      labels: pd.DataFrame) -> Dict[str, Any]:
        """Split dataset into train/validation/test sets."""
        
        n_samples = len(features)
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.validation_ratio))
        
        return {
            'X_train': features.iloc[:train_end],
            'y_train': labels.iloc[:train_end],
            'X_val': features.iloc[train_end:val_end],
            'y_val': labels.iloc[train_end:val_end],
            'X_test': features.iloc[val_end:],
            'y_test': labels.iloc[val_end:],
            'split_indices': {
                'train_end': train_end,
                'val_end': val_end,
                'total': n_samples
            }
        }
    
    def _create_metadata(self, dataset: Dict[str, Any], symbol: str):
        """Create metadata for the dataset."""
        
        features = dataset['features']
        labels = dataset['labels']
        
        # Label distribution
        label_dist = labels['label'].value_counts().to_dict()
        
        # Date range
        date_range = (features.index.min(), features.index.max())
        
        self.metadata = DatasetMetadata(
            created_at=datetime.now(),
            config=self.config,
            symbols=[symbol],
            date_range=date_range,
            n_samples=len(features),
            n_features=len(features.columns),
            label_distribution=label_dist,
            feature_names=features.columns.tolist(),
            performance_metrics={}
        )
    
    def _save_dataset(self, dataset: Dict[str, Any], symbol: str):
        """Save dataset to disk."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{symbol}_{timestamp}.pkl"
        filepath = self.output_dir / filename
        
        # Save main dataset
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
            
        # Save metadata separately
        if self.metadata:
            metadata_file = self.output_dir / f"metadata_{symbol}_{timestamp}.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(asdict(self.metadata), f)
                
        logger.info(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load dataset from disk."""
        
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
            
        logger.info(f"Dataset loaded from {filepath}")
        return dataset
    
    def validate_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset quality and return validation metrics."""
        
        features = dataset['features']
        labels = dataset['labels']
        
        validation_results = {
            'n_samples': len(features),
            'n_features': len(features.columns),
            'missing_values': features.isnull().sum().sum(),
            'label_distribution': labels['label'].value_counts().to_dict(),
            'feature_correlation_max': features.corr().abs().values.max(),
            'sample_weight_stats': {
                'mean': labels['sample_weight'].mean(),
                'std': labels['sample_weight'].std(),
                'min': labels['sample_weight'].min(),
                'max': labels['sample_weight'].max()
            }
        }
        
        # Check for data leakage (future information)
        validation_results['potential_leakage'] = self._check_data_leakage(features, labels)
        
        return validation_results
    
    def _check_data_leakage(self, 
                          features: pd.DataFrame,
                          labels: pd.DataFrame) -> Dict[str, Any]:
        """Check for potential data leakage."""
        
        leakage_checks = {
            'future_price_features': False,
            'suspicious_correlations': [],
            'timestamp_alignment': True
        }
        
        # Check for suspiciously high correlations between features and labels
        if 'return_pct' in labels.columns:
            for col in features.columns:
                if 'return' in col.lower() or 'price' in col.lower():
                    correlation = np.corrcoef(features[col].fillna(0), labels['return_pct'].fillna(0))[0, 1]
                    if abs(correlation) > 0.8:
                        leakage_checks['suspicious_correlations'].append({
                            'feature': col,
                            'correlation': correlation
                        })
        
        return leakage_checks
    
    def validate_vol_scaling(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Validate vol-scaled targets using Levene test and other metrics."""
        
        from scipy.stats import levene
        
        labels = dataset['labels']
        validation_results = {
            'levene_test': {},
            'variance_reduction': {},
            'correlation_analysis': {},
            'training_stability': {}
        }
        
        if 'raw_return' in labels.columns and 'vol_scaled_return' in labels.columns:
            raw_returns = labels['raw_return'].dropna()
            vol_scaled_returns = labels['vol_scaled_return'].dropna()
            
            # Levene test for variance reduction
            try:
                # Split into time periods for Levene test
                mid_point = len(raw_returns) // 2
                raw_first_half = raw_returns.iloc[:mid_point]
                raw_second_half = raw_returns.iloc[mid_point:]
                vol_first_half = vol_scaled_returns.iloc[:mid_point]
                vol_second_half = vol_scaled_returns.iloc[mid_point:]
                
                # Levene test on raw returns
                levene_stat_raw, levene_p_raw = levene(raw_first_half, raw_second_half)
                
                # Levene test on vol-scaled returns
                levene_stat_vol, levene_p_vol = levene(vol_first_half, vol_second_half)
                
                validation_results['levene_test'] = {
                    'raw_returns': {
                        'statistic': levene_stat_raw,
                        'p_value': levene_p_raw,
                        'variance_homogeneous': levene_p_raw > 0.05
                    },
                    'vol_scaled_returns': {
                        'statistic': levene_stat_vol,
                        'p_value': levene_p_vol,
                        'variance_homogeneous': levene_p_vol > 0.05
                    },
                    'variance_reduction_achieved': levene_p_vol < levene_p_raw and levene_p_vol < 0.05
                }
                
            except Exception as e:
                validation_results['levene_test']['error'] = str(e)
            
            # Variance reduction metrics
            validation_results['variance_reduction'] = {
                'raw_variance': raw_returns.var(),
                'vol_scaled_variance': vol_scaled_returns.var(),
                'variance_reduction_ratio': 1 - (vol_scaled_returns.var() / raw_returns.var()),
                'raw_std': raw_returns.std(),
                'vol_scaled_std': vol_scaled_returns.std()
            }
            
            # Correlation preservation
            if 'label' in labels.columns:
                raw_corr = labels[['raw_return', 'label']].corr().iloc[0, 1]
                vol_corr = labels[['vol_scaled_return', 'label']].corr().iloc[0, 1]
                
                validation_results['correlation_analysis'] = {
                    'raw_return_label_correlation': raw_corr,
                    'vol_scaled_label_correlation': vol_corr,
                    'correlation_preservation': abs(vol_corr) >= abs(raw_corr) * 0.9
                }
        
        return validation_results


# Utility functions for common dataset operations
def create_training_dataset(symbol: str,
                          prices: pd.Series,
                          events: pd.DataFrame,
                          config: Optional[DatasetConfig] = None,
                          **kwargs) -> Dict[str, Any]:
    """Convenience function to create a training dataset."""
    
    if config is None:
        config = DatasetConfig()
    
    builder = DatasetBuilder(config)
    return builder.build_dataset(prices, events, symbol, **kwargs)


def create_meta_labeling_dataset(symbol: str,
                               prices: pd.Series,
                               base_signals: pd.DataFrame,
                               config: Optional[DatasetConfig] = None,
                               **kwargs) -> Dict[str, Any]:
    """Convenience function to create a meta-labeling dataset."""
    
    if config is None:
        config = DatasetConfig()
    
    builder = DatasetBuilder(config)
    return builder.build_meta_labeling_dataset(prices, base_signals, symbol, **kwargs)


def create_vol_scaled_dataset(symbol: str,
                             prices: pd.Series,
                             events: pd.DataFrame,
                             volatility_method: str = "realized",
                             window_days: int = 20,
                             config: Optional[DatasetConfig] = None,
                             **kwargs) -> Dict[str, Any]:
    """Convenience function to create a vol-scaled targets dataset."""
    
    if config is None:
        config = DatasetConfig()
    
    # Enable vol-scaled targets
    config.use_vol_scaled_targets = True
    config.vol_scaling_method = volatility_method
    config.vol_window_days = window_days
    config.store_both_returns = True
    
    builder = DatasetBuilder(config)
    dataset = builder.build_dataset(prices, events, symbol, **kwargs)
    
    # Add vol-scaled validation metrics
    if 'labels' in dataset and 'raw_return' in dataset['labels'].columns:
        validation_results = builder.validate_vol_scaling(dataset)
        dataset['vol_scaling_validation'] = validation_results
    
    return dataset


# Example usage
def example_dataset_creation():
    """Example of how to create datasets."""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252))),
        index=dates
    )
    
    # Create events (trading signals)
    signal_dates = dates[::10]
    events = pd.DataFrame({
        'timestamp': signal_dates,
        'side': np.random.choice([1, -1], size=len(signal_dates))
    })
    
    # Configure dataset building
    config = DatasetConfig(
        tb_horizon_days=5,
        tb_upper_sigma=2.0,
        tb_lower_sigma=1.5,
        save_intermediate=False
    )
    
    # Create training dataset
    dataset = create_training_dataset('AAPL', prices, events, config)
    
    print(f"Created standard dataset with {dataset['features'].shape[0]} samples")
    print(f"Label distribution: {dataset['labels']['label'].value_counts()}")
    
    # Test vol-scaled dataset
    print("\n=== Testing Vol-Scaled Targets ===")
    
    # Create mock macro data with IV30
    macro_data = pd.DataFrame({
        'iv30': np.random.uniform(0.15, 0.45, len(dates))
    }, index=dates)
    
    # Create vol-scaled dataset
    vol_dataset = create_vol_scaled_dataset(
        'AAPL', prices, events,
        volatility_method='realized',
        window_days=20,
        macro_data=macro_data
    )
    
    print(f"Created vol-scaled dataset with {vol_dataset['features'].shape[0]} samples")
    
    # Check vol-scaled specific fields
    vol_labels = vol_dataset['labels']
    if 'raw_return' in vol_labels.columns and 'vol_scaled_return' in vol_labels.columns:
        print("✓ Both raw and vol-scaled returns stored")
        print(f"Raw return std: {vol_labels['raw_return'].std():.4f}")
        print(f"Vol-scaled return std: {vol_labels['vol_scaled_return'].std():.4f}")
        
        # Show validation results
        if 'vol_scaling_validation' in vol_dataset:
            validation = vol_dataset['vol_scaling_validation']
            if 'variance_reduction' in validation:
                var_red = validation['variance_reduction']
                reduction = var_red.get('variance_reduction_ratio', 0) * 100
                print(f"Variance reduction: {reduction:.2f}%")
            
            if 'levene_test' in validation:
                levene = validation['levene_test']
                if 'variance_reduction_achieved' in levene:
                    achieved = levene['variance_reduction_achieved']
                    print(f"Levene test variance reduction: {'✓' if achieved else '✗'}")
    
    return dataset


if __name__ == "__main__":
    example_dataset_creation()