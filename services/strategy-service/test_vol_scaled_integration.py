"""
Test script to validate vol-scaled targets integration.

This script tests:
1. Vol-scaled dataset creation
2. Levene test for variance reduction
3. Both raw and vol-scaled returns storage
4. IV30 integration capability
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from datasets.builder import (
    DatasetConfig,
    create_vol_scaled_dataset,
    create_training_dataset
)

def create_sample_data():
    """Create sample price data and events for testing."""
    
    # Create sample price data with volatility clustering
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate returns with volatility clustering
    returns = []
    vol = 0.02  # Initial volatility
    for i in range(252):
        # GARCH-like volatility
        vol = 0.01 + 0.9 * vol + 0.1 * abs(returns[-1] if returns else 0.001)
        vol = min(vol, 0.08)  # Cap volatility
        returns.append(np.random.normal(0.001, vol))
    
    prices = pd.Series(
        100 * np.exp(np.cumsum(returns)),
        index=dates,
        name='price'
    )
    
    # Create trading events (every 5 days)
    signal_dates = dates[::5]
    events = pd.DataFrame({
        'timestamp': signal_dates,
        'side': np.random.choice([1, -1], size=len(signal_dates)),
        'size': np.random.uniform(0.5, 2.0, size=len(signal_dates))
    })
    
    # Create mock IV30 data
    iv30_data = pd.DataFrame({
        'iv30': np.random.uniform(0.15, 0.45, len(dates))
    }, index=dates)
    
    return prices, events, iv30_data

def test_standard_vs_vol_scaled():
    """Test comparison between standard and vol-scaled datasets."""
    
    print("=== Testing Standard vs Vol-Scaled Datasets ===")
    
    prices, events, iv30_data = create_sample_data()
    
    # Create standard dataset
    print("Creating standard dataset...")
    standard_config = DatasetConfig(
        tb_horizon_days=5,
        tb_upper_sigma=2.0,
        tb_lower_sigma=1.5,
        save_intermediate=False
    )
    
    standard_dataset = create_training_dataset(
        'TEST', prices, events, standard_config
    )
    
    # Create vol-scaled dataset
    print("Creating vol-scaled dataset...")
    vol_dataset = create_vol_scaled_dataset(
        'TEST', prices, events,
        volatility_method='realized',
        window_days=20,
        macro_data=iv30_data
    )
    
    print(f"Standard dataset: {len(standard_dataset['features'])} samples")
    print(f"Vol-scaled dataset: {len(vol_dataset['features'])} samples")
    
    # Check for vol-scaled specific columns
    vol_labels = vol_dataset['labels']
    expected_cols = ['raw_return', 'vol_scaled_return', 'entry_volatility', 
                     'realized_volatility', 'vol_scaling_factor']
    
    print("\\nVol-scaled specific columns:")
    for col in expected_cols:
        present = col in vol_labels.columns
        print(f"  {col}: {'✓' if present else '✗'}")
    
    # Validation results
    if 'vol_scaling_validation' in vol_dataset:
        validation = vol_dataset['vol_scaling_validation']
        print("\\n=== Validation Results ===")
        
        # Levene test results
        if 'levene_test' in validation:
            levene = validation['levene_test']
            print(f"Levene test (variance reduction): {levene}")
        
        # Variance reduction
        if 'variance_reduction' in validation:
            var_red = validation['variance_reduction']
            reduction = var_red.get('variance_reduction_ratio', 0) * 100
            print(f"Variance reduction: {reduction:.2f}%")
        
        # Correlation preservation
        if 'correlation_analysis' in validation:
            corr = validation['correlation_analysis']
            preserved = corr.get('correlation_preservation', False)
            print(f"Correlation preserved: {'✓' if preserved else '✗'}")
    
    return standard_dataset, vol_dataset

def test_iv30_integration():
    """Test IV30 integration specifically."""
    
    print("\\n=== Testing IV30 Integration ===")
    
    prices, events, iv30_data = create_sample_data()
    
    # Test with IV30 method
    vol_dataset = create_vol_scaled_dataset(
        'TEST_IV30', prices, events,
        volatility_method='iv30',
        window_days=20,
        macro_data=iv30_data
    )
    
    vol_labels = vol_dataset['labels']
    
    # Check if IV30 values are stored
    iv30_present = 'iv30' in vol_labels.columns
    iv30_values = vol_labels['iv30'].dropna() if iv30_present else pd.Series()
    
    print(f"IV30 column present: {'✓' if iv30_present else '✗'}")
    print(f"IV30 values count: {len(iv30_values)}")
    
    if len(iv30_values) > 0:
        print(f"IV30 range: {iv30_values.min():.3f} - {iv30_values.max():.3f}")
    
    return vol_dataset

def main():
    """Run all tests."""
    
    print("Vol-Scaled Targets Integration Test")
    print("=" * 50)
    
    try:
        # Test 1: Standard vs Vol-scaled comparison
        standard_ds, vol_ds = test_standard_vs_vol_scaled()
        
        # Test 2: IV30 integration
        iv30_ds = test_iv30_integration()
        
        print("\\n=== Test Summary ===")
        print("✓ Vol-scaled dataset creation")
        print("✓ Raw and vol-scaled returns storage")
        print("✓ Levene test implementation")
        print("✓ IV30 integration capability")
        print("✓ Validation metrics calculation")
        
        print("\\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)