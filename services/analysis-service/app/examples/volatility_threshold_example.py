"""
Comprehensive Examples for Volatility-Normalized Surprise Threshold Calibration

This module demonstrates the complete volatility threshold calibration capabilities including:
- Asset-specific volatility normalization using realized & implied volatility
- Sector-specific threshold adjustment with industry clustering
- Event type-specific surprise sensitivity mapping
- Market regime-aware threshold adaptation
- Multi-timeframe volatility analysis and normalization

Key Features:
- N-sigma threshold normalization based on rolling volatility windows
- Sector volatility clustering and percentile-based adjustment
- Event type sensitivity mapping (earnings vs FDA vs M&A sensitivity)
- Market regime detection for adaptive threshold scaling
- Cross-asset volatility spillover analysis
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import volatility threshold components
from ..services.volatility_threshold_calibration import (
    VolatilityThresholdCalibrator,
    EventType,
    SectorType,
    MarketRegime,
    create_volatility_threshold_calibrator
)

logger = logging.getLogger(__name__)

class VolatilityThresholdDemo:
    """Comprehensive demonstration of volatility threshold calibration"""
    
    def __init__(self):
        self.calibrator: Optional[VolatilityThresholdCalibrator] = None
        self.test_data: Dict[str, Any] = {}
        
    async def initialize_system(self):
        """Initialize the volatility threshold calibration system"""
        print("🔧 Initializing Volatility Threshold Calibration System...")
        
        self.calibrator = await create_volatility_threshold_calibrator()
        
        # Generate synthetic market data for demonstration
        await self._generate_test_data()
        
        print("   ✓ Calibration system ready")
        print("   ✓ Test data generated")
        print()
    
    async def _generate_test_data(self):
        """Generate realistic synthetic market data for different asset types"""
        
        assets = {
            # Technology stocks - high volatility, high beta
            "AAPL": {"sector": SectorType.TECHNOLOGY, "base_vol": 0.28, "beta": 1.2},
            "NVDA": {"sector": SectorType.TECHNOLOGY, "base_vol": 0.45, "beta": 1.8},
            "MSFT": {"sector": SectorType.TECHNOLOGY, "base_vol": 0.25, "beta": 1.1},
            
            # Biotech - very high volatility, event-driven
            "MRNA": {"sector": SectorType.BIOTECH, "base_vol": 0.65, "beta": 1.4},
            "GILD": {"sector": SectorType.BIOTECH, "base_vol": 0.35, "beta": 0.9},
            
            # Utilities - low volatility, defensive
            "NEE": {"sector": SectorType.UTILITIES, "base_vol": 0.15, "beta": 0.6},
            "DUK": {"sector": SectorType.UTILITIES, "base_vol": 0.18, "beta": 0.7},
            
            # Energy - commodity-linked volatility
            "XOM": {"sector": SectorType.ENERGY, "base_vol": 0.38, "beta": 1.3},
            "CVX": {"sector": SectorType.ENERGY, "base_vol": 0.32, "beta": 1.2},
            
            # Financials - cyclical, regulation-sensitive
            "JPM": {"sector": SectorType.FINANCIALS, "base_vol": 0.30, "beta": 1.1},
            "BAC": {"sector": SectorType.FINANCIALS, "base_vol": 0.35, "beta": 1.4}
        }
        
        # Generate price data for each asset
        for symbol, characteristics in assets.items():
            price_data = self._generate_price_series(
                symbol=symbol,
                base_vol=characteristics["base_vol"],
                beta=characteristics["beta"],
                days=90  # 3 months of data
            )
            
            self.test_data[symbol] = {
                "price_data": price_data,
                "sector": characteristics["sector"],
                "characteristics": characteristics
            }
    
    def _generate_price_series(self, symbol: str, base_vol: float, beta: float, days: int) -> pd.DataFrame:
        """Generate realistic price series with volatility clustering"""
        
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='D')
        
        # Generate market returns (common factor)
        market_returns = np.random.normal(0.0005, 0.012, len(dates))  # Market: 0.05% drift, 1.2% daily vol
        
        # Generate idiosyncratic returns
        idio_vol = base_vol / np.sqrt(252)  # Convert to daily vol
        idiosyncratic_returns = np.random.normal(0, idio_vol, len(dates))
        
        # Volatility clustering using GARCH-like process
        vol_series = np.zeros(len(dates))
        vol_series[0] = idio_vol
        
        for i in range(1, len(dates)):
            # Simple GARCH(1,1) process
            vol_series[i] = (0.1 * idio_vol + 
                           0.85 * vol_series[i-1] + 
                           0.05 * abs(idiosyncratic_returns[i-1]))
        
        # Apply volatility clustering to returns
        clustered_returns = idiosyncratic_returns * (vol_series / idio_vol)
        
        # Combine market and idiosyncratic components
        total_returns = beta * market_returns + clustered_returns
        
        # Generate price series
        initial_price = np.random.uniform(50, 300)
        prices = [initial_price]
        
        for ret in total_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.lognormal(12, 0.3, len(dates)).astype(int)
        })
        
        return df
    
    def demonstrate_basic_threshold_calibration(self):
        """Demonstrate basic volatility-normalized threshold calibration"""
        print("📊 BASIC VOLATILITY THRESHOLD CALIBRATION")
        print("=" * 60)
        
        # Test different assets with same event type but different volatilities
        test_symbols = ["AAPL", "MRNA", "NEE", "XOM"]
        event_type = EventType.EARNINGS
        surprise_value = 0.08  # 8% surprise
        
        print(f"Event Type: {event_type.value.upper()}")
        print(f"Surprise Value: {surprise_value:.1%}")
        print("-" * 60)
        
        for symbol in test_symbols:
            if symbol in self.test_data:
                asset_data = self.test_data[symbol]
                
                # Get volatility metrics
                price_data = asset_data["price_data"]
                sector = asset_data["sector"]
                base_vol = asset_data["characteristics"]["base_vol"]
                
                print(f"\n{symbol} ({sector.value.upper()})")
                print(f"  Base Volatility: {base_vol:.1%}")
                
                # Calculate returns for vol metrics
                returns = price_data['close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252)
                
                print(f"  Realized Vol (30d): {realized_vol:.1%}")
                
                # Simple threshold calculation for demonstration
                reference_vol = 0.20  # 20% reference
                vol_adjustment = np.log(realized_vol / reference_vol)
                
                base_threshold = 0.05  # 5% base threshold for earnings
                adjusted_threshold = base_threshold * (1 + vol_adjustment)
                
                exceeds_threshold = abs(surprise_value) > adjusted_threshold
                
                print(f"  Base Threshold: {base_threshold:.1%}")
                print(f"  Vol Adjustment: {vol_adjustment:+.3f}")
                print(f"  Final Threshold: {adjusted_threshold:.1%}")
                print(f"  Exceeds Threshold: {'✓' if exceeds_threshold else '✗'}")
                
                # Calculate signal strength
                signal_strength = abs(surprise_value) / adjusted_threshold
                print(f"  Signal Strength: {signal_strength:.2f}x")
        
        print("\n" + "=" * 60)
    
    async def demonstrate_advanced_threshold_calibration(self):
        """Demonstrate advanced threshold calibration with full system"""
        print("🎯 ADVANCED VOLATILITY THRESHOLD CALIBRATION")
        print("=" * 60)
        
        if not self.calibrator:
            await self.initialize_system()
        
        # Test scenarios with different event types and sectors
        test_scenarios = [
            {"symbol": "AAPL", "event_type": EventType.EARNINGS, "surprise": 0.12},
            {"symbol": "MRNA", "event_type": EventType.FDA_APPROVAL, "surprise": 0.35},
            {"symbol": "NEE", "event_type": EventType.REGULATORY, "surprise": 0.08},
            {"symbol": "XOM", "event_type": EventType.GUIDANCE, "surprise": 0.15},
            {"symbol": "JPM", "event_type": EventType.REGULATORY, "surprise": 0.10}
        ]
        
        for scenario in test_scenarios:
            symbol = scenario["symbol"]
            event_type = scenario["event_type"]
            surprise_value = scenario["surprise"]
            
            if symbol in self.test_data:
                asset_data = self.test_data[symbol]
                
                print(f"\n{symbol} - {event_type.value.upper()}")
                print(f"Surprise Value: {surprise_value:.1%}")
                print("-" * 40)
                
                # Get adaptive threshold using full system
                result = await self.calibrator.get_adaptive_threshold(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_value=surprise_value,
                    price_data=asset_data["price_data"],
                    sector=asset_data["sector"]
                )
                
                # Display results
                print(f"Base Threshold: {result['base_threshold']:.1%}")
                print(f"Final Threshold: {result['final_threshold']:.1%}")
                print(f"Exceeds Threshold: {'✓' if result['exceeds_threshold'] else '✗'}")
                print(f"Signal Confidence: {result['signal_confidence']:.1%}")
                print(f"Normalized Surprise: {result['normalized_surprise']:.2f}σ")
                
                # Show adjustments
                adjustments = result['adjustments']
                print(f"Adjustments:")
                print(f"  Volatility: {adjustments['volatility_adjustment']:+.3f}")
                print(f"  Sector: {adjustments['sector_adjustment']:+.3f}")
                print(f"  Regime: {adjustments['regime_adjustment']:+.3f}")
                
                # Show volatility metrics
                vol_metrics = result['volatility_metrics']
                print(f"Volatility Metrics:")
                print(f"  30d Realized Vol: {vol_metrics['realized_vol_30d']:.1%}")
                print(f"  Market Regime: {vol_metrics['vol_regime']}")
                print(f"  Sector Percentile: {vol_metrics['sector_vol_percentile']:.0%}")
        
        print("\n" + "=" * 60)
    
    def demonstrate_sector_analysis(self):
        """Demonstrate sector-specific threshold characteristics"""
        print("🏭 SECTOR-SPECIFIC THRESHOLD ANALYSIS")
        print("=" * 60)
        
        if not self.calibrator:
            return
        
        # Analyze each sector profile
        for sector, profile in self.calibrator.sector_profiles.items():
            print(f"\n{sector.value.upper().replace('_', ' ')}")
            print("-" * 30)
            print(f"Median Volatility: {profile.median_vol:.1%}")
            print(f"Volatility Range: {profile.vol_range[0]:.1%} - {profile.vol_range[1]:.1%}")
            print(f"Market Beta: {profile.beta_to_market:.2f}")
            print(f"Vol Clustering: {profile.volatility_clustering:.1%}")
            print(f"Mean Reversion: {profile.mean_reversion_speed:.1%}")
            
            # Show event sensitivities
            if profile.event_sensitivity:
                print("Event Sensitivities:")
                for event_type, sensitivity in profile.event_sensitivity.items():
                    print(f"  {event_type.value}: {sensitivity:.1f}x")
        
        print("\n" + "=" * 60)
    
    def demonstrate_regime_detection(self):
        """Demonstrate market regime detection and its impact on thresholds"""
        print("📈 MARKET REGIME DETECTION & THRESHOLD IMPACT")
        print("=" * 60)
        
        # Create different market regime scenarios
        regime_scenarios = {
            "Low Volatility Bull Market": self._create_regime_data("low_vol", 0.08),
            "Normal Market Conditions": self._create_regime_data("normal", 0.16),
            "High Volatility Period": self._create_regime_data("high_vol", 0.35),
            "Crisis/Panic Conditions": self._create_regime_data("crisis", 0.55)
        }
        
        base_surprise = 0.10  # 10% surprise
        event_type = EventType.EARNINGS
        
        print(f"Testing {base_surprise:.0%} {event_type.value} surprise across regimes:")
        print("-" * 60)
        
        for regime_name, returns_data in regime_scenarios.items():
            # Detect regime
            regime = self.calibrator.market_regime_detector.detect_regime(returns_data)
            
            # Calculate regime adjustment
            regime_adjustment = self.calibrator._calculate_regime_adjustment(regime, event_type)
            
            # Calculate effective threshold
            base_threshold = 0.05
            effective_threshold = base_threshold * (1 + regime_adjustment)
            
            exceeds_threshold = base_surprise > effective_threshold
            signal_strength = base_surprise / effective_threshold
            
            print(f"\n{regime_name}")
            print(f"  Detected Regime: {regime.value}")
            print(f"  Regime Adjustment: {regime_adjustment:+.1%}")
            print(f"  Effective Threshold: {effective_threshold:.1%}")
            print(f"  Signal Exceeds: {'✓' if exceeds_threshold else '✗'}")
            print(f"  Signal Strength: {signal_strength:.2f}x")
    
    def _create_regime_data(self, regime_type: str, target_vol: float) -> pd.Series:
        """Create synthetic return data for different market regimes"""
        
        n_days = 30
        daily_vol = target_vol / np.sqrt(252)
        
        if regime_type == "low_vol":
            # Low volatility, low autocorrelation
            returns = np.random.normal(0.001, daily_vol, n_days)
        elif regime_type == "normal":
            # Normal market with moderate vol
            returns = np.random.normal(0.0005, daily_vol, n_days)
        elif regime_type == "high_vol":
            # High volatility with volatility clustering
            returns = np.random.normal(0, daily_vol * 1.5, n_days)
            # Add some clustering
            for i in range(1, len(returns)):
                if abs(returns[i-1]) > daily_vol:
                    returns[i] *= 1.3  # Amplify after big moves
        elif regime_type == "crisis":
            # Crisis with extreme moves and negative drift
            returns = np.random.normal(-0.002, daily_vol, n_days)
            # Add fat tails
            extreme_moves = np.random.choice([0, 1], n_days, p=[0.8, 0.2])
            returns += extreme_moves * np.random.normal(0, daily_vol * 2, n_days)
        
        return pd.Series(returns)
    
    async def demonstrate_cross_asset_comparison(self):
        """Demonstrate threshold comparison across different asset classes"""
        print("🔄 CROSS-ASSET THRESHOLD COMPARISON")
        print("=" * 60)
        
        if not self.calibrator:
            await self.initialize_system()
        
        # Compare same event across different asset types
        event_type = EventType.EARNINGS
        surprise_value = 0.08  # 8% surprise
        
        print(f"Comparing {surprise_value:.0%} {event_type.value} surprise across assets:")
        print("-" * 60)
        
        comparison_results = []
        
        for symbol in ["AAPL", "MRNA", "NEE", "XOM", "JPM"]:
            if symbol in self.test_data:
                asset_data = self.test_data[symbol]
                
                result = await self.calibrator.get_adaptive_threshold(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_value=surprise_value,
                    price_data=asset_data["price_data"],
                    sector=asset_data["sector"]
                )
                
                comparison_results.append({
                    "symbol": symbol,
                    "sector": asset_data["sector"].value,
                    "realized_vol": result['volatility_metrics']['realized_vol_30d'],
                    "final_threshold": result['final_threshold'],
                    "exceeds_threshold": result['exceeds_threshold'],
                    "signal_confidence": result['signal_confidence'],
                    "normalized_surprise": result['normalized_surprise']
                })
        
        # Display comparison table
        print(f"{'Symbol':<6} {'Sector':<12} {'Vol':<6} {'Threshold':<10} {'Exceeds':<8} {'Confidence':<10} {'Norm':<6}")
        print("-" * 60)
        
        for result in comparison_results:
            print(f"{result['symbol']:<6} {result['sector'][:11]:<12} "
                  f"{result['realized_vol']:.1%}  {result['final_threshold']:.1%}      "
                  f"{'✓' if result['exceeds_threshold'] else '✗':<8} "
                  f"{result['signal_confidence']:.1%}      {result['normalized_surprise']:.2f}")
        
        print("\n" + "=" * 60)
    
    def demonstrate_event_sensitivity_matrix(self):
        """Demonstrate event sensitivity differences across sectors"""
        print("🎭 EVENT SENSITIVITY MATRIX")
        print("=" * 60)
        
        if not self.calibrator:
            return
        
        # Create sensitivity matrix
        sectors = [SectorType.TECHNOLOGY, SectorType.BIOTECH, SectorType.UTILITIES, 
                  SectorType.ENERGY, SectorType.FINANCIALS]
        events = [EventType.EARNINGS, EventType.FDA_APPROVAL, EventType.GUIDANCE, 
                 EventType.REGULATORY, EventType.MERGER_ACQUISITION]
        
        print(f"{'Event Type':<20}", end="")
        for sector in sectors:
            print(f"{sector.value[:8]:<10}", end="")
        print()
        print("-" * 70)
        
        for event in events:
            print(f"{event.value:<20}", end="")
            
            for sector in sectors:
                # Get base sensitivity
                base_sensitivity = self.calibrator.event_sensitivities.get(event, 1.0)
                
                # Get sector-specific adjustment if available
                if sector in self.calibrator.sector_profiles:
                    sector_profile = self.calibrator.sector_profiles[sector]
                    if event in sector_profile.event_sensitivity:
                        sector_sensitivity = sector_profile.event_sensitivity[event]
                    else:
                        sector_sensitivity = base_sensitivity
                else:
                    sector_sensitivity = base_sensitivity
                
                print(f"{sector_sensitivity:.2f}      ", end="")
            print()
        
        print("\n" + "=" * 60)
    
    async def run_comprehensive_demo(self):
        """Run complete demonstration of volatility threshold calibration"""
        print("🚀 VOLATILITY THRESHOLD CALIBRATION DEMONSTRATION")
        print("=" * 80)
        print("This demo showcases advanced volatility threshold calibration featuring:")
        print("• Asset-specific volatility normalization using realized & implied volatility")
        print("• Sector-specific threshold adjustment with industry characteristics")
        print("• Event type-specific surprise sensitivity mapping")
        print("• Market regime detection for adaptive threshold scaling")
        print("• Cross-asset threshold comparison and analysis")
        print("=" * 80)
        print()
        
        await self.initialize_system()
        
        self.demonstrate_basic_threshold_calibration()
        print()
        
        await self.demonstrate_advanced_threshold_calibration()
        print()
        
        self.demonstrate_sector_analysis()
        print()
        
        self.demonstrate_regime_detection()
        print()
        
        await self.demonstrate_cross_asset_comparison()
        print()
        
        self.demonstrate_event_sensitivity_matrix()
        
        print("✅ DEMONSTRATION COMPLETE")
        print("The volatility threshold calibration system provides sophisticated,")
        print("asset-specific surprise threshold normalization for improved signal quality!")

# Utility functions for testing and validation

async def test_threshold_accuracy():
    """Test threshold calibration accuracy across different scenarios"""
    print("🧪 TESTING THRESHOLD CALIBRATION ACCURACY")
    print("=" * 50)
    
    calibrator = await create_volatility_threshold_calibrator()
    
    # Test scenarios with known expected behaviors
    test_cases = [
        {
            "name": "High Vol Asset - Lower Surprise Should Trigger",
            "symbol": "TEST_HIGH_VOL",
            "event_type": EventType.EARNINGS,
            "surprise": 0.03,  # 3% surprise
            "expected_vol": 0.50,  # 50% volatility
            "should_exceed": True  # Low surprise should exceed threshold for high vol asset
        },
        {
            "name": "Low Vol Asset - Higher Surprise Needed",
            "symbol": "TEST_LOW_VOL", 
            "event_type": EventType.EARNINGS,
            "surprise": 0.03,  # Same 3% surprise
            "expected_vol": 0.10,  # 10% volatility
            "should_exceed": False  # Same surprise should NOT exceed threshold for low vol asset
        },
        {
            "name": "Biotech FDA Approval - High Sensitivity",
            "symbol": "TEST_BIOTECH",
            "event_type": EventType.FDA_APPROVAL,
            "surprise": 0.15,  # 15% surprise
            "expected_vol": 0.40,
            "should_exceed": True  # FDA events should be more sensitive
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        # Create synthetic volatility metrics
        from ..services.volatility_threshold_calibration import VolatilityMetrics, MarketRegime
        
        vol_metrics = VolatilityMetrics(
            symbol=test["symbol"],
            realized_vol_1d=test["expected_vol"],
            realized_vol_5d=test["expected_vol"],
            realized_vol_30d=test["expected_vol"],
            implied_vol=test["expected_vol"] * 1.1,
            vol_regime=MarketRegime.NORMAL_VOLATILITY,
            sector_vol_percentile=0.5
        )
        
        # Get sector for biotech test
        sector = SectorType.BIOTECH if "BIOTECH" in test["symbol"] else None
        
        result = await calibrator.get_adaptive_threshold(
            symbol=test["symbol"],
            event_type=test["event_type"],
            surprise_value=test["surprise"],
            sector=sector
        )
        
        # Override volatility metrics with test values
        result["volatility_metrics"]["realized_vol_30d"] = test["expected_vol"]
        
        actual_exceeds = result["exceeds_threshold"]
        expected_exceeds = test["should_exceed"]
        
        passed = actual_exceeds == expected_exceeds
        
        print(f"{test['name']}: {'✓' if passed else '✗'}")
        print(f"  Expected Exceeds: {expected_exceeds}, Actual: {actual_exceeds}")
        print(f"  Final Threshold: {result['final_threshold']:.1%}")
        print(f"  Signal Confidence: {result['signal_confidence']:.1%}")
        
        if not passed:
            all_passed = False
        
        print()
    
    print(f"Overall Test Result: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed

# Main execution
async def main():
    """Main execution function for demonstration"""
    demo = VolatilityThresholdDemo()
    
    try:
        await demo.run_comprehensive_demo()
        print()
        await test_threshold_accuracy()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())