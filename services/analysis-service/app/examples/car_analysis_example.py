"""
CAR Analysis Example and Testing Framework

This module demonstrates how to use the CAR analysis system for event-driven
trading strategy development with empirical regime identification.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict

from ..services.car_analysis import (
    CARAnalyzer, EventRegimeIdentifier, EventData, EventType, Sector,
    create_car_analyzer, create_regime_identifier
)
from ..services.market_microstructure import (
    create_microstructure_analyzer, create_event_microstructure_integrator,
    OrderFlowData
)

logger = logging.getLogger(__name__)

class CARAnalysisExampleRunner:
    """Example runner for CAR analysis demonstrations"""
    
    def __init__(self):
        self.car_analyzer = None
        self.regime_identifier = None
        
    async def initialize(self):
        """Initialize analysis components"""
        self.car_analyzer = await create_car_analyzer()
        self.regime_identifier = await create_regime_identifier()
        
    def generate_sample_events(self, n_events: int = 100) -> List[EventData]:
        """Generate sample event data for testing"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        event_types = list(EventType)
        sectors = list(Sector)
        
        events = []
        base_date = datetime.now() - timedelta(days=730)  # 2 years ago
        
        for i in range(n_events):
            event_date = base_date + timedelta(days=np.random.randint(0, 700))
            
            event = EventData(
                symbol=np.random.choice(symbols),
                event_type=np.random.choice(event_types),
                event_date=event_date,
                sector=np.random.choice(sectors),
                event_magnitude=np.random.normal(0, 1),  # Standardized magnitude
                pre_event_volume=np.random.lognormal(10, 1),  # Log-normal volume
                event_description=f"Sample {np.random.choice(event_types).value} event",
                metadata={
                    "confidence": np.random.uniform(0.5, 1.0),
                    "source": "simulation",
                    "analyst_rating": np.random.choice(['buy', 'hold', 'sell'])
                }
            )
            events.append(event)
        
        return events
    
    def generate_sample_price_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate sample price data for testing"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        price_data = []
        
        for symbol in symbols:
            # Simulate price with random walk + trend
            initial_price = np.random.uniform(50, 300)
            prices = [initial_price]
            
            for _ in range(len(date_range) - 1):
                # Random walk with slight positive bias
                change = np.random.normal(0.001, 0.02)  # 0.1% drift, 2% daily vol
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1.0))  # Ensure positive prices
            
            # Calculate returns
            returns = [0.0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            for i, (date, price, ret) in enumerate(zip(date_range, prices, returns)):
                price_data.append({
                    'symbol': symbol,
                    'date': date,
                    'price': price,
                    'return': ret,
                    'volume': np.random.lognormal(12, 0.5)  # Random volume
                })
        
        return pd.DataFrame(price_data)
    
    def generate_sample_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate sample market index data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate market index with lower volatility
        initial_value = 3000
        values = [initial_value]
        
        for _ in range(len(date_range) - 1):
            change = np.random.normal(0.0005, 0.015)  # Lower vol for market
            new_value = values[-1] * (1 + change)
            values.append(max(new_value, 100))
        
        returns = [0.0] + [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        market_data = []
        for date, value, ret in zip(date_range, values, returns):
            market_data.append({
                'date': date,
                'index_value': value,
                'return': ret
            })
        
        return pd.DataFrame(market_data)
    
    async def run_earnings_car_analysis(self) -> Dict:
        """Run CAR analysis specifically for earnings events"""
        logger.info("Running earnings CAR analysis example")
        
        # Generate sample data
        events = self.generate_sample_events(200)
        earnings_events = [e for e in events if e.event_type == EventType.EARNINGS]
        
        if len(earnings_events) < 50:
            # Add more earnings events if not enough
            additional_events = []
            for _ in range(60 - len(earnings_events)):
                event_date = datetime.now() - timedelta(days=np.random.randint(30, 700))
                event = EventData(
                    symbol=np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN']),
                    event_type=EventType.EARNINGS,
                    event_date=event_date,
                    sector=np.random.choice([Sector.TECHNOLOGY, Sector.HEALTHCARE]),
                    event_magnitude=np.random.normal(0, 1.5),  # Earnings surprise
                    metadata={"eps_surprise": np.random.normal(0, 0.1)}
                )
                additional_events.append(event)
            earnings_events.extend(additional_events)
        
        # Generate corresponding price and market data
        symbols = list(set(e.symbol for e in earnings_events))
        start_date = min(e.event_date for e in earnings_events) - timedelta(days=300)
        end_date = max(e.event_date for e in earnings_events) + timedelta(days=100)
        
        price_data = self.generate_sample_price_data(symbols, start_date, end_date)
        market_data = self.generate_sample_market_data(start_date, end_date)
        
        try:
            # Perform CAR analysis for earnings in technology sector
            car_results = await self.car_analyzer.calculate_car(
                events=earnings_events,
                price_data=price_data,
                market_data=market_data,
                event_type=EventType.EARNINGS,
                sector=Sector.TECHNOLOGY
            )
            
            return {
                "analysis_type": "earnings_technology_car",
                "events_analyzed": len([e for e in earnings_events if e.sector == Sector.TECHNOLOGY]),
                "optimal_holding_period": car_results.optimal_holding_period,
                "expected_return": car_results.expected_return,
                "return_volatility": car_results.return_volatility,
                "sharpe_ratio": car_results.sharpe_ratio,
                "hit_rate": car_results.hit_rate,
                "skewness": car_results.skewness,
                "kurtosis": car_results.kurtosis,
                "statistical_significance": car_results.statistical_significance,
                "regime_parameters": car_results.regime_parameters,
                "car_timeline": car_results.car_values.tolist()[:10],  # First 10 days
                "interpretation": self._interpret_car_results(car_results)
            }
            
        except Exception as e:
            logger.error(f"CAR analysis failed: {e}")
            return {"error": str(e), "analysis_type": "earnings_technology_car"}
    
    async def run_comprehensive_regime_analysis(self) -> Dict:
        """Run comprehensive regime analysis across multiple event types"""
        logger.info("Running comprehensive regime analysis")
        
        # Generate larger dataset
        events = self.generate_sample_events(500)
        symbols = list(set(e.symbol for e in events))
        
        start_date = min(e.event_date for e in events) - timedelta(days=300)
        end_date = max(e.event_date for e in events) + timedelta(days=100)
        
        price_data = self.generate_sample_price_data(symbols, start_date, end_date)
        market_data = self.generate_sample_market_data(start_date, end_date)
        
        # Identify regimes across all event-sector combinations
        regimes = await self.regime_identifier.identify_regimes_by_event_sector(
            events, price_data, market_data
        )
        
        # Update cache
        await self.regime_identifier.update_regime_cache(regimes)
        
        # Format results
        regime_summary = {}
        for (event_type, sector), results in regimes.items():
            key = f"{event_type.value}_{sector.value if sector else 'all'}"
            regime_summary[key] = {
                "optimal_holding_period": results.optimal_holding_period,
                "expected_return": results.expected_return,
                "volatility": results.return_volatility,
                "sharpe_ratio": results.sharpe_ratio,
                "hit_rate": results.hit_rate,
                "regime_params": results.regime_parameters,
                "trading_recommendation": self._generate_trading_recommendation(results)
            }
        
        return {
            "total_regimes_identified": len(regimes),
            "regime_breakdown": regime_summary,
            "top_performing_regimes": self._identify_top_regimes(regimes),
            "risk_adjusted_rankings": self._rank_by_risk_adjusted_returns(regimes)
        }
    
    async def run_microstructure_integration_example(self) -> Dict:
        """Demonstrate integration with market microstructure analysis"""
        logger.info("Running microstructure integration example")
        
        integrator = await create_event_microstructure_integrator()
        
        # Generate sample order flow data
        event_date = datetime.now() - timedelta(days=30)
        
        pre_event_data = self._generate_sample_order_flow(
            symbol="AAPL", 
            start_time=event_date - timedelta(hours=2),
            end_time=event_date,
            n_snapshots=100
        )
        
        post_event_data = self._generate_sample_order_flow(
            symbol="AAPL",
            start_time=event_date,
            end_time=event_date + timedelta(hours=2),
            n_snapshots=100,
            volatility_multiplier=1.5  # Higher volatility post-event
        )
        
        # Generate sample trade data
        trade_data = pd.DataFrame([
            {
                "timestamp": event_date + timedelta(minutes=i),
                "price": 150 + np.random.normal(0, 2),
                "volume": np.random.lognormal(6, 1),
                "direction": np.random.choice([-1, 1])
            }
            for i in range(-60, 60, 5)  # Every 5 minutes for 2 hours around event
        ])
        
        # Analyze microstructure impact
        impact_analysis = await integrator.analyze_event_microstructure_impact(
            event_type=EventType.EARNINGS,
            sector=Sector.TECHNOLOGY,
            pre_event_data=pre_event_data,
            post_event_data=post_event_data,
            event_trade_data=trade_data
        )
        
        return impact_analysis
    
    def _generate_sample_order_flow(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime, 
        n_snapshots: int,
        volatility_multiplier: float = 1.0
    ) -> List[OrderFlowData]:
        """Generate sample order flow data"""
        
        time_delta = (end_time - start_time).total_seconds() / n_snapshots
        order_flow_data = []
        
        base_price = 150.0  # Starting price
        current_price = base_price
        
        for i in range(n_snapshots):
            timestamp = start_time + timedelta(seconds=i * time_delta)
            
            # Simulate price movement
            price_change = np.random.normal(0, 0.1 * volatility_multiplier)
            current_price = max(current_price + price_change, 50.0)
            
            # Simulate spread
            spread = np.random.uniform(0.01, 0.05) * volatility_multiplier
            
            bid_price = current_price - spread / 2
            ask_price = current_price + spread / 2
            
            # Simulate order sizes
            bid_size = np.random.lognormal(6, 1)
            ask_size = np.random.lognormal(6, 1)
            
            # Simulate trade
            volume = np.random.lognormal(5, 1)
            trade_direction = np.random.choice([-1, 0, 1])
            
            order_flow = OrderFlowData(
                timestamp=timestamp,
                symbol=symbol,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                last_price=current_price,
                volume=volume,
                trade_direction=trade_direction
            )
            
            order_flow_data.append(order_flow)
        
        return order_flow_data
    
    def _interpret_car_results(self, car_results) -> Dict[str, str]:
        """Interpret CAR analysis results for practical trading"""
        interpretation = {}
        
        # Return interpretation
        if car_results.expected_return > 0.02:
            interpretation["return_assessment"] = "Strong positive alpha opportunity"
        elif car_results.expected_return > 0.005:
            interpretation["return_assessment"] = "Moderate alpha opportunity"
        elif car_results.expected_return > -0.005:
            interpretation["return_assessment"] = "Neutral/minimal opportunity"
        else:
            interpretation["return_assessment"] = "Negative expected return - avoid"
        
        # Sharpe ratio interpretation
        if car_results.sharpe_ratio > 1.0:
            interpretation["risk_adjusted"] = "Excellent risk-adjusted returns"
        elif car_results.sharpe_ratio > 0.5:
            interpretation["risk_adjusted"] = "Good risk-adjusted returns"
        else:
            interpretation["risk_adjusted"] = "Poor risk-adjusted returns"
        
        # Hit rate interpretation
        if car_results.hit_rate > 0.65:
            interpretation["consistency"] = "High consistency strategy"
        elif car_results.hit_rate > 0.55:
            interpretation["consistency"] = "Moderate consistency"
        else:
            interpretation["consistency"] = "Low consistency - high risk"
        
        # Statistical significance
        if car_results.statistical_significance.get("is_significant", False):
            interpretation["statistical_validity"] = "Statistically significant results"
        else:
            interpretation["statistical_validity"] = "Results not statistically significant"
        
        return interpretation
    
    def _generate_trading_recommendation(self, car_results) -> Dict[str, Any]:
        """Generate actionable trading recommendations"""
        params = car_results.regime_parameters
        
        return {
            "position_sizing": {
                "kelly_fraction": params.get("position_size_kelly", 0.1),
                "max_position": min(params.get("position_size_kelly", 0.1) * 2, 0.25),
                "volatility_scaling": True
            },
            "entry_strategy": {
                "confidence_threshold": params.get("confidence_threshold", 0.7),
                "pre_event_positioning": car_results.optimal_holding_period > 5,
                "timing": "gradual" if car_results.return_volatility > 0.1 else "aggressive"
            },
            "exit_strategy": {
                "profit_target": params.get("profit_target", 0.05),
                "stop_loss": params.get("stop_loss_threshold", -0.02),
                "holding_period": f"{car_results.optimal_holding_period} days",
                "trailing_stop": car_results.return_volatility > 0.15
            },
            "risk_management": {
                "max_drawdown_tolerance": 0.05,
                "correlation_limit": 0.3,  # Max correlation with other positions
                "sector_concentration_limit": 0.4
            }
        }
    
    def _identify_top_regimes(self, regimes: Dict) -> List[Dict]:
        """Identify top performing regimes"""
        regime_rankings = []
        
        for (event_type, sector), results in regimes.items():
            score = results.sharpe_ratio * results.hit_rate * (1 + results.expected_return)
            
            regime_rankings.append({
                "event_type": event_type.value,
                "sector": sector.value if sector else "all",
                "performance_score": score,
                "expected_return": results.expected_return,
                "sharpe_ratio": results.sharpe_ratio,
                "hit_rate": results.hit_rate
            })
        
        return sorted(regime_rankings, key=lambda x: x["performance_score"], reverse=True)[:5]
    
    def _rank_by_risk_adjusted_returns(self, regimes: Dict) -> List[Dict]:
        """Rank regimes by risk-adjusted returns"""
        risk_rankings = []
        
        for (event_type, sector), results in regimes.items():
            # Modified Sharpe with downside deviation consideration
            downside_penalty = 1.0 if results.skewness >= -0.5 else 0.8
            risk_adjusted_score = results.sharpe_ratio * downside_penalty * results.hit_rate
            
            risk_rankings.append({
                "event_type": event_type.value,
                "sector": sector.value if sector else "all",
                "risk_adjusted_score": risk_adjusted_score,
                "sharpe_ratio": results.sharpe_ratio,
                "skewness": results.skewness,
                "max_drawdown": results.regime_parameters.get("max_drawdown", 0.1)
            })
        
        return sorted(risk_rankings, key=lambda x: x["risk_adjusted_score"], reverse=True)

# Example usage and testing
async def run_car_analysis_examples():
    """Run all CAR analysis examples"""
    runner = CARAnalysisExampleRunner()
    await runner.initialize()
    
    print("=== CAR Analysis Examples ===")
    
    # Run earnings analysis
    print("\n1. Running Earnings CAR Analysis...")
    earnings_results = await runner.run_earnings_car_analysis()
    print(f"Results: {earnings_results}")
    
    # Run comprehensive regime analysis
    print("\n2. Running Comprehensive Regime Analysis...")
    regime_results = await runner.run_comprehensive_regime_analysis()
    print(f"Total regimes identified: {regime_results['total_regimes_identified']}")
    
    # Run microstructure integration
    print("\n3. Running Microstructure Integration Example...")
    microstructure_results = await runner.run_microstructure_integration_example()
    print(f"Microstructure impact: {microstructure_results}")
    
    print("\n=== Analysis Complete ===")
    return {
        "earnings_analysis": earnings_results,
        "regime_analysis": regime_results,
        "microstructure_analysis": microstructure_results
    }

if __name__ == "__main__":
    # Run examples
    asyncio.run(run_car_analysis_examples())