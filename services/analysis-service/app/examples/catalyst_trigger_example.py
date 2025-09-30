"""
Catalyst Trigger Engine Example and Demonstration

This example demonstrates the comprehensive catalyst detection capabilities
combining event occurrence, surprise thresholds, and sentiment spike filters.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from ..services.catalyst_trigger_engine import (
    CatalystTriggerEngine,
    CatalystType,
    SignalStrength,
    EventSignal,
    SentimentSignal,
    TechnicalSignal
)

class CatalystTriggerDemo:
    """Comprehensive demonstration of catalyst trigger detection capabilities"""
    
    def __init__(self):
        self.engine = CatalystTriggerEngine()
        
    async def run_all_demos(self):
        """Run all catalyst trigger demonstrations"""
        print("=== Catalyst Trigger Engine Demonstration ===\n")
        
        print("1. Running Strong Catalyst Detection Demo...")
        await self.demo_strong_catalyst_detection()
        print("\n" + "="*60 + "\n")
        
        print("2. Running Weak Signal Filtering Demo...")
        await self.demo_weak_signal_filtering()
        print("\n" + "="*60 + "\n")
        
        print("3. Running Multi-Factor Alignment Demo...")
        await self.demo_multi_factor_alignment()
        print("\n" + "="*60 + "\n")
        
        print("4. Running Event-Driven Catalyst Demo...")
        await self.demo_event_driven_catalyst()
        print("\n" + "="*60 + "\n")
        
        print("5. Running Sentiment-Driven Catalyst Demo...")
        await self.demo_sentiment_driven_catalyst()
        print("\n" + "="*60 + "\n")
        
        print("6. Running Technical Breakout Catalyst Demo...")
        await self.demo_technical_breakout_catalyst()
        print("\n" + "="*60 + "\n")
        
        print("7. Running Signal Conflict Resolution Demo...")
        await self.demo_signal_conflict_resolution()
        print("\n" + "="*60 + "\n")
        
        print("All catalyst trigger demonstrations completed successfully!")
        
    async def demo_strong_catalyst_detection(self):
        """Demonstrate detection of strong catalyst with aligned signals"""
        print("Strong Catalyst Detection: Biotech FDA Approval")
        print("-" * 50)
        
        # Strong biotech FDA approval event
        event_data = {
            'event_type': 'fda_approval',
            'surprise_value': 0.85,  # Very strong surprise
            'announcement_time': datetime.utcnow(),
            'event_description': 'Unexpected FDA approval for breakthrough therapy',
            'impact_score': 9.5
        }
        
        # Strong positive sentiment spike
        sentiment_data = self.create_sentiment_spike_data(
            base_sentiment=0.8,
            spike_magnitude=0.9,
            volume_multiplier=15,
            duration_hours=2
        )
        
        # Strong technical confirmation
        technical_data = {
            'price': 125.50,
            'volume': 8500000,  # 15x normal volume
            'price_change_1h': 0.22,  # 22% price increase
            'price_change_4h': 0.35,  # 35% price increase
            'volume_ratio': 15.2,
            'unusual_activity': True
        }
        
        catalyst = await self.engine.detect_catalyst_trigger(
            symbol="BIOTECH_FDA",
            event_data=event_data,
            sentiment_data=sentiment_data,
            technical_data=technical_data
        )
        
        self.print_catalyst_result(catalyst, "Strong multi-factor catalyst")
        
    async def demo_weak_signal_filtering(self):
        """Demonstrate filtering of weak signals that don't meet thresholds"""
        print("Weak Signal Filtering: Minor Earnings Beat")
        print("-" * 50)
        
        # Weak earnings event
        event_data = {
            'event_type': 'earnings',
            'surprise_value': 0.03,  # Minor 3% surprise
            'announcement_time': datetime.utcnow(),
            'event_description': 'Slight earnings beat',
            'impact_score': 2.1
        }
        
        # Weak sentiment response
        sentiment_data = self.create_sentiment_spike_data(
            base_sentiment=0.1,
            spike_magnitude=0.2,
            volume_multiplier=1.5,
            duration_hours=1
        )
        
        # Normal technical activity
        technical_data = {
            'price': 85.20,
            'volume': 1200000,  # Normal volume
            'price_change_1h': 0.015,  # 1.5% price increase
            'price_change_4h': 0.025,  # 2.5% price increase
            'volume_ratio': 1.3,
            'unusual_activity': False
        }
        
        catalyst = await self.engine.detect_catalyst_trigger(
            symbol="WEAK_SIGNAL",
            event_data=event_data,
            sentiment_data=sentiment_data,
            technical_data=technical_data
        )
        
        self.print_catalyst_result(catalyst, "Weak signal (should be filtered)")
        
    async def demo_multi_factor_alignment(self):
        """Demonstrate multi-factor signal alignment scoring"""
        print("Multi-Factor Signal Alignment: M&A Announcement")
        print("-" * 50)
        
        # Strong M&A event
        event_data = {
            'event_type': 'merger_acquisition',
            'surprise_value': 0.95,  # Completely unexpected
            'announcement_time': datetime.utcnow() - timedelta(minutes=30),
            'event_description': 'Surprise acquisition by tech giant',
            'impact_score': 9.8
        }
        
        # Aligned sentiment explosion
        sentiment_data = self.create_aligned_sentiment_data()
        
        # Strong technical confirmation
        technical_data = {
            'price': 145.75,
            'volume': 25000000,  # Massive volume
            'price_change_1h': 0.45,  # 45% price jump
            'price_change_4h': 0.48,  # Sustained momentum
            'volume_ratio': 25.8,
            'unusual_activity': True
        }
        
        catalyst = await self.engine.detect_catalyst_trigger(
            symbol="ACQUISITION_TARGET",
            event_data=event_data,
            sentiment_data=sentiment_data,
            technical_data=technical_data
        )
        
        self.print_catalyst_result(catalyst, "Multi-factor aligned catalyst")
        
    async def demo_event_driven_catalyst(self):
        """Demonstrate event-driven catalyst with volatility normalization"""
        print("Event-Driven Catalyst: Regulatory Approval")
        print("-" * 50)
        
        # Regulatory approval event
        event_data = {
            'event_type': 'regulatory_approval',
            'surprise_value': 0.65,
            'announcement_time': datetime.utcnow() - timedelta(hours=1),
            'event_description': 'Unexpected regulatory clearance',
            'impact_score': 8.2
        }
        
        # Moderate sentiment response
        sentiment_data = self.create_sentiment_spike_data(
            base_sentiment=0.4,
            spike_magnitude=0.7,
            volume_multiplier=8,
            duration_hours=3
        )
        
        # Strong technical response
        technical_data = {
            'price': 67.90,
            'volume': 5500000,
            'price_change_1h': 0.18,  # 18% increase
            'price_change_4h': 0.25,  # 25% increase
            'volume_ratio': 8.5,
            'unusual_activity': True
        }
        
        catalyst = await self.engine.detect_catalyst_trigger(
            symbol="REGULATORY_PLAY",
            event_data=event_data,
            sentiment_data=sentiment_data,
            technical_data=technical_data
        )
        
        self.print_catalyst_result(catalyst, "Event-driven catalyst")
        
    async def demo_sentiment_driven_catalyst(self):
        """Demonstrate sentiment-driven catalyst detection"""
        print("Sentiment-Driven Catalyst: Viral Social Media Event")
        print("-" * 50)
        
        # No major corporate event
        event_data = None
        
        # Massive sentiment explosion
        sentiment_data = self.create_viral_sentiment_data()
        
        # Technical response to sentiment
        technical_data = {
            'price': 88.45,
            'volume': 12000000,  # High volume from retail interest
            'price_change_1h': 0.12,  # 12% increase
            'price_change_4h': 0.28,  # 28% increase building momentum
            'volume_ratio': 12.5,
            'unusual_activity': True
        }
        
        catalyst = await self.engine.detect_catalyst_trigger(
            symbol="VIRAL_STOCK",
            event_data=event_data,
            sentiment_data=sentiment_data,
            technical_data=technical_data
        )
        
        self.print_catalyst_result(catalyst, "Sentiment-driven catalyst")
        
    async def demo_technical_breakout_catalyst(self):
        """Demonstrate technical breakout catalyst"""
        print("Technical Breakout Catalyst: Volume Breakout")
        print("-" * 50)
        
        # Minor event
        event_data = {
            'event_type': 'analyst_upgrade',
            'surprise_value': 0.25,
            'announcement_time': datetime.utcnow() - timedelta(hours=4),
            'event_description': 'Analyst price target increase',
            'impact_score': 4.5
        }
        
        # Moderate sentiment
        sentiment_data = self.create_sentiment_spike_data(
            base_sentiment=0.3,
            spike_magnitude=0.5,
            volume_multiplier=3,
            duration_hours=2
        )
        
        # Strong technical breakout
        technical_data = {
            'price': 112.85,
            'volume': 18000000,  # Massive technical volume
            'price_change_1h': 0.08,  # 8% increase
            'price_change_4h': 0.15,  # 15% total move
            'volume_ratio': 20.3,  # Extreme volume ratio
            'unusual_activity': True
        }
        
        catalyst = await self.engine.detect_catalyst_trigger(
            symbol="TECHNICAL_BREAKOUT",
            event_data=event_data,
            sentiment_data=sentiment_data,
            technical_data=technical_data
        )
        
        self.print_catalyst_result(catalyst, "Technical breakout catalyst")
        
    async def demo_signal_conflict_resolution(self):
        """Demonstrate resolution of conflicting signals"""
        print("Signal Conflict Resolution: Mixed Signals")
        print("-" * 50)
        
        # Strong positive event
        event_data = {
            'event_type': 'product_launch',
            'surprise_value': 0.75,
            'announcement_time': datetime.utcnow() - timedelta(minutes=45),
            'event_description': 'Revolutionary product launch',
            'impact_score': 8.8
        }
        
        # Negative sentiment (skepticism about claims)
        sentiment_data = self.create_negative_sentiment_data()
        
        # Weak technical response
        technical_data = {
            'price': 98.20,
            'volume': 2800000,  # Moderate volume
            'price_change_1h': 0.02,  # Only 2% increase
            'price_change_4h': -0.01,  # Actually down over 4h
            'volume_ratio': 2.5,
            'unusual_activity': False
        }
        
        catalyst = await self.engine.detect_catalyst_trigger(
            symbol="CONFLICTED_SIGNALS",
            event_data=event_data,
            sentiment_data=sentiment_data,
            technical_data=technical_data
        )
        
        self.print_catalyst_result(catalyst, "Conflicting signals resolution")
        
    def create_sentiment_spike_data(self, base_sentiment: float, spike_magnitude: float, 
                                  volume_multiplier: float, duration_hours: int) -> pd.DataFrame:
        """Create sentiment spike data for testing"""
        now = datetime.utcnow()
        data = []
        
        # Build sentiment spike over time
        for i in range(duration_hours * 12):  # 5-minute intervals
            timestamp = now - timedelta(minutes=i*5)
            
            # Create spike pattern
            if i < 6:  # First 30 minutes - peak spike
                sentiment = min(1.0, base_sentiment + spike_magnitude)
                volume = int(1000 * volume_multiplier * (1 - i/12))
            elif i < 24:  # Next 1.5 hours - decline
                sentiment = base_sentiment + spike_magnitude * (1 - (i-6)/18)
                volume = int(1000 * volume_multiplier * 0.3)
            else:  # Baseline
                sentiment = base_sentiment
                volume = int(1000)
                
            data.append({
                'timestamp': timestamp,
                'sentiment_score': sentiment,
                'volume': volume,
                'source': f'social_media_{i%3}'
            })
            
        return pd.DataFrame(data)
        
    def create_aligned_sentiment_data(self) -> pd.DataFrame:
        """Create perfectly aligned multi-source sentiment spike"""
        now = datetime.utcnow()
        data = []
        sources = ['twitter', 'reddit', 'news', 'analyst', 'insider']
        
        for i in range(24):  # 2 hours of data
            timestamp = now - timedelta(minutes=i*5)
            
            for source in sources:
                # All sources showing strong positive sentiment
                sentiment = 0.85 + np.random.normal(0, 0.05)
                sentiment = max(0, min(1, sentiment))
                
                volume = int(np.random.normal(2000, 500))
                
                data.append({
                    'timestamp': timestamp,
                    'sentiment_score': sentiment,
                    'volume': volume,
                    'source': source
                })
                
        return pd.DataFrame(data)
        
    def create_viral_sentiment_data(self) -> pd.DataFrame:
        """Create viral sentiment explosion pattern"""
        now = datetime.utcnow()
        data = []
        
        # Viral explosion pattern - exponential growth then plateau
        for i in range(48):  # 4 hours
            timestamp = now - timedelta(minutes=i*5)
            
            # Exponential growth in first hour, then sustained high
            if i < 12:  # First hour - viral explosion
                intensity = min(1.0, 0.1 * np.exp(i/4))
                volume_mult = min(50, np.exp(i/3))
            else:  # Sustained viral activity
                intensity = 0.9 + np.random.normal(0, 0.05)
                volume_mult = 25 + np.random.normal(0, 5)
                
            # Multiple sources all going viral
            for source in ['twitter', 'reddit', 'tiktok', 'instagram', 'youtube']:
                data.append({
                    'timestamp': timestamp,
                    'sentiment_score': max(0, min(1, intensity)),
                    'volume': max(100, int(1000 * volume_mult)),
                    'source': source
                })
                
        return pd.DataFrame(data)
        
    def create_negative_sentiment_data(self) -> pd.DataFrame:
        """Create negative sentiment despite positive event"""
        now = datetime.utcnow()
        data = []
        
        for i in range(24):
            timestamp = now - timedelta(minutes=i*5)
            
            # Skeptical/negative sentiment
            sentiment = 0.2 + np.random.normal(0, 0.1)
            sentiment = max(0, min(1, sentiment))
            
            volume = int(np.random.normal(800, 200))
            
            data.append({
                'timestamp': timestamp,
                'sentiment_score': sentiment,
                'volume': max(10, volume),
                'source': f'skeptical_source_{i%4}'
            })
            
        return pd.DataFrame(data)
        
    def print_catalyst_result(self, catalyst, description: str):
        """Print catalyst detection result in a formatted way"""
        print(f"Analysis: {description}")
        
        if catalyst:
            print(f"CATALYST DETECTED!")
            print(f"  Type: {catalyst.catalyst_type.value}")
            print(f"  Confidence: {catalyst.confidence_score:.1%}")
            print(f"  Risk-Adjusted Score: {catalyst.risk_adjusted_score:.1%}")
            print(f"  Trigger Time: {catalyst.trigger_time}")
            print(f"  Primary Signal: {catalyst.primary_signal_source}")
            
            if catalyst.signal_alignment_score:
                print(f"  Signal Alignment: {catalyst.signal_alignment_score:.1%}")
                
            # Print signal strengths
            if catalyst.event_signal:
                print(f"  Event Signal: {catalyst.event_signal.strength.value} "
                      f"(confidence: {catalyst.event_signal.confidence:.1%})")
                      
            if catalyst.sentiment_signal:
                print(f"  Sentiment Signal: {catalyst.sentiment_signal.strength.value} "
                      f"(confidence: {catalyst.sentiment_signal.confidence:.1%})")
                      
            if catalyst.technical_signal:
                print(f"  Technical Signal: {catalyst.technical_signal.strength.value} "
                      f"(confidence: {catalyst.technical_signal.confidence:.1%})")
                      
            if catalyst.cross_validation_results:
                print(f"  Cross-Validation:")
                for validation, result in catalyst.cross_validation_results.items():
                    print(f"    {validation}: {result}")
        else:
            print("NO CATALYST DETECTED")
            print("  Signals did not meet catalyst threshold requirements")

async def main():
    """Run catalyst trigger demonstrations"""
    demo = CatalystTriggerDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    asyncio.run(main())