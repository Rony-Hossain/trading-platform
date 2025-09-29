"""
Surprise Scoring Service - Calculates event surprise scores
Measures actual outcomes vs expectations for event-driven analysis
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import math

from ..core.database import SurpriseScore, SurpriseData, EventType

logger = logging.getLogger(__name__)


class SurpriseScoringService:
    """Service for calculating and managing surprise scores"""
    
    def __init__(self):
        self.calculation_methods = [
            "earnings_surprise",
            "revenue_surprise", 
            "guidance_surprise",
            "event_timing_surprise",
            "magnitude_surprise",
        ]
    
    async def calculate_earnings_surprise(
        self,
        symbol: str,
        expected_eps: float,
        actual_eps: float,
        historical_volatility: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate earnings surprise score"""
        
        if expected_eps == 0:
            # Handle case where expected EPS is zero
            if actual_eps > 0:
                surprise_magnitude = 1.0
            elif actual_eps < 0:
                surprise_magnitude = -1.0
            else:
                surprise_magnitude = 0.0
        else:
            # Standard surprise calculation
            surprise_magnitude = (actual_eps - expected_eps) / abs(expected_eps)
        
        # Normalize surprise score (0 to 1 scale)
        # Use historical volatility if available for context
        if historical_volatility and historical_volatility > 0:
            # Normalize by historical surprise volatility
            surprise_score = min(1.0, abs(surprise_magnitude) / (historical_volatility * 2))
        else:
            # Simple normalization
            surprise_score = min(1.0, abs(surprise_magnitude))
        
        return {
            "symbol": symbol,
            "event_type": "earnings",
            "expected_value": expected_eps,
            "actual_value": actual_eps,
            "surprise_magnitude": surprise_magnitude,
            "surprise_score": surprise_score,
            "surprise_percent": surprise_magnitude * 100,
            "calculation_method": "earnings_surprise",
            "historical_volatility": historical_volatility,
        }
    
    async def calculate_revenue_surprise(
        self,
        symbol: str,
        expected_revenue: float,
        actual_revenue: float,
        historical_volatility: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate revenue surprise score"""
        
        if expected_revenue == 0:
            surprise_magnitude = 1.0 if actual_revenue > 0 else 0.0
        else:
            surprise_magnitude = (actual_revenue - expected_revenue) / expected_revenue
        
        # Revenue surprises tend to be smaller than EPS surprises
        # so we use a different normalization
        if historical_volatility and historical_volatility > 0:
            surprise_score = min(1.0, abs(surprise_magnitude) / (historical_volatility * 1.5))
        else:
            surprise_score = min(1.0, abs(surprise_magnitude) * 2)  # Amplify smaller revenue surprises
        
        return {
            "symbol": symbol,
            "event_type": "revenue",
            "expected_value": expected_revenue,
            "actual_value": actual_revenue,
            "surprise_magnitude": surprise_magnitude,
            "surprise_score": surprise_score,
            "surprise_percent": surprise_magnitude * 100,
            "calculation_method": "revenue_surprise",
            "historical_volatility": historical_volatility,
        }
    
    async def calculate_guidance_surprise(
        self,
        symbol: str,
        previous_guidance: float,
        new_guidance: float,
        analyst_expectations: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate guidance surprise score"""
        
        # Compare new guidance to previous guidance
        if previous_guidance == 0:
            surprise_magnitude = 1.0 if new_guidance > 0 else 0.0
        else:
            surprise_magnitude = (new_guidance - previous_guidance) / abs(previous_guidance)
        
        # If analyst expectations available, factor them in
        if analyst_expectations and analyst_expectations != 0:
            analyst_surprise = (new_guidance - analyst_expectations) / abs(analyst_expectations)
            # Weight the two surprises
            surprise_magnitude = (surprise_magnitude * 0.6) + (analyst_surprise * 0.4)
        
        surprise_score = min(1.0, abs(surprise_magnitude))
        
        return {
            "symbol": symbol,
            "event_type": "guidance",
            "expected_value": previous_guidance,
            "actual_value": new_guidance,
            "surprise_magnitude": surprise_magnitude,
            "surprise_score": surprise_score,
            "surprise_percent": surprise_magnitude * 100,
            "calculation_method": "guidance_surprise",
            "analyst_expectations": analyst_expectations,
        }
    
    async def calculate_event_timing_surprise(
        self,
        symbol: str,
        expected_date: date,
        actual_date: date,
        event_type: str,
    ) -> Dict[str, Any]:
        """Calculate surprise score for event timing"""
        
        # Calculate days difference
        days_diff = abs((actual_date - expected_date).days)
        
        # Different event types have different timing sensitivity
        timing_sensitivity = {
            "earnings": 7,      # Earnings dates can shift by a week
            "fda_approval": 30, # FDA decisions can be delayed significantly
            "product_launch": 14, # Product launches can be delayed
            "merger": 60,       # Mergers can take months longer than expected
        }
        
        max_days = timing_sensitivity.get(event_type, 14)
        
        # Normalize surprise score
        surprise_score = min(1.0, days_diff / max_days)
        
        return {
            "symbol": symbol,
            "event_type": f"{event_type}_timing",
            "expected_value": expected_date.toordinal(),  # Convert to numeric
            "actual_value": actual_date.toordinal(),
            "surprise_magnitude": days_diff,
            "surprise_score": surprise_score,
            "calculation_method": "event_timing_surprise",
            "days_difference": days_diff,
        }
    
    async def calculate_magnitude_surprise(
        self,
        symbol: str,
        event_type: str,
        expected_magnitude: float,
        actual_magnitude: float,
        market_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate surprise based on event magnitude"""
        
        if expected_magnitude == 0:
            surprise_magnitude = 1.0 if actual_magnitude > 0 else 0.0
        else:
            surprise_magnitude = (actual_magnitude - expected_magnitude) / abs(expected_magnitude)
        
        # Adjust for market context
        context_multiplier = 1.0
        if market_context == "bull_market":
            # In bull markets, positive surprises are less surprising
            if surprise_magnitude > 0:
                context_multiplier = 0.8
        elif market_context == "bear_market":
            # In bear markets, negative surprises are less surprising
            if surprise_magnitude < 0:
                context_multiplier = 0.8
        
        surprise_score = min(1.0, abs(surprise_magnitude) * context_multiplier)
        
        return {
            "symbol": symbol,
            "event_type": event_type,
            "expected_value": expected_magnitude,
            "actual_value": actual_magnitude,
            "surprise_magnitude": surprise_magnitude,
            "surprise_score": surprise_score,
            "calculation_method": "magnitude_surprise",
            "market_context": market_context,
        }
    
    async def get_symbol_surprises(
        self,
        symbol: str,
        since: datetime,
        event_types: Optional[List[str]] = None,
        db: Session = None,
    ) -> List[SurpriseData]:
        """Get recent surprise scores for a symbol"""
        
        # Generate synthetic surprise data
        surprises = []
        
        # Create synthetic surprises for recent events
        event_types_list = event_types or ["earnings", "revenue", "guidance"]
        
        for i, event_type in enumerate(event_types_list):
            event_date = date.today() - timedelta(days=i * 7 + 1)
            
            if datetime.combine(event_date, datetime.min.time()) < since:
                continue
            
            # Generate synthetic surprise data
            if event_type == "earnings":
                expected_eps = 2.50 + (hash(symbol) % 100) / 100
                actual_eps = expected_eps + (hash(symbol + event_type) % 40 - 20) / 100
                surprise_calc = await self.calculate_earnings_surprise(
                    symbol, expected_eps, actual_eps, 0.15
                )
            elif event_type == "revenue":
                expected_revenue = 50000 + (hash(symbol) % 20000)
                actual_revenue = expected_revenue + (hash(symbol + event_type) % 4000 - 2000)
                surprise_calc = await self.calculate_revenue_surprise(
                    symbol, expected_revenue, actual_revenue, 0.08
                )
            else:  # guidance
                previous_guidance = 10.0 + (hash(symbol) % 20) / 10
                new_guidance = previous_guidance + (hash(symbol + event_type) % 20 - 10) / 10
                surprise_calc = await self.calculate_guidance_surprise(
                    symbol, previous_guidance, new_guidance
                )
            
            surprise = SurpriseData(
                id=100 + i,
                symbol=symbol,
                event_type=event_type,
                event_date=event_date,
                expected_value=surprise_calc["expected_value"],
                actual_value=surprise_calc["actual_value"],
                surprise_magnitude=surprise_calc["surprise_magnitude"],
                surprise_score=surprise_calc["surprise_score"],
                historical_volatility=surprise_calc.get("historical_volatility"),
                event_importance=0.8 if event_type == "earnings" else 0.6,
                market_context="neutral",
                calculation_method=surprise_calc["calculation_method"],
                confidence_level=0.85,
                data_sources=["synthetic"],
                metadata={
                    "calculated_at": datetime.now().isoformat(),
                    "surprise_percent": surprise_calc.get("surprise_percent", 0),
                }
            )
            
            surprises.append(surprise)
        
        return surprises
    
    async def get_surprise_history(
        self,
        symbol: str,
        since_date: date,
        event_types: Optional[List[str]] = None,
        min_surprise_score: float = 0.5,
        db: Session = None,
    ) -> List[SurpriseData]:
        """Get historical surprise scores for pattern analysis"""
        
        surprises = []
        current_date = since_date
        today = date.today()
        
        # Generate synthetic historical surprise data
        while current_date <= today:
            # Create surprises every 90 days (quarterly)
            if (today - current_date).days % 90 == 0:
                
                # Earnings surprise
                expected_eps = 2.0 + ((hash(symbol + str(current_date)) % 200) / 100)
                actual_eps = expected_eps + ((hash(symbol + str(current_date) + "eps") % 60 - 30) / 100)
                surprise_calc = await self.calculate_earnings_surprise(symbol, expected_eps, actual_eps, 0.12)
                
                if surprise_calc["surprise_score"] >= min_surprise_score:
                    surprise = SurpriseData(
                        id=hash(f"{symbol}{current_date}earnings") % 10000,
                        symbol=symbol,
                        event_type="earnings",
                        event_date=current_date,
                        expected_value=surprise_calc["expected_value"],
                        actual_value=surprise_calc["actual_value"],
                        surprise_magnitude=surprise_calc["surprise_magnitude"],
                        surprise_score=surprise_calc["surprise_score"],
                        historical_volatility=0.12,
                        event_importance=0.8,
                        market_context="neutral",
                        calculation_method="earnings_surprise",
                        confidence_level=0.85,
                        data_sources=["synthetic"],
                        metadata={
                            "quarter": f"Q{((current_date.month - 1) // 3) + 1}",
                            "fiscal_year": current_date.year,
                        }
                    )
                    surprises.append(surprise)
            
            current_date += timedelta(days=30)  # Check monthly
        
        # Sort by date, most recent first
        surprises.sort(key=lambda x: x.event_date, reverse=True)
        
        return surprises
    
    async def calculate_composite_surprise(
        self,
        symbol: str,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate composite surprise score from multiple factors"""
        
        total_surprise = 0.0
        total_weight = 0.0
        surprise_components = {}
        
        # Earnings component
        if "earnings" in event_data:
            earnings_data = event_data["earnings"]
            earnings_surprise = await self.calculate_earnings_surprise(
                symbol,
                earnings_data["expected"],
                earnings_data["actual"],
                earnings_data.get("volatility")
            )
            weight = 0.4
            total_surprise += earnings_surprise["surprise_score"] * weight
            total_weight += weight
            surprise_components["earnings"] = earnings_surprise
        
        # Revenue component
        if "revenue" in event_data:
            revenue_data = event_data["revenue"]
            revenue_surprise = await self.calculate_revenue_surprise(
                symbol,
                revenue_data["expected"],
                revenue_data["actual"],
                revenue_data.get("volatility")
            )
            weight = 0.3
            total_surprise += revenue_surprise["surprise_score"] * weight
            total_weight += weight
            surprise_components["revenue"] = revenue_surprise
        
        # Guidance component
        if "guidance" in event_data:
            guidance_data = event_data["guidance"]
            guidance_surprise = await self.calculate_guidance_surprise(
                symbol,
                guidance_data["previous"],
                guidance_data["new"],
                guidance_data.get("analyst_expected")
            )
            weight = 0.3
            total_surprise += guidance_surprise["surprise_score"] * weight
            total_weight += weight
            surprise_components["guidance"] = guidance_surprise
        
        # Calculate final composite score
        if total_weight > 0:
            composite_score = total_surprise / total_weight
        else:
            composite_score = 0.0
        
        return {
            "symbol": symbol,
            "composite_surprise_score": composite_score,
            "components": surprise_components,
            "total_weight": total_weight,
            "calculation_method": "composite_surprise",
            "calculated_at": datetime.now().isoformat(),
        }
    
    async def get_stats(self, db: Session = None) -> Dict[str, Any]:
        """Get surprise scoring statistics"""
        
        return {
            "total_surprises_calculated": 450,  # Synthetic
            "average_surprise_score": 0.42,
            "surprise_score_distribution": {
                "0.0-0.2": 125,
                "0.2-0.4": 145,
                "0.4-0.6": 98,
                "0.6-0.8": 56,
                "0.8-1.0": 26,
            },
            "event_types": {
                "earnings": 280,
                "revenue": 85,
                "guidance": 65,
                "timing": 20,
            },
            "calculation_methods": self.calculation_methods,
            "symbols_tracked": 45,
        }


# Global service instance
surprise_scoring_service = SurpriseScoringService()