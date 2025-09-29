"""
Analyst Revision Tracker
Tracks analyst rating changes, price target revisions, and consensus momentum
Captures shifting analyst sentiment ahead of earnings and other catalysts
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from sqlalchemy.orm import Session
import numpy as np

logger = logging.getLogger(__name__)


class RatingAction(Enum):
    """Analyst rating action types"""
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    INITIATE = "initiate"
    REITERATE = "reiterate"
    SUSPEND = "suspend"


class RatingLevel(Enum):
    """Standard rating levels"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AnalystRevision:
    """Individual analyst revision record"""
    revision_date: date
    analyst_firm: str
    analyst_name: Optional[str]
    symbol: str
    company_name: str
    
    # Rating changes
    previous_rating: Optional[str]
    new_rating: str
    rating_action: RatingAction
    
    # Price target changes
    previous_price_target: Optional[float]
    new_price_target: Optional[float]
    price_target_change: Optional[float]
    price_target_change_pct: Optional[float]
    
    # Context
    current_stock_price: Optional[float]
    upside_downside_pct: Optional[float]
    
    # Metadata
    revision_reason: Optional[str]
    event_catalyst: Optional[str]  # earnings, guidance, etc.
    source_url: Optional[str]
    confidence_score: float  # 0-1, based on analyst track record


@dataclass
class ConsensusSnapshot:
    """Consensus metrics at a point in time"""
    snapshot_date: date
    symbol: str
    
    # Rating consensus
    strong_buy_count: int
    buy_count: int
    hold_count: int
    sell_count: int
    strong_sell_count: int
    total_analysts: int
    average_rating: float  # 1-5 scale
    
    # Price target consensus
    mean_price_target: Optional[float]
    median_price_target: Optional[float]
    high_price_target: Optional[float]
    low_price_target: Optional[float]
    price_target_std: Optional[float]
    
    # EPS estimates
    current_quarter_eps_mean: Optional[float]
    current_quarter_eps_count: int
    current_year_eps_mean: Optional[float]
    current_year_eps_count: int
    
    # Revenue estimates
    current_quarter_revenue_mean: Optional[float]
    current_year_revenue_mean: Optional[float]


@dataclass
class RevisionMomentum:
    """30-day revision momentum metrics"""
    symbol: str
    analysis_period_start: date
    analysis_period_end: date
    days_analyzed: int
    
    # Rating momentum
    total_revisions: int
    upgrades: int
    downgrades: int
    initiations: int
    net_rating_changes: int  # upgrades - downgrades
    rating_momentum_score: float  # -1 to 1
    
    # Price target momentum
    price_target_revisions: int
    price_target_increases: int
    price_target_decreases: int
    average_price_target_change_pct: float
    price_target_momentum_score: float  # -1 to 1
    
    # Consensus changes
    consensus_rating_change: float
    consensus_price_target_change_pct: Optional[float]
    consensus_eps_revision_pct: Optional[float]
    
    # Momentum strength
    revision_intensity: float  # revisions per day
    momentum_acceleration: float  # change in revision rate
    conviction_score: float  # weighted by analyst reputation
    
    # Event-driven analysis
    pre_earnings_momentum: bool
    unusual_activity_detected: bool
    smart_money_following: bool


class AnalystRevisionTracker:
    """Service for tracking analyst revisions and consensus momentum"""
    
    def __init__(self):
        self.finnhub_base_url = "https://finnhub.io/api/v1"
        self.alpha_vantage_base_url = "https://www.alphavantage.co/query"
        
        # Rating mappings
        self.rating_to_numeric = {
            "strong buy": 1,
            "buy": 2,
            "hold": 3,
            "sell": 4,
            "strong sell": 5,
            "outperform": 2,
            "market perform": 3,
            "underperform": 4,
            "overweight": 2,
            "equal weight": 3,
            "underweight": 4,
        }
        
        # Top analyst firms (for weighting)
        self.tier_1_firms = [
            "Goldman Sachs", "Morgan Stanley", "JP Morgan", "Bank of America",
            "Barclays", "Credit Suisse", "Deutsche Bank", "Citigroup"
        ]
        
        self.tier_2_firms = [
            "Wells Fargo", "UBS", "Jefferies", "Cowen", "Raymond James",
            "Piper Sandler", "William Blair", "Stifel", "Oppenheimer"
        ]
    
    async def get_recent_revisions(
        self,
        symbol: str,
        days_back: int = 30,
        min_price_target: float = 1.0
    ) -> List[AnalystRevision]:
        """Get recent analyst revisions for a symbol"""
        
        try:
            # For now, generate synthetic revision data
            # In production, this would call real analyst APIs
            
            revisions = []
            analysts_firms = [
                ("Goldman Sachs", "John Smith"),
                ("Morgan Stanley", "Sarah Johnson"), 
                ("JP Morgan", "Michael Brown"),
                ("Bank of America", "Emily Davis"),
                ("Barclays", "David Wilson"),
                ("UBS", "Lisa Anderson"),
                ("Wells Fargo", "Robert Taylor"),
                ("Jefferies", "Jennifer Martinez"),
            ]
            
            # Generate 3-8 revisions over the period
            num_revisions = 3 + (hash(symbol) % 6)
            current_price = 100.0 + (hash(symbol) % 200)  # Mock current price
            
            for i in range(num_revisions):
                firm, analyst = analysts_firms[i % len(analysts_firms)]
                revision_date = date.today() - timedelta(days=(i * days_back // num_revisions))
                
                # Generate revision data
                is_upgrade = (hash(symbol + str(i)) % 100) > 40  # 60% chance of upgrade
                
                if is_upgrade:
                    previous_rating = "hold"
                    new_rating = "buy"
                    rating_action = RatingAction.UPGRADE
                    pt_change_pct = 5 + (hash(symbol + str(i)) % 15)  # 5-20% increase
                else:
                    previous_rating = "buy"
                    new_rating = "hold"
                    rating_action = RatingAction.DOWNGRADE
                    pt_change_pct = -(2 + (hash(symbol + str(i)) % 10))  # 2-12% decrease
                
                previous_pt = current_price * (1.1 + (hash(symbol + str(i)) % 30) / 100)
                new_pt = previous_pt * (1 + pt_change_pct / 100)
                
                revision = AnalystRevision(
                    revision_date=revision_date,
                    analyst_firm=firm,
                    analyst_name=analyst,
                    symbol=symbol,
                    company_name=f"{symbol} Inc.",
                    previous_rating=previous_rating,
                    new_rating=new_rating,
                    rating_action=rating_action,
                    previous_price_target=previous_pt,
                    new_price_target=new_pt,
                    price_target_change=new_pt - previous_pt,
                    price_target_change_pct=pt_change_pct,
                    current_stock_price=current_price,
                    upside_downside_pct=(new_pt - current_price) / current_price * 100,
                    revision_reason="Quarterly results analysis" if i % 2 == 0 else "Sector outlook update",
                    event_catalyst="earnings" if i % 3 == 0 else None,
                    source_url=f"https://research.{firm.lower().replace(' ', '')}.com/{symbol}",
                    confidence_score=0.9 if firm in self.tier_1_firms else 0.7,
                )
                
                revisions.append(revision)
            
            # Sort by date, most recent first
            revisions.sort(key=lambda x: x.revision_date, reverse=True)
            return revisions
            
        except Exception as e:
            logger.error(f"Error getting revisions for {symbol}: {e}")
            return []
    
    async def get_consensus_snapshot(self, symbol: str) -> ConsensusSnapshot:
        """Get current consensus ratings and estimates"""
        
        try:
            # Generate synthetic consensus data
            # In production, this would aggregate from real data sources
            
            total_analysts = 12 + (hash(symbol) % 8)  # 12-20 analysts
            
            # Generate rating distribution
            strong_buy = max(0, 2 + (hash(symbol + "sb") % 4))
            buy = max(0, 5 + (hash(symbol + "b") % 6))
            hold = max(0, 3 + (hash(symbol + "h") % 5))
            sell = max(0, 1 + (hash(symbol + "s") % 2))
            strong_sell = max(0, hash(symbol + "ss") % 2)
            
            # Adjust to match total
            actual_total = strong_buy + buy + hold + sell + strong_sell
            if actual_total != total_analysts:
                hold += (total_analysts - actual_total)
                hold = max(0, hold)
            
            # Calculate average rating
            weighted_sum = (strong_buy * 1 + buy * 2 + hold * 3 + sell * 4 + strong_sell * 5)
            average_rating = weighted_sum / total_analysts if total_analysts > 0 else 3.0
            
            # Generate price targets
            base_price = 100.0 + (hash(symbol) % 200)
            price_targets = []
            for i in range(total_analysts):
                pt = base_price * (0.8 + (hash(symbol + str(i)) % 60) / 100)  # 80% to 140% of base
                price_targets.append(pt)
            
            mean_pt = np.mean(price_targets) if price_targets else None
            median_pt = np.median(price_targets) if price_targets else None
            high_pt = max(price_targets) if price_targets else None
            low_pt = min(price_targets) if price_targets else None
            std_pt = np.std(price_targets) if price_targets else None
            
            return ConsensusSnapshot(
                snapshot_date=date.today(),
                symbol=symbol,
                strong_buy_count=strong_buy,
                buy_count=buy,
                hold_count=hold,
                sell_count=sell,
                strong_sell_count=strong_sell,
                total_analysts=total_analysts,
                average_rating=average_rating,
                mean_price_target=mean_pt,
                median_price_target=median_pt,
                high_price_target=high_pt,
                low_price_target=low_pt,
                price_target_std=std_pt,
                current_quarter_eps_mean=2.5 + (hash(symbol + "eps") % 300) / 100,
                current_quarter_eps_count=total_analysts - 2,
                current_year_eps_mean=10.0 + (hash(symbol + "yeps") % 500) / 100,
                current_year_eps_count=total_analysts - 1,
                current_quarter_revenue_mean=5000 + (hash(symbol + "rev") % 10000),
                current_year_revenue_mean=20000 + (hash(symbol + "yrev") % 30000),
            )
            
        except Exception as e:
            logger.error(f"Error getting consensus for {symbol}: {e}")
            return ConsensusSnapshot(
                snapshot_date=date.today(),
                symbol=symbol,
                strong_buy_count=0, buy_count=0, hold_count=0, sell_count=0, strong_sell_count=0,
                total_analysts=0, average_rating=3.0,
                mean_price_target=None, median_price_target=None,
                high_price_target=None, low_price_target=None, price_target_std=None,
                current_quarter_eps_mean=None, current_quarter_eps_count=0,
                current_year_eps_mean=None, current_year_eps_count=0,
                current_quarter_revenue_mean=None, current_year_revenue_mean=None,
            )
    
    async def calculate_revision_momentum(
        self,
        symbol: str,
        analysis_days: int = 30
    ) -> RevisionMomentum:
        """Calculate 30-day revision momentum metrics"""
        
        try:
            # Get recent revisions
            revisions = await self.get_recent_revisions(symbol, analysis_days)
            
            if not revisions:
                return self._get_empty_momentum(symbol, analysis_days)
            
            # Count revision types
            upgrades = sum(1 for r in revisions if r.rating_action == RatingAction.UPGRADE)
            downgrades = sum(1 for r in revisions if r.rating_action == RatingAction.DOWNGRADE)
            initiations = sum(1 for r in revisions if r.rating_action == RatingAction.INITIATE)
            
            # Price target analysis
            pt_revisions = [r for r in revisions if r.price_target_change is not None]
            pt_increases = sum(1 for r in pt_revisions if r.price_target_change > 0)
            pt_decreases = sum(1 for r in pt_revisions if r.price_target_change < 0)
            
            avg_pt_change = np.mean([r.price_target_change_pct for r in pt_revisions if r.price_target_change_pct])
            if np.isnan(avg_pt_change):
                avg_pt_change = 0.0
            
            # Calculate momentum scores
            net_rating_changes = upgrades - downgrades
            total_revisions = len(revisions)
            
            if total_revisions > 0:
                rating_momentum = net_rating_changes / total_revisions
                pt_momentum = (pt_increases - pt_decreases) / len(pt_revisions) if pt_revisions else 0
            else:
                rating_momentum = 0.0
                pt_momentum = 0.0
            
            # Calculate intensity and acceleration
            revision_intensity = total_revisions / analysis_days
            
            # Check for acceleration (more revisions in recent days)
            recent_revisions = [r for r in revisions if (date.today() - r.revision_date).days <= 7]
            recent_intensity = len(recent_revisions) / 7
            momentum_acceleration = recent_intensity - revision_intensity
            
            # Calculate conviction score (weighted by analyst tier)
            conviction_scores = []
            for revision in revisions:
                if revision.analyst_firm in self.tier_1_firms:
                    conviction_scores.append(revision.confidence_score * 1.0)
                elif revision.analyst_firm in self.tier_2_firms:
                    conviction_scores.append(revision.confidence_score * 0.8)
                else:
                    conviction_scores.append(revision.confidence_score * 0.6)
            
            conviction_score = np.mean(conviction_scores) if conviction_scores else 0.0
            
            # Event-driven analysis
            earnings_related = sum(1 for r in revisions if r.event_catalyst == "earnings")
            pre_earnings_momentum = earnings_related >= 2
            
            unusual_activity = revision_intensity > 0.2  # More than 1 revision per 5 days
            smart_money_following = sum(1 for r in revisions if r.analyst_firm in self.tier_1_firms) >= 2
            
            # Get consensus changes (would compare with historical snapshots)
            current_consensus = await self.get_consensus_snapshot(symbol)
            consensus_rating_change = 0.0  # Would calculate from historical data
            consensus_pt_change_pct = 0.0   # Would calculate from historical data
            consensus_eps_revision = 0.0    # Would calculate from historical data
            
            return RevisionMomentum(
                symbol=symbol,
                analysis_period_start=date.today() - timedelta(days=analysis_days),
                analysis_period_end=date.today(),
                days_analyzed=analysis_days,
                total_revisions=total_revisions,
                upgrades=upgrades,
                downgrades=downgrades,
                initiations=initiations,
                net_rating_changes=net_rating_changes,
                rating_momentum_score=rating_momentum,
                price_target_revisions=len(pt_revisions),
                price_target_increases=pt_increases,
                price_target_decreases=pt_decreases,
                average_price_target_change_pct=avg_pt_change,
                price_target_momentum_score=pt_momentum,
                consensus_rating_change=consensus_rating_change,
                consensus_price_target_change_pct=consensus_pt_change_pct,
                consensus_eps_revision_pct=consensus_eps_revision,
                revision_intensity=revision_intensity,
                momentum_acceleration=momentum_acceleration,
                conviction_score=conviction_score,
                pre_earnings_momentum=pre_earnings_momentum,
                unusual_activity_detected=unusual_activity,
                smart_money_following=smart_money_following,
            )
            
        except Exception as e:
            logger.error(f"Error calculating revision momentum for {symbol}: {e}")
            return self._get_empty_momentum(symbol, analysis_days)
    
    def _get_empty_momentum(self, symbol: str, analysis_days: int) -> RevisionMomentum:
        """Return empty momentum object for error cases"""
        return RevisionMomentum(
            symbol=symbol,
            analysis_period_start=date.today() - timedelta(days=analysis_days),
            analysis_period_end=date.today(),
            days_analyzed=analysis_days,
            total_revisions=0, upgrades=0, downgrades=0, initiations=0,
            net_rating_changes=0, rating_momentum_score=0.0,
            price_target_revisions=0, price_target_increases=0, price_target_decreases=0,
            average_price_target_change_pct=0.0, price_target_momentum_score=0.0,
            consensus_rating_change=0.0, consensus_price_target_change_pct=0.0,
            consensus_eps_revision_pct=0.0, revision_intensity=0.0,
            momentum_acceleration=0.0, conviction_score=0.0,
            pre_earnings_momentum=False, unusual_activity_detected=False,
            smart_money_following=False,
        )
    
    async def get_top_movers_by_revisions(self, symbols: List[str], days: int = 7) -> List[Dict[str, Any]]:
        """Get symbols with most significant revision activity"""
        
        movers = []
        
        for symbol in symbols:
            try:
                momentum = await self.calculate_revision_momentum(symbol, days)
                
                # Calculate significance score
                significance = (
                    abs(momentum.rating_momentum_score) * 0.4 +
                    abs(momentum.price_target_momentum_score) * 0.3 +
                    momentum.revision_intensity * 0.2 +
                    momentum.conviction_score * 0.1
                )
                
                movers.append({
                    "symbol": symbol,
                    "significance_score": significance,
                    "rating_momentum": momentum.rating_momentum_score,
                    "price_target_momentum": momentum.price_target_momentum_score,
                    "total_revisions": momentum.total_revisions,
                    "net_rating_changes": momentum.net_rating_changes,
                    "unusual_activity": momentum.unusual_activity_detected,
                    "smart_money_following": momentum.smart_money_following,
                })
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by significance score
        movers.sort(key=lambda x: x["significance_score"], reverse=True)
        return movers
    
    async def detect_pre_earnings_activity(self, symbol: str, days_to_earnings: int = 30) -> Dict[str, Any]:
        """Detect unusual analyst activity before earnings"""
        
        try:
            momentum = await self.calculate_revision_momentum(symbol, days_to_earnings)
            
            # Analyze patterns
            activity_score = momentum.revision_intensity * 10  # Scale up
            consensus_shift = abs(momentum.rating_momentum_score)
            smart_money_factor = 1.5 if momentum.smart_money_following else 1.0
            
            # Overall pre-earnings signal
            pre_earnings_signal = min(1.0, activity_score * consensus_shift * smart_money_factor)
            
            return {
                "symbol": symbol,
                "days_analyzed": days_to_earnings,
                "pre_earnings_activity_detected": momentum.unusual_activity_detected,
                "pre_earnings_signal_strength": pre_earnings_signal,
                "rating_momentum": momentum.rating_momentum_score,
                "price_target_momentum": momentum.price_target_momentum_score,
                "revision_intensity": momentum.revision_intensity,
                "smart_money_involvement": momentum.smart_money_following,
                "conviction_level": momentum.conviction_score,
                "recommendations": self._generate_pre_earnings_recommendations(momentum, pre_earnings_signal),
            }
            
        except Exception as e:
            logger.error(f"Error detecting pre-earnings activity for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def _generate_pre_earnings_recommendations(self, momentum: RevisionMomentum, signal_strength: float) -> List[str]:
        """Generate recommendations based on pre-earnings analysis"""
        
        recommendations = []
        
        if signal_strength > 0.7:
            if momentum.rating_momentum_score > 0.5:
                recommendations.append("Strong positive analyst momentum suggests potential earnings beat")
            elif momentum.rating_momentum_score < -0.5:
                recommendations.append("Negative analyst revisions may indicate earnings risk")
        
        if momentum.smart_money_following:
            recommendations.append("Tier-1 analysts are revising - high conviction signal")
        
        if momentum.unusual_activity_detected:
            recommendations.append("Unusual revision activity detected - monitor closely")
        
        if momentum.price_target_momentum_score > 0.3:
            recommendations.append("Rising price targets suggest post-earnings upside potential")
        
        if not recommendations:
            recommendations.append("Normal analyst activity levels - no unusual patterns detected")
        
        return recommendations


# Global service instance
analyst_revision_tracker = AnalystRevisionTracker()