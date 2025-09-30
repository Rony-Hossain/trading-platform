"""
Form 13F Holding-Change Aggregation System

This module provides comprehensive analysis of SEC Form 13F institutional holding changes
with sophisticated aggregation methods to generate actionable investment signals.

Key Features:
1. Holding Change Analysis: Track institutional buying/selling patterns
2. Smart Money Identification: Identify high-performing institutional investors
3. Consensus Building: Aggregate holdings across multiple institutions
4. Flow Analysis: Analyze money flows and position changes over time
5. Signal Generation: Create trading signals from institutional activity

Applications:
- Institutional flow analysis for investment decisions
- Smart money following strategies
- Position sizing based on institutional consensus
- Risk assessment through institutional positioning

References:
- Griffin, J. M., & Xu, J. (2009). How smart are the smart guys?
- Sias, R. W., Starks, L. T., & Titman, S. (2006). Changes in institutional ownership.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx

logger = logging.getLogger(__name__)


class InstitutionType(Enum):
    """Types of institutional investors."""
    INVESTMENT_ADVISOR = "investment_advisor"
    BANK = "bank"
    INSURANCE_COMPANY = "insurance_company"
    PENSION_FUND = "pension_fund"
    HEDGE_FUND = "hedge_fund"
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    ENDOWMENT = "endowment"
    SOVEREIGN_WEALTH = "sovereign_wealth"
    OTHER = "other"


class HoldingChangeType(Enum):
    """Types of holding changes."""
    NEW_POSITION = "new_position"
    INCREASED = "increased"
    DECREASED = "decreased"
    ELIMINATED = "eliminated"
    UNCHANGED = "unchanged"


class SignalDirection(Enum):
    """Signal direction for trading."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class Form13FHolding:
    """Represents a single 13F holding record."""
    filing_id: str
    institution_cik: str
    institution_name: str
    cusip: str
    security_name: str
    ticker: Optional[str]
    shares_held: int
    market_value: float           # In USD
    percent_of_portfolio: float   # Percentage of total portfolio
    filing_date: datetime
    report_date: datetime         # Quarter end date
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filing_id': self.filing_id,
            'institution_cik': self.institution_cik,
            'institution_name': self.institution_name,
            'cusip': self.cusip,
            'security_name': self.security_name,
            'ticker': self.ticker,
            'shares_held': self.shares_held,
            'market_value': self.market_value,
            'percent_of_portfolio': self.percent_of_portfolio,
            'filing_date': self.filing_date.isoformat(),
            'report_date': self.report_date.isoformat()
        }


@dataclass
class HoldingChange:
    """Represents a change in institutional holding between periods."""
    institution_cik: str
    institution_name: str
    cusip: str
    ticker: Optional[str]
    security_name: str
    previous_quarter: datetime
    current_quarter: datetime
    previous_shares: int
    current_shares: int
    previous_value: float
    current_value: float
    change_type: HoldingChangeType
    shares_change: int
    value_change: float
    percent_change: float
    
    @property
    def is_meaningful_change(self) -> bool:
        """Determine if change is meaningful (>5% or >$10M)."""
        return abs(self.percent_change) > 0.05 or abs(self.value_change) > 10_000_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'institution_cik': self.institution_cik,
            'institution_name': self.institution_name,
            'cusip': self.cusip,
            'ticker': self.ticker,
            'security_name': self.security_name,
            'previous_quarter': self.previous_quarter.isoformat(),
            'current_quarter': self.current_quarter.isoformat(),
            'previous_shares': self.previous_shares,
            'current_shares': self.current_shares,
            'previous_value': self.previous_value,
            'current_value': self.current_value,
            'change_type': self.change_type.value,
            'shares_change': self.shares_change,
            'value_change': self.value_change,
            'percent_change': self.percent_change,
            'is_meaningful_change': self.is_meaningful_change
        }


@dataclass
class InstitutionProfile:
    """Profile of institutional investor behavior and performance."""
    institution_cik: str
    institution_name: str
    institution_type: InstitutionType
    avg_portfolio_size: float
    total_positions: int
    avg_position_size: float
    concentration_ratio: float      # Top 10 holdings as % of portfolio
    turnover_rate: float           # Annual portfolio turnover
    performance_score: float       # Historical performance rating (0-100)
    smart_money_rank: int         # Rank among all institutions
    sectors_focus: List[str]      # Primary sector focus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'institution_cik': self.institution_cik,
            'institution_name': self.institution_name,
            'institution_type': self.institution_type.value,
            'avg_portfolio_size': self.avg_portfolio_size,
            'total_positions': self.total_positions,
            'avg_position_size': self.avg_position_size,
            'concentration_ratio': self.concentration_ratio,
            'turnover_rate': self.turnover_rate,
            'performance_score': self.performance_score,
            'smart_money_rank': self.smart_money_rank,
            'sectors_focus': self.sectors_focus
        }


@dataclass
class AggregatedSignal:
    """Aggregated institutional signal for a security."""
    cusip: str
    ticker: Optional[str]
    security_name: str
    signal_date: datetime
    signal_direction: SignalDirection
    signal_strength: float         # 0-100
    confidence_level: float        # 0-1
    total_institutions: int
    buying_institutions: int
    selling_institutions: int
    net_flow_value: float         # Net institutional flow
    net_flow_shares: int          # Net share flow
    smart_money_score: float      # Weighted by institution quality
    consensus_score: float        # Agreement among institutions
    contributing_changes: List[str]  # List of change IDs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cusip': self.cusip,
            'ticker': self.ticker,
            'security_name': self.security_name,
            'signal_date': self.signal_date.isoformat(),
            'signal_direction': self.signal_direction.value,
            'signal_strength': self.signal_strength,
            'confidence_level': self.confidence_level,
            'total_institutions': self.total_institutions,
            'buying_institutions': self.buying_institutions,
            'selling_institutions': self.selling_institutions,
            'net_flow_value': self.net_flow_value,
            'net_flow_shares': self.net_flow_shares,
            'smart_money_score': self.smart_money_score,
            'consensus_score': self.consensus_score,
            'contributing_changes': self.contributing_changes
        }


class Form13FAggregator:
    """
    Form 13F Holding-Change Aggregation Engine.
    
    Processes institutional holding changes and generates aggregated
    investment signals based on smart money flows and consensus.
    """
    
    def __init__(self, 
                 min_position_value: float = 1_000_000,
                 min_meaningful_change: float = 0.05,
                 lookback_quarters: int = 8):
        """
        Initialize Form 13F aggregator.
        
        Parameters:
        - min_position_value: Minimum position value to consider
        - min_meaningful_change: Minimum change percentage to consider meaningful
        - lookback_quarters: Number of quarters for analysis
        """
        self.min_position_value = min_position_value
        self.min_meaningful_change = min_meaningful_change
        self.lookback_quarters = lookback_quarters
        
        # Data storage
        self.holdings: List[Form13FHolding] = []
        self.holding_changes: List[HoldingChange] = []
        self.institution_profiles: Dict[str, InstitutionProfile] = {}
        self.aggregated_signals: List[AggregatedSignal] = []
        
        # Performance tracking
        self.performance_cache: Dict[str, float] = {}
        self.smart_money_rankings: Dict[str, int] = {}
    
    def add_13f_holdings(self, holdings: List[Form13FHolding]):
        """
        Add Form 13F holdings for analysis.
        
        Parameters:
        - holdings: List of 13F holding records
        """
        self.holdings.extend(holdings)
        logger.info(f"Added {len(holdings)} 13F holdings")
    
    def calculate_holding_changes(self) -> List[HoldingChange]:
        """
        Calculate holding changes between consecutive quarters.
        
        Returns:
        - List of holding changes
        """
        logger.info("Calculating holding changes between quarters")
        
        # Group holdings by institution and security
        grouped_holdings = {}
        
        for holding in self.holdings:
            key = (holding.institution_cik, holding.cusip)
            if key not in grouped_holdings:
                grouped_holdings[key] = []
            grouped_holdings[key].append(holding)
        
        changes = []
        
        for (institution_cik, cusip), holdings_list in grouped_holdings.items():
            # Sort by report date
            holdings_list.sort(key=lambda x: x.report_date)
            
            # Calculate changes between consecutive quarters
            for i in range(1, len(holdings_list)):
                current = holdings_list[i]
                previous = holdings_list[i-1]
                
                # Skip if not consecutive quarters
                months_diff = (current.report_date.year - previous.report_date.year) * 12 + \
                             (current.report_date.month - previous.report_date.month)
                
                if months_diff > 6:  # Allow up to 6 months gap
                    continue
                
                # Calculate changes
                shares_change = current.shares_held - previous.shares_held
                value_change = current.market_value - previous.market_value
                
                if previous.shares_held > 0:
                    percent_change = shares_change / previous.shares_held
                else:
                    percent_change = 1.0 if shares_change > 0 else 0.0
                
                # Determine change type
                if previous.shares_held == 0 and current.shares_held > 0:
                    change_type = HoldingChangeType.NEW_POSITION
                elif previous.shares_held > 0 and current.shares_held == 0:
                    change_type = HoldingChangeType.ELIMINATED
                elif shares_change > 0:
                    change_type = HoldingChangeType.INCREASED
                elif shares_change < 0:
                    change_type = HoldingChangeType.DECREASED
                else:
                    change_type = HoldingChangeType.UNCHANGED
                
                # Create change record
                change = HoldingChange(
                    institution_cik=institution_cik,
                    institution_name=current.institution_name,
                    cusip=cusip,
                    ticker=current.ticker,
                    security_name=current.security_name,
                    previous_quarter=previous.report_date,
                    current_quarter=current.report_date,
                    previous_shares=previous.shares_held,
                    current_shares=current.shares_held,
                    previous_value=previous.market_value,
                    current_value=current.market_value,
                    change_type=change_type,
                    shares_change=shares_change,
                    value_change=value_change,
                    percent_change=percent_change
                )
                
                changes.append(change)
        
        self.holding_changes = changes
        logger.info(f"Calculated {len(changes)} holding changes")
        
        return changes
    
    def build_institution_profiles(self, 
                                 stock_returns: Optional[Dict[str, pd.Series]] = None) -> Dict[str, InstitutionProfile]:
        """
        Build profiles for institutional investors.
        
        Parameters:
        - stock_returns: Historical stock returns for performance calculation
        
        Returns:
        - Dictionary of institution profiles
        """
        logger.info("Building institutional investor profiles")
        
        # Group holdings by institution
        institution_holdings = {}
        for holding in self.holdings:
            if holding.institution_cik not in institution_holdings:
                institution_holdings[holding.institution_cik] = []
            institution_holdings[holding.institution_cik].append(holding)
        
        profiles = {}
        
        for institution_cik, holdings_list in institution_holdings.items():
            if len(holdings_list) < 4:  # Need at least 4 quarters of data
                continue
            
            # Calculate profile metrics
            institution_name = holdings_list[0].institution_name
            
            # Portfolio statistics
            quarterly_portfolios = {}
            for holding in holdings_list:
                quarter = holding.report_date
                if quarter not in quarterly_portfolios:
                    quarterly_portfolios[quarter] = []
                quarterly_portfolios[quarter].append(holding)
            
            portfolio_sizes = [sum(h.market_value for h in holdings) 
                             for holdings in quarterly_portfolios.values()]
            avg_portfolio_size = np.mean(portfolio_sizes)
            
            total_positions = len(set(h.cusip for h in holdings_list))
            avg_position_size = avg_portfolio_size / max(1, total_positions / len(quarterly_portfolios))
            
            # Concentration ratio (top 10 holdings)
            concentration_ratios = []
            for holdings in quarterly_portfolios.values():
                if len(holdings) >= 10:
                    top_10_value = sum(sorted([h.market_value for h in holdings], reverse=True)[:10])
                    total_value = sum(h.market_value for h in holdings)
                    concentration_ratios.append(top_10_value / total_value if total_value > 0 else 0)
            
            concentration_ratio = np.mean(concentration_ratios) if concentration_ratios else 0
            
            # Turnover rate calculation
            turnover_rate = self._calculate_turnover_rate(institution_cik)
            
            # Performance score
            performance_score = self._calculate_performance_score(
                institution_cik, stock_returns
            )
            
            # Sector focus
            sector_focus = self._identify_sector_focus(holdings_list)
            
            # Determine institution type (simplified heuristic)
            institution_type = self._classify_institution_type(institution_name, avg_portfolio_size)
            
            profile = InstitutionProfile(
                institution_cik=institution_cik,
                institution_name=institution_name,
                institution_type=institution_type,
                avg_portfolio_size=avg_portfolio_size,
                total_positions=total_positions,
                avg_position_size=avg_position_size,
                concentration_ratio=concentration_ratio,
                turnover_rate=turnover_rate,
                performance_score=performance_score,
                smart_money_rank=0,  # Will be calculated after all profiles
                sectors_focus=sector_focus
            )
            
            profiles[institution_cik] = profile
        
        # Calculate smart money rankings
        sorted_institutions = sorted(profiles.values(), 
                                   key=lambda x: x.performance_score, 
                                   reverse=True)
        
        for rank, institution in enumerate(sorted_institutions, 1):
            institution.smart_money_rank = rank
            self.smart_money_rankings[institution.institution_cik] = rank
        
        self.institution_profiles = profiles
        logger.info(f"Built profiles for {len(profiles)} institutions")
        
        return profiles
    
    def _calculate_turnover_rate(self, institution_cik: str) -> float:
        """Calculate annual portfolio turnover rate."""
        institution_changes = [c for c in self.holding_changes 
                             if c.institution_cik == institution_cik]
        
        if not institution_changes:
            return 0.0
        
        # Calculate quarterly turnover
        quarterly_turnover = {}
        
        for change in institution_changes:
            quarter = change.current_quarter
            if quarter not in quarterly_turnover:
                quarterly_turnover[quarter] = {'total_traded': 0, 'avg_portfolio': 0}
            
            quarterly_turnover[quarter]['total_traded'] += abs(change.value_change)
        
        # Get average portfolio values
        institution_holdings = [h for h in self.holdings 
                              if h.institution_cik == institution_cik]
        
        for quarter in quarterly_turnover.keys():
            quarter_holdings = [h for h in institution_holdings 
                              if h.report_date == quarter]
            if quarter_holdings:
                quarterly_turnover[quarter]['avg_portfolio'] = sum(h.market_value for h in quarter_holdings)
        
        # Calculate annualized turnover
        turnover_rates = []
        for quarter_data in quarterly_turnover.values():
            if quarter_data['avg_portfolio'] > 0:
                quarterly_rate = quarter_data['total_traded'] / quarter_data['avg_portfolio']
                annual_rate = quarterly_rate * 4  # Annualize
                turnover_rates.append(annual_rate)
        
        return np.mean(turnover_rates) if turnover_rates else 0.0
    
    def _calculate_performance_score(self, 
                                   institution_cik: str, 
                                   stock_returns: Optional[Dict[str, pd.Series]]) -> float:
        """Calculate institution's performance score based on stock picking ability."""
        if not stock_returns:
            return 50.0  # Default neutral score
        
        institution_changes = [c for c in self.holding_changes 
                             if c.institution_cik == institution_cik 
                             and c.is_meaningful_change]
        
        if not institution_changes:
            return 50.0
        
        performance_scores = []
        
        for change in institution_changes:
            if not change.ticker or change.ticker not in stock_returns:
                continue
            
            returns = stock_returns[change.ticker]
            
            # Look at 3-month forward returns after the change
            start_date = change.current_quarter
            end_date = start_date + timedelta(days=90)
            
            period_returns = returns.loc[start_date:end_date]
            
            if len(period_returns) < 10:  # Need minimum data
                continue
            
            period_performance = (1 + period_returns).prod() - 1
            
            # Score based on change direction and subsequent performance
            if change.change_type in [HoldingChangeType.NEW_POSITION, HoldingChangeType.INCREASED]:
                # Buying - good if performance is positive
                score = 50 + (period_performance * 100)  # Convert to 0-100 scale
            elif change.change_type in [HoldingChangeType.DECREASED, HoldingChangeType.ELIMINATED]:
                # Selling - good if performance is negative
                score = 50 - (period_performance * 100)
            else:
                score = 50
            
            performance_scores.append(max(0, min(100, score)))
        
        if not performance_scores:
            return 50.0
        
        return np.mean(performance_scores)
    
    def _identify_sector_focus(self, holdings: List[Form13FHolding]) -> List[str]:
        """Identify primary sector focus for institution."""
        # This would need sector mapping data - simplified for now
        # In practice, you'd map CUSIPs to sectors using reference data
        return ["Technology", "Healthcare"]  # Placeholder
    
    def _classify_institution_type(self, name: str, portfolio_size: float) -> InstitutionType:
        """Classify institution type based on name and size."""
        name_lower = name.lower()
        
        if 'bank' in name_lower:
            return InstitutionType.BANK
        elif 'insurance' in name_lower:
            return InstitutionType.INSURANCE_COMPANY
        elif 'pension' in name_lower:
            return InstitutionType.PENSION_FUND
        elif 'fund' in name_lower and portfolio_size > 10_000_000_000:
            return InstitutionType.MUTUAL_FUND
        elif 'capital' in name_lower or 'partners' in name_lower:
            return InstitutionType.HEDGE_FUND
        else:
            return InstitutionType.INVESTMENT_ADVISOR
    
    def aggregate_signals(self, 
                         analysis_date: Optional[datetime] = None,
                         min_institutions: int = 3) -> List[AggregatedSignal]:
        """
        Aggregate institutional signals by security.
        
        Parameters:
        - analysis_date: Date for analysis (defaults to latest quarter)
        - min_institutions: Minimum institutions required for signal
        
        Returns:
        - List of aggregated signals
        """
        if analysis_date is None:
            analysis_date = max(h.report_date for h in self.holdings)
        
        logger.info(f"Aggregating institutional signals for {analysis_date}")
        
        # Get recent changes (last quarter)
        recent_changes = [c for c in self.holding_changes 
                         if c.current_quarter == analysis_date 
                         and c.is_meaningful_change]
        
        # Group by security
        security_changes = {}
        for change in recent_changes:
            cusip = change.cusip
            if cusip not in security_changes:
                security_changes[cusip] = []
            security_changes[cusip].append(change)
        
        signals = []
        
        for cusip, changes in security_changes.items():
            if len(changes) < min_institutions:
                continue
            
            signal = self._create_aggregated_signal(cusip, changes, analysis_date)
            if signal:
                signals.append(signal)
        
        self.aggregated_signals = signals
        logger.info(f"Generated {len(signals)} aggregated signals")
        
        return signals
    
    def _create_aggregated_signal(self, 
                                cusip: str, 
                                changes: List[HoldingChange],
                                analysis_date: datetime) -> Optional[AggregatedSignal]:
        """Create aggregated signal for a security."""
        if not changes:
            return None
        
        # Basic aggregation metrics
        total_institutions = len(changes)
        buying_institutions = len([c for c in changes 
                                 if c.change_type in [HoldingChangeType.NEW_POSITION, 
                                                     HoldingChangeType.INCREASED]])
        selling_institutions = len([c for c in changes 
                                  if c.change_type in [HoldingChangeType.DECREASED, 
                                                      HoldingChangeType.ELIMINATED]])
        
        # Net flows
        net_flow_value = sum(c.value_change for c in changes)
        net_flow_shares = sum(c.shares_change for c in changes)
        
        # Signal direction
        if buying_institutions > selling_institutions * 1.5 and net_flow_value > 0:
            signal_direction = SignalDirection.BULLISH
        elif selling_institutions > buying_institutions * 1.5 and net_flow_value < 0:
            signal_direction = SignalDirection.BEARISH
        else:
            signal_direction = SignalDirection.NEUTRAL
        
        # Signal strength (0-100)
        institution_ratio = max(buying_institutions, selling_institutions) / total_institutions
        value_magnitude = min(100, abs(net_flow_value) / 100_000_000)  # Scale by $100M
        signal_strength = min(100, (institution_ratio * 50) + (value_magnitude * 50))
        
        # Smart money score (weighted by institution quality)
        smart_money_score = self._calculate_smart_money_score(changes)
        
        # Consensus score (agreement among institutions)
        consensus_score = max(buying_institutions, selling_institutions) / total_institutions
        
        # Confidence level
        confidence_level = min(1.0, (total_institutions / 10) * consensus_score)
        
        # Get security info from first change
        first_change = changes[0]
        
        signal = AggregatedSignal(
            cusip=cusip,
            ticker=first_change.ticker,
            security_name=first_change.security_name,
            signal_date=analysis_date,
            signal_direction=signal_direction,
            signal_strength=signal_strength,
            confidence_level=confidence_level,
            total_institutions=total_institutions,
            buying_institutions=buying_institutions,
            selling_institutions=selling_institutions,
            net_flow_value=net_flow_value,
            net_flow_shares=net_flow_shares,
            smart_money_score=smart_money_score,
            consensus_score=consensus_score,
            contributing_changes=[f"{c.institution_cik}_{c.current_quarter.strftime('%Y%m%d')}" 
                                for c in changes]
        )
        
        return signal
    
    def _calculate_smart_money_score(self, changes: List[HoldingChange]) -> float:
        """Calculate smart money score based on institution quality."""
        if not self.institution_profiles:
            return 50.0  # Default neutral score
        
        weighted_scores = []
        total_weight = 0
        
        for change in changes:
            institution_cik = change.institution_cik
            
            if institution_cik in self.institution_profiles:
                profile = self.institution_profiles[institution_cik]
                
                # Weight by performance score and portfolio size
                weight = profile.performance_score * np.log10(max(1e6, profile.avg_portfolio_size))
                
                # Score based on change direction
                if change.change_type in [HoldingChangeType.NEW_POSITION, HoldingChangeType.INCREASED]:
                    score = profile.performance_score
                else:
                    score = 100 - profile.performance_score  # Inverse for selling
                
                weighted_scores.append(score * weight)
                total_weight += weight
        
        if total_weight == 0:
            return 50.0
        
        return sum(weighted_scores) / total_weight
    
    def get_top_signals(self, 
                       n: int = 10, 
                       signal_direction: Optional[SignalDirection] = None,
                       min_confidence: float = 0.5) -> List[AggregatedSignal]:
        """
        Get top signals by strength and confidence.
        
        Parameters:
        - n: Number of signals to return
        - signal_direction: Filter by signal direction
        - min_confidence: Minimum confidence level
        
        Returns:
        - List of top signals
        """
        filtered_signals = [s for s in self.aggregated_signals 
                          if s.confidence_level >= min_confidence]
        
        if signal_direction:
            filtered_signals = [s for s in filtered_signals 
                              if s.signal_direction == signal_direction]
        
        # Sort by combined score of strength and confidence
        sorted_signals = sorted(filtered_signals, 
                              key=lambda x: x.signal_strength * x.confidence_level, 
                              reverse=True)
        
        return sorted_signals[:n]
    
    def get_institution_analytics(self) -> Dict[str, Any]:
        """Get analytics about institutional behavior."""
        if not self.institution_profiles:
            return {}
        
        profiles = list(self.institution_profiles.values())
        
        analytics = {
            'total_institutions': len(profiles),
            'avg_portfolio_size': np.mean([p.avg_portfolio_size for p in profiles]),
            'avg_turnover_rate': np.mean([p.turnover_rate for p in profiles]),
            'avg_performance_score': np.mean([p.performance_score for p in profiles]),
            'institution_types': {t.value: len([p for p in profiles if p.institution_type == t]) 
                                for t in InstitutionType},
            'top_performers': [p.institution_name for p in 
                             sorted(profiles, key=lambda x: x.performance_score, reverse=True)[:10]]
        }
        
        return analytics