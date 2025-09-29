"""
Ownership Flow Analysis Service
Analyzes institutional and insider trading patterns to detect smart money flows
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, func
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OwnershipFlowMetrics:
    """Container for ownership flow analysis results"""
    symbol: str
    analysis_date: datetime
    period_days: int
    
    # Insider metrics
    insider_buy_transactions: int = 0
    insider_sell_transactions: int = 0
    insider_net_shares: int = 0
    insider_net_value: float = 0.0
    insider_buy_value: float = 0.0
    insider_sell_value: float = 0.0
    
    # Institutional metrics
    institutions_increasing: int = 0
    institutions_decreasing: int = 0
    institutions_new_positions: int = 0
    institutions_sold_out: int = 0
    institutional_net_shares: int = 0
    institutional_net_value: float = 0.0
    
    # Smart money signals
    cluster_buying_detected: bool = False
    cluster_selling_detected: bool = False
    smart_money_score: float = 0.0
    confidence_level: float = 0.0
    
    metadata: Dict = None

class OwnershipFlowAnalyzer:
    """Analyzes ownership flow patterns and detects smart money signals"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        
    async def analyze_ownership_flow(
        self,
        symbol: str,
        period_days: int = 30,
        analysis_date: Optional[datetime] = None
    ) -> OwnershipFlowMetrics:
        """
        Comprehensive ownership flow analysis
        """
        if analysis_date is None:
            analysis_date = datetime.now().date()
        
        start_date = analysis_date - timedelta(days=period_days)
        
        # Get insider trading data
        insider_metrics = await self._analyze_insider_flow(symbol, start_date, analysis_date)
        
        # Get institutional flow data
        institutional_metrics = await self._analyze_institutional_flow(symbol, start_date, analysis_date)
        
        # Detect clustering patterns
        cluster_signals = await self._detect_cluster_patterns(symbol, start_date, analysis_date)
        
        # Calculate smart money score
        smart_money_score, confidence = self._calculate_smart_money_score(
            insider_metrics, institutional_metrics, cluster_signals
        )
        
        # Create comprehensive metrics
        flow_metrics = OwnershipFlowMetrics(
            symbol=symbol,
            analysis_date=analysis_date,
            period_days=period_days,
            **insider_metrics,
            **institutional_metrics,
            **cluster_signals,
            smart_money_score=smart_money_score,
            confidence_level=confidence,
            metadata={
                'analysis_timestamp': datetime.now().isoformat(),
                'data_sources': ['insider_transactions', 'institutional_holdings'],
                'algorithm_version': '1.0'
            }
        )
        
        # Store results
        await self._store_flow_analysis(flow_metrics)
        
        return flow_metrics
    
    async def _analyze_insider_flow(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict:
        """Analyze insider trading patterns"""
        
        query = text("""
            SELECT 
                COUNT(*) as total_transactions,
                COUNT(CASE WHEN transaction_code IN ('P', 'A') THEN 1 END) as buy_transactions,
                COUNT(CASE WHEN transaction_code IN ('S', 'D') THEN 1 END) as sell_transactions,
                COALESCE(SUM(CASE WHEN transaction_code IN ('P', 'A') THEN transaction_shares ELSE 0 END), 0) as buy_shares,
                COALESCE(SUM(CASE WHEN transaction_code IN ('S', 'D') THEN transaction_shares ELSE 0 END), 0) as sell_shares,
                COALESCE(SUM(CASE WHEN transaction_code IN ('P', 'A') THEN transaction_value ELSE 0 END), 0) as buy_value,
                COALESCE(SUM(CASE WHEN transaction_code IN ('S', 'D') THEN transaction_value ELSE 0 END), 0) as sell_value,
                COUNT(DISTINCT reporting_owner_name) as unique_insiders
            FROM insider_transactions
            WHERE symbol = :symbol 
            AND transaction_date BETWEEN :start_date AND :end_date
            AND transaction_value IS NOT NULL
        """)
        
        result = await self.db.execute(query, {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        row = result.fetchone()
        if not row:
            return self._empty_insider_metrics()
        
        buy_shares = int(row.buy_shares or 0)
        sell_shares = int(row.sell_shares or 0)
        buy_value = float(row.buy_value or 0)
        sell_value = float(row.sell_value or 0)
        
        return {
            'insider_buy_transactions': int(row.buy_transactions or 0),
            'insider_sell_transactions': int(row.sell_transactions or 0),
            'insider_net_shares': buy_shares - sell_shares,
            'insider_net_value': buy_value - sell_value,
            'insider_buy_value': buy_value,
            'insider_sell_value': sell_value
        }
    
    async def _analyze_institutional_flow(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Analyze institutional holdings changes"""
        
        # Get latest quarter data within the period
        query = text("""
            WITH latest_quarter AS (
                SELECT MAX(quarter_end) as max_quarter
                FROM institutional_holdings
                WHERE symbol = :symbol
                AND quarter_end BETWEEN :start_date AND :end_date
            ),
            current_holdings AS (
                SELECT *
                FROM institutional_holdings ih
                JOIN latest_quarter lq ON ih.quarter_end = lq.max_quarter
                WHERE ih.symbol = :symbol
            ),
            previous_quarter AS (
                SELECT quarter_end
                FROM institutional_holdings
                WHERE symbol = :symbol
                AND quarter_end < (SELECT max_quarter FROM latest_quarter)
                ORDER BY quarter_end DESC
                LIMIT 1
            )
            SELECT 
                COUNT(CASE WHEN shares_change > 0 THEN 1 END) as increasing,
                COUNT(CASE WHEN shares_change < 0 THEN 1 END) as decreasing,
                COUNT(CASE WHEN is_new_position = true THEN 1 END) as new_positions,
                COUNT(CASE WHEN is_sold_out = true THEN 1 END) as sold_out,
                COALESCE(SUM(shares_change), 0) as net_shares_change,
                COALESCE(SUM(CASE WHEN shares_change > 0 THEN market_value ELSE 0 END), 0) as increasing_value,
                COALESCE(SUM(CASE WHEN shares_change < 0 THEN ABS(market_value) ELSE 0 END), 0) as decreasing_value
            FROM current_holdings
            WHERE shares_change IS NOT NULL
        """)
        
        result = await self.db.execute(query, {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        row = result.fetchone()
        if not row:
            return self._empty_institutional_metrics()
        
        increasing_value = float(row.increasing_value or 0)
        decreasing_value = float(row.decreasing_value or 0)
        
        return {
            'institutions_increasing': int(row.increasing or 0),
            'institutions_decreasing': int(row.decreasing or 0),
            'institutions_new_positions': int(row.new_positions or 0),
            'institutions_sold_out': int(row.sold_out or 0),
            'institutional_net_shares': int(row.net_shares_change or 0),
            'institutional_net_value': increasing_value - decreasing_value
        }
    
    async def _detect_cluster_patterns(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Detect clustering in ownership changes"""
        
        # Analyze insider clustering
        insider_cluster = await self._detect_insider_clustering(symbol, start_date, end_date)
        
        # Analyze institutional clustering
        institutional_cluster = await self._detect_institutional_clustering(symbol, start_date, end_date)
        
        # Determine overall cluster signals
        cluster_buying = insider_cluster['buy_cluster'] or institutional_cluster['buy_cluster']
        cluster_selling = insider_cluster['sell_cluster'] or institutional_cluster['sell_cluster']
        
        return {
            'cluster_buying_detected': cluster_buying,
            'cluster_selling_detected': cluster_selling
        }
    
    async def _detect_insider_clustering(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Detect clustering in insider transactions"""
        
        query = text("""
            SELECT 
                transaction_date,
                COUNT(*) as transaction_count,
                COUNT(CASE WHEN transaction_code IN ('P', 'A') THEN 1 END) as buy_count,
                COUNT(CASE WHEN transaction_code IN ('S', 'D') THEN 1 END) as sell_count,
                COUNT(DISTINCT reporting_owner_name) as unique_insiders
            FROM insider_transactions
            WHERE symbol = :symbol
            AND transaction_date BETWEEN :start_date AND :end_date
            GROUP BY transaction_date
            HAVING COUNT(*) >= 2  -- At least 2 transactions on same day
            ORDER BY transaction_date
        """)
        
        result = await self.db.execute(query, {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        rows = result.fetchall()
        
        # Look for clustering patterns
        buy_cluster_days = sum(1 for row in rows if row.buy_count >= 2)
        sell_cluster_days = sum(1 for row in rows if row.sell_count >= 2)
        
        # Cluster detected if multiple days with coordinated activity
        buy_cluster = buy_cluster_days >= 2
        sell_cluster = sell_cluster_days >= 2
        
        return {
            'buy_cluster': buy_cluster,
            'sell_cluster': sell_cluster,
            'cluster_days': len(rows)
        }
    
    async def _detect_institutional_clustering(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Detect clustering in institutional changes"""
        
        # For institutional data, clustering is based on percentage of institutions
        # moving in same direction within same quarter
        query = text("""
            SELECT 
                quarter_end,
                COUNT(*) as total_institutions,
                COUNT(CASE WHEN shares_change > 0 THEN 1 END) as increasing_count,
                COUNT(CASE WHEN shares_change < 0 THEN 1 END) as decreasing_count,
                COUNT(CASE WHEN is_new_position = true THEN 1 END) as new_positions,
                COUNT(CASE WHEN is_sold_out = true THEN 1 END) as sold_out
            FROM institutional_holdings
            WHERE symbol = :symbol
            AND quarter_end BETWEEN :start_date AND :end_date
            AND shares_change IS NOT NULL
            GROUP BY quarter_end
        """)
        
        result = await self.db.execute(query, {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        rows = result.fetchall()
        
        buy_cluster = False
        sell_cluster = False
        
        for row in rows:
            if row.total_institutions >= 5:  # Minimum institutions for clustering
                # Buy cluster: >60% of institutions increasing or new positions
                buy_ratio = (row.increasing_count + row.new_positions) / row.total_institutions
                if buy_ratio > 0.6:
                    buy_cluster = True
                
                # Sell cluster: >60% of institutions decreasing or selling out
                sell_ratio = (row.decreasing_count + row.sold_out) / row.total_institutions
                if sell_ratio > 0.6:
                    sell_cluster = True
        
        return {
            'buy_cluster': buy_cluster,
            'sell_cluster': sell_cluster
        }
    
    def _calculate_smart_money_score(
        self,
        insider_metrics: Dict,
        institutional_metrics: Dict,
        cluster_signals: Dict
    ) -> Tuple[float, float]:
        """Calculate overall smart money score and confidence"""
        
        # Base scores from volume and direction
        insider_score = self._calculate_insider_score(insider_metrics)
        institutional_score = self._calculate_institutional_score(institutional_metrics)
        cluster_score = self._calculate_cluster_score(cluster_signals)
        
        # Weighted combination
        weights = {
            'insider': 0.4,
            'institutional': 0.5,
            'cluster': 0.1
        }
        
        smart_money_score = (
            insider_score * weights['insider'] +
            institutional_score * weights['institutional'] +
            cluster_score * weights['cluster']
        )
        
        # Calculate confidence based on data quality and agreement
        confidence = self._calculate_confidence(
            insider_metrics, institutional_metrics, cluster_signals
        )
        
        return round(smart_money_score, 4), round(confidence, 4)
    
    def _calculate_insider_score(self, metrics: Dict) -> float:
        """Calculate insider trading score (-1 to 1)"""
        net_value = metrics.get('insider_net_value', 0)
        buy_value = metrics.get('insider_buy_value', 0)
        sell_value = metrics.get('insider_sell_value', 0)
        
        total_value = buy_value + sell_value
        if total_value == 0:
            return 0.0
        
        # Score based on net direction and magnitude
        direction_score = net_value / total_value if total_value > 0 else 0
        
        # Apply magnitude scaling (larger trades get higher weight)
        magnitude_factor = min(1.0, total_value / 1000000)  # $1M normalization
        
        return direction_score * magnitude_factor
    
    def _calculate_institutional_score(self, metrics: Dict) -> float:
        """Calculate institutional flow score (-1 to 1)"""
        increasing = metrics.get('institutions_increasing', 0)
        decreasing = metrics.get('institutions_decreasing', 0)
        new_positions = metrics.get('institutions_new_positions', 0)
        sold_out = metrics.get('institutions_sold_out', 0)
        
        total_active = increasing + decreasing + new_positions + sold_out
        if total_active == 0:
            return 0.0
        
        # Positive signals: increasing positions, new positions
        positive_signals = increasing + new_positions
        # Negative signals: decreasing positions, sold out
        negative_signals = decreasing + sold_out
        
        net_signals = positive_signals - negative_signals
        score = net_signals / total_active
        
        return score
    
    def _calculate_cluster_score(self, signals: Dict) -> float:
        """Calculate clustering bonus score"""
        buy_cluster = signals.get('cluster_buying_detected', False)
        sell_cluster = signals.get('cluster_selling_detected', False)
        
        if buy_cluster:
            return 0.5  # Positive clustering bonus
        elif sell_cluster:
            return -0.5  # Negative clustering penalty
        else:
            return 0.0
    
    def _calculate_confidence(
        self,
        insider_metrics: Dict,
        institutional_metrics: Dict,
        cluster_signals: Dict
    ) -> float:
        """Calculate confidence level (0 to 1)"""
        
        confidence_factors = []
        
        # Insider data quality
        insider_transactions = (
            insider_metrics.get('insider_buy_transactions', 0) +
            insider_metrics.get('insider_sell_transactions', 0)
        )
        if insider_transactions > 0:
            insider_confidence = min(1.0, insider_transactions / 5)  # Max confidence at 5+ transactions
            confidence_factors.append(insider_confidence * 0.4)
        
        # Institutional data quality
        total_institutions = (
            institutional_metrics.get('institutions_increasing', 0) +
            institutional_metrics.get('institutions_decreasing', 0) +
            institutional_metrics.get('institutions_new_positions', 0) +
            institutional_metrics.get('institutions_sold_out', 0)
        )
        if total_institutions > 0:
            institutional_confidence = min(1.0, total_institutions / 10)  # Max confidence at 10+ institutions
            confidence_factors.append(institutional_confidence * 0.5)
        
        # Clustering confirmation
        if cluster_signals.get('cluster_buying_detected') or cluster_signals.get('cluster_selling_detected'):
            confidence_factors.append(0.1)  # Clustering adds confidence
        
        return sum(confidence_factors) if confidence_factors else 0.0
    
    def _empty_insider_metrics(self) -> Dict:
        """Return empty insider metrics"""
        return {
            'insider_buy_transactions': 0,
            'insider_sell_transactions': 0,
            'insider_net_shares': 0,
            'insider_net_value': 0.0,
            'insider_buy_value': 0.0,
            'insider_sell_value': 0.0
        }
    
    def _empty_institutional_metrics(self) -> Dict:
        """Return empty institutional metrics"""
        return {
            'institutions_increasing': 0,
            'institutions_decreasing': 0,
            'institutions_new_positions': 0,
            'institutions_sold_out': 0,
            'institutional_net_shares': 0,
            'institutional_net_value': 0.0
        }
    
    async def _store_flow_analysis(self, metrics: OwnershipFlowMetrics):
        """Store ownership flow analysis results"""
        
        insert_query = text("""
            INSERT INTO ownership_flow_analysis (
                symbol, analysis_date, period_days,
                insider_buy_transactions, insider_sell_transactions,
                insider_net_shares, insider_net_value,
                insider_buy_value, insider_sell_value,
                institutions_increasing, institutions_decreasing,
                institutions_new_positions, institutions_sold_out,
                institutional_net_shares, institutional_net_value,
                cluster_buying_detected, cluster_selling_detected,
                smart_money_score, confidence_level, metadata
            ) VALUES (
                :symbol, :analysis_date, :period_days,
                :insider_buy_transactions, :insider_sell_transactions,
                :insider_net_shares, :insider_net_value,
                :insider_buy_value, :insider_sell_value,
                :institutions_increasing, :institutions_decreasing,
                :institutions_new_positions, :institutions_sold_out,
                :institutional_net_shares, :institutional_net_value,
                :cluster_buying_detected, :cluster_selling_detected,
                :smart_money_score, :confidence_level, :metadata
            )
            ON CONFLICT (symbol, analysis_date, period_days)
            DO UPDATE SET
                insider_buy_transactions = EXCLUDED.insider_buy_transactions,
                insider_sell_transactions = EXCLUDED.insider_sell_transactions,
                insider_net_shares = EXCLUDED.insider_net_shares,
                insider_net_value = EXCLUDED.insider_net_value,
                insider_buy_value = EXCLUDED.insider_buy_value,
                insider_sell_value = EXCLUDED.insider_sell_value,
                institutions_increasing = EXCLUDED.institutions_increasing,
                institutions_decreasing = EXCLUDED.institutions_decreasing,
                institutions_new_positions = EXCLUDED.institutions_new_positions,
                institutions_sold_out = EXCLUDED.institutions_sold_out,
                institutional_net_shares = EXCLUDED.institutional_net_shares,
                institutional_net_value = EXCLUDED.institutional_net_value,
                cluster_buying_detected = EXCLUDED.cluster_buying_detected,
                cluster_selling_detected = EXCLUDED.cluster_selling_detected,
                smart_money_score = EXCLUDED.smart_money_score,
                confidence_level = EXCLUDED.confidence_level,
                metadata = EXCLUDED.metadata,
                created_at = NOW()
        """)
        
        await self.db.execute(insert_query, {
            'symbol': metrics.symbol,
            'analysis_date': metrics.analysis_date,
            'period_days': metrics.period_days,
            'insider_buy_transactions': metrics.insider_buy_transactions,
            'insider_sell_transactions': metrics.insider_sell_transactions,
            'insider_net_shares': metrics.insider_net_shares,
            'insider_net_value': metrics.insider_net_value,
            'insider_buy_value': metrics.insider_buy_value,
            'insider_sell_value': metrics.insider_sell_value,
            'institutions_increasing': metrics.institutions_increasing,
            'institutions_decreasing': metrics.institutions_decreasing,
            'institutions_new_positions': metrics.institutions_new_positions,
            'institutions_sold_out': metrics.institutions_sold_out,
            'institutional_net_shares': metrics.institutional_net_shares,
            'institutional_net_value': metrics.institutional_net_value,
            'cluster_buying_detected': metrics.cluster_buying_detected,
            'cluster_selling_detected': metrics.cluster_selling_detected,
            'smart_money_score': metrics.smart_money_score,
            'confidence_level': metrics.confidence_level,
            'metadata': metrics.metadata
        })
        
        await self.db.commit()
    
    async def get_smart_money_signals(
        self,
        symbol: Optional[str] = None,
        min_score: float = 0.3,
        limit: int = 50
    ) -> List[Dict]:
        """Get stocks with strong smart money signals"""
        
        where_clause = ""
        params = {'min_score': min_score, 'limit': limit}
        
        if symbol:
            where_clause = "WHERE symbol = :symbol AND"
            params['symbol'] = symbol
        else:
            where_clause = "WHERE"
        
        query = text(f"""
            SELECT 
                symbol,
                analysis_date,
                period_days,
                smart_money_score,
                confidence_level,
                cluster_buying_detected,
                cluster_selling_detected,
                insider_net_value,
                institutional_net_value,
                metadata
            FROM ownership_flow_analysis
            {where_clause} ABS(smart_money_score) >= :min_score
            ORDER BY ABS(smart_money_score) DESC, confidence_level DESC
            LIMIT :limit
        """)
        
        result = await self.db.execute(query, params)
        rows = result.fetchall()
        
        return [
            {
                'symbol': row.symbol,
                'analysis_date': row.analysis_date.isoformat(),
                'period_days': row.period_days,
                'smart_money_score': float(row.smart_money_score),
                'confidence_level': float(row.confidence_level),
                'signal_type': 'bullish' if row.smart_money_score > 0 else 'bearish',
                'cluster_buying': row.cluster_buying_detected,
                'cluster_selling': row.cluster_selling_detected,
                'insider_net_value': float(row.insider_net_value),
                'institutional_net_value': float(row.institutional_net_value),
                'metadata': row.metadata
            }
            for row in rows
        ]