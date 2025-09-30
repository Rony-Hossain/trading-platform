"""
Form 4 Clustering and Analysis System

This module provides sophisticated analysis of SEC Form 4 filings (insider trading reports)
with clustering algorithms to identify meaningful patterns in insider activity.

Key Features:
1. Form 4 Data Processing: Parse and normalize Form 4 filing data
2. Insider Clustering: Group insiders by trading patterns and characteristics
3. Signal Generation: Create actionable trading signals from insider activity
4. Temporal Analysis: Track insider activity over time with trend detection
5. Cross-Security Analysis: Identify patterns across related securities

Applications:
- Insider trading signal generation
- Corporate governance analysis
- Risk assessment for investment decisions
- Regulatory compliance monitoring

References:
- Seyhun, H. N. (1998). Investment Intelligence from Insider Trading.
- Lakonishok, J., & Lee, I. (2001). Are insider trades informative?
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Form 4 transaction types."""
    PURCHASE = "P"
    SALE = "S"
    ACQUISITION = "A"
    DISPOSITION = "D"
    EXERCISE = "M"  # Option exercise
    GRANT = "G"     # Equity grant
    GIFT = "J"      # Gift or inheritance


class InsiderRole(Enum):
    """Insider relationship to company."""
    OFFICER = "officer"
    DIRECTOR = "director"
    TEN_PERCENT_OWNER = "ten_percent_owner"
    OTHER = "other"


class ClusteringMethod(Enum):
    """Clustering algorithms for insider analysis."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    NETWORK = "network"


@dataclass
class Form4Filing:
    """Represents a single Form 4 filing."""
    filing_id: str
    cik: str                    # Company CIK
    ticker: str                 # Stock ticker
    insider_name: str           # Insider name
    insider_cik: str           # Insider CIK
    insider_role: InsiderRole   # Insider role
    filing_date: datetime       # Filing date
    transaction_date: datetime  # Transaction date
    transaction_type: TransactionType
    shares_traded: float        # Number of shares
    price_per_share: float     # Transaction price
    shares_owned_after: float  # Shares owned after transaction
    is_direct_ownership: bool   # Direct vs indirect ownership
    transaction_code: str      # SEC transaction code
    
    @property
    def transaction_value(self) -> float:
        """Calculate total transaction value."""
        return self.shares_traded * self.price_per_share
    
    @property
    def filing_delay_days(self) -> int:
        """Calculate days between transaction and filing."""
        return (self.filing_date - self.transaction_date).days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filing_id': self.filing_id,
            'cik': self.cik,
            'ticker': self.ticker,
            'insider_name': self.insider_name,
            'insider_cik': self.insider_cik,
            'insider_role': self.insider_role.value,
            'filing_date': self.filing_date.isoformat(),
            'transaction_date': self.transaction_date.isoformat(),
            'transaction_type': self.transaction_type.value,
            'shares_traded': self.shares_traded,
            'price_per_share': self.price_per_share,
            'shares_owned_after': self.shares_owned_after,
            'is_direct_ownership': self.is_direct_ownership,
            'transaction_code': self.transaction_code,
            'transaction_value': self.transaction_value,
            'filing_delay_days': self.filing_delay_days
        }


@dataclass
class InsiderProfile:
    """Profile of insider trading behavior."""
    insider_cik: str
    insider_name: str
    total_transactions: int
    total_value_traded: float
    avg_transaction_size: float
    purchase_ratio: float           # Ratio of purchases to total transactions
    avg_filing_delay: float        # Average filing delay in days
    companies_traded: List[str]    # List of company tickers
    preferred_transaction_types: List[TransactionType]
    transaction_frequency: float   # Transactions per month
    volatility_preference: float   # Preference for volatile stocks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'insider_cik': self.insider_cik,
            'insider_name': self.insider_name,
            'total_transactions': self.total_transactions,
            'total_value_traded': self.total_value_traded,
            'avg_transaction_size': self.avg_transaction_size,
            'purchase_ratio': self.purchase_ratio,
            'avg_filing_delay': self.avg_filing_delay,
            'companies_traded': self.companies_traded,
            'preferred_transaction_types': [t.value for t in self.preferred_transaction_types],
            'transaction_frequency': self.transaction_frequency,
            'volatility_preference': self.volatility_preference
        }


@dataclass
class InsiderCluster:
    """Represents a cluster of insiders with similar behavior."""
    cluster_id: int
    cluster_name: str
    insider_ciks: List[str]
    characteristics: Dict[str, float]
    representative_profile: InsiderProfile
    cluster_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster_name,
            'insider_ciks': self.insider_ciks,
            'characteristics': self.characteristics,
            'representative_profile': self.representative_profile.to_dict(),
            'cluster_size': self.cluster_size
        }


@dataclass
class InsiderSignal:
    """Trading signal derived from insider activity."""
    signal_id: str
    ticker: str
    signal_type: str           # "bullish", "bearish", "neutral"
    signal_strength: float     # 0-100 signal strength
    confidence: float          # 0-1 confidence level
    generated_date: datetime
    expiry_date: datetime
    contributing_filings: List[str]  # Filing IDs that contributed
    cluster_analysis: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'signal_id': self.signal_id,
            'ticker': self.ticker,
            'signal_type': self.signal_type,
            'signal_strength': self.signal_strength,
            'confidence': self.confidence,
            'generated_date': self.generated_date.isoformat(),
            'expiry_date': self.expiry_date.isoformat(),
            'contributing_filings': self.contributing_filings,
            'cluster_analysis': self.cluster_analysis,
            'metadata': self.metadata
        }


class Form4Clusterer:
    """
    Form 4 Clustering and Analysis Engine.
    
    Analyzes insider trading patterns using various clustering algorithms
    to identify meaningful groups and generate trading signals.
    """
    
    def __init__(self, 
                 clustering_method: ClusteringMethod = ClusteringMethod.KMEANS,
                 min_transactions: int = 5,
                 lookback_days: int = 365):
        """
        Initialize Form 4 clusterer.
        
        Parameters:
        - clustering_method: Algorithm for clustering insiders
        - min_transactions: Minimum transactions for insider inclusion
        - lookback_days: Lookback period for analysis
        """
        self.clustering_method = clustering_method
        self.min_transactions = min_transactions
        self.lookback_days = lookback_days
        
        # Data storage
        self.filings: List[Form4Filing] = []
        self.insider_profiles: Dict[str, InsiderProfile] = {}
        self.clusters: List[InsiderCluster] = []
        self.signals: List[InsiderSignal] = []
        
        # Feature engineering
        self.scaler = StandardScaler()
        self.feature_names = [
            'total_transactions', 'avg_transaction_size', 'purchase_ratio',
            'avg_filing_delay', 'transaction_frequency', 'volatility_preference',
            'companies_count', 'officer_ratio', 'director_ratio'
        ]
    
    def add_form4_filing(self, filing: Form4Filing):
        """
        Add Form 4 filing for analysis.
        
        Parameters:
        - filing: Form 4 filing data
        """
        self.filings.append(filing)
        logger.debug(f"Added Form 4 filing {filing.filing_id}")
    
    def add_form4_filings_bulk(self, filings: List[Form4Filing]):
        """
        Add multiple Form 4 filings in bulk.
        
        Parameters:
        - filings: List of Form 4 filings
        """
        self.filings.extend(filings)
        logger.info(f"Added {len(filings)} Form 4 filings")
    
    def build_insider_profiles(self, 
                              stock_volatility_data: Optional[Dict[str, float]] = None) -> Dict[str, InsiderProfile]:
        """
        Build profiles for all insiders based on their trading history.
        
        Parameters:
        - stock_volatility_data: Optional volatility data for stocks
        
        Returns:
        - Dictionary of insider profiles by CIK
        """
        logger.info("Building insider profiles from Form 4 filings")
        
        # Group filings by insider
        insider_filings = {}
        for filing in self.filings:
            if filing.insider_cik not in insider_filings:
                insider_filings[filing.insider_cik] = []
            insider_filings[filing.insider_cik].append(filing)
        
        profiles = {}
        
        for insider_cik, filings in insider_filings.items():
            if len(filings) < self.min_transactions:
                continue
            
            # Calculate profile metrics
            total_transactions = len(filings)
            total_value = sum(abs(f.transaction_value) for f in filings)
            avg_transaction_size = total_value / total_transactions
            
            purchases = [f for f in filings if f.transaction_type in [TransactionType.PURCHASE, TransactionType.ACQUISITION]]
            purchase_ratio = len(purchases) / total_transactions
            
            avg_filing_delay = np.mean([f.filing_delay_days for f in filings])
            
            companies = list(set(f.ticker for f in filings))
            
            # Transaction type preferences
            type_counts = {}
            for filing in filings:
                type_counts[filing.transaction_type] = type_counts.get(filing.transaction_type, 0) + 1
            
            preferred_types = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)[:3]
            
            # Transaction frequency (per month)
            date_range = max(f.transaction_date for f in filings) - min(f.transaction_date for f in filings)
            months = max(1, date_range.days / 30)
            transaction_frequency = total_transactions / months
            
            # Volatility preference
            volatility_pref = 0.0
            if stock_volatility_data:
                volatilities = [stock_volatility_data.get(f.ticker, 0.2) for f in filings]
                volatility_pref = np.mean(volatilities)
            
            profile = InsiderProfile(
                insider_cik=insider_cik,
                insider_name=filings[0].insider_name,
                total_transactions=total_transactions,
                total_value_traded=total_value,
                avg_transaction_size=avg_transaction_size,
                purchase_ratio=purchase_ratio,
                avg_filing_delay=avg_filing_delay,
                companies_traded=companies,
                preferred_transaction_types=preferred_types,
                transaction_frequency=transaction_frequency,
                volatility_preference=volatility_pref
            )
            
            profiles[insider_cik] = profile
        
        self.insider_profiles = profiles
        logger.info(f"Built profiles for {len(profiles)} insiders")
        
        return profiles
    
    def extract_features_for_clustering(self) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for clustering analysis.
        
        Returns:
        - Feature matrix and list of insider CIKs
        """
        if not self.insider_profiles:
            raise ValueError("No insider profiles available. Call build_insider_profiles() first.")
        
        features = []
        insider_ciks = []
        
        for insider_cik, profile in self.insider_profiles.items():
            # Calculate additional features
            companies_count = len(profile.companies_traded)
            
            # Role-based features (need to get from filings)
            insider_filings = [f for f in self.filings if f.insider_cik == insider_cik]
            officer_ratio = len([f for f in insider_filings if f.insider_role == InsiderRole.OFFICER]) / len(insider_filings)
            director_ratio = len([f for f in insider_filings if f.insider_role == InsiderRole.DIRECTOR]) / len(insider_filings)
            
            feature_vector = [
                profile.total_transactions,
                profile.avg_transaction_size,
                profile.purchase_ratio,
                profile.avg_filing_delay,
                profile.transaction_frequency,
                profile.volatility_preference,
                companies_count,
                officer_ratio,
                director_ratio
            ]
            
            features.append(feature_vector)
            insider_ciks.append(insider_cik)
        
        features_array = np.array(features)
        
        # Handle missing values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_array)
        
        return features_scaled, insider_ciks
    
    def perform_clustering(self, n_clusters: Optional[int] = None) -> List[InsiderCluster]:
        """
        Perform clustering analysis on insider profiles.
        
        Parameters:
        - n_clusters: Number of clusters (auto-determined if None)
        
        Returns:
        - List of insider clusters
        """
        logger.info(f"Performing clustering using {self.clustering_method.value}")
        
        features, insider_ciks = self.extract_features_for_clustering()
        
        if len(features) < 3:
            logger.warning("Insufficient data for clustering")
            return []
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(features)
        
        # Apply clustering algorithm
        if self.clustering_method == ClusteringMethod.KMEANS:
            cluster_labels = self._kmeans_clustering(features, n_clusters)
        elif self.clustering_method == ClusteringMethod.HIERARCHICAL:
            cluster_labels = self._hierarchical_clustering(features, n_clusters)
        elif self.clustering_method == ClusteringMethod.DBSCAN:
            cluster_labels = self._dbscan_clustering(features)
        elif self.clustering_method == ClusteringMethod.NETWORK:
            cluster_labels = self._network_clustering(features, insider_ciks)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Build cluster objects
        clusters = self._build_cluster_objects(cluster_labels, insider_ciks, features)
        
        self.clusters = clusters
        logger.info(f"Generated {len(clusters)} insider clusters")
        
        return clusters
    
    def _determine_optimal_clusters(self, features: np.ndarray) -> int:
        """Determine optimal number of clusters using elbow method and silhouette analysis."""
        max_clusters = min(10, len(features) // 3)
        
        if max_clusters < 2:
            return 2
        
        silhouette_scores = []
        inertias = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Skip if all points in one cluster
            if len(set(cluster_labels)) < 2:
                continue
            
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        if not silhouette_scores:
            return 2
        
        # Choose k with highest silhouette score
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def _kmeans_clustering(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    
    def _hierarchical_clustering(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform hierarchical clustering."""
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return clustering.fit_predict(features)
    
    def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering."""
        # Auto-tune epsilon using k-distance
        from sklearn.neighbors import NearestNeighbors
        
        k = min(5, len(features) // 2)
        nbrs = NearestNeighbors(n_neighbors=k).fit(features)
        distances, indices = nbrs.kneighbors(features)
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Use 90th percentile as epsilon
        eps = np.percentile(distances, 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=max(2, len(features) // 10))
        return dbscan.fit_predict(features)
    
    def _network_clustering(self, features: np.ndarray, insider_ciks: List[str]) -> np.ndarray:
        """Perform network-based clustering using connection patterns."""
        # Build network based on shared company relationships
        G = nx.Graph()
        
        # Add nodes
        for cik in insider_ciks:
            G.add_node(cik)
        
        # Add edges based on shared companies
        for i, cik1 in enumerate(insider_ciks):
            profile1 = self.insider_profiles[cik1]
            for j, cik2 in enumerate(insider_ciks[i+1:], i+1):
                profile2 = self.insider_profiles[cik2]
                
                # Calculate connection strength based on shared companies
                shared_companies = set(profile1.companies_traded) & set(profile2.companies_traded)
                if shared_companies:
                    weight = len(shared_companies) / max(len(profile1.companies_traded), len(profile2.companies_traded))
                    G.add_edge(cik1, cik2, weight=weight)
        
        # Use community detection
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(G))
            
            # Convert to cluster labels
            cluster_labels = np.zeros(len(insider_ciks))
            for cluster_id, community in enumerate(communities):
                for cik in community:
                    if cik in insider_ciks:
                        idx = insider_ciks.index(cik)
                        cluster_labels[idx] = cluster_id
            
            return cluster_labels
            
        except ImportError:
            logger.warning("NetworkX community detection not available, falling back to K-means")
            return self._kmeans_clustering(features, 3)
    
    def _build_cluster_objects(self, 
                             cluster_labels: np.ndarray, 
                             insider_ciks: List[str],
                             features: np.ndarray) -> List[InsiderCluster]:
        """Build InsiderCluster objects from clustering results."""
        clusters = []
        unique_labels = set(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # DBSCAN noise points
                continue
            
            # Get insiders in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_ciks = [insider_ciks[i] for i in range(len(insider_ciks)) if cluster_mask[i]]
            cluster_features = features[cluster_mask]
            
            if len(cluster_ciks) == 0:
                continue
            
            # Calculate cluster characteristics
            characteristics = {}
            for i, feature_name in enumerate(self.feature_names):
                characteristics[feature_name] = float(np.mean(cluster_features[:, i]))
            
            # Find representative insider (closest to centroid)
            centroid = np.mean(cluster_features, axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            representative_idx = np.argmin(distances)
            representative_cik = cluster_ciks[representative_idx]
            representative_profile = self.insider_profiles[representative_cik]
            
            # Generate cluster name based on characteristics
            cluster_name = self._generate_cluster_name(characteristics)
            
            cluster = InsiderCluster(
                cluster_id=int(cluster_id),
                cluster_name=cluster_name,
                insider_ciks=cluster_ciks,
                characteristics=characteristics,
                representative_profile=representative_profile,
                cluster_size=len(cluster_ciks)
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _generate_cluster_name(self, characteristics: Dict[str, float]) -> str:
        """Generate descriptive name for cluster based on characteristics."""
        # Determine dominant characteristics
        high_transaction_freq = characteristics.get('transaction_frequency', 0) > 2.0
        high_purchase_ratio = characteristics.get('purchase_ratio', 0) > 0.7
        high_transaction_size = characteristics.get('avg_transaction_size', 0) > 100000
        high_volatility_pref = characteristics.get('volatility_preference', 0) > 0.3
        
        if high_transaction_freq and high_purchase_ratio:
            return "Active Buyers"
        elif high_transaction_freq and not high_purchase_ratio:
            return "Active Sellers"
        elif high_transaction_size and high_purchase_ratio:
            return "Large Buyers"
        elif high_transaction_size and not high_purchase_ratio:
            return "Large Sellers"
        elif high_volatility_pref:
            return "Volatility Seekers"
        elif high_purchase_ratio:
            return "Optimistic Insiders"
        else:
            return "Opportunistic Traders"
    
    def generate_signals(self, 
                        lookback_days: int = 30,
                        min_signal_strength: float = 30.0) -> List[InsiderSignal]:
        """
        Generate trading signals based on recent insider activity and clustering.
        
        Parameters:
        - lookback_days: Days to look back for recent activity
        - min_signal_strength: Minimum signal strength to generate
        
        Returns:
        - List of trading signals
        """
        logger.info("Generating trading signals from insider activity")
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_filings = [f for f in self.filings if f.filing_date >= cutoff_date]
        
        if not recent_filings:
            logger.warning("No recent filings for signal generation")
            return []
        
        # Group by ticker
        ticker_filings = {}
        for filing in recent_filings:
            if filing.ticker not in ticker_filings:
                ticker_filings[filing.ticker] = []
            ticker_filings[filing.ticker].append(filing)
        
        signals = []
        
        for ticker, filings in ticker_filings.items():
            signal = self._generate_ticker_signal(ticker, filings, lookback_days)
            
            if signal and signal.signal_strength >= min_signal_strength:
                signals.append(signal)
        
        self.signals = signals
        logger.info(f"Generated {len(signals)} trading signals")
        
        return signals
    
    def _generate_ticker_signal(self, 
                               ticker: str, 
                               filings: List[Form4Filing],
                               lookback_days: int) -> Optional[InsiderSignal]:
        """Generate signal for specific ticker based on filings."""
        if not filings:
            return None
        
        # Calculate basic metrics
        total_value = sum(f.transaction_value for f in filings)
        net_buying = sum(f.transaction_value for f in filings if f.transaction_type in [TransactionType.PURCHASE, TransactionType.ACQUISITION])
        net_selling = sum(abs(f.transaction_value) for f in filings if f.transaction_type in [TransactionType.SALE, TransactionType.DISPOSITION])
        
        net_activity = net_buying - net_selling
        total_activity = abs(net_buying) + abs(net_selling)
        
        if total_activity == 0:
            return None
        
        # Determine signal type and strength
        activity_ratio = net_activity / total_activity
        
        if activity_ratio > 0.3:
            signal_type = "bullish"
            base_strength = min(90, activity_ratio * 100)
        elif activity_ratio < -0.3:
            signal_type = "bearish"
            base_strength = min(90, abs(activity_ratio) * 100)
        else:
            signal_type = "neutral"
            base_strength = 20
        
        # Enhance signal strength based on cluster analysis
        cluster_boost = self._calculate_cluster_signal_boost(filings)
        signal_strength = min(100, base_strength + cluster_boost)
        
        # Calculate confidence based on various factors
        confidence = self._calculate_signal_confidence(filings, total_activity)
        
        # Create cluster analysis summary
        cluster_analysis = self._analyze_filing_clusters(filings)
        
        signal = InsiderSignal(
            signal_id=f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ticker=ticker,
            signal_type=signal_type,
            signal_strength=signal_strength,
            confidence=confidence,
            generated_date=datetime.now(),
            expiry_date=datetime.now() + timedelta(days=30),
            contributing_filings=[f.filing_id for f in filings],
            cluster_analysis=cluster_analysis,
            metadata={
                'total_value': total_activity,
                'net_activity': net_activity,
                'filing_count': len(filings),
                'lookback_days': lookback_days
            }
        )
        
        return signal
    
    def _calculate_cluster_signal_boost(self, filings: List[Form4Filing]) -> float:
        """Calculate signal strength boost based on cluster analysis."""
        if not self.clusters:
            return 0.0
        
        cluster_scores = []
        
        for filing in filings:
            insider_cik = filing.insider_cik
            
            # Find which cluster this insider belongs to
            for cluster in self.clusters:
                if insider_cik in cluster.insider_ciks:
                    # Score based on cluster characteristics
                    if cluster.cluster_name in ["Active Buyers", "Large Buyers", "Optimistic Insiders"]:
                        if filing.transaction_type in [TransactionType.PURCHASE, TransactionType.ACQUISITION]:
                            cluster_scores.append(15.0)
                        else:
                            cluster_scores.append(-5.0)
                    elif cluster.cluster_name in ["Active Sellers", "Large Sellers"]:
                        if filing.transaction_type in [TransactionType.SALE, TransactionType.DISPOSITION]:
                            cluster_scores.append(15.0)
                        else:
                            cluster_scores.append(-5.0)
                    else:
                        cluster_scores.append(5.0)
                    break
        
        return np.mean(cluster_scores) if cluster_scores else 0.0
    
    def _calculate_signal_confidence(self, 
                                   filings: List[Form4Filing], 
                                   total_activity: float) -> float:
        """Calculate confidence level for signal."""
        confidence_factors = []
        
        # Factor 1: Number of filings (more filings = higher confidence)
        filing_factor = min(1.0, len(filings) / 5.0)
        confidence_factors.append(filing_factor)
        
        # Factor 2: Total activity value (higher value = higher confidence)
        value_factor = min(1.0, total_activity / 1000000)  # $1M baseline
        confidence_factors.append(value_factor)
        
        # Factor 3: Insider roles (officers/directors = higher confidence)
        role_factor = len([f for f in filings if f.insider_role in [InsiderRole.OFFICER, InsiderRole.DIRECTOR]]) / len(filings)
        confidence_factors.append(role_factor)
        
        # Factor 4: Filing timing (recent filings = higher confidence)
        avg_delay = np.mean([f.filing_delay_days for f in filings])
        timing_factor = max(0.1, 1.0 - (avg_delay / 10.0))  # Penalize delays > 10 days
        confidence_factors.append(timing_factor)
        
        return np.mean(confidence_factors)
    
    def _analyze_filing_clusters(self, filings: List[Form4Filing]) -> Dict[str, Any]:
        """Analyze cluster distribution of filings."""
        cluster_distribution = {}
        
        if not self.clusters:
            return cluster_distribution
        
        for filing in filings:
            insider_cik = filing.insider_cik
            
            for cluster in self.clusters:
                if insider_cik in cluster.insider_ciks:
                    cluster_name = cluster.cluster_name
                    if cluster_name not in cluster_distribution:
                        cluster_distribution[cluster_name] = 0
                    cluster_distribution[cluster_name] += 1
                    break
        
        return {
            'cluster_distribution': cluster_distribution,
            'dominant_cluster': max(cluster_distribution.keys(), key=cluster_distribution.get) if cluster_distribution else None,
            'cluster_diversity': len(cluster_distribution)
        }
    
    def get_cluster_analytics(self) -> Dict[str, Any]:
        """Get analytics and insights about the clustering results."""
        if not self.clusters:
            return {}
        
        analytics = {
            'total_clusters': len(self.clusters),
            'total_insiders_clustered': sum(c.cluster_size for c in self.clusters),
            'cluster_sizes': [c.cluster_size for c in self.clusters],
            'cluster_names': [c.cluster_name for c in self.clusters],
            'avg_cluster_size': np.mean([c.cluster_size for c in self.clusters]),
            'largest_cluster': max(self.clusters, key=lambda x: x.cluster_size).cluster_name,
            'cluster_characteristics': {c.cluster_name: c.characteristics for c in self.clusters}
        }
        
        return analytics