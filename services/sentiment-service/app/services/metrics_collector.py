"""
Metrics collection for sentiment service monitoring
Provides Prometheus metrics for data collection tracking
"""

import time
import logging
from typing import Dict, Any
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
from datetime import datetime

logger = logging.getLogger(__name__)

class SentimentMetrics:
    """Prometheus metrics for sentiment service"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Collection metrics
        self.posts_collected_total = Counter(
            'sentiment_posts_collected_total',
            'Total number of social media posts collected',
            ['platform', 'symbol'],
            registry=self.registry
        )
        
        self.collection_errors_total = Counter(
            'sentiment_collection_errors_total',
            'Total number of collection errors',
            ['platform', 'error_type'],
            registry=self.registry
        )
        
        self.api_requests_total = Counter(
            'sentiment_api_requests_total',
            'Total number of API requests made',
            ['platform', 'endpoint'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.api_rate_limit_remaining = Gauge(
            'sentiment_api_rate_limit_remaining',
            'Remaining API rate limit',
            ['platform'],
            registry=self.registry
        )
        
        self.api_rate_limit_total = Gauge(
            'sentiment_api_rate_limit_total',
            'Total API rate limit',
            ['platform'],
            registry=self.registry
        )
        
        # Platform health
        self.platform_healthy = Gauge(
            'sentiment_platform_healthy',
            'Platform health status (1=healthy, 0=unhealthy)',
            ['platform'],
            registry=self.registry
        )
        
        # Data quality metrics
        self.sentiment_analysis_duration = Histogram(
            'sentiment_analysis_duration_seconds',
            'Time spent analyzing sentiment',
            ['analyzer_type'],
            registry=self.registry
        )
        
        self.sentiment_confidence_score = Histogram(
            'sentiment_confidence_score',
            'Distribution of sentiment confidence scores',
            ['platform', 'symbol'],
            registry=self.registry
        )
        
        # Business metrics
        self.sentiment_label_count = Counter(
            'sentiment_label_count',
            'Count of sentiment labels assigned',
            ['label', 'platform', 'symbol'],
            registry=self.registry
        )
        
        self.sentiment_average_score = Gauge(
            'sentiment_average_score',
            'Average sentiment score per symbol',
            ['symbol', 'timeframe'],
            registry=self.registry
        )
        
        self.symbol_mentions_total = Counter(
            'sentiment_symbol_mentions_total',
            'Total mentions of each symbol',
            ['symbol', 'platform'],
            registry=self.registry
        )
        
        # Data freshness
        self.last_update_timestamp = Gauge(
            'sentiment_last_update_timestamp',
            'Timestamp of last data update',
            ['platform'],
            registry=self.registry
        )
        
        # Queue metrics
        self.processing_queue_size = Gauge(
            'sentiment_processing_queue_size',
            'Number of items in processing queue',
            ['queue_type'],
            registry=self.registry
        )
        
        # Storage metrics
        self.database_operations_total = Counter(
            'sentiment_database_operations_total',
            'Total database operations',
            ['operation_type', 'table'],
            registry=self.registry
        )
        
        self.database_operation_duration = Histogram(
            'sentiment_database_operation_duration_seconds',
            'Time spent on database operations',
            ['operation_type'],
            registry=self.registry
        )
    
    def record_post_collected(self, platform: str, symbol: str):
        """Record a post collection"""
        self.posts_collected_total.labels(platform=platform, symbol=symbol).inc()
        self.symbol_mentions_total.labels(symbol=symbol, platform=platform).inc()
        self.last_update_timestamp.labels(platform=platform).set(time.time())
    
    def record_collection_error(self, platform: str, error_type: str):
        """Record a collection error"""
        self.collection_errors_total.labels(platform=platform, error_type=error_type).inc()
    
    def record_api_request(self, platform: str, endpoint: str):
        """Record an API request"""
        self.api_requests_total.labels(platform=platform, endpoint=endpoint).inc()
    
    def update_rate_limits(self, platform: str, remaining: int, total: int):
        """Update rate limit metrics"""
        self.api_rate_limit_remaining.labels(platform=platform).set(remaining)
        self.api_rate_limit_total.labels(platform=platform).set(total)
    
    def set_platform_health(self, platform: str, is_healthy: bool):
        """Set platform health status"""
        self.platform_healthy.labels(platform=platform).set(1 if is_healthy else 0)
    
    def record_sentiment_analysis(self, analyzer_type: str, duration: float):
        """Record sentiment analysis timing"""
        self.sentiment_analysis_duration.labels(analyzer_type=analyzer_type).observe(duration)
    
    def record_sentiment_result(self, platform: str, symbol: str, label: str, confidence: float):
        """Record sentiment analysis result"""
        self.sentiment_label_count.labels(label=label, platform=platform, symbol=symbol).inc()
        self.sentiment_confidence_score.labels(platform=platform, symbol=symbol).observe(confidence)
    
    def update_average_sentiment(self, symbol: str, timeframe: str, score: float):
        """Update average sentiment score"""
        self.sentiment_average_score.labels(symbol=symbol, timeframe=timeframe).set(score)
    
    def update_queue_size(self, queue_type: str, size: int):
        """Update processing queue size"""
        self.processing_queue_size.labels(queue_type=queue_type).set(size)
    
    def record_database_operation(self, operation_type: str, table: str, duration: float):
        """Record database operation"""
        self.database_operations_total.labels(operation_type=operation_type, table=table).inc()
        self.database_operation_duration.labels(operation_type=operation_type).observe(duration)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

# Global metrics instance
sentiment_metrics = SentimentMetrics()

class MetricsCollector:
    """Enhanced metrics collector with platform-specific tracking"""
    
    def __init__(self):
        self.metrics = sentiment_metrics
        self.collection_stats = {
            'twitter': {'posts': 0, 'errors': 0, 'last_update': None},
            'reddit': {'posts': 0, 'errors': 0, 'last_update': None},
            'stocktwits': {'posts': 0, 'errors': 0, 'last_update': None},
            'news': {'articles': 0, 'errors': 0, 'last_update': None}
        }
    
    def track_collection_cycle(self, platform: str, symbols: list, success_count: int, error_count: int):
        """Track a complete collection cycle"""
        self.collection_stats[platform]['posts'] += success_count
        self.collection_stats[platform]['errors'] += error_count
        self.collection_stats[platform]['last_update'] = datetime.now()
        
        # Update Prometheus metrics
        for symbol in symbols:
            for _ in range(success_count):
                self.metrics.record_post_collected(platform, symbol)
        
        if error_count > 0:
            self.metrics.record_collection_error(platform, 'collection_failed')
    
    def track_rate_limits(self, platform: str, response_headers: Dict[str, str]):
        """Extract and track rate limits from API response headers"""
        try:
            if platform == 'twitter':
                remaining = int(response_headers.get('x-rate-limit-remaining', 0))
                total = int(response_headers.get('x-rate-limit-limit', 0))
                self.metrics.update_rate_limits(platform, remaining, total)
            
            elif platform == 'reddit':
                remaining = int(response_headers.get('x-ratelimit-remaining', 0))
                total = int(response_headers.get('x-ratelimit-limit', 0))
                self.metrics.update_rate_limits(platform, remaining, total)
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse rate limit headers for {platform}: {e}")
    
    def get_platform_stats(self) -> Dict[str, Any]:
        """Get current platform statistics"""
        stats = {}
        for platform, data in self.collection_stats.items():
            stats[platform] = {
                'total_posts': data['posts'],
                'total_errors': data['errors'],
                'last_update': data['last_update'].isoformat() if data['last_update'] else None,
                'health_status': 'healthy' if data['errors'] / max(data['posts'], 1) < 0.1 else 'degraded'
            }
        return stats
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get collection summary for monitoring dashboard"""
        total_posts = sum(data['posts'] for data in self.collection_stats.values())
        total_errors = sum(data['errors'] for data in self.collection_stats.values())
        
        return {
            'total_posts_collected': total_posts,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_posts, 1),
            'active_platforms': len([p for p, data in self.collection_stats.items() 
                                   if data['last_update'] and 
                                   (datetime.now() - data['last_update']).seconds < 3600]),
            'platform_stats': self.get_platform_stats()
        }

# Global collector instance
metrics_collector = MetricsCollector()