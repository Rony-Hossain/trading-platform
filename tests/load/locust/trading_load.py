"""
Load Testing with Locust

Tests system under load:
- 10x current load capacity
- Graceful degradation
- Latency under load
- Error rates

Usage:
    locust -f trading_load.py --host=http://localhost:8000

Acceptance Criteria:
- System handles 10x current load
- p99 latency < 500ms @ 10x load
- Error rate < 1% @ 10x load
"""
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask
import random
import json
import time
from datetime import datetime


class TradingUser(HttpUser):
    """Simulates a trading system user"""

    wait_time = between(0.1, 0.5)  # 100-500ms between requests

    def on_start(self):
        """Initialize user session"""
        self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]
        self.order_id_counter = 0

    @task(weight=10)
    def get_signals(self):
        """Get latest signals"""
        symbol = random.choice(self.symbols)
        strategy = random.choice(["momentum", "mean_reversion", "pairs"])

        with self.client.get(
            f"/signals/latest",
            params={"symbol": symbol, "strategy": strategy},
            name="/signals/latest",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                response.failure("Signal not found")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=5)
    def submit_order(self):
        """Submit a new order"""
        symbol = random.choice(self.symbols)
        side = random.choice(["buy", "sell"])
        order_type = random.choice(["market", "limit", "ioc"])

        order_data = {
            "symbol": symbol,
            "side": side,
            "quantity": random.randint(10, 500),
            "order_type": order_type,
        }

        if order_type == "limit":
            order_data["limit_price"] = round(random.uniform(100, 200), 2)

        self.order_id_counter += 1

        with self.client.post(
            "/orders",
            json=order_data,
            name="/orders [POST]",
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 429:
                # Rate limited - expected under high load
                response.failure("Rate limited")
                raise RescheduleTask()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=8)
    def get_positions(self):
        """Get current positions"""
        with self.client.get(
            "/positions",
            name="/positions",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=3)
    def get_position_by_symbol(self):
        """Get position for specific symbol"""
        symbol = random.choice(self.symbols)

        with self.client.get(
            f"/positions",
            params={"symbol": symbol},
            name="/positions?symbol=X",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=7)
    def get_market_data(self):
        """Get market data"""
        symbol = random.choice(self.symbols)

        with self.client.get(
            f"/market-data/{symbol}",
            name="/market-data/:symbol",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=2)
    def get_pnl(self):
        """Get P&L"""
        with self.client.get(
            "/pnl",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-31"
            },
            name="/pnl",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=1)
    def cancel_order(self):
        """Cancel an order"""
        order_id = f"ORD-{random.randint(1, 10000)}"

        with self.client.delete(
            f"/orders/{order_id}",
            name="/orders/:id [DELETE]",
            catch_response=True
        ) as response:
            if response.status_code in [200, 204, 404]:
                # 404 is ok - order might not exist
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=4)
    def get_fills(self):
        """Get recent fills"""
        with self.client.get(
            "/fills/recent",
            params={"limit": 10},
            name="/fills/recent",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class HighFrequencyUser(HttpUser):
    """Simulates high-frequency trading user"""

    wait_time = between(0.01, 0.05)  # 10-50ms between requests (very fast)

    def on_start(self):
        """Initialize user session"""
        self.symbols = ["AAPL", "MSFT"]  # Focus on 2 symbols

    @task(weight=20)
    def get_market_data_hft(self):
        """Get market data (high frequency)"""
        symbol = random.choice(self.symbols)

        with self.client.get(
            f"/market-data/{symbol}/l2",
            name="/market-data/:symbol/l2 [HFT]",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=10)
    def submit_order_hft(self):
        """Submit order (high frequency)"""
        symbol = random.choice(self.symbols)

        order_data = {
            "symbol": symbol,
            "side": random.choice(["buy", "sell"]),
            "quantity": random.randint(10, 100),
            "order_type": "ioc",  # Immediate-or-cancel
            "limit_price": round(random.uniform(180, 190), 2)
        }

        with self.client.post(
            "/orders",
            json=order_data,
            name="/orders [POST HFT]",
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 429:
                response.failure("Rate limited (HFT)")
                raise RescheduleTask()
            else:
                response.failure(f"Status: {response.status_code}")


class AnalyticsUser(HttpUser):
    """Simulates analytics/reporting user"""

    wait_time = between(5, 15)  # 5-15 seconds between requests (slow)

    @task(weight=5)
    def get_daily_pnl(self):
        """Get daily P&L summary"""
        with self.client.get(
            "/analytics/pnl/daily",
            name="/analytics/pnl/daily",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=3)
    def get_performance_metrics(self):
        """Get performance metrics"""
        with self.client.get(
            "/analytics/performance",
            name="/analytics/performance",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(weight=2)
    def get_reconciliation(self):
        """Get reconciliation report"""
        with self.client.post(
            "/reconcile",
            params={"reconciliation_date": "2025-01-31"},
            name="/reconcile [POST]",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# Custom event handlers for metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Track request metrics"""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Test start event"""
    print("\n" + "=" * 80)
    print("LOAD TEST STARTED")
    print("=" * 80)
    print(f"Target: {environment.host}")
    print(f"Start time: {datetime.now().isoformat()}")
    print("\nAcceptance Criteria:")
    print("  - System handles 10x current load")
    print("  - p99 latency < 500ms @ 10x load")
    print("  - Error rate < 1% @ 10x load")
    print("=" * 80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Test stop event"""
    stats = environment.stats

    print("\n" + "=" * 80)
    print("LOAD TEST COMPLETED")
    print("=" * 80)
    print(f"End time: {datetime.now().isoformat()}")
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Failure rate: {stats.total.fail_ratio * 100:.2f}%")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Median response time: {stats.total.median_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")
    print("\nAcceptance Criteria Results:")

    # Check acceptance criteria
    p99_latency = stats.total.get_response_time_percentile(0.99)
    error_rate = stats.total.fail_ratio

    if p99_latency < 500:
        print(f"  ✅ p99 latency: {p99_latency:.2f}ms (< 500ms)")
    else:
        print(f"  ❌ p99 latency: {p99_latency:.2f}ms (>= 500ms)")

    if error_rate < 0.01:
        print(f"  ✅ Error rate: {error_rate * 100:.2f}% (< 1%)")
    else:
        print(f"  ❌ Error rate: {error_rate * 100:.2f}% (>= 1%)")

    print("=" * 80 + "\n")


# Load test configurations
class NormalLoad(TradingUser):
    """Normal load (baseline)"""
    weight = 70


class HeavyLoad(TradingUser):
    """Heavy load (10x)"""
    weight = 10


class HFTLoad(HighFrequencyUser):
    """High-frequency trading load"""
    weight = 15


class AnalyticsLoad(AnalyticsUser):
    """Analytics load"""
    weight = 5
