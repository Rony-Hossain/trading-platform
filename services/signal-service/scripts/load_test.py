"""
Locust Load Test for Signal Service

Usage:
    locust -f scripts/load_test.py --host=http://localhost:8000
"""
from locust import HttpUser, task, between, events
import random
import uuid


class SignalServiceUser(HttpUser):
    """
    Simulates a user interacting with Signal Service
    """
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks

    def on_start(self):
        """Called when a user starts"""
        self.user_id = f"user_{random.randint(1, 1000)}"
        self.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    @task(5)
    def get_plan(self):
        """Get trading plan (most common operation)"""
        symbols = random.sample(self.watchlist, k=random.randint(1, 3))
        watchlist_param = ",".join(symbols)

        self.client.get(
            "/api/v1/plan",
            params={
                "watchlist": watchlist_param,
                "mode": random.choice(["beginner", "expert"])
            },
            headers={
                "X-User-ID": self.user_id
            },
            name="/api/v1/plan"
        )

    @task(3)
    def get_alerts(self):
        """Get alerts"""
        self.client.get(
            "/api/v1/alerts",
            params={"mode": "beginner"},
            headers={
                "X-User-ID": self.user_id
            },
            name="/api/v1/alerts"
        )

    @task(2)
    def get_positions(self):
        """Get positions"""
        self.client.get(
            "/api/v1/positions",
            headers={
                "X-User-ID": self.user_id
            },
            name="/api/v1/positions"
        )

    @task(1)
    def execute_action(self):
        """Execute trading action"""
        symbol = random.choice(self.watchlist)
        action = random.choice(["BUY", "SELL"])

        self.client.post(
            "/api/v1/actions/execute",
            json={
                "symbol": symbol,
                "action": action,
                "shares": random.randint(1, 10),
                "limit_price": random.uniform(100, 200)
            },
            headers={
                "X-User-ID": self.user_id,
                "Idempotency-Key": str(uuid.uuid4()),
                "X-Mode": "beginner"
            },
            name="/api/v1/actions/execute"
        )

    @task(1)
    def explain_recommendation(self):
        """Get explanation (less frequent)"""
        self.client.post(
            "/api/v1/explain",
            json={
                "request_id": f"req_{uuid.uuid4().hex[:12]}",
                "symbol": random.choice(self.watchlist)
            },
            headers={
                "X-User-ID": self.user_id
            },
            name="/api/v1/explain"
        )

    @task(1)
    def health_check(self):
        """Health check"""
        self.client.get("/health", name="/health")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    print("=" * 60)
    print("Signal Service Load Test Starting")
    print(f"Target: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    print("=" * 60)
    print("Signal Service Load Test Complete")
    print("=" * 60)


# Custom scenarios for different load patterns
class BeginnerUser(HttpUser):
    """Simulates beginner user behavior"""
    wait_time = between(3, 10)

    def on_start(self):
        self.user_id = f"beginner_{random.randint(1, 100)}"

    @task
    def beginner_flow(self):
        """Typical beginner flow: check plan -> check alerts -> maybe execute"""
        # Get plan
        self.client.get(
            "/api/v1/plan",
            params={"mode": "beginner"},
            headers={"X-User-ID": self.user_id}
        )

        # Get alerts
        self.client.get(
            "/api/v1/alerts",
            params={"mode": "beginner"},
            headers={"X-User-ID": self.user_id}
        )

        # 30% chance to execute action
        if random.random() < 0.3:
            self.client.post(
                "/api/v1/actions/execute",
                json={
                    "symbol": "AAPL",
                    "action": "BUY",
                    "shares": 5,
                    "limit_price": 175.50
                },
                headers={
                    "X-User-ID": self.user_id,
                    "Idempotency-Key": str(uuid.uuid4()),
                    "X-Mode": "beginner"
                }
            )


class ExpertUser(HttpUser):
    """Simulates expert user behavior"""
    wait_time = between(1, 3)

    def on_start(self):
        self.user_id = f"expert_{random.randint(1, 50)}"
        self.watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD", "META"]

    @task
    def expert_flow(self):
        """Expert flow: get plan -> explain -> execute"""
        # Get plan with larger watchlist
        symbols = random.sample(self.watchlist, k=random.randint(3, 5))
        resp = self.client.get(
            "/api/v1/plan",
            params={
                "watchlist": ",".join(symbols),
                "mode": "expert"
            },
            headers={"X-User-ID": self.user_id}
        )

        # Get explanation for one symbol
        if resp.status_code == 200:
            self.client.post(
                "/api/v1/explain",
                json={
                    "request_id": f"req_{uuid.uuid4().hex[:12]}",
                    "symbol": symbols[0]
                },
                headers={"X-User-ID": self.user_id}
            )
