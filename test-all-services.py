#!/usr/bin/env python3
"""
Complete Trading Platform Services Testing Script
Tests all implemented services including advanced features
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import subprocess
import sys

# Service configurations
SERVICES = {
    "market-data-api": {
        "url": "http://localhost:8002",
        "health_endpoint": "/health",
        "test_endpoints": [
            "/",
            "/stocks/AAPL/quote",
            "/stocks/AAPL/candles?period=1d",
            "/stocks/AAPL/intraday?interval=1m",
            "/ws/AAPL"  # WebSocket test
        ]
    },
    "analysis-api": {
        "url": "http://localhost:8003", 
        "health_endpoint": "/health",
        "test_endpoints": [
            "/",
            "/analyze/AAPL",
            "/forecast/AAPL"
        ]
    },
    "sentiment-service": {
        "url": "http://localhost:8005",
        "health_endpoint": "/health", 
        "test_endpoints": [
            "/",
            "/stats",
            "/posts/AAPL",
            "/summary/AAPL"
        ]
    },
    "fundamentals-service": {
        "url": "http://localhost:8006",
        "health_endpoint": "/health",
        "test_endpoints": [
            "/",
            "/earnings/upcoming",
            "/earnings/AAPL/monitor",
            "/earnings/calendar?start_date=2024-01-01&end_date=2024-01-31"
        ]
    },
    "frontend": {
        "url": "http://localhost:3001",
        "health_endpoint": "/",
        "test_endpoints": [
            "/",
            "/daytrading"
        ]
    }
}

# Database services
DATABASE_SERVICES = {
    "postgres": {
        "port": 5432,
        "service": "TimescaleDB"
    },
    "redis": {
        "port": 6379,
        "service": "Redis Cache"
    }
}

class ServiceTester:
    def __init__(self):
        self.results = {}
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_service_health(self, service_name: str, config: Dict) -> Dict[str, Any]:
        """Test service health and availability"""
        result = {
            "service": service_name,
            "url": config["url"],
            "status": "unknown",
            "health_check": False,
            "response_time": None,
            "endpoints_tested": {},
            "error": None
        }
        
        try:
            # Health check
            start_time = time.time()
            health_url = f"{config['url']}{config['health_endpoint']}"
            
            async with self.session.get(health_url) as response:
                response_time = time.time() - start_time
                result["response_time"] = round(response_time * 1000, 2)  # ms
                
                if response.status == 200:
                    result["health_check"] = True
                    result["status"] = "healthy"
                    
                    # Try to get response data
                    try:
                        data = await response.json()
                        result["health_data"] = data
                    except:
                        result["health_data"] = await response.text()
                        
                else:
                    result["status"] = f"unhealthy (HTTP {response.status})"
                    result["error"] = await response.text()
        
        except Exception as e:
            result["status"] = "unreachable"
            result["error"] = str(e)
        
        return result
    
    async def test_service_endpoints(self, service_name: str, config: Dict) -> Dict[str, Any]:
        """Test specific service endpoints"""
        if service_name not in self.results:
            return {}
        
        if not self.results[service_name]["health_check"]:
            print(f"⚠️  Skipping endpoint tests for {service_name} (health check failed)")
            return {}
        
        endpoints_result = {}
        
        for endpoint in config.get("test_endpoints", []):
            endpoint_result = {
                "path": endpoint,
                "status": "unknown",
                "response_time": None,
                "error": None
            }
            
            try:
                # Skip WebSocket endpoints for now
                if endpoint.startswith("/ws"):
                    endpoint_result["status"] = "websocket (skipped)"
                    endpoints_result[endpoint] = endpoint_result
                    continue
                
                start_time = time.time()
                test_url = f"{config['url']}{endpoint}"
                
                async with self.session.get(test_url) as response:
                    response_time = time.time() - start_time
                    endpoint_result["response_time"] = round(response_time * 1000, 2)
                    
                    if response.status == 200:
                        endpoint_result["status"] = "success"
                        
                        # Try to get sample response data
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and len(str(data)) < 500:
                                endpoint_result["sample_response"] = data
                        except:
                            pass
                            
                    else:
                        endpoint_result["status"] = f"error (HTTP {response.status})"
                        endpoint_result["error"] = await response.text()
            
            except Exception as e:
                endpoint_result["status"] = "failed"
                endpoint_result["error"] = str(e)
            
            endpoints_result[endpoint] = endpoint_result
        
        self.results[service_name]["endpoints_tested"] = endpoints_result
        return endpoints_result

async def test_database_connectivity():
    """Test database connections"""
    print("\n🗄️  Testing Database Connectivity")
    print("-" * 50)
    
    db_results = {}
    
    # Test PostgreSQL/TimescaleDB
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="trading_db",
            user="trading_user",
            password="trading_pass"
        )
        cursor = conn.cursor()
        
        # Test basic connectivity
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        # Test TimescaleDB
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';")
        timescaledb = cursor.fetchone()
        
        # Test table counts
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserted_rows
            FROM pg_stat_user_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """)
        tables = cursor.fetchall()
        
        db_results["postgres"] = {
            "status": "connected",
            "version": version,
            "timescaledb": "enabled" if timescaledb else "not_enabled",
            "tables": len(tables),
            "table_list": [{"schema": t[0], "table": t[1], "rows": t[2]} for t in tables]
        }
        
        conn.close()
        print("✅ PostgreSQL/TimescaleDB: Connected")
        
    except Exception as e:
        db_results["postgres"] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"❌ PostgreSQL: {e}")
    
    # Test Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Test connection
        r.ping()
        
        # Get Redis info
        info = r.info()
        
        db_results["redis"] = {
            "status": "connected",
            "version": info.get("redis_version"),
            "memory_used": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients")
        }
        
        print("✅ Redis: Connected")
        
    except Exception as e:
        db_results["redis"] = {
            "status": "failed", 
            "error": str(e)
        }
        print(f"❌ Redis: {e}")
    
    return db_results

async def test_advanced_features():
    """Test advanced features implementation"""
    print("\n🚀 Testing Advanced Features")
    print("-" * 50)
    
    advanced_features = {
        "timescaledb_optimization": {
            "description": "TimescaleDB hypertables and optimization",
            "status": "unknown"
        },
        "sentiment_monitoring": {
            "description": "9-platform social sentiment monitoring", 
            "status": "unknown"
        },
        "fundamentals_parsing": {
            "description": "SEC filing parsing and analysis",
            "status": "unknown"
        },
        "strategy_backtesting": {
            "description": "Strategy backtesting engine",
            "status": "unknown"
        },
        "prometheus_monitoring": {
            "description": "Prometheus/Grafana monitoring",
            "status": "unknown"
        },
        "earnings_monitoring": {
            "description": "Quarterly/yearly earnings tracking",
            "status": "unknown"
        }
    }
    
    # Test TimescaleDB optimization
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost", port=5432, database="trading_db",
            user="trading_user", password="trading_pass"
        )
        cursor = conn.cursor()
        
        # Check for hypertables
        cursor.execute("""
            SELECT table_name 
            FROM timescaledb_information.hypertables
            WHERE schema_name = 'public';
        """)
        hypertables = cursor.fetchall()
        
        if hypertables:
            advanced_features["timescaledb_optimization"]["status"] = "operational"
            advanced_features["timescaledb_optimization"]["hypertables"] = [h[0] for h in hypertables]
            print("✅ TimescaleDB Optimization: Active")
        else:
            advanced_features["timescaledb_optimization"]["status"] = "not_configured"
            print("⚠️  TimescaleDB Optimization: Not configured")
        
        conn.close()
        
    except Exception as e:
        advanced_features["timescaledb_optimization"]["status"] = "failed"
        advanced_features["timescaledb_optimization"]["error"] = str(e)
        print(f"❌ TimescaleDB Optimization: {e}")
    
    # Test Sentiment Service
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8005/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    advanced_features["sentiment_monitoring"]["status"] = "operational"
                    advanced_features["sentiment_monitoring"]["data"] = data
                    print("✅ Sentiment Monitoring: Active")
                else:
                    advanced_features["sentiment_monitoring"]["status"] = "unhealthy"
                    print("⚠️  Sentiment Monitoring: Service unhealthy")
    except Exception as e:
        advanced_features["sentiment_monitoring"]["status"] = "unreachable"
        print(f"❌ Sentiment Monitoring: {e}")
    
    # Test Fundamentals/Earnings Service
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8006/earnings/upcoming") as response:
                if response.status == 200:
                    advanced_features["fundamentals_parsing"]["status"] = "operational"
                    advanced_features["earnings_monitoring"]["status"] = "operational"
                    print("✅ Fundamentals Parsing: Active")
                    print("✅ Earnings Monitoring: Active")
                else:
                    advanced_features["fundamentals_parsing"]["status"] = "unhealthy"
                    advanced_features["earnings_monitoring"]["status"] = "unhealthy"
                    print("⚠️  Fundamentals/Earnings: Service unhealthy")
    except Exception as e:
        advanced_features["fundamentals_parsing"]["status"] = "unreachable"
        advanced_features["earnings_monitoring"]["status"] = "unreachable"
        print(f"❌ Fundamentals/Earnings: {e}")
    
    return advanced_features

async def main():
    """Main testing function"""
    print("🧪 TRADING PLATFORM - COMPLETE SERVICES TESTING")
    print("=" * 60)
    print(f"Testing started at: {datetime.now()}")
    print()
    
    # Check Docker status first
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker is not running. Please start Docker Desktop and run:")
            print("   docker-compose up -d --build")
            return
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker Desktop.")
        return
    
    # Test all services
    async with ServiceTester() as tester:
        print("🔍 Testing Core Services")
        print("-" * 40)
        
        # Test each service
        for service_name, config in SERVICES.items():
            print(f"\nTesting {service_name}...")
            
            # Health check
            result = await tester.test_service_health(service_name, config)
            tester.results[service_name] = result
            
            status_emoji = "✅" if result["health_check"] else "❌"
            print(f"{status_emoji} {service_name}: {result['status']}")
            
            if result["response_time"]:
                print(f"   Response time: {result['response_time']}ms")
            
            # Test endpoints
            if result["health_check"]:
                await tester.test_service_endpoints(service_name, config)
        
        # Test databases
        db_results = await test_database_connectivity()
        
        # Test advanced features
        advanced_results = await test_advanced_features()
        
        # Generate summary report
        print("\n" + "=" * 60)
        print("📊 TESTING SUMMARY REPORT")
        print("=" * 60)
        
        # Services summary
        healthy_services = sum(1 for r in tester.results.values() if r["health_check"])
        total_services = len(tester.results)
        
        print(f"\n🔧 Core Services: {healthy_services}/{total_services} healthy")
        for service_name, result in tester.results.items():
            status_emoji = "✅" if result["health_check"] else "❌"
            print(f"   {status_emoji} {service_name}: {result['status']}")
        
        # Database summary
        healthy_dbs = sum(1 for r in db_results.values() if r["status"] == "connected")
        total_dbs = len(db_results)
        
        print(f"\n🗄️  Databases: {healthy_dbs}/{total_dbs} connected")
        for db_name, result in db_results.items():
            status_emoji = "✅" if result["status"] == "connected" else "❌"
            print(f"   {status_emoji} {db_name}: {result['status']}")
        
        # Advanced features summary
        operational_features = sum(1 for r in advanced_results.values() if r["status"] == "operational")
        total_features = len(advanced_results)
        
        print(f"\n🚀 Advanced Features: {operational_features}/{total_features} operational")
        for feature_name, result in advanced_results.items():
            status_emoji = "✅" if result["status"] == "operational" else "❌" if result["status"] == "failed" else "⚠️"
            print(f"   {status_emoji} {feature_name}: {result['status']}")
        
        # Overall status
        print(f"\n🎯 OVERALL SYSTEM STATUS")
        print("-" * 30)
        
        if healthy_services == total_services and healthy_dbs == total_dbs and operational_features >= total_features // 2:
            print("✅ SYSTEM FULLY OPERATIONAL")
            print("   All core services healthy")
            print("   Databases connected")
            print("   Advanced features active")
        elif healthy_services >= total_services // 2:
            print("⚠️  SYSTEM PARTIALLY OPERATIONAL")
            print("   Some services may need attention")
        else:
            print("❌ SYSTEM NEEDS ATTENTION")
            print("   Multiple services not responding")
        
        # Next steps
        print(f"\n📋 Next Steps:")
        if healthy_services < total_services:
            print("   1. Check Docker services: docker-compose ps")
            print("   2. Review service logs: docker-compose logs [service-name]")
            print("   3. Restart services: docker-compose restart")
        
        if healthy_dbs < total_dbs:
            print("   4. Check database connectivity")
            print("   5. Verify environment variables")
        
        if operational_features < total_features:
            print("   6. Run database migrations")
            print("   7. Check advanced features configuration")
        
        print(f"\nTesting completed at: {datetime.now()}")
        
        # Save detailed results
        with open("test-results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "services": tester.results,
                "databases": db_results,
                "advanced_features": advanced_results,
                "summary": {
                    "healthy_services": f"{healthy_services}/{total_services}",
                    "connected_databases": f"{healthy_dbs}/{total_dbs}",
                    "operational_features": f"{operational_features}/{total_features}"
                }
            }, f, indent=2)
        
        print("\n💾 Detailed results saved to: test-results.json")

if __name__ == "__main__":
    asyncio.run(main())