#!/usr/bin/env python3
"""
Test script for Market Data Service with Finnhub integration

Usage:
    python test_api.py

Before running:
    1. Get free API key from https://finnhub.io/register
    2. Set environment variable: export FINNHUB_API_KEY=your_key_here
    3. Start the service: uvicorn app.main:app --host 0.0.0.0 --port 8001
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8001"

def test_endpoint(endpoint: str, description: str) -> Dict[Any, Any]:
    """Test an API endpoint and return the result"""
    print(f"\n>> Testing: {description}")
    print(f"   GET {endpoint}")
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   [SUCCESS] {response.status_code}")
            return data
        else:
            print(f"   [FAILED] {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Error: {response.text}")
            return {}
            
    except Exception as e:
        print(f"   [ERROR] Connection Error: {e}")
        return {}

def main():
    """Run comprehensive API tests"""
    print(">> Market Data Service API Test Suite")
    print("=" * 50)
    
    # Test service status
    status_data = test_endpoint("/", "Service Status & Stats")
    
    if status_data:
        providers = status_data.get("stats", {}).get("providers", [])
        print(f"\n>> Available Providers:")
        for provider in providers:
            status = "[OK]" if provider["available"] else "[FAIL]"
            print(f"   {status} {provider['name']}: {provider['available']}")
            if provider["last_error"]:
                print(f"      Error: {provider['last_error']}")
    
    # Test stock price endpoints
    symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA"]
    
    print(f"\n>> Testing Stock Prices")
    print("-" * 30)
    
    for symbol in symbols:
        data = test_endpoint(f"/stocks/{symbol}/price", f"{symbol} Stock Price")
        if data:
            print(f"   >> {symbol}: ${data.get('price', 'N/A')} "
                  f"({data.get('change_percent', 0):+.2f}%) "
                  f"[{data.get('source', 'Unknown')}]")
    
    # Test batch endpoint
    print(f"\n>> Testing Batch Quotes")
    print("-" * 30)
    
    try:
        batch_data = {
            "symbols": ["AAPL", "TSLA", "MSFT"]
        }
        response = requests.post(
            f"{BASE_URL}/stocks/batch",
            json=["AAPL", "TSLA", "MSFT"],
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   [SUCCESS] Batch request successful")
            for result in data.get("results", []):
                if result["status"] == "success":
                    print(f"   >> {result['symbol']}: ${result['price']} "
                          f"({result['change_percent']:+.2f}%)")
        else:
            print(f"   [FAILED] Batch request failed: {response.status_code}")
            
    except Exception as e:
        print(f"   [ERROR] Batch request error: {e}")
    
    # Test historical data
    test_endpoint("/stocks/AAPL/history?period=5d", "Historical Data (5 days)")
    
    # Test Finnhub-specific endpoints (these will work only with valid API key)
    print(f"\n>> Testing Finnhub Premium Features")
    print("-" * 40)
    
    test_endpoint("/stocks/AAPL/profile", "Company Profile (Finnhub)")
    test_endpoint("/stocks/AAPL/sentiment", "News Sentiment (Finnhub)")
    
    # Test health endpoint
    test_endpoint("/health", "Health Check")
    
    print(f"\n>> Test Suite Complete!")
    print("\n>> Next Steps:")
    print("   1. Get Finnhub API key: https://finnhub.io/register")
    print("   2. Set env var: export FINNHUB_API_KEY=your_key")  
    print("   3. Restart service to enable premium features")
    print("   4. Test WebSocket: ws://localhost:8001/ws/AAPL")

if __name__ == "__main__":
    main()