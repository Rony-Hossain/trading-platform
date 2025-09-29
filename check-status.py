#!/usr/bin/env python3
"""
Quick Status Check for Trading Platform
Checks what's implemented vs what's running
"""

import os
import subprocess
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {'Found' if exists else 'Missing'}")
    return exists

def check_docker_status():
    """Check Docker Desktop status"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Docker Installed: {result.stdout.strip()}")
            
            # Check if Docker daemon is running
            result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
            if result.returncode == 0:
                print("[OK] Docker Daemon: Running")
                return True
            else:
                print("[MISSING] Docker Daemon: Not running (Start Docker Desktop)")
                return False
        else:
            print("[MISSING] Docker: Not installed")
            return False
    except FileNotFoundError:
        print("[MISSING] Docker: Not found in PATH")
        return False

def check_service_implementations():
    """Check which services are implemented"""
    print("\nService Implementation Status")
    print("-" * 40)
    
    implementations = {
        "Market Data Service": "services/market-data-service/app/main.py",
        "Analysis Service": "services/analysis-service/app/main.py", 
        "Sentiment Service": "services/sentiment-service/app/main.py",
        "Fundamentals Service": "services/fundamentals-service/app/main.py",
        "Strategy Service": "services/strategy-service/app/main.py",
        "Portfolio Service": "services/portfolio-service/app/main.py"
    }
    
    implemented_count = 0
    for service, filepath in implementations.items():
        if check_file_exists(filepath, service):
            implemented_count += 1
    
    print(f"\nServices Implemented: {implemented_count}/{len(implementations)}")
    return implemented_count

def check_database_migrations():
    """Check database migrations"""
    print("\nDatabase Migration Status")
    print("-" * 40)
    
    migrations = {
        "Initial Schema": "migrations/001_initial_schema.sql",
        "Intraday Alerts": "migrations/002_intraday_alerts.sql", 
        "TimescaleDB Optimization": "migrations/003_timescaledb_optimization.sql",
        "Sentiment Tables": "migrations/004_sentiment_tables.sql",
        "Earnings Tables": "migrations/005_fundamentals_earnings_tables.sql"
    }
    
    migration_count = 0
    for migration, filepath in migrations.items():
        if check_file_exists(filepath, migration):
            migration_count += 1
    
    print(f"\nMigrations Ready: {migration_count}/{len(migrations)}")
    return migration_count

def check_advanced_features():
    """Check advanced features implementation"""
    print("\nAdvanced Features Implementation")
    print("-" * 40)
    
    features = {
        "TimescaleDB Optimization": "migrations/003_timescaledb_optimization.sql",
        "Sentiment Service (9 platforms)": "services/sentiment-service/app/services/additional_collectors.py",
        "Fundamentals Parsing": "services/fundamentals-service/app/services/sec_parser.py",
        "Strategy Backtesting": "services/strategy-service/app/main.py",
        "Prometheus Monitoring": "monitoring/prometheus/prometheus.yml",
        "Earnings Monitoring": "services/fundamentals-service/app/services/earnings_monitor.py"
    }
    
    feature_count = 0
    for feature, filepath in features.items():
        if check_file_exists(filepath, feature):
            feature_count += 1
    
    print(f"\nAdvanced Features: {feature_count}/{len(features)}")
    return feature_count

def check_documentation():
    """Check documentation status"""
    print("\nDocumentation Status")
    print("-" * 40)
    
    docs = {
        "Project Flow Guide": "documentation/project-flow.md",
        "Session Tracking": "documentation/claude-chat.md", 
        "TODO Status": "documentation/todo-text.txt",
        "Earnings Documentation": "documentation/earnings-monitoring-complete.md",
        "Testing Instructions": "TESTING-INSTRUCTIONS.md",
        "Manual Test Checklist": "manual-testing-checklist.md"
    }
    
    doc_count = 0
    for doc, filepath in docs.items():
        if check_file_exists(filepath, doc):
            doc_count += 1
    
    print(f"\nDocumentation Complete: {doc_count}/{len(docs)}")
    return doc_count

def main():
    """Main status check"""
    print("TRADING PLATFORM - STATUS CHECK")
    print("=" * 50)
    
    # Check Docker
    docker_ready = check_docker_status()
    
    # Check implementations
    services_implemented = check_service_implementations()
    migrations_ready = check_database_migrations()
    features_implemented = check_advanced_features()
    docs_complete = check_documentation()
    
    # Overall status
    print("\n" + "=" * 50)
    print("OVERALL STATUS SUMMARY")
    print("=" * 50)
    
    print(f"\nCore Services: {services_implemented}/6 implemented")
    print(f"Database Migrations: {migrations_ready}/5 ready")
    print(f"Advanced Features: {features_implemented}/6 implemented")
    print(f"Documentation: {docs_complete}/6 complete")
    print(f"Docker Status: {'Ready' if docker_ready else 'Not Ready'}")
    
    # Calculate readiness percentage
    total_items = 6 + 5 + 6 + 6  # services + migrations + features + docs
    completed_items = services_implemented + migrations_ready + features_implemented + docs_complete
    readiness_percent = (completed_items / total_items) * 100
    
    print(f"\nSystem Readiness: {readiness_percent:.1f}%")
    
    if readiness_percent >= 95:
        print("[OK] SYSTEM FULLY READY FOR TESTING")
        if docker_ready:
            print("   Run: docker-compose up -d --build")
            print("   Then: python test-all-services.py")
        else:
            print("   Start Docker Desktop first")
    elif readiness_percent >= 80:
        print("[WARN] SYSTEM MOSTLY READY - Minor items missing")
    else:
        print("[ERROR] SYSTEM NOT READY - Major implementations missing")
    
    # Next steps
    print(f"\nNext Steps:")
    if not docker_ready:
        print("   1. Start Docker Desktop")
    
    if docker_ready and readiness_percent >= 95:
        print("   1. docker-compose up -d --build")
        print("   2. python test-all-services.py")
        print("   3. Open http://localhost:3001 for frontend")
        print("   4. Test API endpoints (see manual-testing-checklist.md)")
    
    # Save status report
    status_report = {
        "timestamp": "2025-09-19",
        "readiness_percent": readiness_percent,
        "docker_ready": docker_ready,
        "services_implemented": f"{services_implemented}/6",
        "migrations_ready": f"{migrations_ready}/5", 
        "features_implemented": f"{features_implemented}/6",
        "docs_complete": f"{docs_complete}/6",
        "status": "READY" if readiness_percent >= 95 and docker_ready else "NOT_READY"
    }
    
    with open("status-report.json", "w") as f:
        json.dump(status_report, f, indent=2)
    
    print(f"\nStatus report saved to: status-report.json")

if __name__ == "__main__":
    main()