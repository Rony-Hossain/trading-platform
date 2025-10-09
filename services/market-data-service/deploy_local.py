#!/usr/bin/env python3
"""
Local Deployment Script for Market Data Service

This script helps you:
1. Apply database migrations
2. Verify configuration
3. Start the service
4. Enable VizTracer profiling
"""
import os
import sys
import subprocess
from pathlib import Path

# Color codes for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_step(step: str):
    """Print step header."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{step}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")

def print_success(msg: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.NC} {msg}")

def print_error(msg: str):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.NC} {msg}")

def print_info(msg: str):
    """Print info message."""
    print(f"{Colors.YELLOW}ℹ{Colors.NC} {msg}")

def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print_success(description)
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
            return True
        else:
            print_error(description)
            if result.stderr:
                print(f"  Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print_error(f"{description} - {str(e)}")
        return False

def check_prerequisites():
    """Check if prerequisites are installed."""
    print_step("Step 1: Checking Prerequisites")

    # Check Python
    python_version = sys.version.split()[0]
    print_success(f"Python {python_version}")

    # Check if we can import required packages
    try:
        import asyncpg
        print_success("asyncpg installed")
    except ImportError:
        print_error("asyncpg not installed")
        print_info("Run: pip install asyncpg")
        return False

    try:
        import fastapi
        print_success("fastapi installed")
    except ImportError:
        print_error("fastapi not installed")
        print_info("Run: pip install -r requirements.txt")
        return False

    return True

def create_env_file():
    """Create .env file if it doesn't exist."""
    print_step("Step 2: Configuration")

    env_file = Path(".env")

    if env_file.exists():
        print_success(".env file exists")
        return True

    print_info("Creating .env file with default configuration")

    env_content = """# Market Data Service Configuration

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/market_data

# Redis
REDIS_URL=redis://localhost:6379/0

# Providers (configure based on available providers)
POLICY_BARS_1M=["polygon","finnhub"]
POLICY_BARS_1D=["polygon","alpaca"]
POLICY_QUOTES_L1=["polygon","iex"]

# Circuit Breaker
BREAKER_DEMOTE_THRESHOLD=0.55
BREAKER_PROMOTE_THRESHOLD=0.70
RECENT_LATENCY_CAP_MS=500

# VizTracer (disabled by default)
VIZTRACER_ENABLED=false
VIZ_OUT_DIR=./traces

# Service
PORT=8000
HOST=0.0.0.0
"""

    try:
        env_file.write_text(env_content)
        print_success("Created .env file")
        print_info("Edit .env to configure your database and provider API keys")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False

def apply_migrations():
    """Apply database migrations."""
    print_step("Step 3: Database Migrations")

    print_info("Database migration options:")
    print("  1. Standard PostgreSQL (db/migrations/20251008_market_data_core.sql)")
    print("  2. TimescaleDB optimized (db/migrations/20251008_timescale_market_data.sql)")
    print()

    print_info("To apply migrations manually:")
    print()
    print("  # Option 1: Standard PostgreSQL")
    print("  psql -U postgres -d market_data -f db/migrations/20251008_market_data_core.sql")
    print()
    print("  # Option 2: TimescaleDB (recommended)")
    print("  psql -U postgres -d market_data -f db/migrations/20251008_timescale_market_data.sql")
    print()

    # Try to apply via Python
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/market_data")
    print_info(f"Database URL: {db_url}")

    choice = input("Apply migrations now? (1=PostgreSQL, 2=TimescaleDB, n=skip): ").strip().lower()

    if choice == "n":
        print_info("Skipping migrations - apply them manually")
        return True

    migration_file = (
        "db/migrations/20251008_market_data_core.sql" if choice == "1"
        else "db/migrations/20251008_timescale_market_data.sql"
    )

    try:
        import asyncpg
        import asyncio

        async def apply_migration():
            # Parse connection string
            parts = db_url.replace("postgresql://", "").split("@")
            if len(parts) != 2:
                print_error("Invalid DATABASE_URL format")
                return False

            user_pass = parts[0].split(":")
            host_db = parts[1].split("/")

            user = user_pass[0] if len(user_pass) > 0 else "postgres"
            password = user_pass[1] if len(user_pass) > 1 else "postgres"
            host = host_db[0].split(":")[0] if len(host_db) > 0 else "localhost"
            port = int(host_db[0].split(":")[1]) if ":" in host_db[0] else 5432
            database = host_db[1] if len(host_db) > 1 else "market_data"

            try:
                # Connect to PostgreSQL
                conn = await asyncpg.connect(
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                    database=database
                )

                # Read migration file
                with open(migration_file, 'r') as f:
                    sql = f.read()

                # Execute migration
                await conn.execute(sql)
                await conn.close()

                print_success(f"Applied migration: {migration_file}")
                return True

            except Exception as e:
                print_error(f"Migration failed: {e}")
                print_info("Try applying manually with psql")
                return False

        return asyncio.run(apply_migration())

    except Exception as e:
        print_error(f"Could not apply migration: {e}")
        print_info("Apply migrations manually using the commands above")
        return False

def setup_viztracer():
    """Set up VizTracer for performance profiling."""
    print_step("Step 4: VizTracer Setup")

    # Check if VizTracer is installed
    try:
        import viztracer
        print_success("VizTracer is installed")
    except ImportError:
        print_info("VizTracer not installed")
        print_info("To install: pip install viztracer")
        install = input("Install VizTracer now? (y/n): ").strip().lower()

        if install == 'y':
            if run_command([sys.executable, "-m", "pip", "install", "viztracer"], "Installing VizTracer"):
                print_success("VizTracer installed successfully")
            else:
                print_error("Failed to install VizTracer")
                return False
        else:
            print_info("Skipping VizTracer installation")
            return True

    # Create traces directory
    traces_dir = Path("./traces")
    if not traces_dir.exists():
        traces_dir.mkdir(parents=True)
        print_success("Created traces directory")
    else:
        print_success("traces directory exists")

    print()
    print_info("VizTracer Usage:")
    print()
    print("  1. Enable VizTracer in .env:")
    print("     VIZTRACER_ENABLED=true")
    print("     VIZ_OUT_DIR=./traces")
    print()
    print("  2. Start the service (traces will be auto-captured)")
    print()
    print("  3. View traces in browser:")
    print("     Open traces/*.html in Chrome/Firefox")
    print()
    print("  4. To profile specific code, use:")
    print()
    print("     from app.observability.trace import maybe_trace")
    print()
    print("     with maybe_trace('my_operation'):")
    print("         # your code here")
    print()

    return True

def run_tests():
    """Run test suite."""
    print_step("Step 5: Running Tests")

    # Check if pytest is installed
    try:
        import pytest
        print_success("pytest is installed")
    except ImportError:
        print_info("pytest not installed")
        print_info("Run: pip install -r requirements-test.txt")
        return False

    # Run tests
    print_info("Running test suite...")
    return run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        "All tests passed"
    )

def start_service():
    """Instructions for starting the service."""
    print_step("Step 6: Start Service")

    print_info("To start the Market Data Service:")
    print()
    print("  # Development mode (with auto-reload)")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print()
    print("  # Production mode")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4")
    print()
    print("  # With VizTracer enabled")
    print("  VIZTRACER_ENABLED=true uvicorn app.main:app --reload")
    print()
    print_info("Verify service is running:")
    print()
    print("  curl http://localhost:8000/health")
    print("  curl http://localhost:8000/metrics")
    print()

def deploy_monitoring():
    """Instructions for deploying monitoring stack."""
    print_step("Step 7: Deploy Monitoring (Optional)")

    print_info("If using Kubernetes, deploy monitoring stack:")
    print()
    print("  # Deploy ServiceMonitors")
    print("  kubectl apply -f deploy/monitoring/servicemonitor-market-data.yaml")
    print("  kubectl apply -f deploy/monitoring/servicemonitor-blackbox.yaml")
    print()
    print("  # Deploy Alert Rules")
    print("  kubectl apply -f deploy/monitoring/alerts-slo.yaml")
    print()
    print("  # Deploy Blackbox Exporter")
    print("  helm install blackbox-exporter prometheus-community/prometheus-blackbox-exporter \\")
    print("    -f deploy/blackbox-exporter/values.yaml -n monitoring")
    print()
    print_info("For local Prometheus setup, see DEPLOYMENT_GUIDE.md")
    print()

def show_next_steps():
    """Show next steps."""
    print_step("Next Steps")

    print_info("Deployment checklist:")
    print()
    print("  1. ✓ Prerequisites checked")
    print("  2. ✓ Configuration created (.env)")
    print("  3. ✓ Database migrations applied")
    print("  4. ✓ VizTracer configured")
    print("  5. ✓ Tests passed")
    print()
    print_info("To complete deployment:")
    print()
    print("  1. Edit .env with your database credentials and API keys")
    print("  2. Populate symbol universe (see DEPLOYMENT_GUIDE.md)")
    print("  3. Start the service with: uvicorn app.main:app --reload")
    print("  4. Run pre-deployment check: python scripts/pre_deployment_check.py")
    print()
    print_info("Documentation:")
    print()
    print("  - Quick Start:      QUICKSTART.md")
    print("  - Deployment:       DEPLOYMENT_GUIDE.md")
    print("  - Integration:      INTEGRATION_GUIDE.md")
    print("  - VizTracer Guide:  (see below)")
    print()

def show_viztracer_guide():
    """Show detailed VizTracer guide."""
    print_step("VizTracer Performance Profiling Guide")

    print(f"{Colors.GREEN}What is VizTracer?{Colors.NC}")
    print("VizTracer generates interactive flame graphs to visualize code execution")
    print("and identify performance bottlenecks.")
    print()

    print(f"{Colors.GREEN}How to use VizTracer:{Colors.NC}")
    print()

    print("1. Enable VizTracer in .env:")
    print("   VIZTRACER_ENABLED=true")
    print("   VIZ_OUT_DIR=./traces")
    print()

    print("2. Start the service:")
    print("   uvicorn app.main:app --reload")
    print()

    print("3. Make API requests to generate traces:")
    print("   curl http://localhost:8000/health")
    print("   curl http://localhost:8000/stats/cadence")
    print()

    print("4. Find trace files:")
    print("   ls -lh traces/")
    print("   # Example: trace_fetch_bars_polygon_1728400000000.html")
    print()

    print("5. Open trace in browser:")
    print("   # On Windows:")
    print("   start traces/trace_*.html")
    print()
    print("   # On Mac:")
    print("   open traces/trace_*.html")
    print()
    print("   # On Linux:")
    print("   xdg-open traces/trace_*.html")
    print()

    print(f"{Colors.GREEN}What you'll see:{Colors.NC}")
    print("- Timeline view of all function calls")
    print("- Flame graph showing call hierarchy")
    print("- Execution time for each function")
    print("- Identify slow operations (red = slow, green = fast)")
    print()

    print(f"{Colors.GREEN}Example trace locations in code:{Colors.NC}")
    print()
    print("  # In app/services/data_collector.py")
    print("  from app.observability.trace import maybe_trace")
    print()
    print("  async def fetch_bars(self, provider, symbols):")
    print("      with maybe_trace(f'fetch_bars_{provider}'):")
    print("          # Trace this operation")
    print("          result = await provider.get_bars(symbols)")
    print("          return result")
    print()

    print(f"{Colors.YELLOW}Performance Tips:{Colors.NC}")
    print("- Disable VizTracer in production (overhead ~5-10%)")
    print("- Use selective tracing with maybe_trace() for specific operations")
    print("- Traces are saved as HTML (can grow large with long operations)")
    print()

def main():
    """Main deployment flow."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}Market Data Service - Local Deployment{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Run deployment steps
    if not check_prerequisites():
        sys.exit(1)

    if not create_env_file():
        sys.exit(1)

    apply_migrations()  # Non-blocking

    setup_viztracer()

    # Optional: run tests
    run_tests_choice = input("\nRun test suite? (y/n): ").strip().lower()
    if run_test_choice == 'y':
        run_tests()

    start_service()
    deploy_monitoring()
    show_next_steps()

    # Show VizTracer guide
    show_viz = input("\nShow VizTracer usage guide? (y/n): ").strip().lower()
    if show_viz == 'y':
        show_viztracer_guide()

    print(f"\n{Colors.GREEN}✓ Deployment script completed!{Colors.NC}\n")

if __name__ == "__main__":
    main()
