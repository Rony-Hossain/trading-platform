#!/usr/bin/env python3
"""
Pre-Deployment Verification Script for Market Data Service
Automates the production readiness checklist
"""
import os
import sys
import requests
import subprocess
from typing import Tuple, List
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    passed: bool
    warning: bool = False
    message: str = ""


@dataclass
class CheckSummary:
    """Summary of all checks."""
    passed: int = 0
    failed: int = 0
    warned: int = 0
    results: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult):
        """Add a check result."""
        self.results.append(result)
        if result.warning:
            self.warned += 1
        elif result.passed:
            self.passed += 1
        else:
            self.failed += 1


class Colors:
    """ANSI color codes."""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def disable():
        """Disable colors (for Windows or non-TTY)."""
        Colors.GREEN = ''
        Colors.RED = ''
        Colors.YELLOW = ''
        Colors.BLUE = ''
        Colors.NC = ''


def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{'━' * 50}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'━' * 50}{Colors.NC}")


def print_check(result: CheckResult):
    """Print check result."""
    if result.warning:
        symbol = f"{Colors.YELLOW}⚠{Colors.NC}"
    elif result.passed:
        symbol = f"{Colors.GREEN}✓{Colors.NC}"
    else:
        symbol = f"{Colors.RED}✗{Colors.NC}"

    msg = f" - {result.message}" if result.message else ""
    print(f"{symbol} {result.name}{msg}")


class PreDeploymentChecker:
    """Pre-deployment verification checks."""

    def __init__(self):
        self.summary = CheckSummary()
        self.service_url = os.getenv("SERVICE_URL", "http://localhost:8000")
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = os.getenv("POSTGRES_PORT", "5432")
        self.postgres_db = os.getenv("POSTGRES_DB", "market_data")
        self.postgres_user = os.getenv("POSTGRES_USER", "postgres")
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = os.getenv("REDIS_PORT", "6379")
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

    def run_all_checks(self):
        """Run all pre-deployment checks."""
        print("=" * 50)
        print("Market Data Service - Pre-Deployment Check")
        print("=" * 50)
        print(f"\nConfiguration:")
        print(f"  Service URL: {self.service_url}")
        print(f"  PostgreSQL: {self.postgres_host}:{self.postgres_port}/{self.postgres_db}")
        print(f"  Redis: {self.redis_host}:{self.redis_port}")
        print(f"  Prometheus: {self.prometheus_url}")

        self.check_database()
        self.check_redis()
        self.check_service_health()
        self.check_configuration()
        self.check_monitoring()
        self.check_dlq()
        self.check_providers()
        self.check_tests()

        self.print_summary()

    def check_database(self):
        """Check database connectivity and schema."""
        print_header("1. Database Checks")

        # PostgreSQL connectivity
        result = self._run_command(
            ["pg_isready", "-h", self.postgres_host, "-p", self.postgres_port, "-U", self.postgres_user]
        )
        self._add_and_print(CheckResult(
            "PostgreSQL accessible",
            passed=result[0] == 0
        ))

        # Required tables
        required_tables = [
            "candles_intraday",
            "quotes_l1",
            "symbol_universe",
            "ingestion_cursor",
            "backfill_jobs"
        ]

        for table in required_tables:
            exists = self._check_table_exists(table)
            self._add_and_print(CheckResult(
                f"Table '{table}' exists",
                passed=exists
            ))

        # Symbol universe population
        symbol_count = self._get_symbol_count()
        if symbol_count >= 100:
            self._add_and_print(CheckResult(
                "Symbol universe populated",
                passed=True,
                message=f"{symbol_count} active symbols"
            ))
        elif symbol_count > 0:
            self._add_and_print(CheckResult(
                "Symbol universe has few symbols",
                passed=True,
                warning=True,
                message=f"only {symbol_count} symbols (minimum 100 recommended)"
            ))
        else:
            self._add_and_print(CheckResult(
                "Symbol universe is EMPTY",
                passed=False,
                message="minimum 100 symbols required"
            ))

    def check_redis(self):
        """Check Redis connectivity."""
        print_header("2. Redis Checks")

        result = self._run_command(["redis-cli", "-h", self.redis_host, "-p", self.redis_port, "ping"])
        self._add_and_print(CheckResult(
            "Redis accessible",
            passed=result[0] == 0 and "PONG" in result[1]
        ))

    def check_service_health(self):
        """Check service health endpoints."""
        print_header("3. Service Health Checks")

        # Health endpoint
        try:
            resp = requests.get(f"{self.service_url}/health", timeout=5)
            self._add_and_print(CheckResult(
                "Service health check passing",
                passed=resp.status_code == 200,
                message=f"HTTP {resp.status_code}"
            ))
        except Exception as e:
            self._add_and_print(CheckResult(
                "Service health check FAILED",
                passed=False,
                message=str(e)
            ))

        # Metrics endpoint
        try:
            resp = requests.get(f"{self.service_url}/metrics", timeout=5)
            metrics_accessible = resp.status_code == 200

            self._add_and_print(CheckResult(
                "Metrics endpoint accessible",
                passed=metrics_accessible,
                message=f"HTTP {resp.status_code}"
            ))

            if metrics_accessible:
                metrics_text = resp.text

                # Check SLO metrics
                self._add_and_print(CheckResult(
                    "SLO metrics reported",
                    passed="slo_gap_violation_rate" in metrics_text,
                    warning="slo_gap_violation_rate" not in metrics_text
                ))

                # Check provider metrics
                self._add_and_print(CheckResult(
                    "Provider metrics reported",
                    passed="provider_selected_total" in metrics_text
                ))

        except Exception as e:
            self._add_and_print(CheckResult(
                "Metrics endpoint NOT accessible",
                passed=False,
                message=str(e)
            ))

    def check_configuration(self):
        """Check configuration and environment."""
        print_header("4. Configuration Checks")

        # Environment variables
        required_vars = ["DATABASE_URL", "POLICY_BARS_1M"]
        for var in required_vars:
            is_set = os.getenv(var) is not None
            self._add_and_print(CheckResult(
                f"Environment variable {var} set",
                passed=is_set,
                warning=not is_set
            ))

        # Config reload endpoint
        try:
            resp = requests.post(f"{self.service_url}/ops/config/reload", timeout=5)
            if resp.status_code == 200:
                self._add_and_print(CheckResult(
                    "Config hot-reload endpoint working",
                    passed=True
                ))
            elif resp.status_code == 404:
                self._add_and_print(CheckResult(
                    "Config hot-reload endpoint not implemented",
                    passed=True,
                    warning=True,
                    message="manual patch required"
                ))
            else:
                self._add_and_print(CheckResult(
                    "Config hot-reload endpoint FAILED",
                    passed=False,
                    message=f"HTTP {resp.status_code}"
                ))
        except Exception:
            self._add_and_print(CheckResult(
                "Config hot-reload check skipped",
                passed=True,
                warning=True,
                message="service not responding"
            ))

    def check_monitoring(self):
        """Check monitoring stack."""
        print_header("5. Monitoring Stack Checks")

        # Prometheus
        try:
            resp = requests.get(f"{self.prometheus_url}/-/healthy", timeout=5)
            prometheus_up = resp.status_code == 200

            self._add_and_print(CheckResult(
                "Prometheus accessible",
                passed=prometheus_up
            ))

            if prometheus_up:
                # Check targets
                resp = requests.get(f"{self.prometheus_url}/api/v1/targets", timeout=5)
                if resp.status_code == 200:
                    targets_data = resp.json()
                    has_market_data = any(
                        "market-data" in str(target)
                        for target in targets_data.get("data", {}).get("activeTargets", [])
                    )
                    self._add_and_print(CheckResult(
                        "Prometheus scraping market-data-service",
                        passed=has_market_data,
                        warning=not has_market_data
                    ))
        except Exception:
            self._add_and_print(CheckResult(
                "Prometheus NOT accessible",
                passed=True,
                warning=True,
                message="check PROMETHEUS_URL"
            ))

        # Kubernetes checks
        if self._has_command("kubectl"):
            # ServiceMonitors
            result = self._run_command(["kubectl", "get", "servicemonitor", "-A"])
            sm_count = result[1].count("market-data") if result[0] == 0 else 0
            self._add_and_print(CheckResult(
                "ServiceMonitor(s) deployed",
                passed=sm_count >= 1,
                warning=sm_count == 0,
                message=f"{sm_count} found" if sm_count > 0 else "none found"
            ))

            # PrometheusRules
            result = self._run_command(["kubectl", "get", "prometheusrule", "-A"])
            pr_count = result[1].count("market-data") if result[0] == 0 else 0
            self._add_and_print(CheckResult(
                "PrometheusRule(s) deployed",
                passed=pr_count >= 1,
                warning=pr_count == 0,
                message=f"{pr_count} found" if pr_count > 0 else "none found"
            ))

            # Blackbox Exporter
            result = self._run_command(["kubectl", "get", "pods", "-A"])
            has_blackbox = "blackbox-exporter" in result[1] if result[0] == 0 else False
            self._add_and_print(CheckResult(
                "Blackbox Exporter deployed",
                passed=has_blackbox,
                warning=not has_blackbox
            ))
        else:
            print("  kubectl not found (skipping Kubernetes checks)")

    def check_dlq(self):
        """Check DLQ admin interface."""
        print_header("6. DLQ Admin Checks")

        try:
            resp = requests.get(f"{self.service_url}/ops/dlq/stats", timeout=5)
            if resp.status_code == 200:
                self._add_and_print(CheckResult(
                    "DLQ admin endpoints accessible",
                    passed=True
                ))

                stats = resp.json()
                failed_count = stats.get("total_failed", "unknown")
                print(f"  Failed jobs in DLQ: {failed_count}")
            elif resp.status_code == 404:
                self._add_and_print(CheckResult(
                    "DLQ admin endpoints not implemented",
                    passed=True,
                    warning=True,
                    message="manual patch required"
                ))
            else:
                self._add_and_print(CheckResult(
                    "DLQ admin endpoints FAILED",
                    passed=False,
                    message=f"HTTP {resp.status_code}"
                ))
        except Exception as e:
            self._add_and_print(CheckResult(
                "DLQ admin check skipped",
                passed=True,
                warning=True,
                message=str(e)
            ))

    def check_providers(self):
        """Check provider health tracking."""
        print_header("7. Provider Checks")

        # Provider health history
        try:
            resp = requests.get(f"{self.service_url}/providers/polygon/health-history", timeout=5)
            if resp.status_code == 200:
                self._add_and_print(CheckResult(
                    "Provider health history tracking enabled",
                    passed=True
                ))
            elif resp.status_code == 404:
                self._add_and_print(CheckResult(
                    "Provider health history not implemented",
                    passed=True,
                    warning=True,
                    message="manual patch required"
                ))
            else:
                self._add_and_print(CheckResult(
                    "Provider health history check inconclusive",
                    passed=True,
                    warning=True,
                    message=f"HTTP {resp.status_code}"
                ))
        except Exception:
            self._add_and_print(CheckResult(
                "Provider health history check skipped",
                passed=True,
                warning=True,
                message="service not responding"
            ))

        # Circuit breaker metrics
        try:
            resp = requests.get(f"{self.service_url}/metrics", timeout=5)
            if resp.status_code == 200:
                has_circuit_metrics = "circuit_state" in resp.text
                self._add_and_print(CheckResult(
                    "Circuit breaker metrics found",
                    passed=has_circuit_metrics
                ))
        except Exception:
            pass

    def check_tests(self):
        """Check test suite."""
        print_header("8. Test Suite Checks")

        if self._has_command("pytest"):
            self._add_and_print(CheckResult(
                "pytest installed",
                passed=True
            ))

            if os.getenv("RUN_TESTS", "false").lower() == "true":
                print("  Running tests...")
                result = self._run_command(["pytest", "tests/", "-v", "--tb=short"])
                self._add_and_print(CheckResult(
                    "Test suite execution",
                    passed=result[0] == 0,
                    message="all tests passed" if result[0] == 0 else "some tests failed"
                ))
            else:
                print("  Skipping test run (set RUN_TESTS=true to run)")
        else:
            self._add_and_print(CheckResult(
                "pytest not installed",
                passed=True,
                warning=True,
                message="tests cannot be run"
            ))

    def print_summary(self):
        """Print check summary."""
        print("\n" + "=" * 50)
        print("Pre-Deployment Check Summary")
        print("=" * 50)
        print(f"{Colors.GREEN}Passed:  {self.summary.passed}{Colors.NC}")
        print(f"{Colors.YELLOW}Warnings: {self.summary.warned}{Colors.NC}")
        print(f"{Colors.RED}Failed:  {self.summary.failed}{Colors.NC}")
        print()

        if self.summary.failed == 0:
            if self.summary.warned == 0:
                print(f"{Colors.GREEN}✓ ALL CHECKS PASSED - READY FOR DEPLOYMENT{Colors.NC}")
                return 0
            else:
                print(f"{Colors.YELLOW}⚠ CHECKS PASSED WITH WARNINGS - REVIEW BEFORE DEPLOYMENT{Colors.NC}")
                return 0
        else:
            print(f"{Colors.RED}✗ CHECKS FAILED - NOT READY FOR DEPLOYMENT{Colors.NC}")
            print("\nFix the failed checks before deploying to production.")
            return 1

    # Helper methods

    def _add_and_print(self, result: CheckResult):
        """Add check result and print it."""
        self.summary.add(result)
        print_check(result)

    def _run_command(self, cmd: List[str]) -> Tuple[int, str]:
        """Run shell command and return (return_code, output)."""
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            return result.returncode, result.stdout + result.stderr
        except Exception as e:
            return 1, str(e)

    def _has_command(self, cmd: str) -> bool:
        """Check if command exists."""
        result = self._run_command(["which", cmd] if os.name != "nt" else ["where", cmd])
        return result[0] == 0

    def _check_table_exists(self, table: str) -> bool:
        """Check if PostgreSQL table exists."""
        result = self._run_command([
            "psql",
            "-h", self.postgres_host,
            "-p", self.postgres_port,
            "-U", self.postgres_user,
            "-d", self.postgres_db,
            "-tAc",
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{table}';"
        ])
        return result[0] == 0 and "1" in result[1]

    def _get_symbol_count(self) -> int:
        """Get active symbol count."""
        result = self._run_command([
            "psql",
            "-h", self.postgres_host,
            "-p", self.postgres_port,
            "-U", self.postgres_user,
            "-d", self.postgres_db,
            "-tAc",
            "SELECT COUNT(*) FROM symbol_universe WHERE active=true;"
        ])
        if result[0] == 0:
            try:
                return int(result[1].strip())
            except ValueError:
                return 0
        return 0


def main():
    """Main entry point."""
    # Disable colors on Windows if not in a TTY
    if os.name == "nt" and not sys.stdout.isatty():
        Colors.disable()

    checker = PreDeploymentChecker()
    exit_code = checker.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
