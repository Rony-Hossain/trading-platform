#!/usr/bin/env python3
"""
SLO Checker Script
Validates SLO targets and alerts on violations

Usage:
    python scripts/check_slos.py
    python scripts/check_slos.py --alert  # Send alerts on violations
"""
import sys
import argparse
import requests
from typing import Dict, Any


class Colors:
    """Terminal colors"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def check_slo_status(service_url: str) -> Dict[str, Any]:
    """Fetch SLO status from service"""
    try:
        response = requests.get(f"{service_url}/internal/slo/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to fetch SLO status: {e}{Colors.RESET}")
        sys.exit(1)


def check_error_budget(service_url: str, window_days: int = 30) -> Dict[str, Any]:
    """Fetch error budget details"""
    try:
        response = requests.get(
            f"{service_url}/internal/slo/error-budget",
            params={"window_days": window_days},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to fetch error budget: {e}{Colors.RESET}")
        sys.exit(1)


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_metric(name: str, value: Any, status: str = "info"):
    """Print metric with color based on status"""
    color = Colors.RESET
    if status == "success":
        color = Colors.GREEN
        symbol = "✓"
    elif status == "warning":
        color = Colors.YELLOW
        symbol = "⚠"
    elif status == "error":
        color = Colors.RED
        symbol = "✗"
    else:
        symbol = "•"

    print(f"  {color}{symbol} {name}: {value}{Colors.RESET}")


def validate_slos(slo_status: Dict[str, Any], send_alerts: bool = False) -> bool:
    """
    Validate SLOs and return True if all passing

    Returns:
        True if all SLOs met, False otherwise
    """
    all_passing = True

    print_header("SLO Status Report")

    # Overall status
    overall_status = slo_status.get("overall_status", "unknown")
    if overall_status == "healthy":
        print_metric("Overall Status", overall_status.upper(), "success")
    elif overall_status == "warning":
        print_metric("Overall Status", overall_status.upper(), "warning")
        all_passing = False
    else:
        print_metric("Overall Status", overall_status.upper(), "error")
        all_passing = False

    # Error budget
    print_header("Error Budget (30-day window)")
    error_budget = slo_status.get("error_budget", {})

    availability_target = error_budget.get("availability_target", 0)
    current_availability = error_budget.get("current_availability", 0)
    budget_remaining_pct = error_budget.get("error_budget_remaining_pct", 0)
    budget_remaining_min = error_budget.get("error_budget_minutes_remaining", 0)

    print_metric(
        "Availability Target",
        f"{availability_target * 100:.2f}%",
        "info"
    )

    if current_availability >= availability_target:
        print_metric(
            "Current Availability",
            f"{current_availability * 100:.2f}%",
            "success"
        )
    else:
        print_metric(
            "Current Availability",
            f"{current_availability * 100:.2f}%",
            "error"
        )
        all_passing = False

    if budget_remaining_pct >= 25:
        status = "success"
    elif budget_remaining_pct >= 10:
        status = "warning"
    else:
        status = "error"
        all_passing = False

    print_metric(
        "Error Budget Remaining",
        f"{budget_remaining_pct:.1f}% ({budget_remaining_min:.1f} minutes)",
        status
    )

    # Latency
    print_header("Latency SLOs")
    latency = slo_status.get("latency", {})

    for endpoint, metrics in latency.items():
        print(f"\n{Colors.BOLD}Endpoint: {endpoint}{Colors.RESET}")

        p95 = metrics.get("p95", 0)
        p99 = metrics.get("p99", 0)
        p95_target = metrics.get("p95_target", 0)
        p99_target = metrics.get("p99_target", 0)
        p95_met = metrics.get("p95_met", False)
        p99_met = metrics.get("p99_met", False)

        print_metric(
            "p95 Latency",
            f"{p95:.1f}ms (target: {p95_target}ms)",
            "success" if p95_met else "error"
        )

        print_metric(
            "p99 Latency",
            f"{p99:.1f}ms (target: {p99_target}ms)",
            "success" if p99_met else "error"
        )

        if not p95_met or not p99_met:
            all_passing = False

    # Send alerts if requested
    if send_alerts and not all_passing:
        send_alert_notification(slo_status)

    return all_passing


def send_alert_notification(slo_status: Dict[str, Any]):
    """Send alert notification (Slack, PagerDuty, etc.)"""
    print(f"\n{Colors.YELLOW}⚠ SLO VIOLATION - Alerts would be sent here{Colors.RESET}")
    # TODO: Integrate with actual alerting system
    # Examples:
    # - Slack webhook
    # - PagerDuty API
    # - Email via SendGrid
    # - AWS SNS


def main():
    parser = argparse.ArgumentParser(description="Check Signal Service SLOs")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Service URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Send alerts on SLO violations"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Error budget window in days (default: 30)"
    )

    args = parser.parse_args()

    # Fetch SLO status
    slo_status = check_slo_status(args.url)

    # Validate SLOs
    all_passing = validate_slos(slo_status, send_alerts=args.alert)

    # Print summary
    print_header("Summary")
    if all_passing:
        print(f"{Colors.GREEN}✓ All SLOs are PASSING{Colors.RESET}\n")
        sys.exit(0)
    else:
        print(f"{Colors.RED}✗ Some SLOs are FAILING{Colors.RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
