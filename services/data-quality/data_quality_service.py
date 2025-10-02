"""
Data Quality Service
Validates all data pipelines with Great Expectations
Blocks pipeline on critical failures
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import great_expectations as gx
from great_expectations.checkpoint import Checkpoint
from expectations.market_data_suite import create_market_data_suite
from expectations.feature_data_suite import create_feature_data_suite

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation result severity"""
    CRITICAL = "CRITICAL"  # Block pipeline
    WARNING = "WARNING"    # Alert but continue
    INFO = "INFO"          # Log only

class DataQualityStatus(Enum):
    """Overall data quality status"""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

@dataclass
class ValidationResult:
    """Result of a data quality validation"""
    suite_name: str
    timestamp: datetime
    success: bool
    total_expectations: int
    successful_expectations: int
    failed_expectations: int
    success_rate: float
    severity: ValidationSeverity
    failure_details: List[Dict[str, Any]]
    execution_time_ms: float

    @property
    def status(self) -> DataQualityStatus:
        """Determine overall status"""
        if not self.success:
            if self.severity == ValidationSeverity.CRITICAL:
                return DataQualityStatus.FAIL
            else:
                return DataQualityStatus.WARN
        return DataQualityStatus.PASS

class DataQualityService:
    """
    Central data quality validation service

    Responsibilities:
    1. Validate all incoming market data
    2. Validate all generated features
    3. Block pipelines on critical failures
    4. Alert on warnings
    5. Track validation metrics
    """

    def __init__(self, context_root_dir: str = "great_expectations"):
        """
        Initialize data quality service

        Args:
            context_root_dir: Path to GX context root
        """
        self.context = gx.get_context(context_root_dir=context_root_dir)
        self.validation_history = []

        # Create/update expectation suites
        self.market_data_suite = create_market_data_suite(self.context)
        self.feature_data_suite = create_feature_data_suite(self.context)

        # Severity thresholds
        self.critical_threshold = 0.99  # <99% success = CRITICAL
        self.warning_threshold = 0.95   # <95% success = WARNING

        logger.info("Data Quality Service initialized")

    def validate_market_data(
        self,
        data: pd.DataFrame,
        batch_identifier: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate market data quality

        Args:
            data: Market data DataFrame
            batch_identifier: Optional identifier for this batch

        Returns:
            ValidationResult with pass/fail status

        Raises:
            ValueError: If validation fails with CRITICAL severity
        """
        if batch_identifier is None:
            batch_identifier = f"market_data_{datetime.now().isoformat()}"

        logger.info(f"Validating market data batch: {batch_identifier}")

        # Create runtime batch request
        batch_request = gx.core.batch.RuntimeBatchRequest(
            datasource_name="trading_data",
            data_connector_name="default_runtime_data_connector",
            data_asset_name="market_data",
            runtime_parameters={"batch_data": data},
            batch_identifiers={"default_identifier_name": batch_identifier}
        )

        # Run validation
        result = self._run_validation(
            batch_request=batch_request,
            suite_name=self.market_data_suite
        )

        # Block on critical failures
        if result.status == DataQualityStatus.FAIL:
            error_msg = (
                f"Market data validation FAILED (CRITICAL): "
                f"{result.failed_expectations}/{result.total_expectations} "
                f"expectations failed. Pipeline blocked."
            )
            logger.error(error_msg)
            logger.error(f"Failure details: {result.failure_details}")
            raise ValueError(error_msg)

        # Warn on non-critical failures
        if result.status == DataQualityStatus.WARN:
            logger.warning(
                f"Market data validation WARNING: "
                f"{result.failed_expectations}/{result.total_expectations} "
                f"expectations failed. Proceeding with caution."
            )

        return result

    def validate_feature_data(
        self,
        data: pd.DataFrame,
        batch_identifier: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate feature data quality

        Args:
            data: Feature data DataFrame (must include event_timestamp)
            batch_identifier: Optional identifier for this batch

        Returns:
            ValidationResult with pass/fail status

        Raises:
            ValueError: If validation fails with CRITICAL severity
        """
        if batch_identifier is None:
            batch_identifier = f"feature_data_{datetime.now().isoformat()}"

        logger.info(f"Validating feature data batch: {batch_identifier}")

        # Validate PIT compliance first (critical)
        self._validate_pit_compliance(data)

        # Create runtime batch request
        batch_request = gx.core.batch.RuntimeBatchRequest(
            datasource_name="trading_data",
            data_connector_name="default_runtime_data_connector",
            data_asset_name="features",
            runtime_parameters={"batch_data": data},
            batch_identifiers={"default_identifier_name": batch_identifier}
        )

        # Run validation
        result = self._run_validation(
            batch_request=batch_request,
            suite_name=self.feature_data_suite
        )

        # Block on critical failures
        if result.status == DataQualityStatus.FAIL:
            error_msg = (
                f"Feature data validation FAILED (CRITICAL): "
                f"{result.failed_expectations}/{result.total_expectations} "
                f"expectations failed. Pipeline blocked."
            )
            logger.error(error_msg)
            logger.error(f"Failure details: {result.failure_details}")
            raise ValueError(error_msg)

        # Warn on non-critical failures
        if result.status == DataQualityStatus.WARN:
            logger.warning(
                f"Feature data validation WARNING: "
                f"{result.failed_expectations}/{result.total_expectations} "
                f"expectations failed. Proceeding with caution."
            )

        return result

    def _validate_pit_compliance(self, data: pd.DataFrame):
        """
        Validate PIT compliance (critical check)

        Raises:
            ValueError: If any PIT violations detected
        """
        if "event_timestamp" not in data.columns:
            raise ValueError("Feature data must contain 'event_timestamp' column for PIT compliance")

        # Check for future timestamps
        now = datetime.now()
        future_rows = data[data["event_timestamp"] > now]

        if len(future_rows) > 0:
            raise ValueError(
                f"PIT VIOLATION: {len(future_rows)} rows have "
                f"event_timestamp > {now}. Feature leakage detected!"
            )

        # If ingestion_timestamp exists, validate event <= ingestion
        if "ingestion_timestamp" in data.columns:
            leakage_rows = data[data["event_timestamp"] > data["ingestion_timestamp"]]

            if len(leakage_rows) > 0:
                raise ValueError(
                    f"PIT VIOLATION: {len(leakage_rows)} rows have "
                    f"event_timestamp > ingestion_timestamp. Feature leakage detected!"
                )

        logger.info(f"PIT compliance validated: {len(data)} rows, 0 violations")

    def _run_validation(
        self,
        batch_request: gx.core.batch.RuntimeBatchRequest,
        suite_name: str
    ) -> ValidationResult:
        """
        Run validation and parse results

        Args:
            batch_request: GX batch request
            suite_name: Name of expectation suite

        Returns:
            ValidationResult
        """
        import time
        start_time = time.time()

        # Get validator
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )

        # Run validation
        validation_result = validator.validate()

        execution_time_ms = (time.time() - start_time) * 1000

        # Parse results
        success = validation_result.success
        statistics = validation_result.statistics
        total = statistics["evaluated_expectations"]
        successful = statistics["successful_expectations"]
        failed = statistics["unsuccessful_expectations"]
        success_rate = successful / total if total > 0 else 0

        # Extract failure details
        failure_details = []
        for result in validation_result.results:
            if not result.success:
                failure_details.append({
                    "expectation_type": result.expectation_config.expectation_type,
                    "column": result.expectation_config.kwargs.get("column"),
                    "result": result.result
                })

        # Determine severity
        if success_rate < self.critical_threshold:
            severity = ValidationSeverity.CRITICAL
        elif success_rate < self.warning_threshold:
            severity = ValidationSeverity.WARNING
        else:
            severity = ValidationSeverity.INFO

        result = ValidationResult(
            suite_name=suite_name,
            timestamp=datetime.now(),
            success=success,
            total_expectations=total,
            successful_expectations=successful,
            failed_expectations=failed,
            success_rate=success_rate,
            severity=severity,
            failure_details=failure_details,
            execution_time_ms=execution_time_ms
        )

        self.validation_history.append(result)

        logger.info(
            f"Validation complete: {suite_name} - "
            f"{successful}/{total} passed ({success_rate:.1%}) "
            f"in {execution_time_ms:.1f}ms"
        )

        return result

    def get_validation_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get validation statistics for monitoring

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with validation metrics
        """
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        recent = [v for v in self.validation_history if v.timestamp > cutoff]

        if not recent:
            return {"message": "No recent validations"}

        total_validations = len(recent)
        total_failures = sum(1 for v in recent if v.status == DataQualityStatus.FAIL)
        total_warnings = sum(1 for v in recent if v.status == DataQualityStatus.WARN)

        avg_success_rate = sum(v.success_rate for v in recent) / total_validations
        avg_execution_time = sum(v.execution_time_ms for v in recent) / total_validations

        return {
            "window_hours": hours,
            "total_validations": total_validations,
            "total_failures": total_failures,
            "total_warnings": total_warnings,
            "pass_rate": (total_validations - total_failures) / total_validations,
            "avg_success_rate": avg_success_rate,
            "avg_execution_time_ms": avg_execution_time
        }
