"""
Fitness Checker
Validates if picks meet beginner fitness criteria based on reason quality scoring
"""
from typing import Dict, Any, List, Tuple
import structlog

from .contracts import Pick, ReasonCode
from .policy_manager import get_policy_manager

logger = structlog.get_logger(__name__)


class FitnessChecker:
    """
    Checks if trading picks are "fit for beginners"

    Criteria:
    - Reason quality score (based on policy weights)
    - Minimum confidence threshold
    - Required supporting signals
    - Absence of high-risk indicators
    """

    def __init__(self):
        self.policy_manager = get_policy_manager()
        logger.info("fitness_checker_initialized")

    def check_fitness(
        self,
        pick: Pick,
        is_beginner: bool
    ) -> Tuple[bool, float, List[str]]:
        """
        Check if pick meets beginner fitness criteria

        Args:
            pick: Trading pick to validate
            is_beginner: Whether user is in beginner mode

        Returns:
            (is_fit, quality_score, issues) - Whether fit, quality score, list of issues
        """
        if not is_beginner:
            # No fitness checks for non-beginners
            return True, 1.0, []

        issues = []

        # Calculate reason quality score
        quality_score = self._calculate_quality_score(pick.reason_codes)

        # Check minimum quality threshold
        min_quality = self.policy_manager.get("beginner_fitness.min_reason_quality", 0.40)
        if quality_score < min_quality:
            issues.append(f"Reason quality score ({quality_score:.2f}) below threshold ({min_quality})")

        # Check confidence threshold
        confidence_value = self._confidence_to_value(pick.confidence)
        min_confidence = self.policy_manager.get("beginner_fitness.min_confidence", 0.60)
        if confidence_value < min_confidence:
            issues.append(f"Confidence ({pick.confidence}) below threshold ({min_confidence})")

        # Check for required supporting signals
        required_signal_count = self.policy_manager.get("beginner_fitness.min_supporting_signals", 2)
        if len(pick.reason_codes) < required_signal_count:
            issues.append(f"Only {len(pick.reason_codes)} supporting signals (minimum: {required_signal_count})")

        # Check for disqualifying risk factors
        risk_issues = self._check_risk_factors(pick.reason_codes)
        issues.extend(risk_issues)

        is_fit = len(issues) == 0

        if not is_fit:
            logger.warning(
                "fitness_check_failed",
                symbol=pick.symbol,
                action=pick.action,
                quality_score=quality_score,
                issues=issues
            )
        else:
            logger.debug(
                "fitness_check_passed",
                symbol=pick.symbol,
                action=pick.action,
                quality_score=quality_score
            )

        return is_fit, quality_score, issues

    def _calculate_quality_score(self, reason_codes: List[ReasonCode]) -> float:
        """
        Calculate reason quality score based on policy weights

        Higher quality = stronger, more reliable signals
        """
        if not reason_codes:
            return 0.0

        # Get scoring weights from policy
        weights = self.policy_manager.get("reason_scoring.weights", {})

        total_weight = 0.0
        for code in reason_codes:
            # Convert enum to string for lookup
            code_str = code.value if isinstance(code, ReasonCode) else str(code)
            weight = weights.get(code_str, 0.10)  # Default weight: 0.10
            total_weight += weight

        # Normalize by number of signals (prevent gaming by adding many weak signals)
        quality_score = total_weight / max(len(reason_codes), 1)

        return round(quality_score, 2)

    def _confidence_to_value(self, confidence: str) -> float:
        """Convert confidence label to numeric value"""
        mapping = {
            "high": 0.85,
            "medium": 0.65,
            "low": 0.35
        }
        return mapping.get(confidence, 0.50)

    def _check_risk_factors(self, reason_codes: List[ReasonCode]) -> List[str]:
        """Check for disqualifying risk factors"""
        issues = []

        # High-risk reason codes that should block beginners
        high_risk_codes = [
            ReasonCode.HIGH_VOLATILITY,
            ReasonCode.LOW_LIQUIDITY,
            ReasonCode.LOW_CONFIDENCE_PREDICTION,
            ReasonCode.MARKET_UNCERTAINTY
        ]

        for risk_code in high_risk_codes:
            if risk_code in reason_codes:
                issues.append(f"Contains high-risk signal: {risk_code.value}")

        return issues

    def calculate_pick_score(self, pick: Pick) -> Dict[str, Any]:
        """
        Calculate detailed scoring breakdown for a pick

        Returns scoring details for expert mode or debugging
        """
        quality_score = self._calculate_quality_score(pick.reason_codes)
        confidence_value = self._confidence_to_value(pick.confidence)

        # Get individual reason weights
        weights = self.policy_manager.get("reason_scoring.weights", {})
        reason_breakdown = []

        for code in pick.reason_codes:
            code_str = code.value if isinstance(code, ReasonCode) else str(code)
            weight = weights.get(code_str, 0.10)
            reason_breakdown.append({
                "code": code_str,
                "weight": weight,
                "description": code.name if hasattr(code, 'name') else code_str
            })

        return {
            "overall_score": round((quality_score + confidence_value) / 2, 2),
            "quality_score": quality_score,
            "confidence_value": confidence_value,
            "reason_count": len(pick.reason_codes),
            "reason_breakdown": reason_breakdown
        }


# Global fitness checker instance
_fitness_checker: Optional[FitnessChecker] = None


def init_fitness_checker() -> FitnessChecker:
    """Initialize global fitness checker"""
    global _fitness_checker
    _fitness_checker = FitnessChecker()
    logger.info("fitness_checker_initialized")
    return _fitness_checker


def get_fitness_checker() -> FitnessChecker:
    """Get global fitness checker instance"""
    if _fitness_checker is None:
        raise RuntimeError("FitnessChecker not initialized. Call init_fitness_checker() first.")
    return _fitness_checker
