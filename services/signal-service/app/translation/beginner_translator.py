"""
Beginner Translator
Converts technical trading data into plain-English explanations
"""
from typing import Dict, Any, List, Tuple
from enum import Enum
import structlog

from ..core.contracts import ReasonCode

logger = structlog.get_logger(__name__)


class BeginnerTranslator:
    """
    Translates technical trading signals into beginner-friendly language

    Input: Technical data (model predictions, indicators, sentiment scores)
    Output: Plain English with reason codes for machine readability
    """

    # Plain English templates for each reason code
    REASON_TEMPLATES = {
        # Price action reasons
        ReasonCode.SUPPORT_BOUNCE: "Price is holding steady above a recent low point (support level)",
        ReasonCode.RESISTANCE_BREAK: "Price just broke through a recent high point (resistance level)",
        ReasonCode.UPTREND_CONFIRMED: "Price has been moving higher consistently",
        ReasonCode.DOWNTREND_CONFIRMED: "Price has been moving lower consistently",
        ReasonCode.CONSOLIDATION: "Price is moving sideways in a narrow range",

        # Technical indicator reasons
        ReasonCode.OVERSOLD_RSI: "Technical indicator shows the stock may be undervalued",
        ReasonCode.OVERBOUGHT_RSI: "Technical indicator shows the stock may be overvalued",
        ReasonCode.BULLISH_MACD: "Momentum indicator suggests upward price movement",
        ReasonCode.BEARISH_MACD: "Momentum indicator suggests downward price movement",
        ReasonCode.GOLDEN_CROSS: "Short-term average crossed above long-term average (bullish signal)",
        ReasonCode.DEATH_CROSS: "Short-term average crossed below long-term average (bearish signal)",

        # Model/AI reasons
        ReasonCode.HIGH_CONFIDENCE_PREDICTION: "Our AI model has high confidence in this prediction",
        ReasonCode.LOW_CONFIDENCE_PREDICTION: "Our AI model has low confidence in this prediction",
        ReasonCode.POSITIVE_EXPECTED_RETURN: "Model predicts potential price increase",
        ReasonCode.NEGATIVE_EXPECTED_RETURN: "Model predicts potential price decrease",

        # Sentiment reasons
        ReasonCode.POSITIVE_SENTIMENT: "News and social media sentiment is positive",
        ReasonCode.NEGATIVE_SENTIMENT: "News and social media sentiment is negative",
        ReasonCode.ANALYST_UPGRADE: "Analysts recently upgraded their rating",
        ReasonCode.ANALYST_DOWNGRADE: "Analysts recently downgraded their rating",

        # Risk reasons
        ReasonCode.HIGH_VOLATILITY: "Stock price has been swinging more than usual lately",
        ReasonCode.LOW_LIQUIDITY: "This stock doesn't trade as frequently (may be harder to sell)",
        ReasonCode.SECTOR_WEAKNESS: "The sector this stock is in has been underperforming",
        ReasonCode.MARKET_UNCERTAINTY: "Overall market conditions are uncertain right now",

        # Portfolio reasons
        ReasonCode.CONCENTRATION_RISK: "You already own a lot of this stock or sector",
        ReasonCode.DIVERSIFICATION: "This would help spread your investments across different areas",
        ReasonCode.REBALANCING: "This helps rebalance your portfolio to target allocations",

        # Safety reasons
        ReasonCode.STOP_LOSS_TRIGGERED: "Price hit your safety exit point",
        ReasonCode.PROFIT_TARGET_HIT: "Price reached your profit goal",
        ReasonCode.RISK_LIMIT_EXCEEDED: "This trade would exceed your risk limits",

        # Policy reasons
        ReasonCode.QUIET_HOURS: "Market volatility is higher during this time period",
        ReasonCode.FED_DAY: "Federal Reserve announcement today may cause volatility",
        ReasonCode.BEGINNER_RESTRICTED: "This trade doesn't meet beginner safety criteria",
    }

    def __init__(self):
        """Initialize beginner translator"""
        logger.info("beginner_translator_initialized")

    def translate_action(
        self,
        action: str,
        symbol: str,
        prediction_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        portfolio_context: Dict[str, Any]
    ) -> Tuple[str, List[ReasonCode], float]:
        """
        Translate action recommendation into plain English

        Args:
            action: "BUY", "SELL", "HOLD", "AVOID"
            symbol: Ticker symbol
            prediction_data: Model prediction data
            technical_data: Technical indicators
            sentiment_data: Sentiment analysis
            portfolio_context: User portfolio context

        Returns:
            (reason_text, reason_codes, confidence_score)
        """
        reason_codes = []
        contributing_factors = []

        # Analyze prediction data
        if prediction_data:
            pred_codes, pred_factors = self._analyze_prediction(prediction_data)
            reason_codes.extend(pred_codes)
            contributing_factors.extend(pred_factors)

        # Analyze technical indicators
        if technical_data:
            tech_codes, tech_factors = self._analyze_technicals(technical_data, action)
            reason_codes.extend(tech_codes)
            contributing_factors.extend(tech_factors)

        # Analyze sentiment
        if sentiment_data:
            sent_codes, sent_factors = self._analyze_sentiment(sentiment_data)
            reason_codes.extend(sent_codes)
            contributing_factors.extend(sent_factors)

        # Analyze portfolio context
        if portfolio_context:
            port_codes, port_factors = self._analyze_portfolio(portfolio_context)
            reason_codes.extend(port_codes)
            contributing_factors.extend(port_factors)

        # Build plain English explanation
        reason_text = self._build_reason_text(action, symbol, contributing_factors)

        # Calculate overall confidence
        confidence = self._calculate_confidence(prediction_data, reason_codes)

        logger.debug(
            "action_translated",
            symbol=symbol,
            action=action,
            reason_code_count=len(reason_codes),
            confidence=confidence
        )

        return reason_text, reason_codes, confidence

    def _analyze_prediction(self, data: Dict[str, Any]) -> Tuple[List[ReasonCode], List[str]]:
        """Analyze model prediction data"""
        codes = []
        factors = []

        confidence = data.get("confidence", 0)
        expected_return = data.get("expected_return", 0)

        if confidence >= 0.75:
            codes.append(ReasonCode.HIGH_CONFIDENCE_PREDICTION)
            factors.append(self.REASON_TEMPLATES[ReasonCode.HIGH_CONFIDENCE_PREDICTION])
        elif confidence < 0.5:
            codes.append(ReasonCode.LOW_CONFIDENCE_PREDICTION)
            factors.append(self.REASON_TEMPLATES[ReasonCode.LOW_CONFIDENCE_PREDICTION])

        if expected_return > 0.02:  # > 2% expected return
            codes.append(ReasonCode.POSITIVE_EXPECTED_RETURN)
            factors.append(f"{self.REASON_TEMPLATES[ReasonCode.POSITIVE_EXPECTED_RETURN]} (~{expected_return*100:.1f}%)")
        elif expected_return < -0.02:
            codes.append(ReasonCode.NEGATIVE_EXPECTED_RETURN)
            factors.append(f"{self.REASON_TEMPLATES[ReasonCode.NEGATIVE_EXPECTED_RETURN]} (~{abs(expected_return)*100:.1f}%)")

        return codes, factors

    def _analyze_technicals(
        self,
        data: Dict[str, Any],
        action: str
    ) -> Tuple[List[ReasonCode], List[str]]:
        """Analyze technical indicators"""
        codes = []
        factors = []

        indicators = data.get("indicators", {})

        # RSI analysis
        rsi = indicators.get("rsi")
        if rsi:
            if rsi < 30:
                codes.append(ReasonCode.OVERSOLD_RSI)
                factors.append(self.REASON_TEMPLATES[ReasonCode.OVERSOLD_RSI])
            elif rsi > 70:
                codes.append(ReasonCode.OVERBOUGHT_RSI)
                factors.append(self.REASON_TEMPLATES[ReasonCode.OVERBOUGHT_RSI])

        # MACD analysis
        macd = indicators.get("macd", {})
        if macd:
            macd_value = macd.get("value", 0)
            macd_signal = macd.get("signal", 0)
            if macd_value > macd_signal and action == "BUY":
                codes.append(ReasonCode.BULLISH_MACD)
                factors.append(self.REASON_TEMPLATES[ReasonCode.BULLISH_MACD])
            elif macd_value < macd_signal and action == "SELL":
                codes.append(ReasonCode.BEARISH_MACD)
                factors.append(self.REASON_TEMPLATES[ReasonCode.BEARISH_MACD])

        # Trend analysis
        trend = data.get("trend")
        if trend == "bullish" and action == "BUY":
            codes.append(ReasonCode.UPTREND_CONFIRMED)
            factors.append(self.REASON_TEMPLATES[ReasonCode.UPTREND_CONFIRMED])
        elif trend == "bearish" and action == "SELL":
            codes.append(ReasonCode.DOWNTREND_CONFIRMED)
            factors.append(self.REASON_TEMPLATES[ReasonCode.DOWNTREND_CONFIRMED])

        # Support/Resistance
        support_resistance = data.get("support_resistance", {})
        current_price = data.get("current_price", 0)

        support_levels = support_resistance.get("support_levels", [])
        if support_levels and current_price:
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
            if abs(current_price - nearest_support) / current_price < 0.02:  # Within 2%
                codes.append(ReasonCode.SUPPORT_BOUNCE)
                factors.append(self.REASON_TEMPLATES[ReasonCode.SUPPORT_BOUNCE])

        return codes, factors

    def _analyze_sentiment(self, data: Dict[str, Any]) -> Tuple[List[ReasonCode], List[str]]:
        """Analyze sentiment data"""
        codes = []
        factors = []

        overall_score = data.get("overall_score", 0)
        classification = data.get("classification", "neutral")

        if classification == "positive" and overall_score > 0.5:
            codes.append(ReasonCode.POSITIVE_SENTIMENT)
            factors.append(self.REASON_TEMPLATES[ReasonCode.POSITIVE_SENTIMENT])
        elif classification == "negative" and overall_score < -0.5:
            codes.append(ReasonCode.NEGATIVE_SENTIMENT)
            factors.append(self.REASON_TEMPLATES[ReasonCode.NEGATIVE_SENTIMENT])

        return codes, factors

    def _analyze_portfolio(self, data: Dict[str, Any]) -> Tuple[List[ReasonCode], List[str]]:
        """Analyze portfolio context"""
        codes = []
        factors = []

        # Concentration risk
        if data.get("concentration_warning"):
            codes.append(ReasonCode.CONCENTRATION_RISK)
            factors.append(self.REASON_TEMPLATES[ReasonCode.CONCENTRATION_RISK])

        # Diversification benefit
        if data.get("diversification_benefit"):
            codes.append(ReasonCode.DIVERSIFICATION)
            factors.append(self.REASON_TEMPLATES[ReasonCode.DIVERSIFICATION])

        return codes, factors

    def _build_reason_text(
        self,
        action: str,
        symbol: str,
        factors: List[str]
    ) -> str:
        """Build plain English reason text"""
        if not factors:
            return f"Based on our analysis of {symbol}"

        # Take top 3 most important factors
        top_factors = factors[:3]

        if len(top_factors) == 1:
            return top_factors[0]
        elif len(top_factors) == 2:
            return f"{top_factors[0]}, and {top_factors[1].lower()}"
        else:
            return f"{top_factors[0]}, {top_factors[1].lower()}, and {top_factors[2].lower()}"

    def _calculate_confidence(
        self,
        prediction_data: Dict[str, Any],
        reason_codes: List[ReasonCode]
    ) -> float:
        """Calculate overall confidence score"""
        # Start with model confidence
        base_confidence = prediction_data.get("confidence", 0.5) if prediction_data else 0.5

        # Boost confidence if multiple supporting signals
        signal_boost = min(len(reason_codes) * 0.05, 0.2)  # Max 20% boost

        # Penalize if low confidence prediction
        if ReasonCode.LOW_CONFIDENCE_PREDICTION in reason_codes:
            base_confidence *= 0.7

        confidence = min(base_confidence + signal_boost, 1.0)
        return round(confidence, 2)

    def translate_risk_warning(
        self,
        symbol: str,
        risk_data: Dict[str, Any]
    ) -> str:
        """Translate risk data into beginner-friendly warning"""
        warnings = []

        # High volatility
        if risk_data.get("high_volatility"):
            warnings.append(f"{symbol} has been more volatile than usual lately")

        # Low liquidity
        if risk_data.get("low_liquidity"):
            warnings.append("This stock doesn't trade as frequently, so it may be harder to sell quickly")

        # Sector weakness
        if risk_data.get("sector_weakness"):
            sector = risk_data.get("sector", "this sector")
            warnings.append(f"The {sector} sector has been underperforming recently")

        if not warnings:
            return ""

        return ". ".join(warnings) + "."

    def translate_confidence(self, confidence: float) -> str:
        """Translate confidence score to beginner-friendly label"""
        if confidence >= 0.75:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
