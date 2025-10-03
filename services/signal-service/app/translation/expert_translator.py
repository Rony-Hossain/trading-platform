"""
Expert Translator
Provides advanced technical panels and detailed analysis for expert traders
"""
from typing import Dict, Any, List, Optional
import structlog

from ..core.contracts import ReasonCode

logger = structlog.get_logger(__name__)


class ExpertTranslator:
    """
    Translates signals into expert-level technical analysis

    Provides:
    - Technical indicator panels
    - Model confidence breakdowns
    - Risk/reward analysis
    - Sector rotation context
    - Options strategies (future)
    """

    def __init__(self):
        logger.info("expert_translator_initialized")

    def translate_action(
        self,
        symbol: str,
        action: str,
        prediction_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        portfolio_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Translate action recommendation for expert mode

        Returns comprehensive analysis with technical panels
        """
        return {
            "symbol": symbol,
            "action": action,
            "executive_summary": self._build_executive_summary(
                action, prediction_data, technical_data
            ),
            "panels": {
                "technical_indicators": self._build_technical_panel(technical_data),
                "model_analysis": self._build_model_panel(prediction_data),
                "sentiment_analysis": self._build_sentiment_panel(sentiment_data),
                "risk_reward": self._build_risk_reward_panel(
                    prediction_data, technical_data
                ),
                "portfolio_impact": self._build_portfolio_panel(portfolio_context)
            },
            "metadata": {
                "confidence": prediction_data.get("confidence", 0),
                "expected_return": prediction_data.get("expected_return", 0),
                "risk_score": prediction_data.get("risk_score", 0)
            }
        }

    def _build_executive_summary(
        self,
        action: str,
        prediction_data: Dict[str, Any],
        technical_data: Dict[str, Any]
    ) -> str:
        """Build concise executive summary for experts"""
        confidence = prediction_data.get("confidence", 0)
        expected_return = prediction_data.get("expected_return", 0)
        trend = technical_data.get("trend", "neutral")

        summary = f"{action} recommendation with {confidence*100:.0f}% model confidence. "
        summary += f"Expected return: {expected_return*100:+.1f}%. "
        summary += f"Technical trend: {trend}."

        return summary

    def _build_technical_panel(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build technical indicators panel

        Returns detailed breakdown of all technical signals
        """
        indicators = technical_data.get("indicators", {})
        support_resistance = technical_data.get("support_resistance", {})

        # RSI analysis
        rsi = indicators.get("rsi", 50)
        rsi_analysis = self._analyze_rsi(rsi)

        # MACD analysis
        macd = indicators.get("macd", {})
        macd_analysis = self._analyze_macd(macd)

        # Moving averages
        moving_averages = indicators.get("moving_averages", {})
        ma_analysis = self._analyze_moving_averages(moving_averages)

        # Bollinger Bands
        bollinger = indicators.get("bollinger_bands", {})
        bb_analysis = self._analyze_bollinger_bands(
            bollinger,
            technical_data.get("current_price", 0)
        )

        return {
            "rsi": {
                "value": rsi,
                "interpretation": rsi_analysis,
                "signal": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            },
            "macd": {
                "value": macd.get("value", 0),
                "signal": macd.get("signal", 0),
                "histogram": macd.get("histogram", 0),
                "interpretation": macd_analysis,
                "crossover": self._detect_macd_crossover(macd)
            },
            "moving_averages": {
                "sma_20": moving_averages.get("sma_20", 0),
                "sma_50": moving_averages.get("sma_50", 0),
                "ema_12": moving_averages.get("ema_12", 0),
                "interpretation": ma_analysis,
                "golden_cross": self._detect_golden_cross(moving_averages),
                "death_cross": self._detect_death_cross(moving_averages)
            },
            "bollinger_bands": {
                "upper": bollinger.get("upper", 0),
                "middle": bollinger.get("middle", 0),
                "lower": bollinger.get("lower", 0),
                "interpretation": bb_analysis,
                "bandwidth": self._calculate_bb_bandwidth(bollinger)
            },
            "support_resistance": {
                "support_levels": support_resistance.get("support_levels", []),
                "resistance_levels": support_resistance.get("resistance_levels", []),
                "nearest_support": self._find_nearest_level(
                    support_resistance.get("support_levels", []),
                    technical_data.get("current_price", 0),
                    "support"
                ),
                "nearest_resistance": self._find_nearest_level(
                    support_resistance.get("resistance_levels", []),
                    technical_data.get("current_price", 0),
                    "resistance"
                )
            },
            "trend": {
                "direction": technical_data.get("trend", "neutral"),
                "strength": self._calculate_trend_strength(technical_data)
            }
        }

    def _build_model_panel(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build model analysis panel"""
        return {
            "prediction": {
                "action": prediction_data.get("action", "HOLD"),
                "confidence": prediction_data.get("confidence", 0),
                "expected_return": prediction_data.get("expected_return", 0),
                "risk_score": prediction_data.get("risk_score", 0)
            },
            "model_metadata": {
                "version": prediction_data.get("model_version", "unknown"),
                "trained_at": prediction_data.get("trained_at"),
                "features_used": prediction_data.get("features", []),
                "feature_count": len(prediction_data.get("features", []))
            },
            "probability_distribution": {
                "buy": prediction_data.get("prob_buy", 0),
                "hold": prediction_data.get("prob_hold", 0),
                "sell": prediction_data.get("prob_sell", 0)
            },
            "calibration": {
                "is_calibrated": prediction_data.get("is_calibrated", True),
                "calibration_score": prediction_data.get("calibration_score", 0)
            }
        }

    def _build_sentiment_panel(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build sentiment analysis panel"""
        sources = sentiment_data.get("sources", {})

        return {
            "overall": {
                "score": sentiment_data.get("overall_score", 0),
                "classification": sentiment_data.get("classification", "neutral"),
                "confidence": sentiment_data.get("confidence", 0)
            },
            "sources": {
                "news": {
                    "score": sources.get("news", {}).get("score", 0),
                    "article_count": sources.get("news", {}).get("article_count", 0),
                    "trending_topics": sources.get("news", {}).get("topics", [])
                },
                "social": {
                    "score": sources.get("social", {}).get("score", 0),
                    "mention_count": sources.get("social", {}).get("mention_count", 0),
                    "trending_hashtags": sources.get("social", {}).get("hashtags", [])
                },
                "analyst": {
                    "score": sources.get("analyst", {}).get("score", 0),
                    "rating_count": sources.get("analyst", {}).get("rating_count", 0),
                    "recent_upgrades": sources.get("analyst", {}).get("upgrades", 0),
                    "recent_downgrades": sources.get("analyst", {}).get("downgrades", 0)
                }
            },
            "momentum": {
                "direction": sentiment_data.get("momentum", "stable"),
                "is_trending": sentiment_data.get("trending", False)
            }
        }

    def _build_risk_reward_panel(
        self,
        prediction_data: Dict[str, Any],
        technical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build risk/reward analysis panel"""
        expected_return = prediction_data.get("expected_return", 0)
        risk_score = prediction_data.get("risk_score", 0)
        current_price = technical_data.get("current_price", 0)

        # Calculate risk/reward ratio
        potential_upside = expected_return if expected_return > 0 else 0
        potential_downside = abs(expected_return) if expected_return < 0 else risk_score

        risk_reward_ratio = (
            potential_upside / potential_downside if potential_downside > 0 else 0
        )

        return {
            "expected_return": {
                "percentage": expected_return * 100,
                "dollar_value": current_price * expected_return if current_price else 0
            },
            "risk_metrics": {
                "risk_score": risk_score,
                "volatility": technical_data.get("volatility", 0),
                "beta": prediction_data.get("beta", 1.0),
                "max_drawdown": prediction_data.get("max_drawdown", 0)
            },
            "risk_reward_ratio": risk_reward_ratio,
            "sharpe_ratio": prediction_data.get("sharpe_ratio", 0),
            "kelly_criterion": self._calculate_kelly_criterion(
                expected_return,
                risk_score,
                prediction_data.get("confidence", 0)
            ),
            "recommended_position_size": self._calculate_position_size(
                risk_reward_ratio,
                risk_score
            )
        }

    def _build_portfolio_panel(self, portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build portfolio impact panel"""
        return {
            "correlation": {
                "with_existing_positions": portfolio_context.get("correlation", 0),
                "diversification_benefit": portfolio_context.get("diversification_benefit", 0)
            },
            "concentration": {
                "current_sector_allocation": portfolio_context.get("sector_allocation", 0),
                "new_sector_allocation": portfolio_context.get("new_sector_allocation", 0),
                "concentration_warning": portfolio_context.get("concentration_warning", False)
            },
            "portfolio_impact": {
                "expected_portfolio_return": portfolio_context.get("expected_portfolio_return", 0),
                "portfolio_volatility_change": portfolio_context.get("volatility_change", 0),
                "sharpe_ratio_change": portfolio_context.get("sharpe_change", 0)
            }
        }

    # Helper methods for technical analysis
    def _analyze_rsi(self, rsi: float) -> str:
        """Analyze RSI value"""
        if rsi < 30:
            return "Oversold - potential reversal to upside"
        elif rsi > 70:
            return "Overbought - potential reversal to downside"
        elif 40 <= rsi <= 60:
            return "Neutral - no clear directional signal"
        elif rsi < 40:
            return "Bearish momentum - selling pressure"
        else:
            return "Bullish momentum - buying pressure"

    def _analyze_macd(self, macd: Dict[str, Any]) -> str:
        """Analyze MACD signal"""
        value = macd.get("value", 0)
        signal = macd.get("signal", 0)

        if value > signal and value > 0:
            return "Strong bullish signal - MACD above signal and zero line"
        elif value > signal:
            return "Bullish crossover - momentum shifting positive"
        elif value < signal and value < 0:
            return "Strong bearish signal - MACD below signal and zero line"
        elif value < signal:
            return "Bearish crossover - momentum shifting negative"
        else:
            return "Neutral - MACD and signal converging"

    def _analyze_moving_averages(self, ma: Dict[str, Any]) -> str:
        """Analyze moving average configuration"""
        sma_20 = ma.get("sma_20", 0)
        sma_50 = ma.get("sma_50", 0)

        if sma_20 > sma_50:
            return "Bullish configuration - short-term above long-term MA"
        elif sma_20 < sma_50:
            return "Bearish configuration - short-term below long-term MA"
        else:
            return "Neutral - moving averages converging"

    def _analyze_bollinger_bands(self, bb: Dict[str, Any], current_price: float) -> str:
        """Analyze Bollinger Bands position"""
        if not bb or not current_price:
            return "Insufficient data"

        upper = bb.get("upper", 0)
        lower = bb.get("lower", 0)
        middle = bb.get("middle", 0)

        if current_price > upper:
            return "Price above upper band - overbought condition"
        elif current_price < lower:
            return "Price below lower band - oversold condition"
        elif current_price > middle:
            return "Price in upper half - bullish bias"
        else:
            return "Price in lower half - bearish bias"

    def _detect_macd_crossover(self, macd: Dict[str, Any]) -> Optional[str]:
        """Detect MACD crossover"""
        value = macd.get("value", 0)
        signal = macd.get("signal", 0)

        diff = abs(value - signal)
        if diff < 0.1:  # Close to crossover
            if value > signal:
                return "bullish"
            else:
                return "bearish"
        return None

    def _detect_golden_cross(self, ma: Dict[str, Any]) -> bool:
        """Detect golden cross pattern"""
        sma_50 = ma.get("sma_50", 0)
        sma_200 = ma.get("sma_200", 0)
        return sma_50 > sma_200

    def _detect_death_cross(self, ma: Dict[str, Any]) -> bool:
        """Detect death cross pattern"""
        sma_50 = ma.get("sma_50", 0)
        sma_200 = ma.get("sma_200", 0)
        return sma_50 < sma_200

    def _calculate_bb_bandwidth(self, bb: Dict[str, Any]) -> float:
        """Calculate Bollinger Band bandwidth"""
        upper = bb.get("upper", 0)
        lower = bb.get("lower", 0)
        middle = bb.get("middle", 0)

        if middle == 0:
            return 0

        return ((upper - lower) / middle) * 100

    def _find_nearest_level(
        self,
        levels: List[float],
        current_price: float,
        level_type: str
    ) -> Optional[Dict[str, Any]]:
        """Find nearest support or resistance level"""
        if not levels or not current_price:
            return None

        if level_type == "support":
            # Find nearest support below current price
            supports_below = [l for l in levels if l < current_price]
            if supports_below:
                nearest = max(supports_below)
                distance_pct = ((current_price - nearest) / current_price) * 100
                return {"level": nearest, "distance_pct": distance_pct}
        else:  # resistance
            # Find nearest resistance above current price
            resistances_above = [l for l in levels if l > current_price]
            if resistances_above:
                nearest = min(resistances_above)
                distance_pct = ((nearest - current_price) / current_price) * 100
                return {"level": nearest, "distance_pct": distance_pct}

        return None

    def _calculate_trend_strength(self, technical_data: Dict[str, Any]) -> float:
        """Calculate trend strength (0-1)"""
        # Simplified trend strength calculation
        # In production, use ADX or similar indicator
        trend = technical_data.get("trend", "neutral")
        volatility = technical_data.get("volatility", 0)

        if trend == "bullish":
            base_strength = 0.7
        elif trend == "bearish":
            base_strength = 0.7
        else:
            base_strength = 0.3

        # Adjust for volatility (higher volatility = stronger trend)
        strength = base_strength + (volatility * 0.3)
        return min(strength, 1.0)

    def _calculate_kelly_criterion(
        self,
        expected_return: float,
        risk_score: float,
        confidence: float
    ) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if risk_score == 0:
            return 0

        # Kelly % = (probability * expected_return - (1 - probability)) / expected_return
        win_prob = confidence
        loss_prob = 1 - confidence

        kelly_pct = (win_prob * abs(expected_return) - loss_prob) / abs(expected_return) if expected_return != 0 else 0

        # Cap at 25% for safety (fractional Kelly)
        return min(max(kelly_pct * 0.5, 0), 0.25)

    def _calculate_position_size(
        self,
        risk_reward_ratio: float,
        risk_score: float
    ) -> Dict[str, Any]:
        """Calculate recommended position size"""
        # Conservative position sizing
        if risk_reward_ratio >= 3:
            size_pct = 15  # High risk/reward
        elif risk_reward_ratio >= 2:
            size_pct = 10  # Good risk/reward
        elif risk_reward_ratio >= 1:
            size_pct = 5   # Acceptable risk/reward
        else:
            size_pct = 2   # Poor risk/reward

        # Adjust for risk
        if risk_score > 0.7:
            size_pct *= 0.5  # High risk - reduce size
        elif risk_score > 0.5:
            size_pct *= 0.75  # Moderate risk

        return {
            "percentage": round(size_pct, 1),
            "rationale": self._position_size_rationale(risk_reward_ratio, risk_score)
        }

    def _position_size_rationale(self, rrr: float, risk: float) -> str:
        """Explain position sizing rationale"""
        if rrr >= 3 and risk < 0.5:
            return "Strong opportunity with favorable risk/reward and low risk"
        elif rrr >= 2:
            return "Good risk/reward ratio justifies moderate position"
        elif rrr >= 1:
            return "Acceptable risk/reward - conservative sizing recommended"
        else:
            return "Unfavorable risk/reward - minimal position or avoid"
