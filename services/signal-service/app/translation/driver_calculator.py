"""
Driver Calculator
Calculates top feature importances and drivers behind model predictions
"""
from typing import Dict, Any, List, Tuple
import structlog

logger = structlog.get_logger(__name__)


class DriverCalculator:
    """
    Calculates and explains the top drivers behind trading recommendations

    Provides:
    - Feature importance rankings
    - SHAP-like contribution analysis
    - Human-readable explanations
    - Sensitivity analysis
    """

    def __init__(self):
        # Feature importance weights (would come from model in production)
        self.feature_weights = {
            # Price action features
            "price_momentum_5d": 0.12,
            "price_momentum_20d": 0.10,
            "price_volatility": 0.08,
            "price_vs_sma50": 0.09,
            "price_vs_sma200": 0.07,

            # Volume features
            "volume_trend": 0.11,
            "volume_spike": 0.06,
            "relative_volume": 0.05,

            # Technical indicators
            "rsi": 0.09,
            "macd_signal": 0.08,
            "bollinger_position": 0.06,

            # Fundamental features
            "earnings_surprise": 0.10,
            "revenue_growth": 0.08,
            "profit_margin": 0.06,

            # Sentiment features
            "news_sentiment": 0.07,
            "social_sentiment": 0.05,
            "analyst_consensus": 0.08,

            # Market context
            "sector_momentum": 0.06,
            "market_correlation": 0.04,
            "relative_strength": 0.07
        }

        logger.info("driver_calculator_initialized")

    def calculate_top_drivers(
        self,
        prediction_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Calculate top N drivers for the prediction

        Returns list of drivers with:
        - feature_name
        - contribution_score
        - direction (positive/negative)
        - explanation
        """
        # Extract feature values
        feature_values = self._extract_feature_values(
            prediction_data,
            technical_data,
            sentiment_data
        )

        # Calculate contributions
        contributions = self._calculate_contributions(feature_values)

        # Sort by absolute contribution
        sorted_drivers = sorted(
            contributions,
            key=lambda x: abs(x["contribution_score"]),
            reverse=True
        )

        # Return top N with explanations
        top_drivers = sorted_drivers[:top_n]

        for driver in top_drivers:
            driver["explanation"] = self._explain_driver(
                driver["feature_name"],
                driver["contribution_score"],
                feature_values.get(driver["feature_name"], {})
            )

        logger.debug(
            "top_drivers_calculated",
            top_n=top_n,
            drivers=[d["feature_name"] for d in top_drivers]
        )

        return top_drivers

    def _extract_feature_values(
        self,
        prediction_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract and normalize feature values"""
        features = {}

        # Price action features
        current_price = technical_data.get("current_price", 0)
        indicators = technical_data.get("indicators", {})
        ma = indicators.get("moving_averages", {})

        if current_price and ma.get("sma_50"):
            price_vs_sma50 = (current_price - ma["sma_50"]) / ma["sma_50"]
            features["price_vs_sma50"] = {
                "value": price_vs_sma50,
                "normalized": self._normalize_value(price_vs_sma50, -0.1, 0.1)
            }

        if current_price and ma.get("sma_200"):
            price_vs_sma200 = (current_price - ma["sma_200"]) / ma["sma_200"]
            features["price_vs_sma200"] = {
                "value": price_vs_sma200,
                "normalized": self._normalize_value(price_vs_sma200, -0.2, 0.2)
            }

        # Technical indicators
        rsi = indicators.get("rsi", 50)
        features["rsi"] = {
            "value": rsi,
            "normalized": self._normalize_value(rsi, 30, 70, center=50)
        }

        macd = indicators.get("macd", {})
        if macd:
            macd_signal = macd.get("value", 0) - macd.get("signal", 0)
            features["macd_signal"] = {
                "value": macd_signal,
                "normalized": self._normalize_value(macd_signal, -2, 2)
            }

        # Sentiment features
        news_score = sentiment_data.get("sources", {}).get("news", {}).get("score", 0)
        features["news_sentiment"] = {
            "value": news_score,
            "normalized": self._normalize_value(news_score, -1, 1)
        }

        social_score = sentiment_data.get("sources", {}).get("social", {}).get("score", 0)
        features["social_sentiment"] = {
            "value": social_score,
            "normalized": self._normalize_value(social_score, -1, 1)
        }

        analyst_score = sentiment_data.get("sources", {}).get("analyst", {}).get("score", 0)
        features["analyst_consensus"] = {
            "value": analyst_score,
            "normalized": self._normalize_value(analyst_score, -1, 1)
        }

        # Volume features
        features["volume_trend"] = {
            "value": technical_data.get("volume_trend", 0),
            "normalized": self._normalize_value(technical_data.get("volume_trend", 0), -0.5, 0.5)
        }

        # Volatility
        volatility = technical_data.get("volatility", 0)
        features["price_volatility"] = {
            "value": volatility,
            "normalized": self._normalize_value(volatility, 0, 0.1)
        }

        return features

    def _calculate_contributions(
        self,
        feature_values: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate contribution score for each feature"""
        contributions = []

        for feature_name, value_data in feature_values.items():
            # Get feature weight (importance)
            weight = self.feature_weights.get(feature_name, 0.05)

            # Get normalized value
            normalized_value = value_data.get("normalized", 0)

            # Contribution = weight * normalized_value
            contribution_score = weight * normalized_value

            # Determine direction
            direction = "positive" if contribution_score > 0 else "negative" if contribution_score < 0 else "neutral"

            contributions.append({
                "feature_name": feature_name,
                "contribution_score": contribution_score,
                "direction": direction,
                "weight": weight,
                "raw_value": value_data.get("value", 0),
                "normalized_value": normalized_value
            })

        return contributions

    def _normalize_value(
        self,
        value: float,
        min_val: float,
        max_val: float,
        center: Optional[float] = None
    ) -> float:
        """
        Normalize value to [-1, 1] range

        If center provided, normalize around center point
        """
        if center is not None:
            # Normalize around center
            if value > center:
                normalized = (value - center) / (max_val - center)
            else:
                normalized = (value - center) / (center - min_val)
        else:
            # Simple min-max normalization to [-1, 1]
            range_size = max_val - min_val
            if range_size == 0:
                return 0
            normalized = ((value - min_val) / range_size) * 2 - 1

        # Clip to [-1, 1]
        return max(-1.0, min(1.0, normalized))

    def _explain_driver(
        self,
        feature_name: str,
        contribution_score: float,
        feature_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation for driver"""
        raw_value = feature_data.get("value", 0)
        direction = "bullish" if contribution_score > 0 else "bearish"

        explanations = {
            "price_vs_sma50": lambda: f"Price is {abs(raw_value)*100:.1f}% {'above' if raw_value > 0 else 'below'} 50-day average ({direction} signal)",

            "price_vs_sma200": lambda: f"Price is {abs(raw_value)*100:.1f}% {'above' if raw_value > 0 else 'below'} 200-day average (long-term {direction} trend)",

            "rsi": lambda: f"RSI at {raw_value:.0f} indicates {'overbought' if raw_value > 70 else 'oversold' if raw_value < 30 else 'neutral'} conditions",

            "macd_signal": lambda: f"MACD {'above' if raw_value > 0 else 'below'} signal line ({direction} momentum)",

            "news_sentiment": lambda: f"News sentiment is {'positive' if raw_value > 0 else 'negative'} ({abs(raw_value):.2f} score)",

            "social_sentiment": lambda: f"Social media sentiment is {'positive' if raw_value > 0 else 'negative'} ({abs(raw_value):.2f} score)",

            "analyst_consensus": lambda: f"Analyst consensus is {'bullish' if raw_value > 0 else 'bearish'} ({abs(raw_value):.2f} score)",

            "volume_trend": lambda: f"Volume trend is {'increasing' if raw_value > 0 else 'decreasing'} ({direction} confirmation)",

            "price_volatility": lambda: f"Volatility at {raw_value*100:.1f}% ({'high' if raw_value > 0.05 else 'low'} risk)",
        }

        explanation_func = explanations.get(feature_name)
        if explanation_func:
            return explanation_func()
        else:
            return f"{feature_name.replace('_', ' ').title()}: {raw_value:.3f} (contributing {direction})"

    def calculate_sensitivity(
        self,
        feature_name: str,
        current_value: float,
        feature_weight: float
    ) -> Dict[str, Any]:
        """
        Calculate sensitivity analysis for a feature

        Shows how changes in feature value affect prediction
        """
        # Calculate impact of +/- 10% change
        change_pct = 0.10

        plus_10_value = current_value * (1 + change_pct)
        minus_10_value = current_value * (1 - change_pct)

        # Normalized contributions
        current_contribution = feature_weight * current_value
        plus_10_contribution = feature_weight * plus_10_value
        minus_10_contribution = feature_weight * minus_10_value

        # Calculate deltas
        upside_impact = plus_10_contribution - current_contribution
        downside_impact = current_contribution - minus_10_contribution

        return {
            "feature_name": feature_name,
            "current_value": current_value,
            "current_contribution": current_contribution,
            "sensitivity": {
                "plus_10_pct": {
                    "value": plus_10_value,
                    "contribution": plus_10_contribution,
                    "impact": upside_impact
                },
                "minus_10_pct": {
                    "value": minus_10_value,
                    "contribution": minus_10_contribution,
                    "impact": downside_impact
                }
            },
            "volatility": abs(upside_impact) + abs(downside_impact),
            "interpretation": self._interpret_sensitivity(upside_impact, downside_impact)
        }

    def _interpret_sensitivity(self, upside: float, downside: float) -> str:
        """Interpret sensitivity analysis results"""
        total_sensitivity = abs(upside) + abs(downside)

        if total_sensitivity < 0.01:
            return "Low sensitivity - feature has minimal impact on prediction"
        elif total_sensitivity < 0.05:
            return "Moderate sensitivity - feature has some impact on prediction"
        else:
            return "High sensitivity - feature significantly impacts prediction"

    def explain_prediction_change(
        self,
        old_prediction: Dict[str, Any],
        new_prediction: Dict[str, Any],
        old_features: Dict[str, Any],
        new_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain what drove a change in prediction

        Useful for understanding why recommendation changed
        """
        # Calculate old and new contributions
        old_contributions = self._calculate_contributions(old_features)
        new_contributions = self._calculate_contributions(new_features)

        # Find features with largest changes
        feature_changes = []

        for new_contrib in new_contributions:
            feature_name = new_contrib["feature_name"]
            old_contrib = next(
                (c for c in old_contributions if c["feature_name"] == feature_name),
                None
            )

            if old_contrib:
                change = new_contrib["contribution_score"] - old_contrib["contribution_score"]
                feature_changes.append({
                    "feature_name": feature_name,
                    "old_value": old_contrib["raw_value"],
                    "new_value": new_contrib["raw_value"],
                    "contribution_change": change,
                    "explanation": self._explain_change(
                        feature_name,
                        old_contrib["raw_value"],
                        new_contrib["raw_value"],
                        change
                    )
                })

        # Sort by magnitude of change
        feature_changes.sort(key=lambda x: abs(x["contribution_change"]), reverse=True)

        # Get prediction change
        old_action = old_prediction.get("action", "HOLD")
        new_action = new_prediction.get("action", "HOLD")
        action_changed = old_action != new_action

        return {
            "action_changed": action_changed,
            "old_action": old_action,
            "new_action": new_action,
            "top_drivers_of_change": feature_changes[:5],
            "summary": self._summarize_change(action_changed, new_action, feature_changes[:3])
        }

    def _explain_change(
        self,
        feature_name: str,
        old_value: float,
        new_value: float,
        contribution_change: float
    ) -> str:
        """Explain how feature change affected prediction"""
        change_pct = ((new_value - old_value) / old_value * 100) if old_value != 0 else 0
        direction = "increased" if new_value > old_value else "decreased"
        impact = "positive" if contribution_change > 0 else "negative"

        return f"{feature_name.replace('_', ' ').title()} {direction} by {abs(change_pct):.1f}%, contributing {impact} impact ({contribution_change:+.3f})"

    def _summarize_change(
        self,
        action_changed: bool,
        new_action: str,
        top_changes: List[Dict[str, Any]]
    ) -> str:
        """Summarize what drove the prediction change"""
        if not action_changed:
            return "Recommendation unchanged - minor fluctuations in underlying factors"

        if len(top_changes) == 0:
            return f"Recommendation changed to {new_action}"

        top_driver = top_changes[0]
        summary = f"Recommendation changed to {new_action} primarily due to {top_driver['explanation']}"

        if len(top_changes) > 1:
            summary += f", with additional contribution from {top_changes[1]['feature_name'].replace('_', ' ')}"

        return summary
