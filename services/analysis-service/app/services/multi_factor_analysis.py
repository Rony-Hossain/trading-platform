"""
Multi-Factor Analysis Service
Integrates technical, fundamental, sentiment, macro, options, and event data
for comprehensive stock analysis and forecasting
"""

import pandas as pd
import numpy as np
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MultiFactorFeatures:
    """Container for multi-factor feature set"""
    symbol: str
    as_of: datetime
    
    # Technical features
    technical_features: Dict[str, float]
    
    # Fundamental features
    fundamental_features: Dict[str, float]
    
    # Macro environment features
    macro_features: Dict[str, float]
    
    # Options market features
    options_features: Dict[str, float]
    
    # Sentiment features
    sentiment_features: Dict[str, float]
    
    # Event-driven features
    event_features: Dict[str, float]
    
    # Interaction features
    interaction_features: Dict[str, float]


class MultiFactorAnalysis:
    """Service for multi-factor analysis combining all data sources"""
    
    def __init__(self, market_data_url: str, sentiment_url: str, fundamentals_url: str, event_data_url: str):
        self.market_data_url = market_data_url
        self.sentiment_url = sentiment_url
        self.fundamentals_url = fundamentals_url
        self.event_data_url = event_data_url
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Feature weights for composite scoring
        self.feature_weights = {
            "technical": 0.25,
            "fundamental": 0.20,
            "macro": 0.15,
            "options": 0.15,
            "sentiment": 0.15,
            "event": 0.10,
        }
    
    async def get_multi_factor_features(self, symbol: str, period: str = "6mo") -> MultiFactorFeatures:
        """Get comprehensive multi-factor feature set"""
        
        cache_key = f"multi_factor:{symbol}:{period}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            # Gather data from all sources concurrently
            tasks = [
                self._get_technical_features(symbol, period),
                self._get_fundamental_features(symbol),
                self._get_macro_features(),
                self._get_options_features(symbol),
                self._get_sentiment_features(symbol),
                self._get_event_features(symbol),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Extract results with error handling
            technical_features = results[0] if not isinstance(results[0], Exception) else {}
            fundamental_features = results[1] if not isinstance(results[1], Exception) else {}
            macro_features = results[2] if not isinstance(results[2], Exception) else {}
            options_features = results[3] if not isinstance(results[3], Exception) else {}
            sentiment_features = results[4] if not isinstance(results[4], Exception) else {}
            event_features = results[5] if not isinstance(results[5], Exception) else {}
            
            # Calculate interaction features
            interaction_features = self._calculate_interaction_features(
                technical_features, fundamental_features, macro_features,
                options_features, sentiment_features, event_features
            )
            
            # Create multi-factor features object
            multi_factor = MultiFactorFeatures(
                symbol=symbol,
                as_of=datetime.now(),
                technical_features=technical_features,
                fundamental_features=fundamental_features,
                macro_features=macro_features,
                options_features=options_features,
                sentiment_features=sentiment_features,
                event_features=event_features,
                interaction_features=interaction_features,
            )
            
            # Cache the result
            self.cache[cache_key] = {
                "data": multi_factor,
                "timestamp": datetime.now(),
            }
            
            return multi_factor
            
        except Exception as e:
            logger.error(f"Error getting multi-factor features for {symbol}: {e}")
            # Return empty features as fallback
            return MultiFactorFeatures(
                symbol=symbol,
                as_of=datetime.now(),
                technical_features={},
                fundamental_features={},
                macro_features={},
                options_features={},
                sentiment_features={},
                event_features={},
                interaction_features={},
            )
    
    async def _get_technical_features(self, symbol: str, period: str) -> Dict[str, float]:
        """Get technical analysis features"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get technical analysis
                async with session.get(f"{self.market_data_url}/analyze/{symbol}/technical?period={period}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        technical_analysis = data.get("technical_analysis", {})
                        
                        # Extract key technical features
                        features = {}
                        
                        # Moving averages
                        if "moving_averages" in technical_analysis:
                            ma_data = technical_analysis["moving_averages"]
                            features.update({
                                "sma_20": ma_data.get("sma_20", {}).get("current", 0),
                                "sma_50": ma_data.get("sma_50", {}).get("current", 0),
                                "ema_12": ma_data.get("ema_12", {}).get("current", 0),
                                "ema_26": ma_data.get("ema_26", {}).get("current", 0),
                                "ma_trend": 1 if ma_data.get("trend") == "BULLISH" else -1 if ma_data.get("trend") == "BEARISH" else 0,
                            })
                        
                        # Momentum indicators
                        if "momentum" in technical_analysis:
                            momentum = technical_analysis["momentum"]
                            features.update({
                                "rsi": momentum.get("rsi", {}).get("current", 50),
                                "macd_histogram": momentum.get("macd", {}).get("histogram_current", 0),
                                "momentum_score": momentum.get("composite_score", 0),
                            })
                        
                        # Volatility indicators
                        if "volatility" in technical_analysis:
                            volatility = technical_analysis["volatility"]
                            bb = volatility.get("bollinger_bands", {})
                            features.update({
                                "bb_width": bb.get("width_current", 0),
                                "bb_position": bb.get("position", 0.5),
                                "atr": volatility.get("atr", {}).get("current", 0),
                            })
                        
                        # Volume indicators
                        if "volume" in technical_analysis:
                            volume = technical_analysis["volume"]
                            features.update({
                                "volume_sma_ratio": volume.get("volume_sma_ratio", 1),
                                "obv_trend": 1 if volume.get("obv_trend") == "BULLISH" else -1 if volume.get("obv_trend") == "BEARISH" else 0,
                            })
                        
                        return features
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting technical features for {symbol}: {e}")
            return {}
    
    async def _get_fundamental_features(self, symbol: str) -> Dict[str, float]:
        """Get fundamental analysis features"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get earnings data
                async with session.get(f"{self.fundamentals_url}/earnings/{symbol}?periods=4") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        quarters = data.get("quarters", [])
                        
                        features = {}
                        
                        if quarters:
                            latest = quarters[0] if quarters else {}
                            
                            # Growth metrics
                            features.update({
                                "revenue_growth_yoy": latest.get("revenue_growth_yoy", 0),
                                "eps_growth_yoy": latest.get("eps_growth_yoy", 0),
                                "revenue_growth_qoq": latest.get("revenue_growth_qoq", 0),
                            })
                            
                            # Profitability metrics
                            features.update({
                                "gross_margin": latest.get("gross_margin", 0),
                                "operating_margin": latest.get("operating_margin", 0),
                                "net_margin": latest.get("net_margin", 0),
                                "roe": latest.get("roe", 0),
                                "roa": latest.get("roa", 0),
                            })
                            
                            # Calculate trend features
                            if len(quarters) >= 4:
                                revenue_trend = np.polyfit(range(4), [q.get("revenue", 0) for q in quarters[:4]], 1)[0]
                                eps_trend = np.polyfit(range(4), [q.get("earnings_per_share", 0) for q in quarters[:4]], 1)[0]
                                
                                features.update({
                                    "revenue_trend": revenue_trend,
                                    "eps_trend": eps_trend,
                                })
                        
                        return features
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting fundamental features for {symbol}: {e}")
            return {}
    
    async def _get_macro_features(self) -> Dict[str, float]:
        """Get macro environment features"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.market_data_url}/factors/macro") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        factors = data.get("factors", [])
                        
                        features = {}
                        
                        for factor in factors:
                            key = factor.get("key", "").lower()
                            value = factor.get("value", 0)
                            
                            if key == "vix":
                                features["vix"] = value
                                features["vix_regime"] = 1 if value > 30 else -1 if value < 15 else 0
                            elif key == "us10y":
                                features["us10y"] = value
                            elif key == "us02y":
                                features["us02y"] = value
                                # Calculate yield curve slope
                                if "us10y" in features:
                                    features["yield_curve_slope"] = features["us10y"] - value
                            elif key == "eurusd":
                                features["eurusd"] = value
                            elif key == "wti":
                                features["wti_oil"] = value
                        
                        # Add derived macro features
                        if "vix" in features and "us10y" in features:
                            features["risk_appetite"] = features["us10y"] / (features["vix"] + 1)  # Higher = more risk appetite
                        
                        return features
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting macro features: {e}")
            return {}
    
    async def _get_options_features(self, symbol: str) -> Dict[str, float]:
        """Get options market features"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.market_data_url}/options/{symbol}/metrics") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        features = {}
                        
                        # Put/call ratios
                        if "put_call_volume_ratio" in data:
                            pcr_vol = data["put_call_volume_ratio"]
                            features["put_call_volume_ratio"] = pcr_vol
                            features["pcr_vol_regime"] = 1 if pcr_vol > 1.2 else -1 if pcr_vol < 0.8 else 0
                        
                        if "put_call_oi_ratio" in data:
                            pcr_oi = data["put_call_oi_ratio"]
                            features["put_call_oi_ratio"] = pcr_oi
                        
                        # Implied volatility features
                        if "atm_iv" in data:
                            features["atm_iv"] = data["atm_iv"]
                        
                        if "iv_skew_25d" in data:
                            features["iv_skew"] = data["iv_skew_25d"]
                            features["iv_skew_regime"] = 1 if data["iv_skew_25d"] > 0.05 else -1 if data["iv_skew_25d"] < -0.05 else 0
                        
                        # Implied move
                        if "implied_move_pct" in data:
                            features["implied_move_pct"] = data["implied_move_pct"]
                            features["high_implied_move"] = 1 if data["implied_move_pct"] > 0.1 else 0
                        
                        return features
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting options features for {symbol}: {e}")
            return {}
    
    async def _get_sentiment_features(self, symbol: str) -> Dict[str, float]:
        """Get sentiment analysis features"""
        try:
            # For now, return synthetic sentiment features
            # In production, this would call the sentiment service
            
            features = {
                "news_sentiment": 0.2 + (hash(symbol) % 100 - 50) / 100,  # -0.3 to 0.7
                "social_sentiment": 0.1 + (hash(symbol + "social") % 100 - 50) / 100,
                "analyst_sentiment": 0.15 + (hash(symbol + "analyst") % 60 - 30) / 100,
                "sentiment_momentum": (hash(symbol + "momentum") % 20 - 10) / 100,
            }
            
            # Sentiment regime
            avg_sentiment = np.mean(list(features.values()))
            features["sentiment_regime"] = 1 if avg_sentiment > 0.1 else -1 if avg_sentiment < -0.1 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting sentiment features for {symbol}: {e}")
            return {}
    
    async def _get_event_features(self, symbol: str) -> Dict[str, float]:
        """Get event-driven features"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get upcoming events
                async with session.get(f"{self.event_data_url}/events/{symbol}?days_ahead=30") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        events = data.get("events", [])
                        
                        features = {}
                        
                        # Count events by type
                        event_counts = {}
                        high_impact_events = 0
                        days_to_next_earnings = 365
                        
                        for event in events:
                            event_type = event.get("event_type", "unknown")
                            impact = event.get("impact_level", "medium")
                            event_date = datetime.fromisoformat(event.get("event_date"))
                            days_ahead = (event_date - datetime.now()).days
                            
                            if days_ahead > 0:
                                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                                
                                if impact == "high":
                                    high_impact_events += 1
                                
                                if event_type == "earnings" and days_ahead < days_to_next_earnings:
                                    days_to_next_earnings = days_ahead
                        
                        features.update({
                            "earnings_events_30d": event_counts.get("earnings", 0),
                            "product_launch_events_30d": event_counts.get("product_launch", 0),
                            "regulatory_events_30d": event_counts.get("regulatory_decision", 0),
                            "high_impact_events_30d": high_impact_events,
                            "days_to_next_earnings": days_to_next_earnings if days_to_next_earnings < 365 else 0,
                        })
                        
                        # Event clustering (high activity periods)
                        features["event_cluster_intensity"] = min(high_impact_events / 5, 1.0)
                        
                        return features
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting event features for {symbol}: {e}")
            return {}
    
    def _calculate_interaction_features(
        self,
        technical: Dict[str, float],
        fundamental: Dict[str, float],
        macro: Dict[str, float],
        options: Dict[str, float],
        sentiment: Dict[str, float],
        event: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate interaction features between different factor categories"""
        
        interactions = {}
        
        try:
            # Momentum × Rates interaction
            if "momentum_score" in technical and "us10y" in macro:
                interactions["momentum_rates"] = technical["momentum_score"] * (1 / (macro["us10y"] + 0.01))
            
            # Value × Credit spread interaction  
            if "net_margin" in fundamental and "yield_curve_slope" in macro:
                interactions["value_credit"] = fundamental["net_margin"] * macro["yield_curve_slope"]
            
            # Volatility × Options interaction
            if "bb_width" in technical and "atm_iv" in options:
                interactions["vol_options"] = technical["bb_width"] * options["atm_iv"]
            
            # Sentiment × Event interaction
            if "sentiment_regime" in sentiment and "high_impact_events_30d" in event:
                interactions["sentiment_events"] = sentiment["sentiment_regime"] * event["high_impact_events_30d"]
            
            # Growth × Macro interaction
            if "revenue_growth_yoy" in fundamental and "vix_regime" in macro:
                interactions["growth_macro"] = fundamental["revenue_growth_yoy"] * (1 - abs(macro["vix_regime"]))
            
            # Options positioning × Sentiment
            if "put_call_volume_ratio" in options and "news_sentiment" in sentiment:
                interactions["options_sentiment"] = (1 / options["put_call_volume_ratio"]) * sentiment["news_sentiment"]
            
            # Technical momentum × Fundamental momentum
            if "momentum_score" in technical and "eps_trend" in fundamental:
                interactions["tech_fund_momentum"] = technical["momentum_score"] * fundamental["eps_trend"]
                
        except Exception as e:
            logger.error(f"Error calculating interaction features: {e}")
        
        return interactions
    
    async def calculate_composite_score(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """Calculate weighted composite score from all factors"""
        
        try:
            multi_factor = await self.get_multi_factor_features(symbol, period)
            
            # Calculate category scores
            category_scores = {}
            
            # Technical score
            tech_features = multi_factor.technical_features
            if tech_features:
                tech_signals = []
                if "ma_trend" in tech_features:
                    tech_signals.append(tech_features["ma_trend"])
                if "momentum_score" in tech_features:
                    tech_signals.append(tech_features["momentum_score"] / 100)
                if "rsi" in tech_features:
                    rsi_signal = (tech_features["rsi"] - 50) / 50  # Normalize RSI
                    tech_signals.append(rsi_signal)
                
                category_scores["technical"] = np.mean(tech_signals) if tech_signals else 0
            
            # Fundamental score
            fund_features = multi_factor.fundamental_features
            if fund_features:
                fund_signals = []
                if "revenue_growth_yoy" in fund_features:
                    fund_signals.append(min(fund_features["revenue_growth_yoy"] / 20, 1))  # Cap at 20% growth = 1.0
                if "eps_growth_yoy" in fund_features:
                    fund_signals.append(min(fund_features["eps_growth_yoy"] / 25, 1))  # Cap at 25% growth = 1.0
                if "roe" in fund_features:
                    fund_signals.append(min(fund_features["roe"] / 20, 1))  # Cap at 20% ROE = 1.0
                
                category_scores["fundamental"] = np.mean(fund_signals) if fund_signals else 0
            
            # Macro score
            macro_features = multi_factor.macro_features
            if macro_features:
                macro_signals = []
                if "vix_regime" in macro_features:
                    macro_signals.append(-macro_features["vix_regime"])  # Low VIX is positive
                if "risk_appetite" in macro_features:
                    macro_signals.append(min(macro_features["risk_appetite"] / 5, 1))
                
                category_scores["macro"] = np.mean(macro_signals) if macro_signals else 0
            
            # Options score
            options_features = multi_factor.options_features
            if options_features:
                options_signals = []
                if "pcr_vol_regime" in options_features:
                    options_signals.append(-options_features["pcr_vol_regime"])  # Low PCR is bullish
                if "iv_skew_regime" in options_features:
                    options_signals.append(-options_features["iv_skew_regime"] * 0.5)  # Negative skew is bullish
                
                category_scores["options"] = np.mean(options_signals) if options_signals else 0
            
            # Sentiment score
            sentiment_features = multi_factor.sentiment_features
            if sentiment_features:
                sent_signals = []
                if "news_sentiment" in sentiment_features:
                    sent_signals.append(sentiment_features["news_sentiment"])
                if "social_sentiment" in sentiment_features:
                    sent_signals.append(sentiment_features["social_sentiment"])
                if "analyst_sentiment" in sentiment_features:
                    sent_signals.append(sentiment_features["analyst_sentiment"])
                
                category_scores["sentiment"] = np.mean(sent_signals) if sent_signals else 0
            
            # Event score
            event_features = multi_factor.event_features
            if event_features:
                event_signals = []
                if "days_to_next_earnings" in event_features:
                    # Closer to earnings can be positive or negative, normalize
                    days = event_features["days_to_next_earnings"]
                    if days > 0:
                        event_signals.append((30 - min(days, 30)) / 30 * 0.2)  # Small positive for earnings approach
                
                category_scores["event"] = np.mean(event_signals) if event_signals else 0
            
            # Calculate weighted composite score
            composite_score = 0
            total_weight = 0
            
            for category, score in category_scores.items():
                weight = self.feature_weights.get(category, 0)
                composite_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                composite_score = composite_score / total_weight
            
            # Normalize to -1 to 1 range
            composite_score = max(-1, min(1, composite_score))
            
            return {
                "symbol": symbol,
                "composite_score": composite_score,
                "category_scores": category_scores,
                "feature_weights": self.feature_weights,
                "calculation_timestamp": datetime.now().isoformat(),
                "signal": "BULLISH" if composite_score > 0.2 else "BEARISH" if composite_score < -0.2 else "NEUTRAL",
                "confidence": abs(composite_score),
                "multi_factor_features": {
                    "technical": multi_factor.technical_features,
                    "fundamental": multi_factor.fundamental_features,
                    "macro": multi_factor.macro_features,
                    "options": multi_factor.options_features,
                    "sentiment": multi_factor.sentiment_features,
                    "event": multi_factor.event_features,
                    "interactions": multi_factor.interaction_features,
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite score for {symbol}: {e}")
            return {
                "symbol": symbol,
                "composite_score": 0,
                "category_scores": {},
                "error": str(e),
            }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]["timestamp"]
        return (datetime.now() - cache_time).seconds < self.cache_ttl