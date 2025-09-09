"""
Analysis Service
Main service combining technical analysis, statistical analysis, and data pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio

from ..core.indicators import TechnicalAnalysis
from ..core.data_pipeline import DataPipeline
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class StatisticalAnalysis:
    """Statistical analysis methods for market data"""
    
    @staticmethod
    def calculate_volatility_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate various volatility metrics"""
        if returns.empty or len(returns) < 2:
            return {}
        
        # Remove infinite and NaN values
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if clean_returns.empty:
            return {}
        
        return {
            'volatility_daily': float(clean_returns.std()),
            'volatility_annualized': float(clean_returns.std() * np.sqrt(252)),
            'volatility_rolling_30d': float(clean_returns.rolling(30).std().iloc[-1]) if len(clean_returns) >= 30 else None,
            'volatility_rolling_90d': float(clean_returns.rolling(90).std().iloc[-1]) if len(clean_returns) >= 90 else None,
            'skewness': float(stats.skew(clean_returns)),
            'kurtosis': float(stats.kurtosis(clean_returns)),
            'jarque_bera_stat': float(stats.jarque_bera(clean_returns)[0]),
            'jarque_bera_pvalue': float(stats.jarque_bera(clean_returns)[1])
        }
    
    @staticmethod
    def calculate_risk_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        if returns.empty or len(returns) < 2:
            return {}
        
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if clean_returns.empty:
            return {}
        
        # Annualized metrics
        annual_return = clean_returns.mean() * 252
        annual_vol = clean_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = clean_returns[clean_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + clean_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(clean_returns, 5)
        var_99 = np.percentile(clean_returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = clean_returns[clean_returns <= var_95].mean()
        cvar_99 = clean_returns[clean_returns <= var_99].mean()
        
        return {
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95) if not np.isnan(cvar_95) else None,
            'cvar_99': float(cvar_99) if not np.isnan(cvar_99) else None,
            'calmar_ratio': float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else None
        }
    
    @staticmethod
    def correlation_analysis(returns1: pd.Series, returns2: pd.Series) -> Dict[str, float]:
        """Calculate correlation metrics between two return series"""
        if returns1.empty or returns2.empty:
            return {}
        
        # Align the series
        aligned = pd.concat([returns1, returns2], axis=1).dropna()
        if aligned.shape[1] != 2 or len(aligned) < 2:
            return {}
        
        series1, series2 = aligned.iloc[:, 0], aligned.iloc[:, 1]
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(series1, series2)
        
        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(series1, series2)
        
        # Rolling correlation (30-day window)
        rolling_corr = aligned.iloc[:, 0].rolling(30).corr(aligned.iloc[:, 1])
        
        return {
            'pearson_correlation': float(pearson_corr),
            'pearson_pvalue': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_pvalue': float(spearman_p),
            'rolling_correlation_30d': float(rolling_corr.iloc[-1]) if not rolling_corr.empty and not pd.isna(rolling_corr.iloc[-1]) else None,
            'correlation_stability': float(rolling_corr.std()) if len(rolling_corr.dropna()) > 1 else None
        }
    
    @staticmethod
    def trend_analysis(prices: pd.Series) -> Dict[str, Any]:
        """Analyze price trends using statistical methods"""
        if prices.empty or len(prices) < 10:
            return {}
        
        # Linear regression on price vs time
        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        
        # Remove NaN values
        valid_idx = ~np.isnan(y)
        if not valid_idx.any():
            return {}
        
        x_clean = x[valid_idx]
        y_clean = y[valid_idx]
        
        if len(x_clean) < 2:
            return {}
        
        model = LinearRegression()
        model.fit(x_clean, y_clean)
        
        # Calculate trend statistics
        slope = model.coef_[0]
        r_squared = model.score(x_clean, y_clean)
        
        # Trend classification
        slope_normalized = slope / prices.mean() * len(prices)  # Normalize by price level and time
        
        if slope_normalized > 0.05:
            trend_direction = "STRONG_UPTREND"
        elif slope_normalized > 0.01:
            trend_direction = "UPTREND"
        elif slope_normalized < -0.05:
            trend_direction = "STRONG_DOWNTREND"
        elif slope_normalized < -0.01:
            trend_direction = "DOWNTREND"
        else:
            trend_direction = "SIDEWAYS"
        
        # Trend strength based on R-squared
        if r_squared > 0.7:
            trend_strength = "STRONG"
        elif r_squared > 0.4:
            trend_strength = "MODERATE"
        else:
            trend_strength = "WEAK"
        
        # Moving average convergence/divergence analysis
        short_ma = prices.rolling(20).mean()
        long_ma = prices.rolling(50).mean()
        ma_divergence = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1] if len(prices) >= 50 else None
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'slope': float(slope),
            'slope_normalized': float(slope_normalized),
            'r_squared': float(r_squared),
            'ma_divergence': float(ma_divergence) if ma_divergence is not None else None,
            'trend_duration': len(prices),
            'price_momentum': float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) if len(prices) > 0 else None
        }

class ComprehensiveAnalysisService:
    """Main analysis service combining all analysis methods"""
    
    def __init__(self, market_data_url: str = "http://localhost:8002"):
        self.technical_analysis = TechnicalAnalysis()
        self.data_pipeline = DataPipeline(market_data_url)
        self.statistical_analysis = StatisticalAnalysis()
        
    async def comprehensive_analysis(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Perform comprehensive analysis including technical, statistical, and predictive analysis
        
        Args:
            symbol: Stock symbol to analyze
            period: Time period for analysis
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info(f"Starting comprehensive analysis for {symbol}")
            
            # Get prepared data
            df = await self.data_pipeline.prepare_data_for_analysis(symbol, period)
            
            if df is None or df.empty:
                return {
                    'error': f'No data available for {symbol}',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Perform technical analysis
            technical_results = self.technical_analysis.analyze(df, symbol)
            
            # Perform statistical analysis
            returns = df['close'].pct_change().dropna()
            
            statistical_results = {
                'volatility_metrics': self.statistical_analysis.calculate_volatility_metrics(returns),
                'risk_metrics': self.statistical_analysis.calculate_risk_metrics(returns),
                'trend_analysis': self.statistical_analysis.trend_analysis(df['close'])
            }
            
            # Feature importance analysis
            feature_analysis = self._analyze_features(df)
            
            # Market regime detection
            regime_analysis = self._detect_market_regime(df)
            
            # Combine all results
            comprehensive_result = {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_period': period,
                'data_points': len(df),
                'technical_analysis': technical_results,
                'statistical_analysis': statistical_results,
                'feature_analysis': feature_analysis,
                'regime_analysis': regime_analysis,
                'summary': self._generate_summary(technical_results, statistical_results, regime_analysis)
            }
            
            logger.info(f"Completed comprehensive analysis for {symbol}")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    async def quick_analysis(self, symbol: str) -> Dict[str, Any]:
        """Quick analysis with current data and basic indicators"""
        try:
            # Get real-time features
            realtime_features = await self.data_pipeline.get_realtime_features(symbol)
            
            if realtime_features is None:
                return {
                    'error': f'No real-time data available for {symbol}',
                    'symbol': symbol
                }
            
            # Get recent data for technical analysis
            df = await self.data_pipeline.prepare_data_for_analysis(symbol, "1mo", add_features=False)
            
            quick_result = {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'realtime_data': realtime_features,
                'quick_technical': None,
                'market_status': self._assess_market_status(realtime_features)
            }
            
            if df is not None and not df.empty:
                # Quick technical analysis with limited indicators
                quick_technical = {
                    'current_price': float(df['close'].iloc[-1]),
                    'price_change_1d': float(df['close'].iloc[-1] - df['close'].iloc[-2]) if len(df) >= 2 else None,
                    'price_change_5d': float(df['close'].iloc[-1] - df['close'].iloc[-6]) if len(df) >= 6 else None,
                    'volume_ratio': float(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None,
                    'volatility_20d': float(df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)) if len(df) >= 20 else None
                }
                quick_result['quick_technical'] = quick_technical
            
            return quick_result
            
        except Exception as e:
            logger.error(f"Error in quick analysis for {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance and correlations"""
        try:
            # Select numeric features for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_df = df[numeric_cols].dropna()
            
            if feature_df.empty or len(feature_df.columns) < 2:
                return {'error': 'Insufficient feature data'}
            
            # Correlation matrix for key features
            key_features = ['close', 'volume', 'returns'] + [col for col in feature_df.columns if 'sma' in col.lower() or 'ema' in col.lower()]
            key_features = [col for col in key_features if col in feature_df.columns]
            
            if len(key_features) >= 2:
                corr_matrix = feature_df[key_features].corr()
                
                # Find highest correlations with close price
                close_correlations = corr_matrix['close'].abs().sort_values(ascending=False)[1:6]  # Top 5 excluding self
                
                return {
                    'feature_count': len(numeric_cols),
                    'key_correlations_with_price': close_correlations.to_dict(),
                    'highest_correlation_feature': close_correlations.index[0] if len(close_correlations) > 0 else None,
                    'correlation_strength': float(close_correlations.iloc[0]) if len(close_correlations) > 0 else None
                }
            else:
                return {'feature_count': len(numeric_cols)}
                
        except Exception as e:
            logger.error(f"Error in feature analysis: {e}")
            return {'error': str(e)}
    
    def _detect_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime (trending, ranging, volatile)"""
        try:
            if len(df) < 50:
                return {'regime': 'INSUFFICIENT_DATA'}
            
            returns = df['close'].pct_change().dropna()
            prices = df['close']
            
            # Volatility regime
            recent_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            long_term_vol = returns.rolling(60).std().iloc[-1] * np.sqrt(252) if len(returns) >= 60 else recent_vol
            
            vol_regime = "HIGH_VOLATILITY" if recent_vol > long_term_vol * 1.5 else "LOW_VOLATILITY" if recent_vol < long_term_vol * 0.7 else "NORMAL_VOLATILITY"
            
            # Trend regime
            trend_analysis = self.statistical_analysis.trend_analysis(prices.iloc[-50:])  # Last 50 days
            trend_regime = trend_analysis.get('trend_direction', 'UNKNOWN')
            
            # Range-bound detection
            recent_high = prices.rolling(20).max().iloc[-1]
            recent_low = prices.rolling(20).min().iloc[-1]
            current_price = prices.iloc[-1]
            
            price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            if 0.3 <= price_position <= 0.7:
                range_regime = "RANGE_BOUND"
            elif price_position > 0.8:
                range_regime = "NEAR_RESISTANCE"
            elif price_position < 0.2:
                range_regime = "NEAR_SUPPORT"
            else:
                range_regime = "TRENDING"
            
            return {
                'overall_regime': self._classify_overall_regime(trend_regime, vol_regime, range_regime),
                'volatility_regime': vol_regime,
                'trend_regime': trend_regime,
                'range_regime': range_regime,
                'regime_confidence': trend_analysis.get('trend_strength', 'UNKNOWN'),
                'volatility_percentile': float(recent_vol),
                'price_position_in_range': float(price_position)
            }
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return {'error': str(e)}
    
    def _classify_overall_regime(self, trend: str, volatility: str, range_type: str) -> str:
        """Classify overall market regime"""
        if "STRONG" in trend and volatility == "LOW_VOLATILITY":
            return "STRONG_TREND"
        elif "TREND" in trend and volatility == "NORMAL_VOLATILITY":
            return "MODERATE_TREND"
        elif range_type == "RANGE_BOUND":
            return "CONSOLIDATION"
        elif volatility == "HIGH_VOLATILITY":
            return "VOLATILE_MARKET"
        else:
            return "UNCERTAIN"
    
    def _assess_market_status(self, realtime_features: Dict[str, Any]) -> Dict[str, str]:
        """Assess current market status based on real-time features"""
        try:
            current_price = realtime_features.get('current_price', 0)
            price_change_5d = realtime_features.get('price_change_5d', 0)
            
            # Simple status assessment
            if abs(price_change_5d / current_price) > 0.1:  # 10% move in 5 days
                momentum = "HIGH"
            elif abs(price_change_5d / current_price) > 0.05:  # 5% move
                momentum = "MODERATE"
            else:
                momentum = "LOW"
            
            direction = "UP" if price_change_5d > 0 else "DOWN" if price_change_5d < 0 else "NEUTRAL"
            
            return {
                'momentum': momentum,
                'direction': direction,
                'status': f"{momentum}_{direction}" if direction != "NEUTRAL" else "STABLE"
            }
            
        except Exception as e:
            logger.error(f"Error assessing market status: {e}")
            return {'status': 'UNKNOWN'}
    
    def _generate_summary(self, technical: Dict[str, Any], statistical: Dict[str, Any], regime: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the analysis"""
        try:
            # Extract key metrics
            current_price = technical.get('current_price', 0)
            overall_signal = technical.get('signals', {}).get('overall_signal', 'NEUTRAL')
            signal_strength = technical.get('signals', {}).get('strength', 0)
            
            trend_direction = statistical.get('trend_analysis', {}).get('trend_direction', 'UNKNOWN')
            annual_return = statistical.get('risk_metrics', {}).get('annual_return', 0)
            sharpe_ratio = statistical.get('risk_metrics', {}).get('sharpe_ratio', 0)
            max_drawdown = statistical.get('risk_metrics', {}).get('max_drawdown', 0)
            
            overall_regime = regime.get('overall_regime', 'UNKNOWN')
            
            # Generate summary
            summary = {
                'overall_assessment': self._get_overall_assessment(overall_signal, trend_direction, overall_regime),
                'key_metrics': {
                    'current_price': current_price,
                    'signal': overall_signal,
                    'signal_strength': signal_strength,
                    'trend': trend_direction,
                    'regime': overall_regime
                },
                'performance_summary': {
                    'annual_return': f"{annual_return:.2%}" if annual_return else "N/A",
                    'sharpe_ratio': f"{sharpe_ratio:.2f}" if sharpe_ratio else "N/A",
                    'max_drawdown': f"{max_drawdown:.2%}" if max_drawdown else "N/A"
                },
                'risk_level': self._assess_risk_level(statistical.get('risk_metrics', {})),
                'recommendation': self._generate_recommendation(overall_signal, signal_strength, overall_regime)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {'error': str(e)}
    
    def _get_overall_assessment(self, signal: str, trend: str, regime: str) -> str:
        """Generate overall assessment string"""
        assessments = []
        
        if "BUY" in signal:
            assessments.append("technically bullish")
        elif "SELL" in signal:
            assessments.append("technically bearish")
        else:
            assessments.append("technically neutral")
        
        if "UPTREND" in trend:
            assessments.append("upward trending")
        elif "DOWNTREND" in trend:
            assessments.append("downward trending")
        
        if regime in ["STRONG_TREND", "MODERATE_TREND"]:
            assessments.append("trending market")
        elif regime == "CONSOLIDATION":
            assessments.append("consolidating")
        elif regime == "VOLATILE_MARKET":
            assessments.append("high volatility")
        
        return ", ".join(assessments)
    
    def _assess_risk_level(self, risk_metrics: Dict[str, Any]) -> str:
        """Assess overall risk level"""
        volatility = risk_metrics.get('annual_volatility', 0)
        max_drawdown = risk_metrics.get('max_drawdown', 0)
        
        if volatility > 0.4 or max_drawdown < -0.3:  # 40% annual vol or 30% drawdown
            return "HIGH"
        elif volatility > 0.25 or max_drawdown < -0.15:  # 25% vol or 15% drawdown
            return "MODERATE"
        else:
            return "LOW"
    
    def _generate_recommendation(self, signal: str, strength: float, regime: str) -> str:
        """Generate trading recommendation"""
        if "STRONG_BUY" in signal and regime in ["STRONG_TREND", "MODERATE_TREND"]:
            return "Strong Buy - Technical and regime analysis align bullishly"
        elif "BUY" in signal and strength > 20:
            return "Buy - Positive technical signals with good strength"
        elif "STRONG_SELL" in signal and regime in ["STRONG_TREND", "MODERATE_TREND"]:
            return "Strong Sell - Technical and regime analysis align bearishly"
        elif "SELL" in signal and strength < -20:
            return "Sell - Negative technical signals with good strength"
        elif regime == "VOLATILE_MARKET":
            return "Caution - High volatility environment, consider smaller position sizes"
        elif regime == "CONSOLIDATION":
            return "Wait - Market is consolidating, wait for clearer direction"
        else:
            return "Hold - Mixed signals, maintain current position"