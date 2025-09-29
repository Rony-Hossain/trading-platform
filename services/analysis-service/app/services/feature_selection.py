#!/usr/bin/env python3
"""
Advanced Feature Selection and Pruning Service

Implements SHAP-based feature importance analysis and Recursive Feature Elimination (RFE)
to reduce model complexity and improve generalization for financial forecasting models.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced functionality
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance from different methods."""
    feature_name: str
    shap_importance: Optional[float] = None
    rfe_rank: Optional[int] = None
    model_importance: Optional[float] = None
    correlation_score: Optional[float] = None
    composite_score: float = 0.0
    selected: bool = False


@dataclass
class SelectionResults:
    """Results from feature selection process."""
    original_feature_count: int
    selected_feature_count: int
    removed_feature_count: int
    feature_importances: List[FeatureImportance]
    selected_features: List[str]
    removed_features: List[str]
    performance_metrics: Dict[str, float]
    selection_method: str
    selection_params: Dict[str, Any]
    execution_time: float


@dataclass
class CollinearityAnalysis:
    """Results from collinearity analysis."""
    correlation_matrix: Dict[str, Dict[str, float]]
    high_correlation_pairs: List[Tuple[str, str, float]]
    vif_scores: Dict[str, float]
    collinear_features: List[str]
    recommendations: List[str]


class AdvancedFeatureSelector:
    """
    Advanced feature selection using SHAP values, RFE, and collinearity analysis.
    
    Combines multiple feature selection techniques to identify the most important
    features while removing redundant and low-impact features.
    """
    
    def __init__(self, 
                 correlation_threshold: float = 0.9,
                 vif_threshold: float = 10.0,
                 shap_threshold: float = 0.001,
                 min_features: int = 5,
                 max_features: Optional[int] = None):
        """
        Initialize feature selector.
        
        Args:
            correlation_threshold: Remove features with correlation > threshold
            vif_threshold: Remove features with VIF > threshold  
            shap_threshold: Remove features with SHAP importance < threshold
            min_features: Minimum number of features to retain
            max_features: Maximum number of features to retain
        """
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.shap_threshold = shap_threshold
        self.min_features = min_features
        self.max_features = max_features
        
        # Model for feature selection
        self.base_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.selection_history = []
        
    def analyze_collinearity(self, X: pd.DataFrame) -> CollinearityAnalysis:
        """
        Analyze feature collinearity using correlation matrix and VIF.
        
        Args:
            X: Feature matrix
            
        Returns:
            CollinearityAnalysis with correlation and VIF results
        """
        logger.info(f"Analyzing collinearity for {X.shape[1]} features")
        
        # Calculate correlation matrix
        corr_matrix = X.corr()
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > self.correlation_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        # Calculate VIF scores (simplified version)
        vif_scores = {}
        collinear_features = []
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # Prepare data for VIF calculation
            X_scaled = self.scaler.fit_transform(X.fillna(0))
            
            for i, feature in enumerate(X.columns):
                try:
                    vif = variance_inflation_factor(X_scaled, i)
                    vif_scores[feature] = vif
                    if vif > self.vif_threshold:
                        collinear_features.append(feature)
                except:
                    vif_scores[feature] = 0.0
                    
        except ImportError:
            # Fallback: use correlation-based approach
            logger.warning("statsmodels not available, using correlation-based VIF approximation")
            for feature in X.columns:
                max_corr = corr_matrix[feature].abs().drop(feature).max()
                vif_approx = 1 / (1 - max_corr**2) if max_corr < 0.99 else 999.0
                vif_scores[feature] = vif_approx
                if vif_approx > self.vif_threshold:
                    collinear_features.append(feature)
        
        # Generate recommendations
        recommendations = []
        if high_corr_pairs:
            recommendations.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        if collinear_features:
            recommendations.append(f"Found {len(collinear_features)} features with high VIF scores")
        if not high_corr_pairs and not collinear_features:
            recommendations.append("No significant collinearity detected")
            
        return CollinearityAnalysis(
            correlation_matrix=corr_matrix.to_dict(),
            high_correlation_pairs=high_corr_pairs,
            vif_scores=vif_scores,
            collinear_features=collinear_features,
            recommendations=recommendations
        )
    
    def calculate_shap_importance(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                model=None) -> Dict[str, float]:
        """
        Calculate SHAP feature importance values.
        
        Args:
            X: Feature matrix
            y: Target values  
            model: Trained model (optional)
            
        Returns:
            Dictionary of feature names to SHAP importance scores
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping SHAP analysis")
            return {}
            
        logger.info("Calculating SHAP feature importance")
        
        try:
            # Use provided model or train a new one
            if model is None:
                model = self.base_model
                X_scaled = self.scaler.fit_transform(X.fillna(0))
                model.fit(X_scaled, y)
            else:
                X_scaled = X.fillna(0)
            
            # Create SHAP explainer
            if hasattr(model, 'feature_importances_'):
                # Tree-based model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
            else:
                # Use KernelExplainer for other models
                explainer = shap.KernelExplainer(model.predict, X_scaled[:100])  # Sample for speed
                shap_values = explainer.shap_values(X_scaled[:100])
            
            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-output models
                
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dictionary
            shap_importance = {}
            for i, feature in enumerate(X.columns):
                shap_importance[feature] = float(mean_shap[i])
                
            logger.info(f"Calculated SHAP importance for {len(shap_importance)} features")
            return shap_importance
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return {}
    
    def recursive_feature_elimination(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    cv_folds: int = 3) -> Tuple[List[str], Dict[str, int]]:
        """
        Perform Recursive Feature Elimination with Cross-Validation.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Tuple of (selected_features, feature_rankings)
        """
        logger.info(f"Performing RFE with {cv_folds}-fold CV")
        
        # Prepare data
        X_clean = X.fillna(0)
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Set up cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Determine target number of features
        n_features_target = self.max_features or max(self.min_features, X.shape[1] // 3)
        n_features_target = min(n_features_target, X.shape[1])
        
        try:
            # Use RFECV for automatic feature selection
            rfecv = RFECV(
                estimator=self.base_model,
                step=1,
                cv=tscv,
                scoring='neg_mean_squared_error',
                min_features_to_select=self.min_features,
                n_jobs=-1
            )
            
            rfecv.fit(X_scaled, y)
            
            selected_features = X.columns[rfecv.support_].tolist()
            feature_rankings = {
                feature: int(rank) 
                for feature, rank in zip(X.columns, rfecv.ranking_)
            }
            
            logger.info(f"RFE selected {len(selected_features)} features")
            
        except Exception as e:
            logger.warning(f"RFECV failed: {e}, falling back to simple RFE")
            
            # Fallback to simple RFE
            rfe = RFE(
                estimator=self.base_model,
                n_features_to_select=n_features_target,
                step=1
            )
            
            rfe.fit(X_scaled, y)
            
            selected_features = X.columns[rfe.support_].tolist()
            feature_rankings = {
                feature: int(rank)
                for feature, rank in zip(X.columns, rfe.ranking_)
            }
        
        return selected_features, feature_rankings
    
    def calculate_composite_score(self, 
                                feature_importances: List[FeatureImportance]) -> List[FeatureImportance]:
        """
        Calculate composite feature importance scores.
        
        Args:
            feature_importances: List of FeatureImportance objects
            
        Returns:
            Updated list with composite scores
        """
        logger.info("Calculating composite feature importance scores")
        
        # Normalize different importance measures
        shap_scores = [fi.shap_importance or 0.0 for fi in feature_importances]
        model_scores = [fi.model_importance or 0.0 for fi in feature_importances]
        rfe_ranks = [fi.rfe_rank or 999 for fi in feature_importances]
        
        # Normalize to 0-1 scale
        if max(shap_scores) > 0:
            shap_scores = [s / max(shap_scores) for s in shap_scores]
        
        if max(model_scores) > 0:
            model_scores = [s / max(model_scores) for s in model_scores]
            
        # Convert RFE ranks (lower is better) to scores (higher is better)
        if max(rfe_ranks) > 1:
            rfe_scores = [1.0 - (r - 1) / (max(rfe_ranks) - 1) for r in rfe_ranks]
        else:
            rfe_scores = [1.0] * len(rfe_ranks)
        
        # Calculate composite scores (weighted average)
        weights = {
            'shap': 0.4 if SHAP_AVAILABLE else 0.0,
            'model': 0.3,
            'rfe': 0.3
        }
        
        # Adjust weights if SHAP not available
        if not SHAP_AVAILABLE:
            weights['model'] = 0.5
            weights['rfe'] = 0.5
        
        for i, fi in enumerate(feature_importances):
            composite = (
                weights['shap'] * shap_scores[i] +
                weights['model'] * model_scores[i] +
                weights['rfe'] * rfe_scores[i]
            )
            fi.composite_score = composite
            
        return feature_importances
    
    async def select_features(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            method: str = 'composite',
                            cv_folds: int = 3) -> SelectionResults:
        """
        Perform comprehensive feature selection.
        
        Args:
            X: Feature matrix
            y: Target values
            method: Selection method ('composite', 'shap', 'rfe', 'correlation')
            cv_folds: Number of CV folds
            
        Returns:
            SelectionResults with feature selection outcomes
        """
        start_time = datetime.now()
        logger.info(f"Starting feature selection with method '{method}' for {X.shape[1]} features")
        
        # Initialize feature importance tracking
        feature_importances = []
        
        # Step 1: Calculate model-based feature importance
        logger.info("Calculating model-based feature importance")
        X_clean = X.fillna(0)
        X_scaled = self.scaler.fit_transform(X_clean)
        
        model = self.base_model
        model.fit(X_scaled, y)
        
        model_importance = {}
        if hasattr(model, 'feature_importances_'):
            for feature, importance in zip(X.columns, model.feature_importances_):
                model_importance[feature] = float(importance)
        
        # Step 2: Calculate SHAP importance (if available)
        shap_importance = {}
        if method in ['composite', 'shap'] and SHAP_AVAILABLE:
            shap_importance = self.calculate_shap_importance(X, y, model)
        
        # Step 3: Perform RFE
        rfe_features, rfe_rankings = [], {}
        if method in ['composite', 'rfe']:
            rfe_features, rfe_rankings = self.recursive_feature_elimination(X, y, cv_folds)
        
        # Step 4: Analyze collinearity
        collinearity = self.analyze_collinearity(X)
        
        # Step 5: Build feature importance objects
        for feature in X.columns:
            fi = FeatureImportance(
                feature_name=feature,
                shap_importance=shap_importance.get(feature),
                rfe_rank=rfe_rankings.get(feature),
                model_importance=model_importance.get(feature),
                correlation_score=1.0 - min(collinearity.vif_scores.get(feature, 1.0) / 10.0, 1.0)
            )
            feature_importances.append(fi)
        
        # Step 6: Calculate composite scores
        feature_importances = self.calculate_composite_score(feature_importances)
        
        # Step 7: Select features based on method
        selected_features = self._apply_selection_method(
            feature_importances, method, rfe_features, collinearity
        )
        
        # Step 8: Ensure minimum and maximum feature counts
        selected_features = self._enforce_feature_limits(feature_importances, selected_features)
        
        # Update selection status
        for fi in feature_importances:
            fi.selected = fi.feature_name in selected_features
        
        removed_features = [f for f in X.columns if f not in selected_features]
        
        # Step 9: Evaluate performance
        performance_metrics = self._evaluate_selection_performance(
            X, y, selected_features, cv_folds
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        results = SelectionResults(
            original_feature_count=X.shape[1],
            selected_feature_count=len(selected_features),
            removed_feature_count=len(removed_features),
            feature_importances=feature_importances,
            selected_features=selected_features,
            removed_features=removed_features,
            performance_metrics=performance_metrics,
            selection_method=method,
            selection_params={
                'correlation_threshold': self.correlation_threshold,
                'vif_threshold': self.vif_threshold,
                'shap_threshold': self.shap_threshold,
                'min_features': self.min_features,
                'max_features': self.max_features,
                'cv_folds': cv_folds
            },
            execution_time=execution_time
        )
        
        self.selection_history.append(results)
        logger.info(f"Feature selection completed: {len(selected_features)}/{X.shape[1]} features selected")
        
        return results
    
    def _apply_selection_method(self, 
                              feature_importances: List[FeatureImportance],
                              method: str,
                              rfe_features: List[str],
                              collinearity: CollinearityAnalysis) -> List[str]:
        """Apply the specified selection method."""
        
        if method == 'composite':
            # Sort by composite score and select top features
            sorted_features = sorted(
                feature_importances, 
                key=lambda x: x.composite_score, 
                reverse=True
            )
            
            # Remove highly collinear features
            selected = []
            for fi in sorted_features:
                if fi.feature_name not in collinearity.collinear_features:
                    selected.append(fi.feature_name)
                elif len(selected) < self.min_features:
                    selected.append(fi.feature_name)  # Keep if needed for minimum
            
            return selected
            
        elif method == 'shap':
            # Select based on SHAP importance
            if SHAP_AVAILABLE:
                sorted_features = sorted(
                    [fi for fi in feature_importances if fi.shap_importance is not None],
                    key=lambda x: x.shap_importance or 0.0,
                    reverse=True
                )
                return [fi.feature_name for fi in sorted_features 
                       if (fi.shap_importance or 0.0) >= self.shap_threshold]
            else:
                logger.warning("SHAP not available, falling back to model importance")
                return self._apply_selection_method(feature_importances, 'model', rfe_features, collinearity)
                
        elif method == 'rfe':
            # Use RFE selected features
            return rfe_features
            
        elif method == 'correlation':
            # Remove only highly correlated features
            return [fi.feature_name for fi in feature_importances 
                   if fi.feature_name not in collinearity.collinear_features]
        
        elif method == 'model':
            # Select based on model feature importance
            sorted_features = sorted(
                feature_importances,
                key=lambda x: x.model_importance or 0.0,
                reverse=True
            )
            # Select top 50% or minimum features
            n_select = max(self.min_features, len(sorted_features) // 2)
            return [fi.feature_name for fi in sorted_features[:n_select]]
        
        else:
            logger.warning(f"Unknown selection method '{method}', using composite")
            return self._apply_selection_method(feature_importances, 'composite', rfe_features, collinearity)
    
    def _enforce_feature_limits(self, 
                               feature_importances: List[FeatureImportance],
                               selected_features: List[str]) -> List[str]:
        """Enforce minimum and maximum feature count limits."""
        
        # Sort all features by composite score
        sorted_features = sorted(
            feature_importances,
            key=lambda x: x.composite_score,
            reverse=True
        )
        
        # Enforce minimum
        if len(selected_features) < self.min_features:
            for fi in sorted_features:
                if fi.feature_name not in selected_features:
                    selected_features.append(fi.feature_name)
                    if len(selected_features) >= self.min_features:
                        break
        
        # Enforce maximum
        if self.max_features and len(selected_features) > self.max_features:
            # Keep top features by composite score
            top_features = [fi.feature_name for fi in sorted_features[:self.max_features]]
            selected_features = [f for f in selected_features if f in top_features]
        
        return selected_features
    
    def _evaluate_selection_performance(self, 
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      selected_features: List[str],
                                      cv_folds: int) -> Dict[str, float]:
        """Evaluate the performance of feature selection."""
        
        try:
            # Compare performance with all features vs selected features
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            X_all = self.scaler.fit_transform(X.fillna(0))
            X_selected = self.scaler.fit_transform(X[selected_features].fillna(0))
            
            # Evaluate with all features
            all_scores = []
            for train_idx, test_idx in tscv.split(X_all):
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_all[train_idx], y.iloc[train_idx])
                pred = model.predict(X_all[test_idx])
                score = r2_score(y.iloc[test_idx], pred)
                all_scores.append(score)
            
            # Evaluate with selected features
            selected_scores = []
            for train_idx, test_idx in tscv.split(X_selected):
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_selected[train_idx], y.iloc[train_idx])
                pred = model.predict(X_selected[test_idx])
                score = r2_score(y.iloc[test_idx], pred)
                selected_scores.append(score)
            
            return {
                'all_features_r2': float(np.mean(all_scores)),
                'selected_features_r2': float(np.mean(selected_scores)),
                'r2_difference': float(np.mean(selected_scores) - np.mean(all_scores)),
                'feature_reduction_ratio': len(selected_features) / X.shape[1],
                'complexity_reduction': 1.0 - (len(selected_features) / X.shape[1])
            }
            
        except Exception as e:
            logger.error(f"Error evaluating selection performance: {e}")
            return {
                'all_features_r2': 0.0,
                'selected_features_r2': 0.0,
                'r2_difference': 0.0,
                'feature_reduction_ratio': len(selected_features) / X.shape[1],
                'complexity_reduction': 1.0 - (len(selected_features) / X.shape[1])
            }
    
    async def save_selection_artifacts(self, 
                                     results: SelectionResults,
                                     output_dir: str = "artifacts/feature_selection") -> Path:
        """Save feature selection artifacts to disk."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_path = Path(output_dir) / f"feature_selection_{timestamp}"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(artifacts_path / "selection_results.json", "w") as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        # Save feature importance details
        importance_data = [asdict(fi) for fi in results.feature_importances]
        with open(artifacts_path / "feature_importance.json", "w") as f:
            json.dump(importance_data, f, indent=2, default=str)
        
        # Save selected features list
        with open(artifacts_path / "selected_features.json", "w") as f:
            json.dump(results.selected_features, f, indent=2)
        
        # Save summary
        summary = {
            "selection_timestamp": timestamp,
            "method": results.selection_method,
            "original_features": results.original_feature_count,
            "selected_features": results.selected_feature_count,
            "reduction_ratio": results.removed_feature_count / results.original_feature_count,
            "performance_improvement": results.performance_metrics.get('r2_difference', 0.0),
            "execution_time": results.execution_time
        }
        
        with open(artifacts_path / "selection_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Feature selection artifacts saved to {artifacts_path}")
        return artifacts_path


# Demo function for testing
async def run_feature_selection_demo():
    """Demo of the feature selection framework."""
    print("Advanced Feature Selection and Pruning Demo")
    print("=" * 60)
    
    # Generate synthetic financial data with noise features
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    n_informative = 15
    
    # Create informative features
    X_informative = np.random.randn(n_samples, n_informative)
    
    # Create target based on informative features
    y = (
        X_informative[:, 0] * 0.5 +
        X_informative[:, 1] * 0.3 +
        X_informative[:, 2] * 0.2 +
        np.random.randn(n_samples) * 0.1
    )
    
    # Add noise features
    X_noise = np.random.randn(n_samples, n_features - n_informative)
    X = np.column_stack([X_informative, X_noise])
    
    # Create DataFrame
    feature_names = [f"informative_{i}" for i in range(n_informative)] + \
                   [f"noise_{i}" for i in range(n_features - n_informative)]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"True informative features: {n_informative}")
    print()
    
    # Initialize feature selector
    selector = AdvancedFeatureSelector(
        correlation_threshold=0.9,
        vif_threshold=10.0,
        shap_threshold=0.001,
        min_features=5,
        max_features=20
    )
    
    # Test different selection methods
    methods = ['composite', 'rfe', 'correlation']
    if SHAP_AVAILABLE:
        methods.append('shap')
    
    for method in methods:
        print(f"Testing {method.upper()} method...")
        
        results = await selector.select_features(
            X_df, y_series, method=method, cv_folds=3
        )
        
        print(f"Selected {results.selected_feature_count}/{results.original_feature_count} features")
        print(f"R² change: {results.performance_metrics['r2_difference']:.4f}")
        print(f"Complexity reduction: {results.performance_metrics['complexity_reduction']:.2%}")
        
        # Show top selected features
        top_features = sorted(
            results.feature_importances,
            key=lambda x: x.composite_score,
            reverse=True
        )[:10]
        
        print("Top 10 features by composite score:")
        for fi in top_features:
            status = "SELECTED" if fi.selected else "REMOVED"
            print(f"  {fi.feature_name}: {fi.composite_score:.4f} ({status})")
        
        print(f"Execution time: {results.execution_time:.2f}s")
        print()
    
    print("Feature selection demo completed!")


if __name__ == "__main__":
    asyncio.run(run_feature_selection_demo())