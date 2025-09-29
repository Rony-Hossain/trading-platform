"""
Model Interpretability Service
Implements SHAP/LIME for local model interpretability and feature attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available, feature attribution will be limited")

# LIME for explainability
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available, alternative explanations will be used")

# Additional interpretability tools
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ExplainerType(Enum):
    """Types of model explainers"""
    SHAP_TREE = "shap_tree"
    SHAP_LINEAR = "shap_linear"
    SHAP_KERNEL = "shap_kernel"
    SHAP_DEEP = "shap_deep"
    LIME_TABULAR = "lime_tabular"
    PERMUTATION = "permutation"

class AttributionScope(Enum):
    """Scope of feature attribution analysis"""
    GLOBAL = "global"
    LOCAL = "local"
    REGIME_SPECIFIC = "regime_specific"
    COMPARATIVE = "comparative"

@dataclass
class FeatureAttribution:
    """Feature attribution result for a single prediction"""
    feature_names: List[str]
    feature_values: List[float]
    attribution_scores: List[float]
    prediction: float
    baseline: float
    explainer_type: ExplainerType
    confidence: float
    timestamp: datetime

@dataclass
class GlobalAttribution:
    """Global feature importance across multiple predictions"""
    feature_names: List[str]
    mean_importance: List[float]
    std_importance: List[float]
    importance_rank: List[int]
    sample_count: int
    explainer_type: ExplainerType
    regime_context: Optional[str] = None

@dataclass
class RegimeAttribution:
    """Feature attribution comparison across market regimes"""
    regime_name: str
    feature_attributions: Dict[str, float]
    dominant_features: List[str]
    regime_confidence: float
    sample_count: int

@dataclass
class InterpretabilityReport:
    """Comprehensive model interpretability report"""
    model_name: str
    global_attribution: GlobalAttribution
    local_attributions: List[FeatureAttribution]
    regime_attributions: Dict[str, RegimeAttribution]
    stability_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    compliance_summary: Dict[str, Any]
    timestamp: datetime

class ModelExplainer:
    """Model explainability service with SHAP/LIME integration"""
    
    def __init__(self, explainer_types: List[ExplainerType] = None):
        self.explainer_types = explainer_types or [
            ExplainerType.SHAP_TREE, ExplainerType.LIME_TABULAR, ExplainerType.PERMUTATION
        ]
        self.explainers = {}
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def _create_shap_explainer(self, model: BaseEstimator, X_train: np.ndarray, 
                              explainer_type: ExplainerType) -> Optional[Any]:
        """Create SHAP explainer based on model type"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping SHAP explainer creation")
            return None
        
        try:
            if explainer_type == ExplainerType.SHAP_TREE:
                # For tree-based models (RandomForest, XGBoost, etc.)
                if hasattr(model, 'estimators_') or hasattr(model, 'tree_'):
                    return shap.TreeExplainer(model)
                else:
                    logger.warning("Model not compatible with TreeExplainer, using KernelExplainer")
                    return shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
                    
            elif explainer_type == ExplainerType.SHAP_LINEAR:
                # For linear models
                if hasattr(model, 'coef_'):
                    return shap.LinearExplainer(model, X_train)
                else:
                    logger.warning("Model not compatible with LinearExplainer, using KernelExplainer")
                    return shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
                    
            elif explainer_type == ExplainerType.SHAP_KERNEL:
                # General purpose explainer (slower but works with any model)
                background = shap.sample(X_train, min(100, len(X_train)))
                return shap.KernelExplainer(model.predict, background)
                
            elif explainer_type == ExplainerType.SHAP_DEEP:
                # For deep learning models
                logger.warning("Deep learning SHAP explainer not implemented yet")
                return shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
                
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {str(e)}")
            return None
    
    def _create_lime_explainer(self, X_train: np.ndarray, feature_names: List[str]) -> Optional[Any]:
        """Create LIME explainer"""
        if not LIME_AVAILABLE:
            logger.warning("LIME not available, skipping LIME explainer creation")
            return None
        
        try:
            return lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                mode='regression',
                discretize_continuous=True,
                random_state=42
            )
        except Exception as e:
            logger.error(f"Error creating LIME explainer: {str(e)}")
            return None
    
    def fit_explainers(self, model: BaseEstimator, X_train: np.ndarray, 
                      feature_names: List[str]) -> Dict[ExplainerType, Any]:
        """Fit multiple explainers to the model"""
        logger.info("Fitting model explainers")
        
        self.feature_names = feature_names
        explainers = {}
        
        for explainer_type in self.explainer_types:
            try:
                if explainer_type in [ExplainerType.SHAP_TREE, ExplainerType.SHAP_LINEAR, 
                                    ExplainerType.SHAP_KERNEL, ExplainerType.SHAP_DEEP]:
                    explainer = self._create_shap_explainer(model, X_train, explainer_type)
                    if explainer:
                        explainers[explainer_type] = explainer
                        logger.info(f"Created {explainer_type.value} explainer")
                
                elif explainer_type == ExplainerType.LIME_TABULAR:
                    explainer = self._create_lime_explainer(X_train, feature_names)
                    if explainer:
                        explainers[explainer_type] = explainer
                        logger.info(f"Created {explainer_type.value} explainer")
                
                elif explainer_type == ExplainerType.PERMUTATION:
                    # Permutation importance doesn't need fitting
                    explainers[explainer_type] = model
                    logger.info(f"Registered {explainer_type.value} explainer")
                    
            except Exception as e:
                logger.warning(f"Failed to create {explainer_type.value} explainer: {str(e)}")
        
        self.explainers = explainers
        return explainers
    
    def explain_prediction(self, X_sample: np.ndarray, explainer_type: ExplainerType = None,
                          num_features: int = 10) -> Optional[FeatureAttribution]:
        """Explain a single prediction using specified explainer"""
        
        if explainer_type is None:
            # Use first available explainer
            explainer_type = list(self.explainers.keys())[0] if self.explainers else None
        
        if explainer_type not in self.explainers:
            logger.error(f"Explainer {explainer_type} not available")
            return None
        
        explainer = self.explainers[explainer_type]
        
        try:
            if explainer_type in [ExplainerType.SHAP_TREE, ExplainerType.SHAP_LINEAR, 
                                ExplainerType.SHAP_KERNEL, ExplainerType.SHAP_DEEP]:
                return self._explain_with_shap(explainer, X_sample, explainer_type)
                
            elif explainer_type == ExplainerType.LIME_TABULAR:
                return self._explain_with_lime(explainer, X_sample)
                
            elif explainer_type == ExplainerType.PERMUTATION:
                logger.warning("Permutation importance is for global explanations only")
                return None
                
        except Exception as e:
            logger.error(f"Error explaining prediction with {explainer_type.value}: {str(e)}")
            return None
    
    def _explain_with_shap(self, explainer: Any, X_sample: np.ndarray, 
                          explainer_type: ExplainerType) -> FeatureAttribution:
        """Explain prediction using SHAP"""
        
        # Get SHAP values
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For classification or multi-output
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Take first sample
        
        # Get baseline and prediction
        if hasattr(explainer, 'expected_value'):
            baseline = explainer.expected_value
            if isinstance(baseline, np.ndarray):
                baseline = baseline[0] if len(baseline) > 0 else 0
        else:
            baseline = 0
        
        # Calculate prediction (baseline + sum of SHAP values)
        prediction = baseline + np.sum(shap_values)
        
        # Calculate confidence (based on SHAP value concentration)
        total_importance = np.sum(np.abs(shap_values))
        top_features_importance = np.sum(np.abs(np.sort(shap_values)[-5:]))
        confidence = top_features_importance / total_importance if total_importance > 0 else 0.5
        
        return FeatureAttribution(
            feature_names=self.feature_names[:len(shap_values)],
            feature_values=X_sample[0][:len(shap_values)].tolist(),
            attribution_scores=shap_values.tolist(),
            prediction=prediction,
            baseline=baseline,
            explainer_type=explainer_type,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def _explain_with_lime(self, explainer: Any, X_sample: np.ndarray) -> FeatureAttribution:
        """Explain prediction using LIME"""
        
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # Get model prediction function
        model = self.explainers.get(ExplainerType.PERMUTATION)
        if model is None:
            logger.error("No model available for LIME explanation")
            return None
        
        def predict_fn(X):
            return model.predict(X)
        
        # Generate explanation
        explanation = explainer.explain_instance(
            X_sample[0], 
            predict_fn,
            num_features=len(self.feature_names)
        )
        
        # Extract feature importances
        feature_importance_dict = dict(explanation.as_list())
        
        # Map to our feature names and get scores
        attribution_scores = []
        for feature_name in self.feature_names:
            score = feature_importance_dict.get(feature_name, 0.0)
            attribution_scores.append(score)
        
        # Get prediction
        prediction = predict_fn(X_sample)[0]
        baseline = np.mean([predict_fn(self.explainers[ExplainerType.LIME_TABULAR].training_data[i:i+1]) 
                           for i in range(min(10, len(self.explainers[ExplainerType.LIME_TABULAR].training_data)))])
        
        # Calculate confidence
        total_importance = np.sum(np.abs(attribution_scores))
        top_features_importance = np.sum(np.abs(np.sort(attribution_scores)[-5:]))
        confidence = top_features_importance / total_importance if total_importance > 0 else 0.5
        
        return FeatureAttribution(
            feature_names=self.feature_names,
            feature_values=X_sample[0].tolist(),
            attribution_scores=attribution_scores,
            prediction=prediction,
            baseline=baseline,
            explainer_type=ExplainerType.LIME_TABULAR,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def global_feature_importance(self, model: BaseEstimator, X_test: np.ndarray, 
                                 y_test: np.ndarray, num_repeats: int = 10) -> GlobalAttribution:
        """Calculate global feature importance using permutation importance"""
        
        try:
            logger.info("Calculating global feature importance")
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test, 
                n_repeats=num_repeats, 
                random_state=42,
                n_jobs=-1
            )
            
            # Get mean and std importance
            mean_importance = perm_importance.importances_mean
            std_importance = perm_importance.importances_std
            
            # Calculate rank
            importance_rank = np.argsort(mean_importance)[::-1]
            
            return GlobalAttribution(
                feature_names=self.feature_names,
                mean_importance=mean_importance.tolist(),
                std_importance=std_importance.tolist(),
                importance_rank=importance_rank.tolist(),
                sample_count=len(X_test),
                explainer_type=ExplainerType.PERMUTATION
            )
            
        except Exception as e:
            logger.error(f"Error calculating global feature importance: {str(e)}")
            return None
    
    def regime_specific_attribution(self, model: BaseEstimator, X_data: np.ndarray, 
                                  regime_labels: List[str], 
                                  regime_names: List[str]) -> Dict[str, RegimeAttribution]:
        """Calculate feature attribution for each market regime"""
        
        regime_attributions = {}
        unique_regimes = list(set(regime_labels))
        
        for regime in unique_regimes:
            try:
                # Get data for this regime
                regime_mask = np.array([label == regime for label in regime_labels])
                regime_X = X_data[regime_mask]
                
                if len(regime_X) < 10:
                    logger.warning(f"Insufficient data for regime {regime} ({len(regime_X)} samples)")
                    continue
                
                # Calculate SHAP values for this regime if possible
                if ExplainerType.SHAP_KERNEL in self.explainers:
                    try:
                        explainer = self.explainers[ExplainerType.SHAP_KERNEL]
                        shap_values = explainer.shap_values(regime_X[:min(50, len(regime_X))])
                        
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        
                        # Calculate mean attribution for each feature
                        mean_attribution = np.mean(np.abs(shap_values), axis=0)
                        
                        # Create attribution dictionary
                        feature_attributions = {
                            self.feature_names[i]: float(mean_attribution[i]) 
                            for i in range(len(self.feature_names))
                        }
                        
                        # Get dominant features (top 5)
                        top_indices = np.argsort(mean_attribution)[-5:][::-1]
                        dominant_features = [self.feature_names[i] for i in top_indices]
                        
                        regime_attributions[regime] = RegimeAttribution(
                            regime_name=regime,
                            feature_attributions=feature_attributions,
                            dominant_features=dominant_features,
                            regime_confidence=0.8,  # Placeholder
                            sample_count=len(regime_X)
                        )
                        
                    except Exception as e:
                        logger.warning(f"SHAP attribution failed for regime {regime}: {str(e)}")
                        continue
                
                # Fallback to permutation importance for this regime
                else:
                    logger.info(f"Using permutation importance for regime {regime}")
                    # This would require regime-specific model evaluation
                    # For now, create placeholder
                    regime_attributions[regime] = RegimeAttribution(
                        regime_name=regime,
                        feature_attributions={name: 0.1 for name in self.feature_names},
                        dominant_features=self.feature_names[:5],
                        regime_confidence=0.5,
                        sample_count=len(regime_X)
                    )
                    
            except Exception as e:
                logger.error(f"Error calculating attribution for regime {regime}: {str(e)}")
                continue
        
        return regime_attributions
    
    def validate_model_stability(self, model: BaseEstimator, X_test: np.ndarray, 
                                num_samples: int = 100) -> Dict[str, float]:
        """Validate model stability through interpretability analysis"""
        
        try:
            logger.info("Validating model stability through interpretability")
            
            # Sample random subsets and calculate attribution consistency
            if len(X_test) < num_samples:
                num_samples = len(X_test)
            
            attribution_consistency = []
            prediction_consistency = []
            
            # Use SHAP if available, otherwise use permutation
            explainer_type = ExplainerType.SHAP_KERNEL if ExplainerType.SHAP_KERNEL in self.explainers else ExplainerType.PERMUTATION
            
            if explainer_type == ExplainerType.SHAP_KERNEL:
                explainer = self.explainers[ExplainerType.SHAP_KERNEL]
                
                # Sample multiple subsets
                for i in range(5):
                    sample_indices = np.random.choice(len(X_test), min(20, len(X_test)), replace=False)
                    sample_X = X_test[sample_indices]
                    
                    try:
                        shap_values = explainer.shap_values(sample_X)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        
                        # Calculate mean attribution
                        mean_attr = np.mean(np.abs(shap_values), axis=0)
                        attribution_consistency.append(mean_attr)
                        
                        # Get predictions
                        predictions = model.predict(sample_X)
                        prediction_consistency.append(np.std(predictions))
                        
                    except Exception as e:
                        logger.warning(f"Error in stability validation iteration {i}: {str(e)}")
                        continue
            
            # Calculate stability metrics
            if attribution_consistency:
                attr_consistency = np.array(attribution_consistency)
                feature_stability = 1 - np.mean(np.std(attr_consistency, axis=0) / (np.mean(attr_consistency, axis=0) + 1e-8))
                
                prediction_stability = 1 - np.mean(prediction_consistency) / (np.std([model.predict(X_test[i:i+1])[0] for i in range(min(100, len(X_test)))]) + 1e-8)
                
                return {
                    'feature_attribution_stability': float(np.clip(feature_stability, 0, 1)),
                    'prediction_stability': float(np.clip(prediction_stability, 0, 1)),
                    'overall_stability': float(np.clip((feature_stability + prediction_stability) / 2, 0, 1)),
                    'sample_count': num_samples
                }
            else:
                return {
                    'feature_attribution_stability': 0.5,
                    'prediction_stability': 0.5,
                    'overall_stability': 0.5,
                    'sample_count': 0
                }
                
        except Exception as e:
            logger.error(f"Error in model stability validation: {str(e)}")
            return {
                'feature_attribution_stability': 0.0,
                'prediction_stability': 0.0,
                'overall_stability': 0.0,
                'sample_count': 0
            }
    
    def generate_compliance_report(self, model_name: str, global_attribution: GlobalAttribution,
                                 stability_metrics: Dict[str, float], 
                                 regime_attributions: Dict[str, RegimeAttribution]) -> Dict[str, Any]:
        """Generate regulatory compliance report for model validation"""
        
        try:
            # Feature importance analysis
            top_features = [global_attribution.feature_names[i] for i in global_attribution.importance_rank[:10]]
            feature_concentration = np.sum(np.array(global_attribution.mean_importance)[global_attribution.importance_rank[:5]]) / np.sum(global_attribution.mean_importance)
            
            # Stability assessment
            stability_score = stability_metrics.get('overall_stability', 0.0)
            stability_grade = 'A' if stability_score > 0.8 else 'B' if stability_score > 0.6 else 'C' if stability_score > 0.4 else 'D'
            
            # Regime consistency
            regime_consistency = self._calculate_regime_consistency(regime_attributions)
            
            # Risk assessment
            risk_factors = []
            if feature_concentration > 0.8:
                risk_factors.append("High feature concentration - model relies heavily on few features")
            if stability_score < 0.6:
                risk_factors.append("Low model stability - predictions may be unreliable")
            if len(regime_attributions) > 0 and regime_consistency < 0.5:
                risk_factors.append("Low regime consistency - feature importance varies significantly across market conditions")
            
            return {
                'model_name': model_name,
                'compliance_status': 'COMPLIANT' if len(risk_factors) == 0 else 'REVIEW_REQUIRED',
                'overall_score': (stability_score + regime_consistency + (1 - feature_concentration)) / 3,
                'stability_grade': stability_grade,
                'feature_analysis': {
                    'top_features': top_features,
                    'feature_concentration': feature_concentration,
                    'total_features': len(global_attribution.feature_names)
                },
                'stability_metrics': stability_metrics,
                'regime_consistency': regime_consistency,
                'risk_factors': risk_factors,
                'recommendations': self._generate_recommendations(stability_score, feature_concentration, regime_consistency),
                'validation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {str(e)}")
            return {
                'model_name': model_name,
                'compliance_status': 'ERROR',
                'error': str(e),
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def _calculate_regime_consistency(self, regime_attributions: Dict[str, RegimeAttribution]) -> float:
        """Calculate consistency of feature importance across regimes"""
        
        if len(regime_attributions) < 2:
            return 1.0
        
        # Get common features across all regimes
        all_features = set()
        for regime_attr in regime_attributions.values():
            all_features.update(regime_attr.feature_attributions.keys())
        
        # Calculate consistency for each feature
        feature_consistencies = []
        
        for feature in all_features:
            feature_values = []
            for regime_attr in regime_attributions.values():
                feature_values.append(regime_attr.feature_attributions.get(feature, 0))
            
            if len(feature_values) > 1:
                # Calculate coefficient of variation (lower is more consistent)
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                consistency = 1 - (std_val / (mean_val + 1e-8))
                feature_consistencies.append(max(0, consistency))
        
        return np.mean(feature_consistencies) if feature_consistencies else 0.5
    
    def _generate_recommendations(self, stability_score: float, feature_concentration: float, 
                                regime_consistency: float) -> List[str]:
        """Generate recommendations based on interpretability analysis"""
        
        recommendations = []
        
        if stability_score < 0.6:
            recommendations.append("Improve model stability through regularization or ensemble methods")
        
        if feature_concentration > 0.8:
            recommendations.append("Consider feature engineering to reduce dependency on few features")
        
        if regime_consistency < 0.5:
            recommendations.append("Implement regime-aware modeling to handle varying feature importance")
        
        if stability_score > 0.8 and feature_concentration < 0.6 and regime_consistency > 0.7:
            recommendations.append("Model shows good interpretability characteristics for production use")
        
        return recommendations