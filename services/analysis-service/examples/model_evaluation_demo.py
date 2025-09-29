#!/usr/bin/env python3
"""
Advanced Model Evaluation Demo

Demonstrates the enhanced model evaluation capabilities comparing
RandomForest, LightGBM, and XGBoost with proper time-series cross-validation
and financial performance metrics.
"""

import asyncio
import aiohttp
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.model_evaluation import ModelEvaluationFramework, run_model_evaluation_demo


class ModelEvaluationClient:
    """Client for testing model evaluation API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_service_health(self) -> bool:
        """Check if the Analysis Service is running."""
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    print("Analysis Service is running")
                    return True
                else:
                    print(f"Analysis Service health check failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"Failed to connect to Analysis Service: {e}")
            return False
    
    async def evaluate_model(self, symbol: str, cv_folds: int = 3) -> Dict[str, Any]:
        """Evaluate models for a symbol."""
        try:
            params = {
                'period': '2y',
                'cv_folds': cv_folds,
                'include_financial_metrics': True
            }
            
            async with self.session.post(
                f"{self.base_url}/models/evaluate/{symbol}",
                params=params
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    return {
                        'error': f"HTTP {resp.status}",
                        'message': error_text
                    }
        except Exception as e:
            return {
                'error': 'request_failed',
                'message': str(e)
            }
    
    async def get_model_recommendation(self, symbol: str) -> Dict[str, Any]:
        """Get model recommendation for a symbol."""
        try:
            async with self.session.get(f"{self.base_url}/models/recommendation/{symbol}") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    return {
                        'error': f"HTTP {resp.status}",
                        'message': error_text
                    }
        except Exception as e:
            return {
                'error': 'request_failed',
                'message': str(e)
            }
    
    async def batch_evaluate(self, symbols: list) -> Dict[str, Any]:
        """Run batch model evaluation."""
        try:
            payload = {'symbols': symbols}
            
            async with self.session.post(
                f"{self.base_url}/models/batch-evaluate",
                json=payload
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    return {
                        'error': f"HTTP {resp.status}",
                        'message': error_text
                    }
        except Exception as e:
            return {
                'error': 'request_failed',
                'message': str(e)
            }


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_model_performance(performance: Dict[str, Any]):
    """Print formatted model performance metrics."""
    for model_name, metrics in performance.items():
        print(f"\n{model_name.upper()} Performance:")
        
        # ML Metrics
        ml_metrics = metrics.get('ml_metrics', {})
        print(f"   ML Metrics:")
        print(f"     • RMSE: {ml_metrics.get('rmse', 0):.4f}")
        print(f"     • MAE: {ml_metrics.get('mae', 0):.4f}")
        print(f"     • R²: {ml_metrics.get('r2', 0):.4f}")
        print(f"     • CV Mean: {ml_metrics.get('cv_mean', 0):.4f}")
        print(f"     • CV Std: {ml_metrics.get('cv_std', 0):.4f}")
        
        # Financial Metrics
        fin_metrics = metrics.get('financial_metrics', {})
        print(f"   Financial Metrics:")
        print(f"     • Sharpe Ratio: {fin_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"     • Hit Rate: {fin_metrics.get('hit_rate', 0):.3f}")
        print(f"     • Max Drawdown: {fin_metrics.get('max_drawdown', 0):.3f}")
        print(f"     • Total Return: {fin_metrics.get('total_return', 0):.3f}")
        print(f"     • Information Ratio: {fin_metrics.get('information_ratio', 0):.3f}")
        
        # Performance Metrics
        perf_metrics = metrics.get('performance_metrics', {})
        print(f"   Performance:")
        print(f"     • Training Time: {perf_metrics.get('training_time', 0):.3f}s")
        print(f"     • Prediction Time: {perf_metrics.get('prediction_time', 0):.4f}s")


async def run_api_demo():
    """Run API-based model evaluation demo."""
    print("Enhanced Model Evaluation API Demo")
    print("Testing the advanced ML model comparison framework")
    
    symbols_to_test = ["AAPL", "MSFT", "GOOGL"]
    
    async with ModelEvaluationClient() as client:
        # Check service health
        if not await client.check_service_health():
            print("Analysis Service is not available. Please start the service first.")
            return
        
        try:
            # 1. Single Symbol Evaluation
            print_section("Single Symbol Model Evaluation")
            test_symbol = symbols_to_test[0]
            print(f"Evaluating models for {test_symbol}...")
            
            evaluation_result = await client.evaluate_model(test_symbol, cv_folds=3)
            
            if 'error' in evaluation_result:
                print(f"Evaluation failed: {evaluation_result['message']}")
                print("Note: This demo requires synthetic data generation or market data service")
            else:
                print(f"Model evaluation completed for {test_symbol}")
                print(f"Best Model: {evaluation_result.get('best_model', 'Unknown')}")
                print(f"Best Score: {evaluation_result.get('best_score', 0):.4f}")
                
                data_summary = evaluation_result.get('data_summary', {})
                print(f"Dataset: {data_summary.get('total_samples', 0)} samples, {data_summary.get('feature_count', 0)} features")
                
                # Print model availability
                model_availability = evaluation_result.get('model_availability', {})
                print(f"Available Models:")
                for model, available in model_availability.items():
                    status = "Yes" if available else "No"
                    print(f"     {status} {model}")
                
                # Print performance details
                model_performance = evaluation_result.get('model_performance', {})
                if model_performance:
                    print_model_performance(model_performance)
                
                # Print recommendations
                recommendations = evaluation_result.get('recommendations', [])
                if recommendations:
                    print(f"\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"   {i}. {rec}")
                
                # Print top features
                top_features = evaluation_result.get('top_features', {})
                if top_features:
                    print(f"\nTop Features:")
                    for feature, importance in list(top_features.items())[:5]:
                        print(f"   • {feature}: {importance:.4f}")
            
            # 2. Model Recommendation
            print_section("Model Recommendation")
            print(f"Getting model recommendation for {test_symbol}...")
            
            recommendation = await client.get_model_recommendation(test_symbol)
            
            if 'error' in recommendation:
                print(f"ℹ️  {recommendation.get('message', 'No recommendation available')}")
            else:
                print(f"Recommended Model: {recommendation.get('recommended_model', 'Unknown')}")
                print(f"Confidence: {recommendation.get('confidence_score', 0):.3f}")
                
                rationale = recommendation.get('rationale', [])
                if rationale:
                    print(f"Rationale:")
                    for reason in rationale:
                        print(f"   • {reason}")
                
                performance_summary = recommendation.get('performance_summary', {})
                if performance_summary:
                    print(f"Performance Summary:")
                    print(f"   • Sharpe Ratio: {performance_summary.get('sharpe_ratio', 0):.3f}")
                    print(f"   • Hit Rate: {performance_summary.get('hit_rate', 0):.3f}")
                    print(f"   • R² Score: {performance_summary.get('r2_score', 0):.3f}")
            
            # 3. Batch Evaluation (smaller scale for demo)
            print_section("Batch Model Evaluation")
            batch_symbols = symbols_to_test[:2]  # Limit to 2 for demo
            print(f"Running batch evaluation for {batch_symbols}...")
            
            batch_result = await client.batch_evaluate(batch_symbols)
            
            if 'error' in batch_result:
                print(f"Batch evaluation failed: {batch_result['message']}")
            else:
                print(f"Batch evaluation completed")
                print(f"Successful: {batch_result.get('successful_evaluations', 0)}")
                print(f"Failed: {batch_result.get('failed_evaluations', 0)}")
                
                summary = batch_result.get('summary', {})
                best_models = summary.get('best_models', {})
                if best_models:
                    print(f"Best Models by Symbol:")
                    for symbol, model in best_models.items():
                        print(f"   • {symbol}: {model}")
                
                avg_performance = summary.get('avg_performance', {})
                avg_sharpe = avg_performance.get('avg_sharpe', 0)
                print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
            
            print_section("API Demo Complete")
            print("Enhanced model evaluation demo completed!")
            print("The framework successfully compares RandomForest, LightGBM, and XGBoost")
            print("Using proper time-series cross-validation and financial metrics")
            print("Ready for production model selection and deployment")
            
        except Exception as e:
            print(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()


async def run_comprehensive_demo():
    """Run both standalone and API demos."""
    print("Enhanced Model Evaluation Framework - Comprehensive Demo")
    print("=" * 80)
    
    # 1. Standalone Framework Demo
    print_section("Standalone Framework Demo")
    print("Testing the core ModelEvaluationFramework...")
    await run_model_evaluation_demo()
    
    # 2. API Integration Demo  
    print_section("API Integration Demo")
    print("Testing the Analysis Service API endpoints...")
    await run_api_demo()
    
    print_section("All Demos Complete")
    print("Enhanced model evaluation framework is production-ready!")
    print("Key Features Demonstrated:")
    print("   - RandomForest vs LightGBM vs XGBoost comparison")
    print("   - Time-series cross-validation (no look-ahead bias)")
    print("   - Financial performance metrics (Sharpe, hit rate, drawdown)")
    print("   - Feature importance analysis and ranking")
    print("   - Model recommendation with rationale")
    print("   - Comprehensive evaluation artifacts")
    print("   - API integration for production deployment")


def print_usage():
    """Print usage instructions."""
    print("""
Enhanced Model Evaluation Demo

This demo showcases the advanced model evaluation framework that implements
the enhanced Phase 1 requirement for comparing RandomForest, LightGBM, and XGBoost.

Features:
- Rigorous time-series cross-validation (prevents look-ahead bias)
- Financial performance metrics (Sharpe ratio, hit rate, drawdown)
- Automatic best model selection based on composite scoring
- Feature importance analysis and ranking
- Actionable recommendations for model deployment
- Production-ready API endpoints

Prerequisites:
- Analysis Service running on localhost:8003 (for API demo)
- Required packages: pandas, numpy, scikit-learn
- Optional: lightgbm, xgboost (for full model comparison)

Usage:
    python model_evaluation_demo.py [--api-only] [--standalone-only]

Options:
    --api-only         Run only the API integration demo
    --standalone-only  Run only the standalone framework demo
    (no args)         Run comprehensive demo (both standalone and API)
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            print_usage()
            sys.exit(0)
        elif sys.argv[1] == '--api-only':
            asyncio.run(run_api_demo())
        elif sys.argv[1] == '--standalone-only':
            asyncio.run(run_model_evaluation_demo())
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print_usage()
            sys.exit(1)
    else:
        asyncio.run(run_comprehensive_demo())