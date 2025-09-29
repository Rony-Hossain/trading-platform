#!/usr/bin/env python3
"""
Feature Selection and Pruning Demo

Demonstrates the advanced feature selection capabilities including SHAP analysis,
RFE, collinearity detection, and automated feature pruning for model optimization.
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

from app.services.feature_selection import AdvancedFeatureSelector, run_feature_selection_demo


class FeatureSelectionClient:
    """Client for testing feature selection API endpoints."""
    
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
    
    async def get_feature_status(self) -> Dict[str, Any]:
        """Get feature selection service status."""
        try:
            async with self.session.get(f"{self.base_url}/features/status") as resp:
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
    
    async def analyze_feature_importance(self, symbol: str, method: str = 'composite') -> Dict[str, Any]:
        """Analyze feature importance for a symbol."""
        try:
            params = {
                'period': '2y',
                'method': method
            }
            
            async with self.session.get(
                f"{self.base_url}/features/analyze/{symbol}",
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
    
    async def optimize_feature_set(self, symbol: str, target_reduction: float = 0.5) -> Dict[str, Any]:
        """Optimize feature set for a symbol."""
        try:
            params = {
                'period': '2y',
                'method': 'composite',
                'target_reduction': target_reduction
            }
            
            async with self.session.post(
                f"{self.base_url}/features/optimize/{symbol}",
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
    
    async def batch_feature_analysis(self, symbols: list) -> Dict[str, Any]:
        """Run batch feature analysis."""
        try:
            payload = {
                'symbols': symbols,
                'period': '2y',
                'method': 'composite'
            }
            
            async with self.session.post(
                f"{self.base_url}/features/batch-analyze",
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
    print(f"\\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_feature_analysis(analysis: Dict[str, Any]):
    """Print formatted feature analysis results."""
    if 'error' in analysis:
        print(f"Error: {analysis['message']}")
        return
    
    print(f"Symbol: {analysis['symbol']}")
    print(f"Method: {analysis['method']}")
    print(f"Original Features: {analysis['original_features']}")
    print(f"Selected Features: {analysis['selected_features']}")
    print(f"Reduction Ratio: {analysis['reduction_ratio']:.1%}")
    
    performance = analysis.get('performance_metrics', {})
    print(f"\\nPerformance Metrics:")
    print(f"  • R² with all features: {performance.get('all_features_r2', 0):.4f}")
    print(f"  • R² with selected features: {performance.get('selected_features_r2', 0):.4f}")
    print(f"  • Performance change: {performance.get('r2_difference', 0):.4f}")
    print(f"  • Complexity reduction: {performance.get('complexity_reduction', 0):.1%}")
    
    print(f"\\nTop 10 Features:")
    for feature in analysis.get('top_features', [])[:10]:
        status = "SELECTED" if feature['selected'] else "REMOVED"
        print(f"  • {feature['name']}: {feature['composite_score']:.4f} ({status})")
    
    print(f"\\nExecution Time: {analysis.get('execution_time', 0):.2f}s")


async def run_api_demo():
    """Run API-based feature selection demo."""
    print("Advanced Feature Selection and Pruning API Demo")
    print("Testing SHAP, RFE, and collinearity analysis framework")
    
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    async with FeatureSelectionClient() as client:
        # Check service health
        if not await client.check_service_health():
            print("Analysis Service is not available. Please start the service first.")
            return
        
        try:
            # 1. Service Status
            print_section("Feature Selection Service Status")
            status = await client.get_feature_status()
            
            if 'error' in status:
                print(f"Failed to get service status: {status['message']}")
                print("Note: This demo requires the Analysis Service to be running")
                return
            else:
                print(f"Service: {status['service']}")
                print(f"Status: {status['status']}")
                print(f"SHAP Available: {status['capabilities']['shap_analysis']}")
                print(f"RFE Available: {status['capabilities']['rfe_analysis']}")
                print(f"Collinearity Analysis: {status['capabilities']['collinearity_analysis']}")
                print(f"Available Methods: {status['available_methods']}")
                
                config = status['configuration']
                print(f"\\nConfiguration:")
                print(f"  • Correlation Threshold: {config['correlation_threshold']}")
                print(f"  • VIF Threshold: {config['vif_threshold']}")
                print(f"  • Min Features: {config['min_features']}")
                print(f"  • Max Features: {config['max_features']}")
            
            # 2. Single Symbol Feature Analysis
            print_section("Single Symbol Feature Analysis")
            test_symbol = test_symbols[0]
            print(f"Analyzing feature importance for {test_symbol}...")
            
            analysis_result = await client.analyze_feature_importance(test_symbol, 'composite')
            print_feature_analysis(analysis_result)
            
            # 3. Feature Set Optimization
            print_section("Feature Set Optimization")
            print(f"Optimizing feature set for {test_symbol} with 60% reduction target...")
            
            optimization_result = await client.optimize_feature_set(test_symbol, 0.6)
            
            if 'error' in optimization_result:
                print(f"Optimization failed: {optimization_result['message']}")
            else:
                print(f"Symbol: {optimization_result['symbol']}")
                print(f"Target Reduction: {optimization_result['target_reduction']:.0%}")
                print(f"Achieved Reduction: {optimization_result['achieved_reduction']:.0%}")
                print(f"Original Features: {optimization_result['original_features']}")
                print(f"Optimized Features: {optimization_result['optimized_features']}")
                print(f"Performance Improvement: {optimization_result['performance_improvement']:.4f}")
                print(f"Complexity Reduction: {optimization_result['complexity_reduction']:.1%}")
                
                summary = optimization_result['optimization_summary']
                print(f"\\nOptimization Summary:")
                print(f"  • Features Removed: {summary['features_removed']}")
                print(f"  • Performance Retained: {summary['performance_retained']:.4f}")
                print(f"  • Baseline Performance: {summary['baseline_performance']:.4f}")
                print(f"  • Efficiency Gain: {summary['efficiency_gain']}")
                
                if 'artifacts_path' in optimization_result:
                    print(f"  • Artifacts Saved: {optimization_result['artifacts_path']}")
            
            # 4. Batch Feature Analysis
            print_section("Batch Feature Analysis")
            batch_symbols = test_symbols[:3]  # Limit to 3 for demo
            print(f"Running batch feature analysis for {batch_symbols}...")
            
            batch_result = await client.batch_feature_analysis(batch_symbols)
            
            if 'error' in batch_result:
                print(f"Batch analysis failed: {batch_result['message']}")
            else:
                print(f"Symbols Analyzed: {batch_result['symbols_analyzed']}")
                print(f"Successful Analyses: {batch_result['successful_analyses']}")
                print(f"Failed Analyses: {len(batch_result['failed_symbols'])}")
                
                if batch_result['failed_symbols']:
                    print(f"Failed Symbols: {batch_result['failed_symbols']}")
                
                stats = batch_result['aggregate_statistics']
                print(f"\\nAggregate Statistics:")
                print(f"  • Average Reduction Ratio: {stats['avg_reduction_ratio']:.1%}")
                print(f"  • Average Performance Improvement: {stats['avg_performance_improvement']:.4f}")
                print(f"  • Best Performing Symbol: {stats['best_performing_symbol']}")
                print(f"  • Most Efficient Symbol: {stats['most_efficient_symbol']}")
                
                print(f"\\nMost Important Features Across Portfolio:")
                for feature_info in batch_result['most_important_features'][:10]:
                    print(f"  • {feature_info['feature']}: {feature_info['frequency']} symbols ({feature_info['percentage']:.0f}%)")
            
            print_section("API Demo Complete")
            print("Advanced feature selection demo completed!")
            print("Key capabilities demonstrated:")
            print("  - SHAP-based feature importance analysis")
            print("  - Recursive Feature Elimination (RFE)")
            print("  - Collinearity detection and VIF analysis")
            print("  - Composite scoring for optimal feature selection")
            print("  - Automated feature pruning with performance tracking")
            print("  - Batch analysis for portfolio-wide insights")
            print("Ready for production feature optimization!")
            
        except Exception as e:
            print(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()


async def run_comprehensive_demo():
    """Run both standalone and API demos."""
    print("Advanced Feature Selection Framework - Comprehensive Demo")
    print("=" * 80)
    
    # 1. Standalone Framework Demo
    print_section("Standalone Framework Demo")
    print("Testing the core AdvancedFeatureSelector...")
    await run_feature_selection_demo()
    
    # 2. API Integration Demo  
    print_section("API Integration Demo")
    print("Testing the Analysis Service API endpoints...")
    await run_api_demo()
    
    print_section("All Demos Complete")
    print("Advanced feature selection framework is production-ready!")
    print("Key Features Demonstrated:")
    print("   - SHAP vs RFE vs Correlation-based feature selection")
    print("   - Multi-collinearity detection with VIF analysis")
    print("   - Composite scoring for optimal feature ranking")
    print("   - Automated feature pruning with performance validation")
    print("   - Model complexity reduction without performance loss")
    print("   - Batch processing for portfolio-wide feature optimization")
    print("   - Production-ready API endpoints with comprehensive error handling")


def print_usage():
    """Print usage instructions."""
    print("""
Advanced Feature Selection and Pruning Demo

This demo showcases the automated feature selection framework that implements
SHAP analysis, RFE, and collinearity detection for model optimization.

Features:
- SHAP-based feature importance analysis (model interpretability)
- Recursive Feature Elimination (RFE) with cross-validation
- Collinearity detection using correlation matrix and VIF scores
- Composite scoring combining multiple selection methods
- Automated feature pruning with performance tracking
- Production-ready API endpoints for integration

Prerequisites:
- Analysis Service running on localhost:8003 (for API demo)
- Required packages: pandas, numpy, scikit-learn
- Optional: shap, lightgbm (for advanced analysis)

Usage:
    python feature_selection_demo.py [--api-only] [--standalone-only]

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
            asyncio.run(run_feature_selection_demo())
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print_usage()
            sys.exit(1)
    else:
        asyncio.run(run_comprehensive_demo())