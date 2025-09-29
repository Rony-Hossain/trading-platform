"""
Transformer training orchestration service for financial sentiment analysis.
Manages the complete training pipeline from data preparation to model evaluation.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np
from sqlalchemy.orm import Session

from .financial_transformer import (
    FinancialTransformerTrainer, TrainingConfig, TargetConfig,
    ModelArchitecture, FinancialTarget, ModelComparison
)
from .financial_dataset_builder import FinancialDatasetBuilder, DatasetConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingJobConfig:
    job_id: str
    symbols: List[str]
    training_approach: str  # "multi_target" or "single_target" or "comparison"
    model_architecture: ModelArchitecture
    dataset_config: DatasetConfig
    training_config: TrainingConfig
    
    # Experiment tracking
    experiment_name: str
    description: str
    tags: List[str]
    
    # Output settings
    save_models: bool = True
    save_datasets: bool = True
    output_dir: str = "./training_experiments"

@dataclass
class TrainingResult:
    job_id: str
    status: str  # "running", "completed", "failed"
    start_time: datetime
    end_time: Optional[datetime]
    
    # Dataset info
    dataset_stats: Dict[str, Any]
    train_samples: int
    val_samples: int
    test_samples: int
    
    # Training results
    training_metrics: Dict[str, Any]
    evaluation_results: Dict[str, Any]
    
    # Model paths
    model_paths: Dict[str, str]
    
    # Performance comparison (if applicable)
    comparison_results: Optional[Dict[str, Any]] = None
    
    # Error info
    error_message: Optional[str] = None

class TransformerTrainingService:
    """Orchestrates transformer training for financial sentiment analysis"""
    
    def __init__(self):
        self.active_jobs = {}  # job_id -> asyncio.Task
        self.job_results = {}  # job_id -> TrainingResult
        self.dataset_builder = FinancialDatasetBuilder(DatasetConfig())
        
    async def start_training_job(self, job_config: TrainingJobConfig, db: Session) -> str:
        """Start a new training job"""
        try:
            logger.info(f"Starting training job: {job_config.job_id}")
            
            # Create initial result
            result = TrainingResult(
                job_id=job_config.job_id,
                status="running",
                start_time=datetime.now(),
                end_time=None,
                dataset_stats={},
                train_samples=0,
                val_samples=0,
                test_samples=0,
                training_metrics={},
                evaluation_results={},
                model_paths={}
            )
            
            self.job_results[job_config.job_id] = result
            
            # Start training task
            task = asyncio.create_task(self._run_training_job(job_config, db))
            self.active_jobs[job_config.job_id] = task
            
            return job_config.job_id
            
        except Exception as e:
            logger.error(f"Failed to start training job {job_config.job_id}: {e}")
            raise
    
    async def _run_training_job(self, job_config: TrainingJobConfig, db: Session):
        """Execute the complete training pipeline"""
        result = self.job_results[job_config.job_id]
        
        try:
            logger.info(f"Executing training job {job_config.job_id}")
            
            # Step 1: Build dataset
            logger.info("Building dataset...")
            dataset_builder = FinancialDatasetBuilder(job_config.dataset_config)
            
            data_points = await dataset_builder.build_dataset(
                db=db,
                symbols=job_config.symbols
            )
            
            if len(data_points) == 0:
                raise ValueError("No data points generated from dataset builder")
            
            # Split dataset
            train_data, val_data, test_data = dataset_builder.create_train_val_test_split(data_points)
            
            # Update dataset stats
            result.dataset_stats = dataset_builder.get_dataset_statistics(data_points)
            result.train_samples = len(train_data)
            result.val_samples = len(val_data)
            result.test_samples = len(test_data)
            
            # Save dataset if requested
            if job_config.save_datasets:
                dataset_dir = Path(job_config.output_dir) / job_config.job_id / "datasets"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                dataset_builder.save_dataset(train_data, dataset_dir / "train_data.pkl")
                dataset_builder.save_dataset(val_data, dataset_dir / "val_data.pkl")
                dataset_builder.save_dataset(test_data, dataset_dir / "test_data.pkl")
            
            # Step 2: Execute training based on approach
            if job_config.training_approach == "multi_target":
                training_results, eval_results, model_paths = await self._train_multi_target_model(
                    job_config, train_data, val_data, test_data
                )
                result.training_metrics = training_results
                result.evaluation_results = eval_results
                result.model_paths = model_paths
                
            elif job_config.training_approach == "single_target":
                training_results, eval_results, model_paths = await self._train_single_target_models(
                    job_config, train_data, val_data, test_data
                )
                result.training_metrics = training_results
                result.evaluation_results = eval_results
                result.model_paths = model_paths
                
            elif job_config.training_approach == "comparison":
                comparison_results = await self._run_model_comparison(
                    job_config, train_data, val_data, test_data
                )
                result.comparison_results = comparison_results
                
            else:
                raise ValueError(f"Unknown training approach: {job_config.training_approach}")
            
            # Mark as completed
            result.status = "completed"
            result.end_time = datetime.now()
            
            logger.info(f"Training job {job_config.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job_config.job_id} failed: {e}")
            result.status = "failed"
            result.error_message = str(e)
            result.end_time = datetime.now()
            
        finally:
            # Clean up
            if job_config.job_id in self.active_jobs:
                del self.active_jobs[job_config.job_id]
    
    async def _train_multi_target_model(self, job_config: TrainingJobConfig,
                                      train_data, val_data, test_data) -> tuple:
        """Train a multi-target model"""
        try:
            logger.info("Training multi-target model...")
            
            # Create trainer
            trainer = FinancialTransformerTrainer(job_config.training_config)
            trainer.prepare_model_and_tokenizer()
            
            # Prepare datasets
            train_dataset, val_dataset = trainer.prepare_dataset(train_data, val_data)
            test_dataset = trainer.prepare_dataset([test_data[0]], [test_data[0]])[0]  # Single item for format
            
            # Train
            training_metrics = trainer.train(train_dataset, val_dataset)
            
            # Evaluate
            eval_results = trainer.evaluate(test_dataset)
            
            # Format results
            formatted_eval_results = {}
            for target, result in eval_results.items():
                formatted_eval_results[target.value] = {
                    'target_type': result.target_type,
                    'metrics': result.metrics
                }
            
            # Save model
            model_paths = {}
            if job_config.save_models:
                model_dir = Path(job_config.output_dir) / job_config.job_id / "multi_target_model"
                model_dir.mkdir(parents=True, exist_ok=True)
                trainer.trainer.save_model(str(model_dir))
                trainer.tokenizer.save_pretrained(str(model_dir))
                model_paths['multi_target'] = str(model_dir)
            
            return training_metrics, formatted_eval_results, model_paths
            
        except Exception as e:
            logger.error(f"Multi-target training failed: {e}")
            raise
    
    async def _train_single_target_models(self, job_config: TrainingJobConfig,
                                        train_data, val_data, test_data) -> tuple:
        """Train separate single-target models"""
        try:
            logger.info("Training single-target models...")
            
            all_training_metrics = {}
            all_eval_results = {}
            all_model_paths = {}
            
            # Define targets to train
            targets = [
                TargetConfig(FinancialTarget.SENTIMENT, "classification", num_classes=3),
                TargetConfig(FinancialTarget.PRICE_DIRECTION, "classification", num_classes=3),
                TargetConfig(FinancialTarget.VOLATILITY, "classification", num_classes=3),
                TargetConfig(FinancialTarget.PRICE_MAGNITUDE, "regression")
            ]
            
            for target_config in targets:
                target_name = target_config.target_name.value
                logger.info(f"Training single-target model for {target_name}")
                
                # Create single-target training config
                single_config = TrainingConfig(
                    model_name=job_config.model_architecture,
                    target_configs=[target_config],
                    max_length=job_config.training_config.max_length,
                    batch_size=job_config.training_config.batch_size,
                    learning_rate=job_config.training_config.learning_rate,
                    num_epochs=job_config.training_config.num_epochs,
                    output_dir=f"{job_config.output_dir}/{job_config.job_id}/single_target_{target_name}"
                )
                
                # Train model
                trainer = FinancialTransformerTrainer(single_config)
                trainer.prepare_model_and_tokenizer()
                
                train_dataset, val_dataset = trainer.prepare_dataset(train_data, val_data)
                training_metrics = trainer.train(train_dataset, val_dataset)
                
                # Evaluate
                test_dataset = trainer.prepare_dataset([test_data[0]], [test_data[0]])[0]
                eval_results = trainer.evaluate(test_dataset)
                
                # Store results
                all_training_metrics[target_name] = training_metrics
                all_eval_results[target_name] = {
                    'target_type': target_config.target_type,
                    'metrics': eval_results[target_config.target_name].metrics
                }
                
                # Save model
                if job_config.save_models:
                    model_dir = Path(single_config.output_dir)
                    all_model_paths[target_name] = str(model_dir)
            
            return all_training_metrics, all_eval_results, all_model_paths
            
        except Exception as e:
            logger.error(f"Single-target training failed: {e}")
            raise
    
    async def _run_model_comparison(self, job_config: TrainingJobConfig,
                                  train_data, val_data, test_data) -> Dict[str, Any]:
        """Run comprehensive comparison between multi-target and single-target approaches"""
        try:
            logger.info("Running model comparison...")
            
            comparison = ModelComparison()
            results = comparison.compare_approaches(train_data, val_data, test_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job"""
        if job_id not in self.job_results:
            return None
        
        result = self.job_results[job_id]
        
        status = {
            'job_id': result.job_id,
            'status': result.status,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'dataset_stats': result.dataset_stats,
            'samples': {
                'train': result.train_samples,
                'validation': result.val_samples,
                'test': result.test_samples
            }
        }
        
        if result.status == "completed":
            status['training_metrics'] = result.training_metrics
            status['evaluation_results'] = result.evaluation_results
            status['model_paths'] = result.model_paths
            
            if result.comparison_results:
                status['comparison_results'] = result.comparison_results
        
        elif result.status == "failed":
            status['error_message'] = result.error_message
        
        return status
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs"""
        jobs = []
        for job_id, result in self.job_results.items():
            job_info = {
                'job_id': job_id,
                'status': result.status,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'total_samples': result.train_samples + result.val_samples + result.test_samples
            }
            jobs.append(job_info)
        
        return sorted(jobs, key=lambda x: x['start_time'], reverse=True)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job"""
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            
            # Update result
            if job_id in self.job_results:
                result = self.job_results[job_id]
                result.status = "cancelled"
                result.end_time = datetime.now()
            
            return True
        
        return False
    
    def get_training_recommendations(self, symbol_list: List[str]) -> Dict[str, Any]:
        """Get training recommendations based on available data"""
        recommendations = {
            'recommended_approach': 'multi_target',
            'recommended_architecture': ModelArchitecture.FINBERT.value,
            'estimated_training_time': '2-4 hours',
            'recommendations': [
                'Use multi-target approach for better feature sharing',
                'Start with FinBERT for financial domain adaptation',
                'Use 3 epochs to avoid overfitting',
                'Monitor validation loss for early stopping'
            ],
            'data_requirements': {
                'min_samples_per_symbol': 50,
                'recommended_total_samples': 1000,
                'min_date_range_days': 90
            }
        }
        
        # Estimate based on symbol count
        if len(symbol_list) > 10:
            recommendations['estimated_training_time'] = '4-8 hours'
            recommendations['recommendations'].append('Consider distributed training for large symbol sets')
        
        if len(symbol_list) < 3:
            recommendations['recommendations'].append('Add more symbols for better generalization')
        
        return recommendations
    
    def create_default_job_config(self, symbols: List[str], approach: str = "multi_target") -> TrainingJobConfig:
        """Create a default training job configuration"""
        
        # Create target configurations
        target_configs = [
            TargetConfig(
                target_name=FinancialTarget.SENTIMENT,
                target_type="classification",
                num_classes=3,
                loss_weight=1.0
            ),
            TargetConfig(
                target_name=FinancialTarget.PRICE_DIRECTION,
                target_type="classification",
                num_classes=3,
                loss_weight=1.2  # Higher weight for financial prediction
            ),
            TargetConfig(
                target_name=FinancialTarget.VOLATILITY,
                target_type="classification",
                num_classes=3,
                loss_weight=1.0
            ),
            TargetConfig(
                target_name=FinancialTarget.PRICE_MAGNITUDE,
                target_type="regression",
                loss_weight=0.8
            )
        ]
        
        # Create training config
        training_config = TrainingConfig(
            model_name=ModelArchitecture.FINBERT,
            target_configs=target_configs,
            max_length=512,
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=3,
            early_stopping_patience=3
        )
        
        # Create dataset config
        dataset_config = DatasetConfig(
            lookback_days=365,
            min_sentiment_per_symbol=50,
            min_text_length=10,
            max_text_length=512
        )
        
        job_id = f"training_{approach}_{'_'.join(symbols)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return TrainingJobConfig(
            job_id=job_id,
            symbols=symbols,
            training_approach=approach,
            model_architecture=ModelArchitecture.FINBERT,
            dataset_config=dataset_config,
            training_config=training_config,
            experiment_name=f"Financial Sentiment {approach.title()}",
            description=f"Training {approach} model for symbols: {', '.join(symbols)}",
            tags=["financial", "sentiment", approach],
            save_models=True,
            save_datasets=True
        )