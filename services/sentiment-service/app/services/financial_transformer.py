"""
Multi-target transformer fine-tuning for financial sentiment analysis.
Implements FinBERT/DistilBERT fine-tuning with multiple financial objectives.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset
import evaluate
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import wandb

logger = logging.getLogger(__name__)

class FinancialTarget(str, Enum):
    SENTIMENT = "sentiment"  # Traditional sentiment classification
    PRICE_DIRECTION = "price_direction"  # Next-day price direction (up/down/flat)
    VOLATILITY = "volatility"  # Expected volatility level (low/medium/high)
    PRICE_MAGNITUDE = "price_magnitude"  # Magnitude of price change (regression)

class ModelArchitecture(str, Enum):
    FINBERT = "yiyanghkust/finbert-tone"
    DISTILBERT_FINANCIAL = "distilbert-base-uncased"
    ROBERTA_FINANCIAL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@dataclass
class TargetConfig:
    target_name: FinancialTarget
    target_type: str  # "classification" or "regression"
    num_classes: Optional[int] = None  # For classification
    class_weights: Optional[Dict[str, float]] = None
    loss_weight: float = 1.0  # Weight in multi-target loss
    
@dataclass
class TrainingConfig:
    model_name: ModelArchitecture
    target_configs: List[TargetConfig]
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    early_stopping_patience: int = 3
    output_dir: str = "./financial_transformer_models"
    
@dataclass
class FinancialDataPoint:
    text: str
    symbol: str
    timestamp: datetime
    targets: Dict[FinancialTarget, Union[int, float]]  # Target values
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelEvaluationResult:
    target_name: FinancialTarget
    target_type: str
    metrics: Dict[str, float]
    predictions: List[Union[int, float]]
    true_values: List[Union[int, float]]
    confusion_matrix: Optional[np.ndarray] = None

class FinancialDataset(Dataset):
    """Custom dataset for multi-target financial sentiment analysis"""
    
    def __init__(self, data_points: List[FinancialDataPoint], 
                 tokenizer, target_configs: List[TargetConfig], max_length: int = 512):
        self.data_points = data_points
        self.tokenizer = tokenizer
        self.target_configs = target_configs
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        data_point = self.data_points[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            data_point.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        # Add target labels
        for target_config in self.target_configs:
            target_name = target_config.target_name
            if target_name in data_point.targets:
                target_value = data_point.targets[target_name]
                
                if target_config.target_type == "classification":
                    item[f'labels_{target_name.value}'] = torch.tensor(target_value, dtype=torch.long)
                else:  # regression
                    item[f'labels_{target_name.value}'] = torch.tensor(target_value, dtype=torch.float)
        
        return item

class MultiTargetFinancialModel(nn.Module):
    """Multi-target transformer model for financial analysis"""
    
    def __init__(self, model_name: str, target_configs: List[TargetConfig], dropout_rate: float = 0.1):
        super().__init__()
        
        self.target_configs = target_configs
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        
        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
        for target_config in target_configs:
            if target_config.target_type == "classification":
                self.task_heads[target_config.target_name.value] = nn.Linear(
                    self.config.hidden_size, target_config.num_classes
                )
            else:  # regression
                self.task_heads[target_config.target_name.value] = nn.Linear(
                    self.config.hidden_size, 1
                )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of task-specific heads"""
        for head in self.task_heads.values():
            if isinstance(head, nn.Linear):
                torch.nn.init.normal_(head.weight, std=0.02)
                torch.nn.init.zeros_(head.bias)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        # Calculate outputs for each task
        task_outputs = {}
        for target_config in self.target_configs:
            task_name = target_config.target_name.value
            task_outputs[task_name] = self.task_heads[task_name](pooled_output)
        
        # Calculate losses if labels are provided
        losses = {}
        total_loss = 0.0
        
        for target_config in self.target_configs:
            task_name = target_config.target_name.value
            label_key = f'labels_{task_name}'
            
            if label_key in kwargs:
                labels = kwargs[label_key]
                logits = task_outputs[task_name]
                
                if target_config.target_type == "classification":
                    loss_fn = nn.CrossEntropyLoss(
                        weight=self._get_class_weights(target_config)
                    )
                    task_loss = loss_fn(logits, labels)
                else:  # regression
                    loss_fn = nn.MSELoss()
                    task_loss = loss_fn(logits.squeeze(), labels.float())
                
                # Apply task weight
                weighted_loss = task_loss * target_config.loss_weight
                losses[task_name] = weighted_loss
                total_loss += weighted_loss
        
        return {
            'loss': total_loss if losses else None,
            'task_losses': losses,
            'task_outputs': task_outputs,
            'hidden_states': pooled_output
        }
    
    def _get_class_weights(self, target_config: TargetConfig) -> Optional[torch.Tensor]:
        """Get class weights for imbalanced classification"""
        if target_config.class_weights:
            weights = torch.tensor(list(target_config.class_weights.values()))
            return weights.to(next(self.parameters()).device)
        return None

class FinancialTransformerTrainer:
    """Trainer for multi-target financial transformer models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_history = []
        
    def prepare_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name.value)
        
        # Add special tokens if needed
        special_tokens = ["[PRICE_UP]", "[PRICE_DOWN]", "[HIGH_VOL]", "[LOW_VOL]"]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # Initialize model
        self.model = MultiTargetFinancialModel(
            model_name=self.config.model_name.value,
            target_configs=self.config.target_configs
        )
        
        # Resize token embeddings for new special tokens
        self.model.transformer.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Initialized {self.config.model_name.value} with {len(self.config.target_configs)} targets")
    
    def prepare_dataset(self, train_data: List[FinancialDataPoint], 
                       val_data: List[FinancialDataPoint]) -> Tuple[FinancialDataset, FinancialDataset]:
        """Prepare training and validation datasets"""
        train_dataset = FinancialDataset(
            train_data, self.tokenizer, self.config.target_configs, self.config.max_length
        )
        val_dataset = FinancialDataset(
            val_data, self.tokenizer, self.config.target_configs, self.config.max_length
        )
        
        logger.info(f"Prepared datasets: {len(train_dataset)} train, {len(val_dataset)} validation")
        return train_dataset, val_dataset
    
    def train(self, train_dataset: FinancialDataset, val_dataset: FinancialDataset) -> Dict[str, Any]:
        """Train the multi-target model"""
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            dataloader_drop_last=False,
            report_to="wandb" if wandb.run else None,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )
        
        # Start training
        logger.info("Starting multi-target training...")
        train_result = self.trainer.train()
        
        # Save model and tokenizer
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'training_loss': train_result.training_loss,
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'config': self._config_to_dict()
        })
        
        logger.info(f"Training completed. Final loss: {train_result.training_loss:.4f}")
        return train_result.metrics
    
    def evaluate(self, eval_dataset: FinancialDataset) -> Dict[FinancialTarget, ModelEvaluationResult]:
        """Evaluate model on each target separately"""
        if not self.trainer:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        predictions = self.trainer.predict(eval_dataset)
        
        results = {}
        for target_config in self.config.target_configs:
            target_name = target_config.target_name
            task_name = target_name.value
            
            # Extract predictions and labels for this target
            if target_config.target_type == "classification":
                pred_logits = predictions.predictions[task_name]
                pred_classes = np.argmax(pred_logits, axis=1)
                true_labels = predictions.label_ids[f'labels_{task_name}']
                
                # Calculate metrics
                accuracy = accuracy_score(true_labels, pred_classes)
                f1 = f1_score(true_labels, pred_classes, average='weighted')
                
                metrics = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'num_samples': len(true_labels)
                }
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(true_labels, pred_classes)
                
                results[target_name] = ModelEvaluationResult(
                    target_name=target_name,
                    target_type=target_config.target_type,
                    metrics=metrics,
                    predictions=pred_classes.tolist(),
                    true_values=true_labels.tolist(),
                    confusion_matrix=cm
                )
                
            else:  # regression
                pred_values = predictions.predictions[task_name].squeeze()
                true_values = predictions.label_ids[f'labels_{task_name}']
                
                # Calculate metrics
                mse = mean_squared_error(true_values, pred_values)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(true_values - pred_values))
                
                # R-squared
                ss_res = np.sum((true_values - pred_values) ** 2)
                ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'num_samples': len(true_values)
                }
                
                results[target_name] = ModelEvaluationResult(
                    target_name=target_name,
                    target_type=target_config.target_type,
                    metrics=metrics,
                    predictions=pred_values.tolist(),
                    true_values=true_values.tolist()
                )
        
        return results
    
    def predict(self, texts: List[str]) -> Dict[FinancialTarget, List[Union[int, float]]]:
        """Generate predictions for new texts"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be initialized")
        
        self.model.eval()
        predictions = {target.target_name: [] for target in self.config.target_configs}
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                # Forward pass
                outputs = self.model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
                
                # Extract predictions for each target
                for target_config in self.config.target_configs:
                    task_name = target_config.target_name.value
                    task_output = outputs['task_outputs'][task_name]
                    
                    if target_config.target_type == "classification":
                        pred = torch.argmax(task_output, dim=1).item()
                    else:  # regression
                        pred = task_output.squeeze().item()
                    
                    predictions[target_config.target_name].append(pred)
        
        return predictions
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics during training"""
        # This is a simplified version for training monitoring
        # Full evaluation is done separately
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        metrics = {}
        total_samples = 0
        
        for target_config in self.config.target_configs:
            task_name = target_config.target_name.value
            
            if task_name in predictions and f'labels_{task_name}' in labels:
                if target_config.target_type == "classification":
                    pred_classes = np.argmax(predictions[task_name], axis=1)
                    true_labels = labels[f'labels_{task_name}']
                    accuracy = accuracy_score(true_labels, pred_classes)
                    metrics[f'{task_name}_accuracy'] = accuracy
                else:  # regression
                    pred_values = predictions[task_name].squeeze()
                    true_values = labels[f'labels_{task_name}']
                    mse = mean_squared_error(true_values, pred_values)
                    metrics[f'{task_name}_mse'] = mse
                
                total_samples = len(labels[f'labels_{task_name}'])
        
        metrics['total_samples'] = total_samples
        return metrics
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert training config to dictionary"""
        return {
            'model_name': self.config.model_name.value,
            'targets': [
                {
                    'name': tc.target_name.value,
                    'type': tc.target_type,
                    'num_classes': tc.num_classes,
                    'loss_weight': tc.loss_weight
                }
                for tc in self.config.target_configs
            ],
            'max_length': self.config.max_length,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'num_epochs': self.config.num_epochs
        }
    
    def save_training_history(self, filepath: str):
        """Save training history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = MultiTargetFinancialModel(
            model_name=model_path,
            target_configs=self.config.target_configs
        )
        
        # Load state dict if available
        model_file = Path(model_path) / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location='cpu')
            self.model.load_state_dict(state_dict)
        
        logger.info(f"Loaded model from {model_path}")

class ModelComparison:
    """Compare multi-target vs single-target model performance"""
    
    def __init__(self):
        self.comparison_results = []
    
    def compare_approaches(self, train_data: List[FinancialDataPoint], 
                          val_data: List[FinancialDataPoint],
                          test_data: List[FinancialDataPoint]) -> Dict[str, Any]:
        """
        Compare multi-target vs single-target approaches.
        
        Args:
            train_data: Training data
            val_data: Validation data  
            test_data: Test data for final evaluation
            
        Returns:
            Comparison results
        """
        
        results = {
            'multi_target': {},
            'single_target': {},
            'comparison_metrics': {}
        }
        
        # Define target configurations
        target_configs = [
            TargetConfig(
                target_name=FinancialTarget.SENTIMENT,
                target_type="classification",
                num_classes=3,  # positive, negative, neutral
                loss_weight=1.0
            ),
            TargetConfig(
                target_name=FinancialTarget.PRICE_DIRECTION,
                target_type="classification", 
                num_classes=3,  # up, down, flat
                loss_weight=1.2  # Higher weight for financial target
            ),
            TargetConfig(
                target_name=FinancialTarget.VOLATILITY,
                target_type="classification",
                num_classes=3,  # low, medium, high
                loss_weight=1.0
            ),
            TargetConfig(
                target_name=FinancialTarget.PRICE_MAGNITUDE,
                target_type="regression",
                loss_weight=0.8
            )
        ]
        
        # Train multi-target model
        logger.info("Training multi-target model...")
        multi_config = TrainingConfig(
            model_name=ModelArchitecture.FINBERT,
            target_configs=target_configs,
            num_epochs=3,
            output_dir="./multi_target_model"
        )
        
        multi_trainer = FinancialTransformerTrainer(multi_config)
        multi_trainer.prepare_model_and_tokenizer()
        train_dataset, val_dataset = multi_trainer.prepare_dataset(train_data, val_data)
        
        multi_train_results = multi_trainer.train(train_dataset, val_dataset)
        
        # Evaluate multi-target model
        test_dataset = FinancialDataset(
            test_data, multi_trainer.tokenizer, target_configs, multi_config.max_length
        )
        multi_eval_results = multi_trainer.evaluate(test_dataset)
        
        results['multi_target'] = {
            'training_metrics': multi_train_results,
            'evaluation_results': {
                target.value: {
                    'metrics': result.metrics,
                    'target_type': result.target_type
                }
                for target, result in multi_eval_results.items()
            }
        }
        
        # Train single-target models
        logger.info("Training single-target models...")
        single_target_results = {}
        
        for target_config in target_configs:
            logger.info(f"Training single-target model for {target_config.target_name.value}")
            
            single_config = TrainingConfig(
                model_name=ModelArchitecture.FINBERT,
                target_configs=[target_config],
                num_epochs=3,
                output_dir=f"./single_target_{target_config.target_name.value}"
            )
            
            single_trainer = FinancialTransformerTrainer(single_config)
            single_trainer.prepare_model_and_tokenizer()
            
            single_train_dataset, single_val_dataset = single_trainer.prepare_dataset(train_data, val_data)
            single_train_results = single_trainer.train(single_train_dataset, single_val_dataset)
            
            # Evaluate single-target model
            single_test_dataset = FinancialDataset(
                test_data, single_trainer.tokenizer, [target_config], single_config.max_length
            )
            single_eval_results = single_trainer.evaluate(single_test_dataset)
            
            single_target_results[target_config.target_name.value] = {
                'training_metrics': single_train_results,
                'evaluation_results': single_eval_results[target_config.target_name].metrics
            }
        
        results['single_target'] = single_target_results
        
        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(
            results['multi_target']['evaluation_results'],
            results['single_target']
        )
        results['comparison_metrics'] = comparison_metrics
        
        # Store results
        self.comparison_results.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        return results
    
    def _calculate_comparison_metrics(self, multi_results: Dict, single_results: Dict) -> Dict[str, Any]:
        """Calculate performance comparison metrics"""
        comparison = {}
        
        for target_name in multi_results.keys():
            if target_name in single_results:
                multi_metrics = multi_results[target_name]['metrics']
                single_metrics = single_results[target_name]['evaluation_results']
                
                target_comparison = {}
                for metric_name in multi_metrics.keys():
                    if metric_name in single_metrics and metric_name != 'num_samples':
                        multi_value = multi_metrics[metric_name]
                        single_value = single_metrics[metric_name]
                        
                        # Calculate improvement
                        improvement = ((multi_value - single_value) / single_value) * 100
                        target_comparison[metric_name] = {
                            'multi_target': multi_value,
                            'single_target': single_value,
                            'improvement_pct': improvement
                        }
                
                comparison[target_name] = target_comparison
        
        return comparison