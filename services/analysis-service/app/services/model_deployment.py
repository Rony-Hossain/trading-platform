#!/usr/bin/env python3
"""
Model Deployment Pipeline
Production-grade model deployment system with staging and production environments.
Integrates with MLflow model registry for automated deployment workflows.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import os
import pickle
import time
import hashlib
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# MLflow imports with fallback
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Model deployment stages."""
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    SHADOW = "Shadow"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"


class HealthStatus(Enum):
    """Model health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_name: str
    version: str
    stage: DeploymentStage
    strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    promotion_criteria: Optional[Dict[str, float]] = None
    shadow_traffic_pct: float = 0.1
    canary_traffic_pct: float = 0.05
    rollback_threshold: float = 0.8  # Performance threshold for automatic rollback
    health_check_interval: int = 300  # seconds
    max_deployment_time: int = 1800  # seconds
    tags: Optional[Dict[str, str]] = None


@dataclass
class PromotionCriteria:
    """Criteria for promoting models between stages."""
    min_sharpe_ratio: float = 1.0
    min_hit_rate: float = 0.55
    max_drawdown: float = 0.15
    min_r2_score: float = 0.4
    min_samples: int = 100
    min_days_in_staging: int = 7
    max_latency_ms: float = 500.0
    min_uptime_pct: float = 99.0


@dataclass
class ModelEndpoint:
    """Model serving endpoint configuration."""
    endpoint_id: str
    model_name: str
    version: str
    stage: DeploymentStage
    url: str
    port: int
    health_url: str
    last_health_check: Optional[datetime] = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    traffic_weight: float = 0.0
    deployment_time: Optional[datetime] = None
    metrics: Dict[str, float] = None


@dataclass
class DeploymentResult:
    """Result of model deployment operation."""
    deployment_id: str
    model_name: str
    version: str
    stage: DeploymentStage
    strategy: DeploymentStrategy
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    endpoints: List[ModelEndpoint] = None
    rollback_performed: bool = False
    error_message: Optional[str] = None
    promotion_eligible: bool = False
    next_stage: Optional[DeploymentStage] = None


class ModelDeploymentPipeline:
    """
    Production-grade model deployment pipeline with staging and production environments.
    
    Features:
    - MLflow model registry integration
    - Blue-green and canary deployment strategies
    - Automated promotion criteria validation
    - Shadow testing capabilities
    - Health monitoring and automatic rollback
    - Zero-downtime deployments
    """
    
    def __init__(self, tracking_uri: str = "./mlruns", 
                 staging_port: int = 8004, production_port: int = 8005):
        """
        Initialize model deployment pipeline.
        
        Args:
            tracking_uri: MLflow tracking URI
            staging_port: Port for staging environment
            production_port: Port for production environment
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        self.staging_port = staging_port
        self.production_port = production_port
        self.client = None
        
        # Active deployments tracking
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.staging_endpoints: Dict[str, ModelEndpoint] = {}
        self.production_endpoints: Dict[str, ModelEndpoint] = {}
        self.shadow_endpoints: Dict[str, ModelEndpoint] = {}
        
        # Default promotion criteria
        self.default_promotion_criteria = PromotionCriteria()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        logger.info(f"Model deployment pipeline initialized with URI: {self.tracking_uri}")
    
    def _get_client(self) -> MlflowClient:
        """Get MLflow client instance."""
        if self.client is None:
            self.client = MlflowClient()
        return self.client
    
    async def deploy_to_staging(self, model_name: str, version: str, 
                               config: Optional[DeploymentConfig] = None) -> DeploymentResult:
        """
        Deploy model to staging environment.
        
        Args:
            model_name: Name of the model to deploy
            version: Version of the model to deploy
            config: Deployment configuration
            
        Returns:
            Deployment result with status and endpoint information
        """
        try:
            logger.info(f"Starting staging deployment for {model_name} v{version}")
            
            if config is None:
                config = DeploymentConfig(
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.STAGING,
                    strategy=DeploymentStrategy.BLUE_GREEN
                )
            
            deployment_id = f"staging_{model_name}_{version}_{int(time.time())}"
            start_time = datetime.now(timezone.utc)
            
            # Validate model exists in registry
            client = self._get_client()
            try:
                model_version = client.get_model_version(model_name, version)
            except Exception as e:
                logger.error(f"Model {model_name} v{version} not found in registry: {e}")
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.STAGING,
                    strategy=config.strategy,
                    status="failed",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message=f"Model not found in registry: {str(e)}"
                )
            
            # Create staging endpoint
            endpoint = ModelEndpoint(
                endpoint_id=f"staging_{model_name}_{version}",
                model_name=model_name,
                version=version,
                stage=DeploymentStage.STAGING,
                url=f"http://localhost:{self.staging_port}/models/{model_name}/v{version}",
                port=self.staging_port,
                health_url=f"http://localhost:{self.staging_port}/health",
                deployment_time=start_time,
                traffic_weight=1.0,
                metrics={}
            )
            
            # Deploy model (simulated)
            success = await self._deploy_model_endpoint(endpoint, config)
            
            if success:
                # Transition model to Staging in MLflow
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Staging"
                )
                
                # Store staging endpoint
                self.staging_endpoints[endpoint.endpoint_id] = endpoint
                
                # Create deployment result
                end_time = datetime.now(timezone.utc)
                result = DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.STAGING,
                    strategy=config.strategy,
                    status="success",
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    endpoints=[endpoint],
                    rollback_performed=False
                )
                
                self.active_deployments[deployment_id] = result
                
                logger.info(f"Successfully deployed {model_name} v{version} to staging")
                return result
            
            else:
                logger.error(f"Failed to deploy {model_name} v{version} to staging")
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.STAGING,
                    strategy=config.strategy,
                    status="failed",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message="Endpoint deployment failed"
                )
                
        except Exception as e:
            logger.error(f"Error deploying to staging: {e}")
            return DeploymentResult(
                deployment_id=deployment_id if 'deployment_id' in locals() else f"error_{int(time.time())}",
                model_name=model_name,
                version=version,
                stage=DeploymentStage.STAGING,
                strategy=config.strategy if config else DeploymentStrategy.BLUE_GREEN,
                status="error",
                start_time=start_time if 'start_time' in locals() else datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
    
    async def promote_to_production(self, model_name: str, version: str,
                                   criteria: Optional[PromotionCriteria] = None,
                                   config: Optional[DeploymentConfig] = None) -> DeploymentResult:
        """
        Promote model from staging to production with validation.
        
        Args:
            model_name: Name of the model to promote
            version: Version of the model to promote
            criteria: Promotion criteria to validate
            config: Deployment configuration
            
        Returns:
            Deployment result with promotion status
        """
        try:
            logger.info(f"Starting promotion to production for {model_name} v{version}")
            
            if criteria is None:
                criteria = self.default_promotion_criteria
            
            if config is None:
                config = DeploymentConfig(
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=DeploymentStrategy.BLUE_GREEN
                )
            
            deployment_id = f"production_{model_name}_{version}_{int(time.time())}"
            start_time = datetime.now(timezone.utc)
            
            # Validate promotion criteria
            validation_result = await self._validate_promotion_criteria(
                model_name, version, criteria
            )
            
            if not validation_result["eligible"]:
                logger.warning(f"Model {model_name} v{version} does not meet promotion criteria")
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=config.strategy,
                    status="rejected",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message=f"Promotion criteria not met: {validation_result['reasons']}",
                    promotion_eligible=False
                )
            
            # Deploy to production using specified strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._blue_green_deployment(model_name, version, config)
            elif config.strategy == DeploymentStrategy.CANARY:
                result = await self._canary_deployment(model_name, version, config)
            else:
                result = await self._immediate_deployment(model_name, version, config)
            
            if result.status == "success":
                # Transition model to Production in MLflow
                client = self._get_client()
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Production"
                )
                
                logger.info(f"Successfully promoted {model_name} v{version} to production")
            
            return result
            
        except Exception as e:
            logger.error(f"Error promoting to production: {e}")
            return DeploymentResult(
                deployment_id=deployment_id if 'deployment_id' in locals() else f"error_{int(time.time())}",
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                strategy=config.strategy if config else DeploymentStrategy.BLUE_GREEN,
                status="error",
                start_time=start_time if 'start_time' in locals() else datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
    
    async def _validate_promotion_criteria(self, model_name: str, version: str,
                                         criteria: PromotionCriteria) -> Dict[str, Any]:
        """
        Validate if model meets promotion criteria.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            criteria: Promotion criteria to validate
            
        Returns:
            Validation result with eligibility status and reasons
        """
        try:
            client = self._get_client()
            
            # Get model version details
            model_version = client.get_model_version(model_name, version)
            
            # Check if model is in staging
            if model_version.current_stage != "Staging":
                return {
                    "eligible": False,
                    "reasons": ["Model must be in Staging stage"]
                }
            
            # Get run metrics
            run = client.get_run(model_version.run_id)
            metrics = run.data.metrics
            
            reasons = []
            
            # Validate performance criteria
            if 'sharpe_ratio' in metrics:
                if metrics['sharpe_ratio'] < criteria.min_sharpe_ratio:
                    reasons.append(f"Sharpe ratio {metrics['sharpe_ratio']:.3f} < {criteria.min_sharpe_ratio}")
            
            if 'hit_rate' in metrics:
                if metrics['hit_rate'] < criteria.min_hit_rate:
                    reasons.append(f"Hit rate {metrics['hit_rate']:.3f} < {criteria.min_hit_rate}")
            
            if 'max_drawdown' in metrics:
                if metrics['max_drawdown'] > criteria.max_drawdown:
                    reasons.append(f"Max drawdown {metrics['max_drawdown']:.3f} > {criteria.max_drawdown}")
            
            if 'r2' in metrics:
                if metrics['r2'] < criteria.min_r2_score:
                    reasons.append(f"R² score {metrics['r2']:.3f} < {criteria.min_r2_score}")
            
            # Check time in staging
            creation_time = datetime.fromtimestamp(model_version.creation_timestamp / 1000, tz=timezone.utc)
            days_in_staging = (datetime.now(timezone.utc) - creation_time).days
            
            if days_in_staging < criteria.min_days_in_staging:
                reasons.append(f"Only {days_in_staging} days in staging < {criteria.min_days_in_staging} required")
            
            # Check health metrics from staging endpoint
            staging_endpoint_id = f"staging_{model_name}_{version}"
            if staging_endpoint_id in self.staging_endpoints:
                endpoint = self.staging_endpoints[staging_endpoint_id]
                if endpoint.health_status != HealthStatus.HEALTHY:
                    reasons.append(f"Staging endpoint health: {endpoint.health_status.value}")
            
            eligible = len(reasons) == 0
            
            return {
                "eligible": eligible,
                "reasons": reasons,
                "metrics": metrics,
                "days_in_staging": days_in_staging
            }
            
        except Exception as e:
            logger.error(f"Error validating promotion criteria: {e}")
            return {
                "eligible": False,
                "reasons": [f"Validation error: {str(e)}"]
            }
    
    async def _blue_green_deployment(self, model_name: str, version: str,
                                    config: DeploymentConfig) -> DeploymentResult:
        """
        Perform blue-green deployment to production.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            config: Deployment configuration
            
        Returns:
            Deployment result
        """
        deployment_id = f"bluegreen_{model_name}_{version}_{int(time.time())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Create new production endpoint (green)
            green_endpoint = ModelEndpoint(
                endpoint_id=f"production_green_{model_name}_{version}",
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                url=f"http://localhost:{self.production_port}/models/{model_name}/v{version}",
                port=self.production_port,
                health_url=f"http://localhost:{self.production_port}/health",
                deployment_time=start_time,
                traffic_weight=0.0,  # Start with no traffic
                metrics={}
            )
            
            # Deploy green endpoint
            success = await self._deploy_model_endpoint(green_endpoint, config)
            
            if not success:
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=DeploymentStrategy.BLUE_GREEN,
                    status="failed",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message="Green endpoint deployment failed"
                )
            
            # Health check green endpoint
            await asyncio.sleep(5)  # Allow time for startup
            health_ok = await self._health_check_endpoint(green_endpoint)
            
            if not health_ok:
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=DeploymentStrategy.BLUE_GREEN,
                    status="failed",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message="Green endpoint health check failed"
                )
            
            # Switch traffic to green (blue-green switch)
            green_endpoint.traffic_weight = 1.0
            
            # Store new production endpoint
            old_endpoints = list(self.production_endpoints.values())
            self.production_endpoints[green_endpoint.endpoint_id] = green_endpoint
            
            # Remove old blue endpoints
            for old_endpoint in old_endpoints:
                if old_endpoint.endpoint_id in self.production_endpoints:
                    del self.production_endpoints[old_endpoint.endpoint_id]
            
            end_time = datetime.now(timezone.utc)
            
            return DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                strategy=DeploymentStrategy.BLUE_GREEN,
                status="success",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                endpoints=[green_endpoint],
                rollback_performed=False,
                promotion_eligible=True,
                next_stage=None
            )
            
        except Exception as e:
            logger.error(f"Blue-green deployment error: {e}")
            return DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                strategy=DeploymentStrategy.BLUE_GREEN,
                status="error",
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
    
    async def _canary_deployment(self, model_name: str, version: str,
                                config: DeploymentConfig) -> DeploymentResult:
        """
        Perform canary deployment to production.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            config: Deployment configuration
            
        Returns:
            Deployment result
        """
        deployment_id = f"canary_{model_name}_{version}_{int(time.time())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Create canary endpoint
            canary_endpoint = ModelEndpoint(
                endpoint_id=f"production_canary_{model_name}_{version}",
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                url=f"http://localhost:{self.production_port + 1}/models/{model_name}/v{version}",
                port=self.production_port + 1,
                health_url=f"http://localhost:{self.production_port + 1}/health",
                deployment_time=start_time,
                traffic_weight=config.canary_traffic_pct,
                metrics={}
            )
            
            # Deploy canary endpoint
            success = await self._deploy_model_endpoint(canary_endpoint, config)
            
            if not success:
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=DeploymentStrategy.CANARY,
                    status="failed",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message="Canary endpoint deployment failed"
                )
            
            # Monitor canary for specified time
            monitoring_duration = 300  # 5 minutes
            await asyncio.sleep(monitoring_duration)
            
            # Check canary performance
            canary_healthy = await self._health_check_endpoint(canary_endpoint)
            
            if canary_healthy:
                # Gradually increase traffic to canary
                for weight in [0.25, 0.5, 0.75, 1.0]:
                    canary_endpoint.traffic_weight = weight
                    await asyncio.sleep(60)  # Wait 1 minute between increases
                    
                    health_ok = await self._health_check_endpoint(canary_endpoint)
                    if not health_ok:
                        # Rollback on health failure
                        canary_endpoint.traffic_weight = 0.0
                        return DeploymentResult(
                            deployment_id=deployment_id,
                            model_name=model_name,
                            version=version,
                            stage=DeploymentStage.PRODUCTION,
                            strategy=DeploymentStrategy.CANARY,
                            status="rolled_back",
                            start_time=start_time,
                            end_time=datetime.now(timezone.utc),
                            endpoints=[canary_endpoint],
                            rollback_performed=True,
                            error_message="Canary health check failed during ramp-up"
                        )
                
                # Successful canary deployment
                self.production_endpoints[canary_endpoint.endpoint_id] = canary_endpoint
                
                end_time = datetime.now(timezone.utc)
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=DeploymentStrategy.CANARY,
                    status="success",
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    endpoints=[canary_endpoint],
                    rollback_performed=False
                )
            
            else:
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=DeploymentStrategy.CANARY,
                    status="failed",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message="Canary health check failed"
                )
            
        except Exception as e:
            logger.error(f"Canary deployment error: {e}")
            return DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                strategy=DeploymentStrategy.CANARY,
                status="error",
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
    
    async def _immediate_deployment(self, model_name: str, version: str,
                                   config: DeploymentConfig) -> DeploymentResult:
        """
        Perform immediate deployment to production (use with caution).
        
        Args:
            model_name: Name of the model
            version: Version of the model
            config: Deployment configuration
            
        Returns:
            Deployment result
        """
        deployment_id = f"immediate_{model_name}_{version}_{int(time.time())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Create production endpoint
            endpoint = ModelEndpoint(
                endpoint_id=f"production_{model_name}_{version}",
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                url=f"http://localhost:{self.production_port}/models/{model_name}/v{version}",
                port=self.production_port,
                health_url=f"http://localhost:{self.production_port}/health",
                deployment_time=start_time,
                traffic_weight=1.0,
                metrics={}
            )
            
            # Deploy endpoint
            success = await self._deploy_model_endpoint(endpoint, config)
            
            if success:
                self.production_endpoints[endpoint.endpoint_id] = endpoint
                
                end_time = datetime.now(timezone.utc)
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=DeploymentStrategy.IMMEDIATE,
                    status="success",
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    endpoints=[endpoint],
                    rollback_performed=False
                )
            else:
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.PRODUCTION,
                    strategy=DeploymentStrategy.IMMEDIATE,
                    status="failed",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message="Endpoint deployment failed"
                )
            
        except Exception as e:
            logger.error(f"Immediate deployment error: {e}")
            return DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                version=version,
                stage=DeploymentStage.PRODUCTION,
                strategy=DeploymentStrategy.IMMEDIATE,
                status="error",
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
    
    async def _deploy_model_endpoint(self, endpoint: ModelEndpoint, 
                                    config: DeploymentConfig) -> bool:
        """
        Deploy model to endpoint (simulated deployment).
        
        Args:
            endpoint: Model endpoint configuration
            config: Deployment configuration
            
        Returns:
            True if deployment successful, False otherwise
        """
        try:
            logger.info(f"Deploying endpoint {endpoint.endpoint_id}")
            
            # Simulate deployment time
            await asyncio.sleep(2)
            
            # In a real implementation, this would:
            # 1. Pull model from MLflow registry
            # 2. Create container/process with model serving
            # 3. Configure load balancer
            # 4. Set up monitoring
            
            endpoint.health_status = HealthStatus.HEALTHY
            endpoint.last_health_check = datetime.now(timezone.utc)
            
            logger.info(f"Successfully deployed endpoint {endpoint.endpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying endpoint {endpoint.endpoint_id}: {e}")
            endpoint.health_status = HealthStatus.UNHEALTHY
            return False
    
    async def _health_check_endpoint(self, endpoint: ModelEndpoint) -> bool:
        """
        Perform health check on model endpoint.
        
        Args:
            endpoint: Model endpoint to check
            
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simulate health check
            await asyncio.sleep(1)
            
            # In a real implementation, this would:
            # 1. Make HTTP request to health endpoint
            # 2. Check response time and status
            # 3. Validate model predictions
            # 4. Check resource utilization
            
            endpoint.last_health_check = datetime.now(timezone.utc)
            endpoint.health_status = HealthStatus.HEALTHY
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for {endpoint.endpoint_id}: {e}")
            endpoint.health_status = HealthStatus.UNHEALTHY
            return False
    
    async def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Rollback a deployment to previous version.
        
        Args:
            deployment_id: ID of deployment to rollback
            
        Returns:
            Rollback result
        """
        try:
            if deployment_id not in self.active_deployments:
                return {
                    "success": False,
                    "error": f"Deployment {deployment_id} not found"
                }
            
            deployment = self.active_deployments[deployment_id]
            
            logger.info(f"Rolling back deployment {deployment_id}")
            
            # Remove current endpoints
            if deployment.stage == DeploymentStage.PRODUCTION:
                for endpoint in deployment.endpoints or []:
                    if endpoint.endpoint_id in self.production_endpoints:
                        del self.production_endpoints[endpoint.endpoint_id]
            elif deployment.stage == DeploymentStage.STAGING:
                for endpoint in deployment.endpoints or []:
                    if endpoint.endpoint_id in self.staging_endpoints:
                        del self.staging_endpoints[endpoint.endpoint_id]
            
            # Mark deployment as rolled back
            deployment.rollback_performed = True
            deployment.status = "rolled_back"
            deployment.end_time = datetime.now(timezone.utc)
            
            logger.info(f"Successfully rolled back deployment {deployment_id}")
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "rollback_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error rolling back deployment {deployment_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """
        Get status of a deployment.
        
        Args:
            deployment_id: ID of deployment to check
            
        Returns:
            Deployment result or None if not found
        """
        return self.active_deployments.get(deployment_id)
    
    async def list_active_deployments(self) -> List[DeploymentResult]:
        """
        List all active deployments.
        
        Returns:
            List of active deployment results
        """
        return list(self.active_deployments.values())
    
    async def get_endpoint_status(self, stage: DeploymentStage) -> List[ModelEndpoint]:
        """
        Get status of endpoints for a stage.
        
        Args:
            stage: Deployment stage to check
            
        Returns:
            List of endpoints for the stage
        """
        if stage == DeploymentStage.STAGING:
            return list(self.staging_endpoints.values())
        elif stage == DeploymentStage.PRODUCTION:
            return list(self.production_endpoints.values())
        elif stage == DeploymentStage.SHADOW:
            return list(self.shadow_endpoints.values())
        else:
            return []
    
    async def health_check_all_endpoints(self) -> Dict[str, Any]:
        """
        Perform health checks on all active endpoints.
        
        Returns:
            Health check results for all endpoints
        """
        results = {
            "staging": {},
            "production": {},
            "shadow": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check staging endpoints
        for endpoint_id, endpoint in self.staging_endpoints.items():
            healthy = await self._health_check_endpoint(endpoint)
            results["staging"][endpoint_id] = {
                "healthy": healthy,
                "status": endpoint.health_status.value,
                "last_check": endpoint.last_health_check.isoformat() if endpoint.last_health_check else None
            }
        
        # Check production endpoints
        for endpoint_id, endpoint in self.production_endpoints.items():
            healthy = await self._health_check_endpoint(endpoint)
            results["production"][endpoint_id] = {
                "healthy": healthy,
                "status": endpoint.health_status.value,
                "last_check": endpoint.last_health_check.isoformat() if endpoint.last_check else None
            }
        
        # Check shadow endpoints
        for endpoint_id, endpoint in self.shadow_endpoints.items():
            healthy = await self._health_check_endpoint(endpoint)
            results["shadow"][endpoint_id] = {
                "healthy": healthy,
                "status": endpoint.health_status.value,
                "last_check": endpoint.last_health_check.isoformat() if endpoint.last_health_check else None
            }
        
        return results
    
    async def create_shadow_deployment(self, model_name: str, version: str,
                                      traffic_pct: float = 0.1) -> DeploymentResult:
        """
        Create shadow deployment for testing without affecting production traffic.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            traffic_pct: Percentage of traffic to shadow (for logging/testing)
            
        Returns:
            Deployment result
        """
        try:
            deployment_id = f"shadow_{model_name}_{version}_{int(time.time())}"
            start_time = datetime.now(timezone.utc)
            
            # Create shadow endpoint
            shadow_endpoint = ModelEndpoint(
                endpoint_id=f"shadow_{model_name}_{version}",
                model_name=model_name,
                version=version,
                stage=DeploymentStage.SHADOW,
                url=f"http://localhost:{self.production_port + 2}/models/{model_name}/v{version}",
                port=self.production_port + 2,
                health_url=f"http://localhost:{self.production_port + 2}/health",
                deployment_time=start_time,
                traffic_weight=traffic_pct,
                metrics={}
            )
            
            # Deploy shadow endpoint
            config = DeploymentConfig(
                model_name=model_name,
                version=version,
                stage=DeploymentStage.SHADOW,
                shadow_traffic_pct=traffic_pct
            )
            
            success = await self._deploy_model_endpoint(shadow_endpoint, config)
            
            if success:
                self.shadow_endpoints[shadow_endpoint.endpoint_id] = shadow_endpoint
                
                end_time = datetime.now(timezone.utc)
                result = DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.SHADOW,
                    strategy=DeploymentStrategy.IMMEDIATE,
                    status="success",
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    endpoints=[shadow_endpoint],
                    rollback_performed=False
                )
                
                self.active_deployments[deployment_id] = result
                return result
            
            else:
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    version=version,
                    stage=DeploymentStage.SHADOW,
                    strategy=DeploymentStrategy.IMMEDIATE,
                    status="failed",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    error_message="Shadow endpoint deployment failed"
                )
            
        except Exception as e:
            logger.error(f"Error creating shadow deployment: {e}")
            return DeploymentResult(
                deployment_id=deployment_id if 'deployment_id' in locals() else f"error_{int(time.time())}",
                model_name=model_name,
                version=version,
                stage=DeploymentStage.SHADOW,
                strategy=DeploymentStrategy.IMMEDIATE,
                status="error",
                start_time=start_time if 'start_time' in locals() else datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )


# Demo function for testing
async def demo_deployment_pipeline():
    """Demo of model deployment pipeline."""
    print("Model Deployment Pipeline Demo")
    print("=" * 50)
    
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Please install with: pip install mlflow")
        return
    
    # Initialize deployment pipeline
    pipeline = ModelDeploymentPipeline(
        tracking_uri="file:./demo_mlruns",
        staging_port=8004,
        production_port=8005
    )
    
    # Demo deployment workflow
    model_name = "demo_trading_model"
    version = "1"
    
    print(f"\n1. Deploying {model_name} v{version} to staging...")
    staging_result = await pipeline.deploy_to_staging(model_name, version)
    print(f"Staging deployment status: {staging_result.status}")
    
    if staging_result.status == "success":
        print(f"\n2. Promoting {model_name} v{version} to production...")
        production_result = await pipeline.promote_to_production(model_name, version)
        print(f"Production deployment status: {production_result.status}")
        
        print(f"\n3. Creating shadow deployment for testing...")
        shadow_result = await pipeline.create_shadow_deployment(model_name, "2")
        print(f"Shadow deployment status: {shadow_result.status}")
        
        print(f"\n4. Health checking all endpoints...")
        health_results = await pipeline.health_check_all_endpoints()
        print(f"Health check results: {json.dumps(health_results, indent=2, default=str)}")
    
    print("\nDeployment pipeline demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_deployment_pipeline())