"""
API endpoints for Governance & Compliance Framework.

Provides REST API access to comprehensive compliance monitoring:
- Point-in-Time (PIT) audit framework with leakage detection
- Model evaluation hygiene monitoring
- Deploy gate enforcement and compliance tracking
- Live vs simulation performance gap monitoring
- Risk policy enforcement with violation tracking
- Model governance with cards and deployment memos
- Phase 3 Definition of Done compliance dashboard
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, File, UploadFile
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json

from services.governance_compliance import (
    ComprehenseiveComplianceSystem, PITAuditor, EvaluationHygieneMonitor,
    RealityGapMonitor, RiskPolicyEnforcer, ModelGovernanceSystem,
    ViolationSeverity, ModelStatus, PITViolation, EvaluationAudit,
    RealityGapMetrics, RiskPolicyViolation, ModelCard, DeploymentMemo
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class PITAuditRequest(BaseModel):
    """Request model for PIT compliance audit."""
    model_id: str = Field(..., description="Model identifier")
    features: List[str] = Field(..., description="List of feature names")
    feature_timestamps: Dict[str, datetime] = Field(..., description="Feature availability timestamps")
    prediction_timestamp: datetime = Field(..., description="Model prediction timestamp")


class EvaluationAuditRequest(BaseModel):
    """Request model for evaluation hygiene audit."""
    model_id: str = Field(..., description="Model identifier")
    spa_results: Optional[Dict[str, float]] = Field(None, description="SPA test results")
    deflated_sharpe_results: Optional[Dict[str, float]] = Field(None, description="Deflated Sharpe results")
    pbo_results: Optional[Dict[str, float]] = Field(None, description="PBO test results")


class RealityGapRequest(BaseModel):
    """Request model for reality gap monitoring."""
    model_id: str = Field(..., description="Model identifier")
    live_slippage_bps: float = Field(..., description="Live trading slippage in bps")
    sim_slippage_bps: float = Field(..., description="Simulated slippage in bps")
    trade_count: int = Field(..., ge=1, description="Number of trades in comparison")


class RiskComplianceRequest(BaseModel):
    """Request model for risk policy compliance check."""
    model_id: str = Field("portfolio", description="Model/portfolio identifier")
    risk_metrics: Dict[str, float] = Field(..., description="Risk metrics to check")


class ModelCardRequest(BaseModel):
    """Request model for creating model card."""
    model_id: str = Field(..., description="Unique model identifier")
    
    # Model details
    model_details: Dict[str, Any] = Field(..., description="Model specification details")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    governance_info: Dict[str, Any] = Field(..., description="Governance and approval information")
    
    @validator('performance_metrics')
    def validate_performance_metrics(cls, v):
        required_metrics = ['sharpe_ratio', 'max_drawdown', 'spa_p_value', 'pbo_estimate', 'var_95', 'volatility']
        for metric in required_metrics:
            if metric not in v:
                raise ValueError(f"Required performance metric '{metric}' missing")
        return v


class DeploymentMemoRequest(BaseModel):
    """Request model for creating deployment memo."""
    model_id: str = Field(..., description="Model identifier")
    deployment_details: Dict[str, Any] = Field(..., description="Deployment details and justification")
    approvals: Dict[str, Any] = Field(..., description="Approval information")
    
    @validator('approvals')
    def validate_approvals(cls, v):
        required_fields = ['risk_reviewer', 'risk_approval_date', 'final_approver', 'final_approval_date']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Required approval field '{field}' missing")
        return v


class ComplianceResponse(BaseModel):
    """Response model for compliance operations."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime
    processing_time_ms: Optional[float] = None


# Global compliance system instance
compliance_system = ComprehenseiveComplianceSystem()


# API Endpoints

@router.post("/pit-audit", response_model=ComplianceResponse)
async def conduct_pit_audit(request: PITAuditRequest):
    """
    Conduct Point-in-Time compliance audit for model features.
    
    Detects potential data leakage by checking for features that use
    future information not available at prediction time.
    """
    start_time = datetime.now()
    
    try:
        # Conduct PIT audit
        violations = compliance_system.pit_auditor.audit_feature_set(
            features=request.features,
            feature_timestamps=request.feature_timestamps,
            prediction_timestamp=request.prediction_timestamp,
            model_id=request.model_id
        )
        
        # Format violations for response
        violations_data = []
        for violation in violations:
            violations_data.append({
                'timestamp': violation.timestamp,
                'model_id': violation.model_id,
                'violation_type': violation.violation_type,
                'severity': violation.severity.value,
                'description': violation.description,
                'feature_name': violation.feature_name,
                'future_data_detected': violation.future_data_detected,
                'remediation_required': violation.remediation_required
            })
        
        # Get violation summary
        summary = compliance_system.pit_auditor.get_violation_summary(30)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ComplianceResponse(
            success=True,
            message=f"PIT audit completed for {request.model_id}",
            data={
                'audit_results': {
                    'violations_found': len(violations),
                    'violations': violations_data,
                    'compliance_summary': summary
                },
                'model_id': request.model_id,
                'features_audited': len(request.features),
                'prediction_timestamp': request.prediction_timestamp
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"PIT audit failed: {e}")
        raise HTTPException(status_code=500, detail=f"PIT audit failed: {str(e)}")


@router.post("/evaluation-audit", response_model=ComplianceResponse)
async def conduct_evaluation_audit(request: EvaluationAuditRequest):
    """
    Audit model evaluation for required statistical tests and deploy gates.
    
    Verifies that models have completed SPA, Deflated Sharpe, and PBO tests
    and meet deployment gate criteria.
    """
    start_time = datetime.now()
    
    try:
        # Conduct evaluation audit
        audit_result = compliance_system.eval_monitor.audit_model_evaluation(
            model_id=request.model_id,
            spa_results=request.spa_results,
            deflated_sharpe_results=request.deflated_sharpe_results,
            pbo_results=request.pbo_results
        )
        
        # Get hygiene compliance summary
        compliance_summary = compliance_system.eval_monitor.get_hygiene_compliance()
        
        # Format audit result
        audit_data = {
            'model_id': audit_result.model_id,
            'evaluation_timestamp': audit_result.evaluation_timestamp,
            'test_completion': {
                'spa_test_completed': audit_result.spa_test_completed,
                'deflated_sharpe_completed': audit_result.deflated_sharpe_completed,
                'pbo_test_completed': audit_result.pbo_test_completed
            },
            'test_results': {
                'spa_p_value': audit_result.spa_p_value,
                'deflated_sharpe_value': audit_result.deflated_sharpe_value,
                'pbo_estimate': audit_result.pbo_estimate
            },
            'deployment_status': {
                'deploy_gates_passed': audit_result.deploy_gates_passed,
                'deployment_approved': audit_result.deployment_approved
            },
            'compliance_summary': compliance_summary
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ComplianceResponse(
            success=True,
            message=f"Evaluation audit completed for {request.model_id}",
            data=audit_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Evaluation audit failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation audit failed: {str(e)}")


@router.post("/reality-gap", response_model=ComplianceResponse)
async def record_reality_gap(request: RealityGapRequest):
    """
    Record live vs simulation performance gap for monitoring.
    
    Tracks the difference between live trading performance and simulation
    to ensure realistic backtesting assumptions.
    """
    start_time = datetime.now()
    
    try:
        # Record reality gap metrics
        gap_metrics = compliance_system.reality_monitor.record_slippage_comparison(
            model_id=request.model_id,
            live_slippage_bps=request.live_slippage_bps,
            sim_slippage_bps=request.sim_slippage_bps,
            trade_count=request.trade_count
        )
        
        # Get compliance summary
        compliance_summary = compliance_system.reality_monitor.get_reality_gap_compliance()
        
        # Format response
        gap_data = {
            'gap_metrics': {
                'model_id': gap_metrics.model_id,
                'live_slippage_bps': gap_metrics.live_slippage_bps,
                'sim_slippage_bps': gap_metrics.sim_slippage_bps,
                'slippage_gap_pct': gap_metrics.slippage_gap_pct,
                'trade_count': gap_metrics.trade_count,
                'gap_within_tolerance': gap_metrics.gap_within_tolerance,
                'tolerance_threshold_pct': gap_metrics.tolerance_threshold_pct
            },
            'compliance_summary': compliance_summary
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ComplianceResponse(
            success=True,
            message=f"Reality gap recorded for {request.model_id}",
            data=gap_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Reality gap recording failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reality gap recording failed: {str(e)}")


@router.post("/risk-compliance", response_model=ComplianceResponse)
async def check_risk_compliance(request: RiskComplianceRequest):
    """
    Check risk metrics against policy limits and generate violations.
    
    Monitors portfolio risk metrics against predefined limits and
    triggers RCA requirements for policy violations.
    """
    start_time = datetime.now()
    
    try:
        # Check risk compliance
        violations = compliance_system.risk_enforcer.check_risk_compliance(
            metrics=request.risk_metrics,
            model_id=request.model_id
        )
        
        # Get compliance summary
        compliance_summary = compliance_system.risk_enforcer.get_risk_compliance_summary()
        
        # Format violations
        violations_data = []
        for violation in violations:
            violations_data.append({
                'timestamp': violation.timestamp,
                'policy_name': violation.policy_name,
                'metric_name': violation.metric_name,
                'actual_value': violation.actual_value,
                'policy_limit': violation.policy_limit,
                'violation_magnitude': violation.violation_magnitude,
                'severity': violation.severity.value,
                'rca_required': violation.rca_required,
                'rca_deadline': violation.rca_deadline
            })
        
        compliance_data = {
            'risk_check_results': {
                'violations_found': len(violations),
                'violations': violations_data,
                'metrics_checked': list(request.risk_metrics.keys()),
                'compliance_summary': compliance_summary
            },
            'model_id': request.model_id,
            'risk_metrics': request.risk_metrics
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ComplianceResponse(
            success=True,
            message=f"Risk compliance check completed for {request.model_id}",
            data=compliance_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Risk compliance check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk compliance check failed: {str(e)}")


@router.post("/create-model-card", response_model=ComplianceResponse)
async def create_model_card(request: ModelCardRequest):
    """
    Create comprehensive model documentation card.
    
    Documents model specifications, performance metrics, risk characteristics,
    and governance information for compliance tracking.
    """
    start_time = datetime.now()
    
    try:
        # Create model card
        model_card = compliance_system.governance_system.create_model_card(
            model_id=request.model_id,
            model_details=request.model_details,
            performance_metrics=request.performance_metrics,
            governance_info=request.governance_info
        )
        
        # Get governance compliance summary
        governance_compliance = compliance_system.governance_system.get_governance_compliance()
        
        # Format model card data
        card_data = {
            'model_card': {
                'model_id': model_card.model_id,
                'model_name': model_card.model_name,
                'version': model_card.version,
                'created_date': model_card.created_date,
                'model_type': model_card.model_type,
                'algorithm': model_card.algorithm,
                'features_count': len(model_card.features),
                'target_variable': model_card.target_variable,
                'performance_metrics': {
                    'backtest_sharpe': model_card.backtest_sharpe,
                    'backtest_max_dd': model_card.backtest_max_dd,
                    'spa_p_value': model_card.spa_p_value,
                    'pbo_estimate': model_card.pbo_estimate
                },
                'compliance_status': {
                    'pit_audit_passed': model_card.pit_audit_passed,
                    'evaluation_hygiene_passed': model_card.evaluation_hygiene_passed,
                    'deploy_gates_passed': model_card.deploy_gates_passed
                },
                'governance': {
                    'model_owner': model_card.model_owner,
                    'risk_approver': model_card.risk_approver,
                    'deployment_approver': model_card.deployment_approver,
                    'review_frequency': model_card.review_frequency,
                    'next_review_date': model_card.next_review_date
                }
            },
            'governance_compliance': governance_compliance
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ComplianceResponse(
            success=True,
            message=f"Model card created for {request.model_id}",
            data=card_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Model card creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model card creation failed: {str(e)}")


@router.post("/create-deployment-memo", response_model=ComplianceResponse)
async def create_deployment_memo(request: DeploymentMemoRequest):
    """
    Create deployment memo with required approvals.
    
    Documents deployment justification, risk assessment, approvals,
    and post-deployment monitoring requirements.
    """
    start_time = datetime.now()
    
    try:
        # Convert approval dates from strings to datetime if needed
        approvals = request.approvals.copy()
        if isinstance(approvals['risk_approval_date'], str):
            approvals['risk_approval_date'] = datetime.fromisoformat(approvals['risk_approval_date'])
        if isinstance(approvals['final_approval_date'], str):
            approvals['final_approval_date'] = datetime.fromisoformat(approvals['final_approval_date'])
        
        # Create deployment memo
        deployment_memo = compliance_system.governance_system.create_deployment_memo(
            model_id=request.model_id,
            deployment_details=request.deployment_details,
            approvals=approvals
        )
        
        # Get governance compliance summary
        governance_compliance = compliance_system.governance_system.get_governance_compliance()
        
        # Format memo data
        memo_data = {
            'deployment_memo': {
                'model_id': deployment_memo.model_id,
                'deployment_date': deployment_memo.deployment_date,
                'deployment_version': deployment_memo.deployment_version,
                'business_case': deployment_memo.business_case,
                'expected_pnl_impact': deployment_memo.expected_pnl_impact,
                'risk_assessment': deployment_memo.risk_assessment,
                'approvals': {
                    'risk_reviewer': deployment_memo.risk_reviewer,
                    'risk_approval_date': deployment_memo.risk_approval_date,
                    'final_approver': deployment_memo.final_approver,
                    'final_approval_date': deployment_memo.final_approval_date
                },
                'monitoring_plan': deployment_memo.monitoring_plan,
                'rollback_criteria': deployment_memo.rollback_criteria,
                'success_metrics': deployment_memo.success_metrics
            },
            'governance_compliance': governance_compliance
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ComplianceResponse(
            success=True,
            message=f"Deployment memo created for {request.model_id}",
            data=memo_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Deployment memo creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deployment memo creation failed: {str(e)}")


@router.get("/phase3-dod-status", response_model=ComplianceResponse)
async def get_phase3_dod_status():
    """
    Get comprehensive Phase 3 Definition of Done compliance status.
    
    Provides complete compliance dashboard showing status of all
    Phase 3 DoD requirements and overall compliance rating.
    """
    start_time = datetime.now()
    
    try:
        # Get comprehensive DoD compliance status
        dod_compliance = compliance_system.get_phase3_dod_compliance()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ComplianceResponse(
            success=True,
            message="Phase 3 DoD compliance status retrieved",
            data=dod_compliance,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"DoD status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"DoD status retrieval failed: {str(e)}")


@router.get("/compliance-dashboard")
async def get_compliance_dashboard():
    """Get high-level compliance dashboard with key metrics."""
    
    try:
        # Get all compliance summaries
        pit_summary = compliance_system.pit_auditor.get_violation_summary(30)
        eval_summary = compliance_system.eval_monitor.get_hygiene_compliance()
        reality_summary = compliance_system.reality_monitor.get_reality_gap_compliance()
        risk_summary = compliance_system.risk_enforcer.get_risk_compliance_summary()
        governance_summary = compliance_system.governance_system.get_governance_compliance()
        
        # Calculate overall compliance score
        compliance_scores = [
            1.0 if pit_summary['dod_compliance'] else 0.0,
            1.0 if eval_summary['dod_compliance'] else 0.0,
            1.0 if reality_summary['dod_compliance'] else 0.0,
            1.0 if risk_summary['compliance_status'] == 'compliant' else 0.0,
            1.0 if governance_summary['dod_compliance'] else 0.0
        ]
        
        overall_score = sum(compliance_scores) / len(compliance_scores) * 100
        
        dashboard_data = {
            'overall_compliance_score': overall_score,
            'compliance_status': 'compliant' if overall_score == 100 else 'non_compliant',
            'compliance_breakdown': {
                'pit_leakage': {
                    'status': 'compliant' if pit_summary['dod_compliance'] else 'violations',
                    'critical_high_violations': pit_summary['critical_high_violations'],
                    'requirement': pit_summary['dod_requirement']
                },
                'evaluation_hygiene': {
                    'status': 'compliant' if eval_summary['dod_compliance'] else 'incomplete',
                    'compliance_rate': eval_summary.get('all_tests_compliance_rate', 0.0),
                    'requirement': eval_summary['dod_requirement']
                },
                'reality_gap': {
                    'status': 'compliant' if reality_summary['dod_compliance'] else 'gaps_detected',
                    'compliance_rate': reality_summary.get('compliance_rate', 0.0),
                    'requirement': reality_summary['dod_requirement']
                },
                'risk_compliance': {
                    'status': risk_summary['compliance_status'],
                    'outstanding_rcas': risk_summary['outstanding_rcas'],
                    'requirement': 'Portfolio MaxDD ≤ policy; VaR breaches documented with RCAs'
                },
                'governance': {
                    'status': 'compliant' if governance_summary['dod_compliance'] else 'incomplete',
                    'compliance_rate': governance_summary['governance_compliance_rate'],
                    'requirement': governance_summary['dod_requirement']
                }
            },
            'action_items': _generate_action_items(
                pit_summary, eval_summary, reality_summary, risk_summary, governance_summary
            )
        }
        
        return ComplianceResponse(
            success=True,
            message="Compliance dashboard retrieved",
            data=dashboard_data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Compliance dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance dashboard failed: {str(e)}")


def _generate_action_items(pit_summary, eval_summary, reality_summary, risk_summary, governance_summary) -> List[str]:
    """Generate action items based on compliance status."""
    
    action_items = []
    
    if not pit_summary['dod_compliance']:
        action_items.append(f"Address {pit_summary['critical_high_violations']} critical/high PIT violations")
    
    if not eval_summary['dod_compliance']:
        action_items.append("Ensure all models complete SPA, DSR, and PBO tests")
    
    if not reality_summary['dod_compliance']:
        action_items.append("Investigate reality gaps exceeding 10% threshold")
    
    if risk_summary['compliance_status'] != 'compliant':
        action_items.append(f"Complete {risk_summary['outstanding_rcas']} outstanding RCAs")
    
    if not governance_summary['dod_compliance']:
        action_items.append("Create missing model cards and deployment memos")
    
    if not action_items:
        action_items.append("All compliance requirements met - maintain monitoring")
    
    return action_items


@router.get("/health")
async def health_check():
    """Health check endpoint for compliance service."""
    return {
        'status': 'healthy',
        'service': 'governance-compliance',
        'timestamp': datetime.now(),
        'available_endpoints': [
            'pit-audit',
            'evaluation-audit', 
            'reality-gap',
            'risk-compliance',
            'create-model-card',
            'create-deployment-memo',
            'phase3-dod-status',
            'compliance-dashboard'
        ]
    }


@router.get("/documentation")
async def get_api_documentation():
    """Get comprehensive API documentation for governance and compliance endpoints."""
    
    return {
        'overview': 'Governance & Compliance Framework for Phase 3 Definition of Done requirements',
        'purpose': 'Ensure production-ready models meet all compliance, risk, and governance standards',
        'phase3_dod_requirements': {
            'leakage': {
                'requirement': 'Zero Critical/High PIT violations over 30 days',
                'endpoint': '/pit-audit',
                'description': 'Detects future data leakage in model features'
            },
            'eval_hygiene': {
                'requirement': '100% models log SPA + DSR + PBO; deploy gates enforced',
                'endpoint': '/evaluation-audit',
                'description': 'Ensures statistical significance testing is complete'
            },
            'reality_gap': {
                'requirement': 'Live vs sim slippage gap <10% for ≥80% trades',
                'endpoint': '/reality-gap',
                'description': 'Monitors execution realism and backtesting accuracy'
            },
            'risk': {
                'requirement': 'Portfolio MaxDD ≤ policy; VaR breaches documented with RCAs',
                'endpoint': '/risk-compliance',
                'description': 'Enforces risk limits with violation tracking'
            },
            'governance': {
                'requirement': '100% production models have model cards & deployment memos',
                'endpoints': ['/create-model-card', '/create-deployment-memo'],
                'description': 'Comprehensive model documentation and approval tracking'
            }
        },
        'compliance_framework': {
            'pit_auditing': {
                'description': 'Point-in-Time compliance auditing',
                'checks': ['Future data detection', 'Feature timing validation', 'Survivorship bias'],
                'violation_levels': ['Critical', 'High', 'Medium', 'Low']
            },
            'evaluation_hygiene': {
                'description': 'Model evaluation completeness monitoring',
                'required_tests': ['SPA (Superior Predictive Ability)', 'DSR (Deflated Sharpe Ratio)', 'PBO (Probability of Backtest Overfitting)'],
                'deploy_gates': ['SPA p-value < 0.05', 'DSR > 0', 'PBO ≤ 0.2']
            },
            'reality_gap_monitoring': {
                'description': 'Live vs simulation performance tracking',
                'metrics': ['Slippage comparison', 'Execution timing', 'Fill rates'],
                'tolerance': '10% gap threshold for 80% of trades'
            },
            'risk_enforcement': {
                'description': 'Risk policy enforcement with violation management',
                'policies': ['Max drawdown', 'VaR limits', 'Volatility limits', 'Concentration limits'],
                'rca_process': 'Automated RCA requirements and deadline tracking'
            },
            'governance_system': {
                'description': 'Model lifecycle documentation and approval tracking',
                'components': ['Model cards', 'Deployment memos', 'Approval workflows'],
                'storage': 'Persistent documentation with version control'
            }
        },
        'endpoints': {
            '/phase3-dod-status': {
                'method': 'GET',
                'description': 'Complete Phase 3 DoD compliance dashboard',
                'use_cases': ['Compliance reporting', 'Audit preparation', 'Management dashboards']
            },
            '/compliance-dashboard': {
                'method': 'GET', 
                'description': 'High-level compliance status with action items',
                'use_cases': ['Daily monitoring', 'Compliance reviews', 'Issue tracking']
            },
            '/pit-audit': {
                'method': 'POST',
                'description': 'Audit model features for data leakage',
                'use_cases': ['Pre-deployment validation', 'Feature engineering review']
            },
            '/evaluation-audit': {
                'method': 'POST',
                'description': 'Verify statistical testing completeness',
                'use_cases': ['Deploy gate validation', 'Model approval process']
            }
        },
        'integration_guidance': {
            'deployment_workflow': [
                '1. Conduct PIT audit on model features',
                '2. Complete statistical evaluation (SPA, DSR, PBO)',
                '3. Create model card with performance metrics',
                '4. Obtain approvals and create deployment memo',
                '5. Monitor live performance vs simulation',
                '6. Maintain risk compliance with RCA process'
            ],
            'monitoring_cadence': {
                'daily': 'Risk compliance checks, reality gap monitoring',
                'weekly': 'PIT audit reviews, evaluation hygiene checks',
                'monthly': 'Governance compliance reviews, model card updates',
                'quarterly': 'Comprehensive DoD compliance assessment'
            }
        },
        'compliance_scoring': {
            'overall_score': 'Percentage of DoD requirements met (0-100%)',
            'requirement_weighting': 'Equal weight (20%) for each of 5 DoD requirements',
            'pass_criteria': '100% compliance required for production deployment',
            'action_items': 'Automatically generated based on compliance gaps'
        }
    }