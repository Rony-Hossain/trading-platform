-- Migration 010: Point-in-Time (PIT) Enforcement Tables
-- Creates tables for tracking and monitoring PIT violations across the trading platform

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================
-- PIT Violations Tracking Table
-- =====================================================

CREATE TABLE IF NOT EXISTS pit_violations (
    id SERIAL PRIMARY KEY,
    violation_id VARCHAR(255) UNIQUE NOT NULL,
    violation_type VARCHAR(50) NOT NULL CHECK (violation_type IN (
        'future_leak',
        'timestamp_drift', 
        'arrival_delay',
        'revision_backdating',
        'contract_violation'
    )),
    service_name VARCHAR(100) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    metadata JSONB,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    resolution_method VARCHAR(100),
    created_by VARCHAR(100) DEFAULT 'pit_enforcement_system',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('pit_violations', 'detected_at', if_not_exists => TRUE);

-- Add data retention policy (keep 2 years of violation data)
SELECT add_retention_policy('pit_violations', INTERVAL '2 years', if_not_exists => TRUE);

-- =====================================================
-- PIT Enforcement Metrics Table  
-- =====================================================

CREATE TABLE IF NOT EXISTS pit_enforcement_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    metric_type VARCHAR(50) NOT NULL CHECK (metric_type IN (
        'violation_count',
        'latency_ms',
        'compliance_score',
        'data_quality_score',
        'processing_time_ms'
    )),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable for metrics
SELECT create_hypertable('pit_enforcement_metrics', 'timestamp', if_not_exists => TRUE);

-- Add data retention policy (keep 1 year of metrics)
SELECT add_retention_policy('pit_enforcement_metrics', INTERVAL '1 year', if_not_exists => TRUE);

-- =====================================================
-- Timestamp Normalization Audit Table
-- =====================================================

CREATE TABLE IF NOT EXISTS timestamp_normalization_audit (
    id SERIAL PRIMARY KEY,
    source_service VARCHAR(100) NOT NULL,
    source_table VARCHAR(100),
    source_column VARCHAR(100),
    original_timestamp TEXT NOT NULL,
    normalized_timestamp TIMESTAMPTZ NOT NULL,
    original_timezone VARCHAR(50),
    detected_timezone VARCHAR(50),
    normalization_method VARCHAR(100) NOT NULL,
    precision_level VARCHAR(20) NOT NULL CHECK (precision_level IN (
        'seconds',
        'milliseconds', 
        'microseconds',
        'nanoseconds'
    )),
    validation_errors TEXT[],
    confidence_score DOUBLE PRECISION NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- Create hypertable for audit logs
SELECT create_hypertable('timestamp_normalization_audit', 'processed_at', if_not_exists => TRUE);

-- Add data retention policy (keep 6 months of audit data)
SELECT add_retention_policy('timestamp_normalization_audit', INTERVAL '6 months', if_not_exists => TRUE);

-- =====================================================
-- Feature Contract Compliance Table
-- =====================================================

CREATE TABLE IF NOT EXISTS feature_contract_compliance (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(255) NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    contract_version VARCHAR(20) NOT NULL,
    compliance_check_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    compliance_status VARCHAR(20) NOT NULL CHECK (compliance_status IN (
        'compliant',
        'non_compliant',
        'warning',
        'unknown'
    )),
    violation_details JSONB,
    pit_rule_violations TEXT[],
    sla_violations TEXT[],
    data_quality_score DOUBLE PRECISION CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    last_violation_timestamp TIMESTAMPTZ,
    auto_remediation_applied BOOLEAN DEFAULT FALSE,
    remediation_details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable for compliance tracking
SELECT create_hypertable('feature_contract_compliance', 'compliance_check_timestamp', if_not_exists => TRUE);

-- =====================================================
-- Temporal Data Quality Metrics
-- =====================================================

CREATE TABLE IF NOT EXISTS temporal_data_quality (
    id SERIAL PRIMARY KEY,
    measurement_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    service_name VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    timestamp_column VARCHAR(100),
    
    -- Quality metrics
    total_records BIGINT NOT NULL,
    null_timestamps BIGINT NOT NULL DEFAULT 0,
    future_timestamps BIGINT NOT NULL DEFAULT 0,
    duplicate_timestamps BIGINT NOT NULL DEFAULT 0,
    out_of_order_timestamps BIGINT NOT NULL DEFAULT 0,
    timezone_inconsistencies BIGINT NOT NULL DEFAULT 0,
    precision_issues BIGINT NOT NULL DEFAULT 0,
    
    -- Calculated scores
    completeness_score DOUBLE PRECISION NOT NULL CHECK (completeness_score >= 0 AND completeness_score <= 100),
    timeliness_score DOUBLE PRECISION NOT NULL CHECK (timeliness_score >= 0 AND timeliness_score <= 100),
    consistency_score DOUBLE PRECISION NOT NULL CHECK (consistency_score >= 0 AND consistency_score <= 100),
    overall_quality_score DOUBLE PRECISION NOT NULL CHECK (overall_quality_score >= 0 AND overall_quality_score <= 100),
    
    -- Time range analyzed
    analysis_start_time TIMESTAMPTZ NOT NULL,
    analysis_end_time TIMESTAMPTZ NOT NULL,
    
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable for quality metrics
SELECT create_hypertable('temporal_data_quality', 'measurement_timestamp', if_not_exists => TRUE);

-- =====================================================
-- Indexes for Performance
-- =====================================================

-- PIT Violations indexes
CREATE INDEX IF NOT EXISTS idx_pit_violations_service_time 
    ON pit_violations(service_name, detected_at DESC);
    
CREATE INDEX IF NOT EXISTS idx_pit_violations_type_severity 
    ON pit_violations(violation_type, severity, detected_at DESC);
    
CREATE INDEX IF NOT EXISTS idx_pit_violations_unresolved 
    ON pit_violations(detected_at DESC) 
    WHERE resolved_at IS NULL;
    
CREATE INDEX IF NOT EXISTS idx_pit_violations_feature 
    ON pit_violations(feature_name, detected_at DESC);

-- Metrics indexes  
CREATE INDEX IF NOT EXISTS idx_pit_metrics_service_metric 
    ON pit_enforcement_metrics(service_name, metric_name, timestamp DESC);
    
CREATE INDEX IF NOT EXISTS idx_pit_metrics_type 
    ON pit_enforcement_metrics(metric_type, timestamp DESC);

-- Audit indexes
CREATE INDEX IF NOT EXISTS idx_timestamp_audit_service 
    ON timestamp_normalization_audit(source_service, processed_at DESC);
    
CREATE INDEX IF NOT EXISTS idx_timestamp_audit_confidence 
    ON timestamp_normalization_audit(confidence_score, processed_at DESC);

-- Compliance indexes
CREATE INDEX IF NOT EXISTS idx_compliance_feature_status 
    ON feature_contract_compliance(feature_name, compliance_status, compliance_check_timestamp DESC);
    
CREATE INDEX IF NOT EXISTS idx_compliance_service 
    ON feature_contract_compliance(service_name, compliance_check_timestamp DESC);

-- Quality indexes
CREATE INDEX IF NOT EXISTS idx_quality_service_table 
    ON temporal_data_quality(service_name, table_name, measurement_timestamp DESC);
    
CREATE INDEX IF NOT EXISTS idx_quality_overall_score 
    ON temporal_data_quality(overall_quality_score, measurement_timestamp DESC);

-- =====================================================
-- Views for Monitoring and Reporting
-- =====================================================

-- Current PIT violations summary
CREATE OR REPLACE VIEW current_pit_violations AS
SELECT 
    service_name,
    violation_type,
    severity,
    COUNT(*) as violation_count,
    MIN(detected_at) as earliest_violation,
    MAX(detected_at) as latest_violation
FROM pit_violations 
WHERE resolved_at IS NULL
GROUP BY service_name, violation_type, severity
ORDER BY 
    CASE severity 
        WHEN 'critical' THEN 1 
        WHEN 'high' THEN 2 
        WHEN 'medium' THEN 3 
        WHEN 'low' THEN 4 
    END,
    violation_count DESC;

-- Service compliance dashboard
CREATE OR REPLACE VIEW service_compliance_dashboard AS
SELECT 
    service_name,
    COUNT(DISTINCT feature_name) as total_features,
    SUM(CASE WHEN compliance_status = 'compliant' THEN 1 ELSE 0 END) as compliant_features,
    SUM(CASE WHEN compliance_status = 'non_compliant' THEN 1 ELSE 0 END) as non_compliant_features,
    AVG(data_quality_score) as avg_quality_score,
    MAX(compliance_check_timestamp) as last_check
FROM feature_contract_compliance fcc
WHERE compliance_check_timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY service_name
ORDER BY avg_quality_score DESC;

-- Temporal data quality trends
CREATE OR REPLACE VIEW temporal_quality_trends AS
SELECT 
    DATE_TRUNC('hour', measurement_timestamp) as hour,
    service_name,
    AVG(overall_quality_score) as avg_quality_score,
    MIN(overall_quality_score) as min_quality_score,
    MAX(overall_quality_score) as max_quality_score,
    COUNT(*) as measurement_count
FROM temporal_data_quality
WHERE measurement_timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', measurement_timestamp), service_name
ORDER BY hour DESC, service_name;

-- =====================================================
-- Functions for PIT Enforcement
-- =====================================================

-- Function to log PIT violations
CREATE OR REPLACE FUNCTION log_pit_violation(
    p_violation_id VARCHAR(255),
    p_violation_type VARCHAR(50),
    p_service_name VARCHAR(100),
    p_feature_name VARCHAR(255),
    p_timestamp TIMESTAMPTZ,
    p_as_of_timestamp TIMESTAMPTZ,
    p_description TEXT,
    p_severity VARCHAR(20),
    p_metadata JSONB DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO pit_violations (
        violation_id, violation_type, service_name, feature_name,
        timestamp, as_of_timestamp, description, severity, metadata
    ) VALUES (
        p_violation_id, p_violation_type, p_service_name, p_feature_name,
        p_timestamp, p_as_of_timestamp, p_description, p_severity, p_metadata
    ) RETURNING id INTO v_id;
    
    -- Log metric for violation count
    INSERT INTO pit_enforcement_metrics (
        service_name, metric_name, metric_value, metric_type, metadata
    ) VALUES (
        p_service_name, 'violation_detected', 1, 'violation_count',
        jsonb_build_object('violation_type', p_violation_type, 'severity', p_severity)
    );
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to resolve PIT violations
CREATE OR REPLACE FUNCTION resolve_pit_violation(
    p_violation_id VARCHAR(255),
    p_resolution_notes TEXT DEFAULT NULL,
    p_resolution_method VARCHAR(100) DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE pit_violations 
    SET 
        resolved_at = NOW(),
        resolution_notes = p_resolution_notes,
        resolution_method = p_resolution_method,
        updated_at = NOW()
    WHERE violation_id = p_violation_id
    AND resolved_at IS NULL;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate service compliance score
CREATE OR REPLACE FUNCTION calculate_service_compliance_score(
    p_service_name VARCHAR(100),
    p_time_window INTERVAL DEFAULT '24 hours'
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_score DOUBLE PRECISION := 0;
    v_total_checks INTEGER := 0;
    v_compliant_checks INTEGER := 0;
BEGIN
    SELECT 
        COUNT(*),
        SUM(CASE WHEN compliance_status = 'compliant' THEN 1 ELSE 0 END)
    INTO v_total_checks, v_compliant_checks
    FROM feature_contract_compliance
    WHERE service_name = p_service_name
    AND compliance_check_timestamp >= NOW() - p_time_window;
    
    IF v_total_checks > 0 THEN
        v_score := (v_compliant_checks::DOUBLE PRECISION / v_total_checks::DOUBLE PRECISION) * 100;
    END IF;
    
    RETURN v_score;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Triggers for Automatic Maintenance
-- =====================================================

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_pit_violations_updated_at
    BEFORE UPDATE ON pit_violations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- Initial Data and Configuration
-- =====================================================

-- Insert initial configuration data
INSERT INTO pit_enforcement_metrics (service_name, metric_name, metric_value, metric_type, metadata)
VALUES 
    ('system', 'pit_enforcement_initialized', 1, 'violation_count', '{"version": "1.0.0", "initialized_at": "' || NOW() || '"}')
ON CONFLICT DO NOTHING;

-- Grant permissions (adjust as needed for your setup)
GRANT SELECT, INSERT, UPDATE ON pit_violations TO trading_user;
GRANT SELECT, INSERT ON pit_enforcement_metrics TO trading_user;
GRANT SELECT, INSERT ON timestamp_normalization_audit TO trading_user;
GRANT SELECT, INSERT, UPDATE ON feature_contract_compliance TO trading_user;
GRANT SELECT, INSERT ON temporal_data_quality TO trading_user;

GRANT SELECT ON current_pit_violations TO trading_user;
GRANT SELECT ON service_compliance_dashboard TO trading_user; 
GRANT SELECT ON temporal_quality_trends TO trading_user;

GRANT EXECUTE ON FUNCTION log_pit_violation TO trading_user;
GRANT EXECUTE ON FUNCTION resolve_pit_violation TO trading_user;
GRANT EXECUTE ON FUNCTION calculate_service_compliance_score TO trading_user;

-- =====================================================
-- Comments for Documentation
-- =====================================================

COMMENT ON TABLE pit_violations IS 'Tracks point-in-time violations across all trading services';
COMMENT ON TABLE pit_enforcement_metrics IS 'Performance and compliance metrics for PIT enforcement';
COMMENT ON TABLE timestamp_normalization_audit IS 'Audit trail for timestamp normalization operations';
COMMENT ON TABLE feature_contract_compliance IS 'Feature contract compliance tracking and validation';
COMMENT ON TABLE temporal_data_quality IS 'Data quality metrics for temporal data consistency';

COMMENT ON FUNCTION log_pit_violation IS 'Logs a new PIT violation with automatic metrics recording';
COMMENT ON FUNCTION resolve_pit_violation IS 'Marks a PIT violation as resolved with resolution details';
COMMENT ON FUNCTION calculate_service_compliance_score IS 'Calculates compliance score for a service over time window';

-- Migration completion
INSERT INTO schema_migrations (version, description, executed_at) 
VALUES ('010', 'Point-in-Time (PIT) Enforcement Tables', NOW())
ON CONFLICT (version) DO NOTHING;