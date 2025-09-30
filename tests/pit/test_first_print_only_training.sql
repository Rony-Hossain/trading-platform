-- Test Suite: First-Print Only Training Data Compliance
-- Validates that training data access enforces Point-in-Time constraints
-- Ensures no look-ahead bias through proper fundamentals data usage

\echo 'Starting First-Print Only Training Tests...'
\echo '================================================='

-- Test Setup: Create test data with different scenarios
DO $$
BEGIN
    -- Clean up any existing test data
    DELETE FROM fundamentals_first_print WHERE symbol LIKE 'TEST%';
    DELETE FROM fundamentals_latest WHERE symbol LIKE 'TEST%';
    DELETE FROM fundamentals_access_log WHERE user_id = 'test_user';
    
    -- Insert test data: First-print data (original filings)
    INSERT INTO fundamentals_first_print (
        symbol, report_date, filing_date, first_print_timestamp, period_type, fiscal_year,
        revenue, net_income, earnings_per_share, roe, data_source, filing_type, 
        data_quality_score, amendment_flag, revision_count
    ) VALUES 
    -- TEST1: Compliant data (45+ days old, high quality, not amended)
    ('TEST1', '2024-03-31', '2024-05-10', '2024-05-10 16:30:00-04'::TIMESTAMPTZ, 'Q1', 2024,
     100000000, 10000000, 1.25, 15.5, 'sec_edgar', '10-Q', 0.95, FALSE, 0),
    
    -- TEST2: Too recent (violates lag requirement)
    ('TEST2', '2024-09-30', '2024-11-01', NOW() - INTERVAL '10 days', 'Q3', 2024,
     120000000, 12000000, 1.50, 16.0, 'sec_edgar', '10-Q', 0.95, FALSE, 0),
    
    -- TEST3: Low quality data (below threshold)
    ('TEST3', '2024-06-30', '2024-08-09', '2024-08-09 16:30:00-04'::TIMESTAMPTZ, 'Q2', 2024,
     90000000, 9000000, 1.10, 14.0, 'sec_edgar', '10-Q', 0.70, FALSE, 0),
    
    -- TEST4: Amendment filing (should be excluded)
    ('TEST4', '2024-03-31', '2024-05-15', '2024-05-15 16:30:00-04'::TIMESTAMPTZ, 'Q1', 2024,
     105000000, 10500000, 1.30, 15.8, 'sec_edgar', '10-Q/A', 0.95, TRUE, 1),
    
    -- TEST5: High revision count
    ('TEST5', '2024-03-31', '2024-05-12', '2024-05-12 16:30:00-04'::TIMESTAMPTZ, 'Q1', 2024,
     110000000, 11000000, 1.35, 16.2, 'sec_edgar', '10-Q', 0.85, FALSE, 5);
    
    -- Insert corresponding latest data with revisions
    INSERT INTO fundamentals_latest (
        symbol, report_date, filing_date, first_print_timestamp, latest_revision_timestamp,
        period_type, fiscal_year, revenue, net_income, earnings_per_share, roe, 
        data_source, filing_type, data_quality_score, amendment_flag, revision_count,
        major_revision, revision_magnitude, revision_reason
    ) VALUES 
    -- TEST1: Minor revision
    ('TEST1', '2024-03-31', '2024-05-10', '2024-05-10 16:30:00-04'::TIMESTAMPTZ, '2024-06-01 10:00:00-04'::TIMESTAMPTZ,
     'Q1', 2024, 101000000, 10100000, 1.26, 15.6, 'sec_edgar', '10-Q', 0.95, FALSE, 1,
     FALSE, 1.0, 'Minor rounding adjustment'),
    
    -- TEST2: Same as first print (no revision yet)
    ('TEST2', '2024-09-30', '2024-11-01', NOW() - INTERVAL '10 days', NOW() - INTERVAL '10 days',
     'Q3', 2024, 120000000, 12000000, 1.50, 16.0, 'sec_edgar', '10-Q', 0.95, FALSE, 0,
     FALSE, 0.0, NULL),
    
    -- TEST3: Major revision improving quality
    ('TEST3', '2024-06-30', '2024-08-09', '2024-08-09 16:30:00-04'::TIMESTAMPTZ, '2024-09-01 14:00:00-04'::TIMESTAMPTZ,
     'Q2', 2024, 95000000, 9500000, 1.16, 14.5, 'sec_edgar', '10-Q', 0.90, FALSE, 2,
     TRUE, 5.6, 'Error correction in revenue recognition'),
    
    -- TEST4: Amendment with significant changes
    ('TEST4', '2024-03-31', '2024-05-15', '2024-05-15 16:30:00-04'::TIMESTAMPTZ, '2024-06-15 16:30:00-04'::TIMESTAMPTZ,
     'Q1', 2024, 108000000, 10800000, 1.33, 16.1, 'sec_edgar', '10-Q/A', 0.95, TRUE, 3,
     TRUE, 2.9, 'Amendment filing with corrections'),
    
    -- TEST5: Multiple major revisions
    ('TEST5', '2024-03-31', '2024-05-12', '2024-05-12 16:30:00-04'::TIMESTAMPTZ, '2024-08-20 12:00:00-04'::TIMESTAMPTZ,
     'Q1', 2024, 115000000, 11500000, 1.42, 16.8, 'sec_edgar', '10-Q', 0.85, FALSE, 5,
     TRUE, 4.5, 'Multiple restatements');
     
    RAISE NOTICE 'Test data setup completed';
END $$;

-- Test 1: Verify training view only includes compliant data
\echo ''
\echo 'Test 1: Training View Compliance Check'
\echo '-------------------------------------'

DO $$
DECLARE
    compliant_count INTEGER;
    total_test_count INTEGER;
    expected_compliant INTEGER := 1; -- Only TEST1 should be compliant
BEGIN
    -- Count compliant records in training view
    SELECT COUNT(*) INTO compliant_count 
    FROM vw_fundamentals_training 
    WHERE symbol LIKE 'TEST%' AND training_compliant = TRUE;
    
    -- Count total test records
    SELECT COUNT(*) INTO total_test_count 
    FROM fundamentals_first_print 
    WHERE symbol LIKE 'TEST%';
    
    RAISE NOTICE 'Total test records: %, Compliant records: %', total_test_count, compliant_count;
    
    IF compliant_count = expected_compliant THEN
        RAISE NOTICE '✓ PASS: Training view correctly filters non-compliant data';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Expected % compliant records, found %', expected_compliant, compliant_count;
    END IF;
END $$;

-- Test 2: Verify lag requirement enforcement
\echo ''
\echo 'Test 2: Lag Requirement Enforcement'
\echo '-----------------------------------'

DO $$
DECLARE
    recent_data_count INTEGER;
BEGIN
    -- Check that recent data (TEST2) is not available for training
    SELECT COUNT(*) INTO recent_data_count
    FROM vw_fundamentals_training 
    WHERE symbol = 'TEST2' AND available_for_training = TRUE;
    
    IF recent_data_count = 0 THEN
        RAISE NOTICE '✓ PASS: Recent data correctly excluded from training';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Recent data should not be available for training';
    END IF;
    
    -- Verify lag calculation
    DECLARE
        lag_days INTEGER;
        expected_lag INTEGER := 45;
    BEGIN
        SELECT EXTRACT(DAYS FROM NOW() - first_print_timestamp) INTO lag_days
        FROM fundamentals_first_print 
        WHERE symbol = 'TEST2';
        
        IF lag_days < expected_lag THEN
            RAISE NOTICE '✓ PASS: Lag calculation correct - % days (< % required)', lag_days, expected_lag;
        ELSE
            RAISE EXCEPTION '✗ FAIL: Lag calculation error - % days should be < %', lag_days, expected_lag;
        END IF;
    END;
END $$;

-- Test 3: Verify quality threshold enforcement
\echo ''
\echo 'Test 3: Data Quality Threshold Enforcement'
\echo '------------------------------------------'

DO $$
DECLARE
    low_quality_count INTEGER;
BEGIN
    -- Check that low quality data (TEST3) is excluded
    SELECT COUNT(*) INTO low_quality_count
    FROM vw_fundamentals_training 
    WHERE symbol = 'TEST3' AND meets_quality_threshold = TRUE;
    
    IF low_quality_count = 0 THEN
        RAISE NOTICE '✓ PASS: Low quality data correctly excluded';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Low quality data should be excluded from training';
    END IF;
END $$;

-- Test 4: Verify amendment exclusion
\echo ''
\echo 'Test 4: Amendment Filing Exclusion'
\echo '----------------------------------'

DO $$
DECLARE
    amendment_count INTEGER;
BEGIN
    -- Check that amendment filings (TEST4) are excluded
    SELECT COUNT(*) INTO amendment_count
    FROM vw_fundamentals_training 
    WHERE symbol = 'TEST4' AND meets_amendment_rules = TRUE;
    
    IF amendment_count = 0 THEN
        RAISE NOTICE '✓ PASS: Amendment filings correctly excluded';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Amendment filings should be excluded from training';
    END IF;
END $$;

-- Test 5: Verify training-safe function compliance
\echo ''
\echo 'Test 5: Training-Safe Function Compliance'
\echo '-----------------------------------------'

DO $$
DECLARE
    safe_records_count INTEGER;
    compliance_record RECORD;
BEGIN
    -- Test the training-safe function
    SELECT COUNT(*) INTO safe_records_count
    FROM get_training_safe_fundamentals(ARRAY['TEST1', 'TEST2', 'TEST3', 'TEST4', 'TEST5']);
    
    -- Should only return TEST1 data
    IF safe_records_count = 1 THEN
        RAISE NOTICE '✓ PASS: Training-safe function returns only compliant data';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Training-safe function returned % records, expected 1', safe_records_count;
    END IF;
    
    -- Verify the returned record is TEST1
    SELECT symbol INTO compliance_record
    FROM get_training_safe_fundamentals(ARRAY['TEST1', 'TEST2', 'TEST3', 'TEST4', 'TEST5'])
    LIMIT 1;
    
    IF compliance_record.symbol = 'TEST1' THEN
        RAISE NOTICE '✓ PASS: Correct symbol returned (TEST1)';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Wrong symbol returned: %', compliance_record.symbol;
    END IF;
END $$;

-- Test 6: Verify compliance validation function
\echo ''
\echo 'Test 6: Compliance Validation Function'
\echo '--------------------------------------'

DO $$
DECLARE
    validation_result RECORD;
    violations_found TEXT[];
BEGIN
    -- Test compliant data (TEST1)
    SELECT * INTO validation_result
    FROM validate_training_data_compliance(ARRAY['TEST1'])
    WHERE symbol = 'TEST1';
    
    IF validation_result.compliance_status = 'COMPLIANT' THEN
        RAISE NOTICE '✓ PASS: TEST1 correctly identified as compliant';
    ELSE
        RAISE EXCEPTION '✗ FAIL: TEST1 should be compliant, status: %', validation_result.compliance_status;
    END IF;
    
    -- Test non-compliant data (TEST2 - insufficient lag)
    SELECT * INTO validation_result
    FROM validate_training_data_compliance(ARRAY['TEST2'])
    WHERE symbol = 'TEST2';
    
    IF validation_result.compliance_status IN ('PENDING', 'VIOLATION') THEN
        RAISE NOTICE '✓ PASS: TEST2 correctly identified as non-compliant (%)', validation_result.compliance_status;
    ELSE
        RAISE EXCEPTION '✗ FAIL: TEST2 should be non-compliant, status: %', validation_result.compliance_status;
    END IF;
    
    -- Test low quality data (TEST3)
    SELECT * INTO validation_result
    FROM validate_training_data_compliance(ARRAY['TEST3'])
    WHERE symbol = 'TEST3';
    
    violations_found := validation_result.violation_details;
    IF 'low_quality' = ANY(SELECT unnest(violations_found) LIKE 'low_quality%') THEN
        RAISE NOTICE '✓ PASS: TEST3 correctly flagged for low quality';
    ELSE
        RAISE EXCEPTION '✗ FAIL: TEST3 should be flagged for low quality';
    END IF;
END $$;

-- Test 7: Verify access logging
\echo ''
\echo 'Test 7: Access Logging Verification'
\echo '-----------------------------------'

DO $$
DECLARE
    log_count INTEGER;
    recent_log RECORD;
BEGIN
    -- Trigger a logged access
    PERFORM * FROM get_training_safe_fundamentals(ARRAY['TEST1']);
    
    -- Check if access was logged
    SELECT COUNT(*) INTO log_count
    FROM fundamentals_access_log 
    WHERE user_id = session_user 
    AND query_type = 'training_data'
    AND timestamp >= NOW() - INTERVAL '1 minute';
    
    IF log_count > 0 THEN
        RAISE NOTICE '✓ PASS: Training data access properly logged';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Training data access not logged';
    END IF;
    
    -- Check log details
    SELECT * INTO recent_log
    FROM fundamentals_access_log 
    WHERE user_id = session_user 
    AND query_type = 'training_data'
    ORDER BY timestamp DESC
    LIMIT 1;
    
    IF recent_log.data_version = 'first_print' AND recent_log.access_granted = TRUE THEN
        RAISE NOTICE '✓ PASS: Access log contains correct details';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Access log details incorrect';
    END IF;
END $$;

-- Test 8: Verify row-level security policies
\echo ''
\echo 'Test 8: Row-Level Security Policies'
\echo '-----------------------------------'

DO $$
DECLARE
    rls_enabled BOOLEAN;
    policy_count INTEGER;
BEGIN
    -- Check if RLS is enabled on fundamentals tables
    SELECT relrowsecurity INTO rls_enabled
    FROM pg_class 
    WHERE relname = 'fundamentals_first_print';
    
    IF rls_enabled THEN
        RAISE NOTICE '✓ PASS: Row-level security enabled on first_print table';
    ELSE
        RAISE NOTICE '⚠ WARNING: Row-level security not enabled on first_print table';
    END IF;
    
    -- Check if policies exist
    SELECT COUNT(*) INTO policy_count
    FROM pg_policies 
    WHERE tablename IN ('fundamentals_first_print', 'fundamentals_latest');
    
    IF policy_count > 0 THEN
        RAISE NOTICE '✓ PASS: Security policies configured (% policies found)', policy_count;
    ELSE
        RAISE NOTICE '⚠ WARNING: No security policies found';
    END IF;
END $$;

-- Test 9: Verify materialized view performance
\echo ''
\echo 'Test 9: Materialized View Performance Test'
\echo '------------------------------------------'

DO $$
DECLARE
    mv_count INTEGER;
    view_count INTEGER;
    performance_ratio NUMERIC;
    start_time TIMESTAMPTZ;
    mv_time INTERVAL;
    view_time INTERVAL;
BEGIN
    -- Test materialized view
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO mv_count FROM mvw_fundamentals_training_fast WHERE symbol LIKE 'TEST%';
    mv_time := clock_timestamp() - start_time;
    
    -- Test regular view
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO view_count FROM vw_fundamentals_training WHERE symbol LIKE 'TEST%';
    view_time := clock_timestamp() - start_time;
    
    IF mv_count = view_count THEN
        RAISE NOTICE '✓ PASS: Materialized view returns same results as regular view';
    ELSE
        RAISE EXCEPTION '✗ FAIL: Materialized view count (%) != regular view count (%)', mv_count, view_count;
    END IF;
    
    -- Performance comparison (materialized view should be faster)
    IF mv_time <= view_time THEN
        RAISE NOTICE '✓ PASS: Materialized view performance acceptable (% vs %)', mv_time, view_time;
    ELSE
        RAISE NOTICE '⚠ WARNING: Materialized view slower than regular view (% vs %)', mv_time, view_time;
    END IF;
END $$;

-- Test 10: Verify no access to latest data in training context
\echo ''
\echo 'Test 10: Latest Data Access Prevention'
\echo '--------------------------------------'

DO $$
DECLARE
    latest_access_count INTEGER;
BEGIN
    -- Attempt to access latest data (should be restricted in training context)
    -- This test assumes application context is set
    
    BEGIN
        -- Set training context
        PERFORM set_config('app.context', 'training', false);
        
        -- Try to access latest data
        SELECT COUNT(*) INTO latest_access_count
        FROM fundamentals_latest 
        WHERE symbol LIKE 'TEST%';
        
        -- Reset context
        PERFORM set_config('app.context', '', false);
        
    EXCEPTION WHEN insufficient_privilege THEN
        latest_access_count := -1; -- Access properly denied
        PERFORM set_config('app.context', '', false);
    END;
    
    IF latest_access_count = -1 OR latest_access_count = 0 THEN
        RAISE NOTICE '✓ PASS: Latest data access properly restricted in training context';
    ELSE
        RAISE NOTICE '⚠ WARNING: Latest data access not properly restricted (% records accessible)', latest_access_count;
    END IF;
END $$;

-- Test 11: Cross-validation with historical data
\echo ''
\echo 'Test 11: Historical Data Cross-Validation'
\echo '-----------------------------------------'

DO $$
DECLARE
    historical_violations INTEGER;
    total_historical INTEGER;
    violation_rate NUMERIC;
BEGIN
    -- Check compliance of existing historical data
    SELECT 
        COUNT(CASE WHEN compliance_status != 'COMPLIANT' THEN 1 END),
        COUNT(*)
    INTO historical_violations, total_historical
    FROM validate_training_data_compliance(
        ARRAY(SELECT DISTINCT symbol FROM fundamentals_first_print WHERE symbol NOT LIKE 'TEST%' LIMIT 10),
        CURRENT_DATE,
        365
    );
    
    IF total_historical > 0 THEN
        violation_rate := (historical_violations::NUMERIC / total_historical::NUMERIC) * 100;
        
        IF violation_rate <= 10 THEN -- Allow up to 10% violations in historical data
            RAISE NOTICE '✓ PASS: Historical data compliance acceptable (%.1f%% violations)', violation_rate;
        ELSE
            RAISE NOTICE '⚠ WARNING: High historical data violation rate (%.1f%%)', violation_rate;
        END IF;
    ELSE
        RAISE NOTICE '⚠ INFO: No historical data available for validation';
    END IF;
END $$;

-- Test Cleanup
\echo ''
\echo 'Test Cleanup'
\echo '------------'

DO $$
BEGIN
    -- Clean up test data
    DELETE FROM fundamentals_first_print WHERE symbol LIKE 'TEST%';
    DELETE FROM fundamentals_latest WHERE symbol LIKE 'TEST%';
    DELETE FROM fundamentals_access_log WHERE user_id = 'test_user';
    
    RAISE NOTICE '✓ Test data cleanup completed';
END $$;

-- Performance and statistics summary
\echo ''
\echo 'Test Summary and Statistics'
\echo '============================'

DO $$
DECLARE
    total_first_print INTEGER;
    total_latest INTEGER;
    compliant_percentage NUMERIC;
    view_count INTEGER;
    access_log_count INTEGER;
BEGIN
    -- Get statistics
    SELECT COUNT(*) INTO total_first_print FROM fundamentals_first_print;
    SELECT COUNT(*) INTO total_latest FROM fundamentals_latest;
    SELECT COUNT(*) INTO view_count FROM vw_fundamentals_training WHERE training_compliant = TRUE;
    SELECT COUNT(*) INTO access_log_count FROM fundamentals_access_log WHERE timestamp >= CURRENT_DATE;
    
    IF total_first_print > 0 THEN
        compliant_percentage := (view_count::NUMERIC / total_first_print::NUMERIC) * 100;
    ELSE
        compliant_percentage := 0;
    END IF;
    
    RAISE NOTICE 'Database Statistics:';
    RAISE NOTICE '- Total first-print records: %', total_first_print;
    RAISE NOTICE '- Total latest records: %', total_latest;
    RAISE NOTICE '- Training-compliant records: % (%.1f%%)', view_count, compliant_percentage;
    RAISE NOTICE '- Access log entries today: %', access_log_count;
    
    -- Compliance recommendations
    IF compliant_percentage < 80 THEN
        RAISE NOTICE '⚠ RECOMMENDATION: Low compliance rate - review data quality processes';
    END IF;
    
    IF total_first_print != total_latest THEN
        RAISE NOTICE '⚠ RECOMMENDATION: First-print and latest record counts differ - investigate data loading';
    END IF;
END $$;

\echo ''
\echo '================================================='
\echo 'First-Print Only Training Tests Completed'
\echo '================================================='

-- Final validation query that can be run manually
\echo ''
\echo 'Manual Validation Queries:'
\echo '--------------------------'
\echo 'To manually verify training view compliance:'
\echo 'SELECT symbol, report_date, training_compliant, available_for_training FROM vw_fundamentals_training LIMIT 10;'
\echo ''
\echo 'To check recent access patterns:'
\echo 'SELECT * FROM vw_training_compliance_monitor WHERE access_date >= CURRENT_DATE - INTERVAL ''7 days'';'
\echo ''
\echo 'To audit user access patterns:'
\echo 'SELECT * FROM audit_fundamentals_access_patterns(30);'