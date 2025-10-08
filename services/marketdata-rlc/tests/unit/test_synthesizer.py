from app.policy.synthesizer import PolicySynthesizer


def test_token_bucket_from_rate() -> None:
    synthesizer = PolicySynthesizer()
    params = synthesizer.token_bucket_from_rate(target_rps=120.0, desired_burst_seconds=2.0)

    assert 0.5 <= params.refill_rate <= 500.0
    assert params.burst >= int(params.refill_rate * 1.9)
    assert 0 <= params.jitter_ms <= 2000
    assert params.ttl_s >= 10


def test_quotas_from_budget() -> None:
    synthesizer = PolicySynthesizer()
    ok = synthesizer.quotas_from_budget(t0_demand=150, t1_demand=200, budget_breach=False)
    assert ok.t0_max == 150
    assert ok.t1_max == 200
    assert ok.t2_mode == "60s"

    breaching = synthesizer.quotas_from_budget(t0_demand=150, t1_demand=200, budget_breach=True)
    assert breaching.t0_max >= 50
    assert breaching.t1_max == 120
    assert breaching.t2_mode == "EOD"
