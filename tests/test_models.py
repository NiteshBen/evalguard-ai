"""Unit tests for EvalGuard data models and schemas."""

import pytest
from evalguard.models import (
    ClaimStatus,
    ClaimVerification,
    EvalResult,
    MetricScore,
    MetricType,
    SuiteResult,
    SuiteSummary,
    TestCase,
)


def test_metric_score_evaluation():
    # Passed threshold
    score_pass = MetricScore.evaluate(name="test_metric", score=0.85, threshold=0.70)
    assert score_pass.passed is True
    assert score_pass.score == 0.85

    # Failed threshold
    score_fail = MetricScore.evaluate(name="test_metric", score=0.65, threshold=0.70)
    assert score_fail.passed is False
    assert score_fail.score == 0.65

    # Clamping behavior
    score_high = MetricScore.evaluate(name="test_metric", score=1.5)
    assert score_high.score == 1.0

    score_low = MetricScore.evaluate(name="test_metric", score=-0.5)
    assert score_low.score == 0.0


def test_test_case_creation():
    tc = TestCase(
        id="tc_1",
        query="What is AI?",
        actual_output="Artificial Intelligence.",
        retrieved_contexts=["AI is Artificial Intelligence."],
        expected_output="Artificial Intelligence.",
        latency_ms=120.5,
        tokens_used=45,
    )
    assert tc.id == "tc_1"
    assert len(tc.retrieved_contexts) == 1
    assert tc.latency_ms == 120.5


def test_claim_verification_model():
    cv = ClaimVerification(
        claim="The warranty lasts 3 years.",
        status=ClaimStatus.SUPPORTED,
        confidence=0.92,
        evidence="3-year limited warranty included.",
    )
    assert cv.status == ClaimStatus.SUPPORTED
    assert cv.confidence == 0.92
    assert cv.evidence is not None


def test_suite_result_computed_summary():
    tc1 = TestCase(id="t1", query="q1", actual_output="a1")
    tc2 = TestCase(id="t2", query="q2", actual_output="a2")

    r1 = EvalResult(
        test_case=tc1,
        metrics={"faithfulness": MetricScore.evaluate("faithfulness", 0.9, 0.7)},
        passed=True,
        overall_score=0.9,
    )
    r2 = EvalResult(
        test_case=tc2,
        metrics={"faithfulness": MetricScore.evaluate("faithfulness", 0.5, 0.7)},
        passed=False,
        overall_score=0.5,
    )

    suite = SuiteResult(
        suite_name="Unit Test Suite",
        results=[r1, r2],
        timestamp="2026-10-15T00:00:00Z",
    )

    summary = suite.summary
    assert summary.total_tests == 2
    assert summary.passed_tests == 1
    assert summary.failed_tests == 1
    assert summary.pass_rate == 0.5
    assert summary.average_scores["faithfulness"] == 0.7
    assert suite.passed is False


def test_empty_suite_summary():
    suite = SuiteResult(
        suite_name="Empty Suite",
        results=[],
        timestamp="2026-10-15T00:00:00Z",
    )
    assert suite.summary.total_tests == 0
    assert suite.summary.pass_rate == 1.0
    assert suite.passed is True
