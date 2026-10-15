"""Unit tests for pytest plugin fixtures and quality gate assertions."""

import pytest
from evalguard.evaluators.deterministic import DeterministicEvaluator
from evalguard.models import TestCase
from evalguard.pytest_plugin import assert_eval_passed


def test_pytest_plugin_assertion_pass():
    evaluator = DeterministicEvaluator()
    tc = TestCase(
        id="gate_pass_test",
        query="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        retrieved_contexts=["Geography: Paris is the capital and largest city of France."],
    )
    result = evaluator.evaluate(tc)
    # Should not raise
    assert_eval_passed(result, min_overall_score=0.60)


def test_pytest_plugin_assertion_fail():
    evaluator = DeterministicEvaluator()
    tc = TestCase(
        id="gate_fail_test",
        query="What is the refund policy?",
        actual_output="Refunds are impossible and denied forever.",
        retrieved_contexts=["All customers can request a full refund within 30 days."],
    )
    result = evaluator.evaluate(tc)
    with pytest.raises(AssertionError) as exc_info:
        assert_eval_passed(result, min_overall_score=0.70)
    assert "EvalGuard quality gate failed" in str(exc_info.value)
