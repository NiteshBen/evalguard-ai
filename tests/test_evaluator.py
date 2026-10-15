"""Unit tests for DeterministicEvaluator and LLMJudgeEvaluator."""

import pytest
from evalguard.evaluators.deterministic import DeterministicEvaluator
from evalguard.evaluators.llm_judge import LLMJudgeEvaluator
from evalguard.examples.sample_rag_data import get_sample_test_suite
from evalguard.models import TestCase


def test_deterministic_evaluator_single_case():
    evaluator = DeterministicEvaluator()
    tc = TestCase(
        id="eval_test_1",
        query="What is the speed of light in vacuum?",
        actual_output="The speed of light in vacuum is approximately 299,792 kilometers per second.",
        retrieved_contexts=[
            "Physical Constants: The speed of light in vacuum, denoted c, is exactly 299,792,458 metres per second."
        ],
        expected_output="Light travels at approximately 300,000 km/s in vacuum.",
        latency_ms=150.0,
        tokens_used=60,
    )

    result = evaluator.evaluate(tc)
    assert result.test_case.id == "eval_test_1"
    assert "faithfulness" in result.metrics
    assert "answer_relevance" in result.metrics
    assert "context_precision" in result.metrics
    assert "guardrail_safety" in result.metrics
    assert "latency_cost" in result.metrics
    assert result.overall_score > 0.0
    assert result.execution_time_ms >= 0.0


def test_deterministic_evaluator_suite_run():
    evaluator = DeterministicEvaluator()
    test_cases = get_sample_test_suite()

    suite = evaluator.evaluate_suite("Sample Production Benchmark", test_cases)
    assert suite.suite_name == "Sample Production Benchmark"
    assert len(suite.results) == len(test_cases)
    assert suite.summary.total_tests == len(test_cases)
    assert 0.0 <= suite.summary.pass_rate <= 1.0


def test_llm_judge_offline_fallback():
    # When no API key is provided, LLMJudgeEvaluator must fall back to DeterministicEvaluator without errors
    judge = LLMJudgeEvaluator(api_key=None)
    assert judge.is_online is False

    tc = TestCase(
        id="judge_fallback_test",
        query="Explain TCP handshake.",
        actual_output="TCP uses a three-way handshake: SYN, SYN-ACK, and ACK.",
        retrieved_contexts=["Networking: TCP establishes a connection using SYN, SYN-ACK, ACK."],
    )

    result = judge.evaluate(tc)
    assert result.test_case.id == "judge_fallback_test"
    assert result.passed is True
