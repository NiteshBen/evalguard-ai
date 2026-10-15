"""Unit tests for FaithfulnessMetric and hallucination detection."""

import pytest
from evalguard.metrics.faithfulness import FaithfulnessMetric
from evalguard.models import ClaimStatus, TestCase


def test_faithfulness_grounded_output():
    metric = FaithfulnessMetric(threshold=0.70)
    tc = TestCase(
        id="test_grounded",
        query="What is the refund period?",
        actual_output="Customers can cancel within 30 days of purchase for a full refund.",
        retrieved_contexts=[
            "Subscription Refund Policy: Customers are eligible for a 100% full refund if cancelled within 30 days of purchase."
        ],
    )
    score = metric.evaluate(tc)
    assert score.passed is True
    assert score.score >= 0.70
    assert "verifications" in score.metadata
    assert len(score.metadata["verifications"]) >= 1


def test_faithfulness_hallucinated_output():
    metric = FaithfulnessMetric(threshold=0.70)
    tc = TestCase(
        id="test_hallucinated",
        query="What is the warranty period?",
        actual_output="The product includes a 25-year lifetime unconditional replacement guarantee with free home repairs.",
        retrieved_contexts=[
            "Limited Hardware Warranty: Each server unit includes a standard 1-year limited warranty covering manufacturer defects."
        ],
    )
    score = metric.evaluate(tc)
    # The output makes claims not supported by the 1-year limited warranty context
    assert score.score < 0.70
    assert score.passed is False


def test_faithfulness_contradiction_detection():
    metric = FaithfulnessMetric(threshold=0.70)
    tc = TestCase(
        id="test_contradiction",
        query="Is water damage covered?",
        actual_output="The warranty does cover accidental water damage and spills.",
        retrieved_contexts=[
            "The warranty does not cover accidental water damage and spills under any circumstances."
        ],
    )
    score = metric.evaluate(tc)
    assert score.passed is False
    assert score.metadata.get("contradicted_claims", 0) > 0


def test_faithfulness_empty_context():
    metric = FaithfulnessMetric()
    tc = TestCase(
        id="test_no_context",
        query="What is quantum computing?",
        actual_output="Quantum computing uses qubits and superposition.",
        retrieved_contexts=[],
    )
    score = metric.evaluate(tc)
    assert score.score == 0.0
    assert score.passed is False


def test_faithfulness_empty_output():
    metric = FaithfulnessMetric()
    tc = TestCase(
        id="test_empty_output",
        query="What is AI?",
        actual_output="",
        retrieved_contexts=["AI stands for artificial intelligence."],
    )
    score = metric.evaluate(tc)
    assert score.score == 1.0  # Empty output has no hallucinations
