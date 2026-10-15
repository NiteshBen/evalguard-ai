"""Unit tests for ContextPrecisionMetric and ContextRecallMetric."""

import pytest
from evalguard.metrics.context_precision import ContextPrecisionMetric
from evalguard.metrics.context_recall import ContextRecallMetric
from evalguard.models import TestCase


def test_context_precision_relevant_at_top():
    metric = ContextPrecisionMetric(threshold=0.60)
    tc = TestCase(
        id="prec_high",
        query="What encryption standard is used?",
        actual_output="AES-256 is used for storage encryption.",
        retrieved_contexts=[
            "Data Security: All persistent storage is encrypted using industry standard AES-256.",
            "Office hours are 9am to 5pm Monday through Friday.",
            "Cafeteria menu options include vegetarian dishes.",
        ],
    )
    score = metric.evaluate(tc)
    assert score.passed is True
    assert score.metadata["first_relevant_rank"] == 1


def test_context_precision_irrelevant_chunks():
    metric = ContextPrecisionMetric(threshold=0.60)
    tc = TestCase(
        id="prec_low",
        query="What encryption standard is used?",
        actual_output="AES-256 is used.",
        retrieved_contexts=[
            "Office hours are 9am to 5pm Monday through Friday.",
            "Cafeteria menu options include vegetarian dishes.",
        ],
    )
    score = metric.evaluate(tc)
    assert score.passed is False


def test_context_recall_full_coverage():
    metric = ContextRecallMetric(threshold=0.70)
    tc = TestCase(
        id="recall_high",
        query="What are the two key requirements for admission?",
        actual_output="Applicants must have a bachelor's degree and achieve minimum 80% marks.",
        expected_output="Applicants must have a bachelor's degree and achieve minimum 80% marks.",
        retrieved_contexts=[
            "Eligibility Criteria: All applicants must hold an accredited bachelor's degree.",
            "Academic Cutoff: Students must achieve a minimum of 80% marks in their prior degree.",
        ],
    )
    score = metric.evaluate(tc)
    assert score.passed is True
    assert score.score >= 0.70


def test_context_recall_missing_critical_context():
    metric = ContextRecallMetric(threshold=0.70)
    tc = TestCase(
        id="recall_low",
        query="What are the admission requirements and deadlines?",
        expected_output="Admission requires a bachelor degree and submission before December 1st.",
        retrieved_contexts=[
            "Our campus has modern sports facilities and green parks.",
        ],
        actual_output="The deadline is December 1st.",
    )
    score = metric.evaluate(tc)
    assert score.passed is False
    assert score.score < 0.50
