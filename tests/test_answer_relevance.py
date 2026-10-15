"""Unit tests for AnswerRelevanceMetric."""

import pytest
from evalguard.metrics.answer_relevance import AnswerRelevanceMetric
from evalguard.models import TestCase


def test_answer_relevance_high():
    metric = AnswerRelevanceMetric(threshold=0.65)
    tc = TestCase(
        id="rel_high",
        query="How do I reset my account password?",
        actual_output="To reset your account password, click on 'Forgot Password' on the login screen, enter your registered email address, and follow the link sent to your inbox.",
        expected_output="Navigate to the login page and select forgot password to receive a reset link.",
    )
    score = metric.evaluate(tc)
    assert score.passed is True
    assert score.score >= 0.65
    assert score.metadata["intent_keyword_overlap"] > 0.5


def test_answer_relevance_evasive_or_irrelevant():
    metric = AnswerRelevanceMetric(threshold=0.60)
    tc = TestCase(
        id="rel_low",
        query="How do I configure OAuth2 authentication with Azure Active Directory?",
        actual_output="Bananas are rich in potassium and grow in tropical climates around the equator.",
    )
    score = metric.evaluate(tc)
    assert score.passed is False
    assert score.score < 0.40


def test_answer_relevance_empty_output():
    metric = AnswerRelevanceMetric()
    tc = TestCase(
        id="rel_empty",
        query="What is the pricing model?",
        actual_output="",
    )
    score = metric.evaluate(tc)
    assert score.score == 0.0
    assert score.passed is False


def test_answer_relevance_short_evasive_answer():
    metric = AnswerRelevanceMetric(threshold=0.60)
    tc = TestCase(
        id="rel_one_word",
        query="Can you explain the detailed technical difference between TCP and UDP networking protocols?",
        actual_output="Yes.",
    )
    score = metric.evaluate(tc)
    assert score.passed is False
