"""EvalGuard AI - Production-grade LLM & RAG Evaluation, Guardrails & Quality Gates.

Developed by Nitesh Beniwal.
"""

from evalguard.evaluators.deterministic import DeterministicEvaluator
from evalguard.evaluators.llm_judge import LLMJudgeEvaluator
from evalguard.metrics.answer_relevance import AnswerRelevanceMetric
from evalguard.metrics.base import BaseMetric
from evalguard.metrics.context_precision import ContextPrecisionMetric
from evalguard.metrics.context_recall import ContextRecallMetric
from evalguard.metrics.faithfulness import FaithfulnessMetric
from evalguard.metrics.guardrails import GuardrailSafetyMetric
from evalguard.metrics.latency_cost import LatencyCostMetric
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
from evalguard.reporting.console import ConsoleReporter
from evalguard.reporting.html_reporter import HTMLReporter
from evalguard.reporting.json_reporter import JSONReporter

__version__ = "0.1.0"
__author__ = "Nitesh Beniwal"

# Convenient top-level alias
EvalGuard = DeterministicEvaluator

__all__ = [
    "EvalGuard",
    "DeterministicEvaluator",
    "LLMJudgeEvaluator",
    "TestCase",
    "EvalResult",
    "SuiteResult",
    "SuiteSummary",
    "MetricScore",
    "MetricType",
    "ClaimStatus",
    "ClaimVerification",
    "BaseMetric",
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "GuardrailSafetyMetric",
    "LatencyCostMetric",
    "ConsoleReporter",
    "HTMLReporter",
    "JSONReporter",
]
