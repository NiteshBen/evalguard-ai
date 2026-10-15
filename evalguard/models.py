"""Pydantic data models and schemas for EvalGuard AI."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, computed_field


class MetricType(str, Enum):
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    GUARDRAIL_SAFETY = "guardrail_safety"
    LATENCY_COST = "latency_cost"
    CUSTOM = "custom"


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class ClaimVerification(BaseModel):
    """Detailed verification record for a specific factual claim extracted from LLM output."""
    claim: str = Field(..., description="Extracted factual proposition from model response")
    status: ClaimStatus = Field(..., description="Verification result against retrieved context")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    evidence: Optional[str] = Field(default=None, description="Relevant context sentence backing claim")

    model_config = {"frozen": False}


class MetricScore(BaseModel):
    """Evaluation score for a single metric with threshold validation and diagnostics."""
    name: str = Field(..., description="Metric identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0.0 and 1.0")
    threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Minimum acceptable threshold")
    passed: bool = Field(..., description="Whether score meets or exceeds threshold")
    reason: Optional[str] = Field(default=None, description="Diagnostic explanation of the score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metric-specific debugging artifacts")

    @classmethod
    def evaluate(
        cls,
        name: str,
        score: float,
        threshold: float = 0.70,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MetricScore":
        """Factory constructor that automatically computes boolean pass/fail status."""
        score_clamped = max(0.0, min(1.0, float(score)))
        return cls(
            name=name,
            score=round(score_clamped, 4),
            threshold=round(threshold, 4),
            passed=score_clamped >= threshold,
            reason=reason,
            metadata=metadata or {},
        )


class TestCase(BaseModel):
    """A single evaluation sample representing a RAG or LLM interaction."""
    __test__ = False  # Prevent pytest from treating TestCase as a test class
    id: str = Field(..., description="Unique identifier for the test case")
    query: str = Field(..., description="User prompt or question submitted to the system")
    actual_output: str = Field(..., description="Generated text response from the LLM")
    retrieved_contexts: List[str] = Field(
        default_factory=list,
        description="Chunks or context documents retrieved by the RAG retriever",
    )
    expected_output: Optional[str] = Field(
        default=None,
        description="Ground truth response for reference-based benchmarking",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Execution latency in milliseconds",
    )
    tokens_used: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total tokens consumed by the generation step",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary tags (e.g. model name, environment, pipeline version)",
    )


class EvalResult(BaseModel):
    """Aggregate evaluation result for a single test case across all configured metrics."""
    test_case: TestCase = Field(..., description="The evaluated test case")
    metrics: Dict[str, MetricScore] = Field(
        default_factory=dict,
        description="Dictionary mapping metric name to its score",
    )
    claims: List[ClaimVerification] = Field(
        default_factory=list,
        description="Detailed claim-level breakdown for hallucination inspection",
    )
    passed: bool = Field(
        ...,
        description="True if every evaluated metric passed its required threshold",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Arithmetic mean of all evaluated metric scores",
    )
    execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to run the evaluation algorithms",
    )


class SuiteSummary(BaseModel):
    """High-level summary of an entire test suite run."""
    total_tests: int = Field(..., ge=0)
    passed_tests: int = Field(..., ge=0)
    failed_tests: int = Field(..., ge=0)
    pass_rate: float = Field(..., ge=0.0, le=1.0)
    average_scores: Dict[str, float] = Field(default_factory=dict)
    average_latency_ms: Optional[float] = None
    total_tokens: Optional[int] = None


class SuiteResult(BaseModel):
    """Complete collection of evaluated test cases and overall test suite health summary."""
    suite_name: str = Field(default="Default Suite")
    results: List[EvalResult] = Field(default_factory=list)
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")

    @computed_field  # type: ignore[misc]
    @property
    def summary(self) -> SuiteSummary:
        total = len(self.results)
        if total == 0:
            return SuiteSummary(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                pass_rate=1.0,
                average_scores={},
            )

        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        # Calculate average score per metric
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        for r in self.results:
            for name, m in r.metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + m.score
                metric_counts[name] = metric_counts.get(name, 0) + 1

        avg_scores = {
            name: round(metric_sums[name] / metric_counts[name], 4)
            for name in metric_sums
        }

        # Average latency if available
        latencies = [
            r.test_case.latency_ms
            for r in self.results
            if r.test_case.latency_ms is not None
        ]
        avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else None

        # Total tokens
        tokens = [
            r.test_case.tokens_used
            for r in self.results
            if r.test_case.tokens_used is not None
        ]
        tot_tok = sum(tokens) if tokens else None

        return SuiteSummary(
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            pass_rate=round(passed / total, 4),
            average_scores=avg_scores,
            average_latency_ms=avg_lat,
            total_tokens=tot_tok,
        )

    @property
    def passed(self) -> bool:
        """Returns True only if all individual test results passed."""
        return all(r.passed for r in self.results) if self.results else True
