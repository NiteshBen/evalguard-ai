"""Latency and token cost efficiency metric.

Tracks execution performance against defined SLA thresholds and calculates
estimated inference expenditures.
"""

from typing import Dict, Optional
from evalguard.metrics.base import BaseMetric
from evalguard.models import MetricScore, TestCase


# Pricing table per 1,000,000 tokens (USD)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "blended": 0.25},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "blended": 3.125},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "blended": 0.375},
    "gpt-4o": {"input": 2.50, "output": 10.00, "blended": 6.25},
}


class LatencyCostMetric(BaseMetric):
    """Measures latency against SLA target (ms) and monitors token budget efficiency."""

    def __init__(
        self,
        max_latency_ms: float = 3000.0,
        max_tokens: int = 4000,
        target_model: str = "gemini-2.0-flash",
        threshold: float = 0.70,
    ):
        super().__init__(
            name="latency_cost",
            threshold=threshold,
            description=f"Efficiency evaluation: SLA latency <= {max_latency_ms}ms, tokens <= {max_tokens}",
        )
        self.max_latency_ms = max_latency_ms
        self.max_tokens = max_tokens
        self.target_model = target_model

    def evaluate(self, test_case: TestCase) -> MetricScore:
        latency = test_case.latency_ms
        tokens = test_case.tokens_used

        if latency is None and tokens is None:
            return MetricScore.evaluate(
                name=self.name,
                score=1.0,
                threshold=self.threshold,
                reason="Latency and token usage not recorded for test case.",
                metadata={"latency_ms": None, "tokens": None, "estimated_cost_usd": 0.0},
            )

        # 1. Latency sub-score
        if latency is not None:
            if latency <= self.max_latency_ms:
                # Linear score: 1.0 if fast, down to 0.7 at max_latency
                latency_score = 1.0 - (0.3 * (latency / self.max_latency_ms))
            else:
                # Over SLA: rapid decay
                overshoot = (latency - self.max_latency_ms) / self.max_latency_ms
                latency_score = max(0.0, 0.70 - (0.5 * overshoot))
        else:
            latency_score = 1.0

        # 2. Token budget sub-score
        if tokens is not None:
            if tokens <= self.max_tokens:
                token_score = 1.0 - (0.2 * (tokens / self.max_tokens))
            else:
                overshoot = (tokens - self.max_tokens) / self.max_tokens
                token_score = max(0.0, 0.80 - (0.5 * overshoot))
        else:
            token_score = 1.0

        # Estimate cost
        pricing = MODEL_PRICING.get(self.target_model, MODEL_PRICING["gemini-2.0-flash"])
        token_count = tokens or 0
        est_cost = (token_count / 1_000_000.0) * pricing["blended"]

        combined_score = round((0.6 * latency_score) + (0.4 * token_score), 4)
        reason = (
            f"Latency: {latency or 'N/A'}ms (SLA: {self.max_latency_ms}ms), "
            f"Tokens: {tokens or 'N/A'} (Est. cost: ${est_cost:.6f} via {self.target_model})"
        )

        return MetricScore.evaluate(
            name=self.name,
            score=combined_score,
            threshold=self.threshold,
            reason=reason,
            metadata={
                "latency_ms": latency,
                "max_latency_ms": self.max_latency_ms,
                "tokens": tokens,
                "max_tokens": self.max_tokens,
                "model_profile": self.target_model,
                "estimated_cost_usd": round(est_cost, 7),
            },
        )
