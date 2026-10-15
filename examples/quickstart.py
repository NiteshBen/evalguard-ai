import os
import sys

# Ensure evalguard package is importable when run directly from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evalguard import EvalGuard, TestCase, ConsoleReporter

# 1. Define a RAG interaction test case
test_case = TestCase(
    id="quickstart_demo_1",
    query="What is the data retention policy for enterprise accounts?",
    actual_output="For enterprise accounts, the customer data retention policy specifies that logs and telemetry are retained for 90 days by default, with custom retention up to 7 years available upon request.",
    retrieved_contexts=[
        "Compliance & Retention Guide: Enterprise tier logs and activity telemetry are preserved for a baseline 90 days.",
        "Extended retention periods up to a maximum of 7 years can be configured via customer support contracts.",
    ],
    expected_output="Enterprise telemetry is retained for 90 days by default, expandable up to 7 years.",
    latency_ms=280.0,
    tokens_used=95,
)

# 2. Initialize the evaluation engine with production battery
evaluator = EvalGuard()

# 3. Run evaluation
print("Evaluating test case...")
result = evaluator.evaluate(test_case)

# 4. Print rich terminal diagnostics
ConsoleReporter.print_result(result)
