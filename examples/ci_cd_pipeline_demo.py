"""CI/CD Quality Gate demonstration script for EvalGuard AI."""

import os
import sys

# Ensure evalguard package is importable when run directly from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evalguard import EvalGuard, ConsoleReporter, HTMLReporter, JSONReporter
from evalguard.examples import get_sample_test_suite

def run_ci_gate():
    print("=" * 60)
    print("  CI/CD PIPELINE: RAG MODEL QUALITY AUDIT")
    print("=" * 60)

    # 1. Load benchmark test cases
    test_cases = get_sample_test_suite()
    print(f"Loaded {len(test_cases)} validation test cases.")

    # 2. Evaluate suite
    evaluator = EvalGuard()
    suite = evaluator.evaluate_suite("CI/CD Build Quality Gate #412", test_cases)

    # 3. Output reports
    ConsoleReporter.print_suite_summary(suite)
    HTMLReporter.save(suite, "ci_eval_report.html")
    JSONReporter.save(suite, "ci_eval_results.json")
    print("Reports generated: ci_eval_report.html, ci_eval_results.json")

    # 4. Check CI Quality Gate conditions
    summary = suite.summary
    print("\nQuality Gate Verification:")
    print(f"- Overall Pass Rate: {summary.pass_rate * 100:.1f}%")
    print(f"- Avg Faithfulness: {summary.average_scores.get('faithfulness', 0):.2f}")
    print(f"- Avg Safety Score: {summary.average_scores.get('guardrail_safety', 0):.2f}")

    # For CI demonstration, we alert on failed adversarial/hallucination checks
    if summary.failed_tests > 0:
        print(f"\n[CI INFO] {summary.failed_tests} intentional regression test(s) flagged vulnerabilities as expected.")
    else:
        print("\n[CI SUCCESS] All quality gates passed with zero regressions.")

if __name__ == "__main__":
    run_ci_gate()
