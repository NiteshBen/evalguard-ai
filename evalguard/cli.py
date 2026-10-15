"""Command-line interface (CLI) for EvalGuard AI."""

import json
import os
import sys
import click
from evalguard.evaluators.deterministic import DeterministicEvaluator
from evalguard.examples.sample_rag_data import get_sample_test_suite
from evalguard.models import TestCase
from evalguard.reporting.console import ConsoleReporter
from evalguard.reporting.html_reporter import HTMLReporter
from evalguard.reporting.json_reporter import JSONReporter


@click.group()
@click.version_option(version="0.1.0", prog_name="evalguard")
def main() -> None:
    """EvalGuard AI - Production-grade LLM & RAG Evaluation, Guardrails & Quality Gates."""
    pass


@main.command(name="benchmark")
@click.option("--html", "html_out", default="evalguard_report.html", help="Path to output HTML report.")
@click.option("--json", "json_out", default=None, help="Path to output JSON results.")
@click.option("--strict", is_flag=True, default=False, help="Fail with non-zero exit code if any test fails.")
def benchmark(html_out: str, json_out: str, strict: bool) -> None:
    """Run curated benchmark test suite and generate interactive evaluation reports."""
    click.echo(click.style("\n[*] Running EvalGuard AI Benchmark Test Suite...", fg="cyan", bold=True))
    test_cases = get_sample_test_suite()

    evaluator = DeterministicEvaluator()
    suite_result = evaluator.evaluate_suite("RAG Production Baseline Benchmark", test_cases)

    # Print console summary
    ConsoleReporter.print_suite_summary(suite_result)

    # Save HTML report
    if html_out:
        HTMLReporter.save(suite_result, html_out)
        click.echo(click.style(f"[+] Interactive HTML report generated: {os.path.abspath(html_out)}", fg="green"))

    # Save JSON report if requested
    if json_out:
        JSONReporter.save(suite_result, json_out)
        click.echo(click.style(f"[+] JSON results exported: {os.path.abspath(json_out)}", fg="green"))

    if strict and not suite_result.passed:
        click.echo(click.style("\n[!] Quality gate failed: Some benchmark tests failed thresholds.", fg="red", bold=True))
        sys.exit(1)


@main.command(name="test")
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option("--suite-name", default="Custom Dataset Suite", help="Title for the evaluation suite.")
@click.option("--html", "html_out", default=None, help="Path to output HTML report.")
@click.option("--json", "json_out", default=None, help="Path to output JSON results.")
@click.option("--strict", is_flag=True, default=False, help="Fail with non-zero exit code if any test fails.")
def test_dataset(dataset_file: str, suite_name: str, html_out: str, json_out: str, strict: bool) -> None:
    """Evaluate a custom JSON or JSONL dataset of test cases."""
    click.echo(click.style(f"\n[*] Loading test cases from {dataset_file}...", fg="cyan"))

    test_cases = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        if dataset_file.endswith(".jsonl"):
            for line in f:
                if line.strip():
                    test_cases.append(TestCase(**json.loads(line)))
        else:
            raw_data = json.load(f)
            if isinstance(raw_data, list):
                test_cases = [TestCase(**item) for item in raw_data]
            elif isinstance(raw_data, dict) and "test_cases" in raw_data:
                test_cases = [TestCase(**item) for item in raw_data["test_cases"]]

    click.echo(click.style(f"[+] Loaded {len(test_cases)} test cases. Evaluating...", fg="green"))
    evaluator = DeterministicEvaluator()
    suite_result = evaluator.evaluate_suite(suite_name, test_cases)

    ConsoleReporter.print_suite_summary(suite_result)

    if html_out:
        HTMLReporter.save(suite_result, html_out)
        click.echo(click.style(f"[+] HTML report saved: {os.path.abspath(html_out)}", fg="green"))

    if json_out:
        JSONReporter.save(suite_result, json_out)
        click.echo(click.style(f"[+] JSON results saved: {os.path.abspath(json_out)}", fg="green"))

    if strict and not suite_result.passed:
        click.echo(click.style("\n[!] Quality gate failed: One or more test cases did not pass thresholds.", fg="red", bold=True))
        sys.exit(1)


if __name__ == "__main__":
    main()
