"""Unit tests for Console, HTML, and JSON reporters."""

import json
import os
import pytest
from evalguard.evaluators.deterministic import DeterministicEvaluator
from evalguard.examples.sample_rag_data import get_sample_test_suite
from evalguard.reporting.console import ConsoleReporter
from evalguard.reporting.html_reporter import HTMLReporter
from evalguard.reporting.json_reporter import JSONReporter


@pytest.fixture
def sample_suite():
    evaluator = DeterministicEvaluator()
    cases = get_sample_test_suite()[:2]
    return evaluator.evaluate_suite("Reporter Test Suite", cases)


def test_console_reporter(sample_suite, capsys):
    ConsoleReporter.print_suite_summary(sample_suite)
    captured = capsys.readouterr()
    assert "EVALGUARD AI" in captured.out
    assert "Reporter Test Suite" in captured.out


def test_html_reporter_render_and_save(sample_suite, tmp_path):
    html_str = HTMLReporter.render(sample_suite)
    assert "<!DOCTYPE html>" in html_str
    assert "EvalGuard AI" in html_str
    assert "Multi-Metric Radar Breakdown" in html_str
    assert "<svg" in html_str

    out_file = tmp_path / "test_report.html"
    HTMLReporter.save(sample_suite, str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_json_reporter(sample_suite, tmp_path):
    data = JSONReporter.to_dict(sample_suite)
    assert "suite_name" in data
    assert "summary" in data
    assert "results" in data

    json_file = tmp_path / "test_results.json"
    JSONReporter.save(sample_suite, str(json_file))
    assert json_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["suite_name"] == "Reporter Test Suite"
