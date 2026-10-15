"""Unit tests for EvalGuard CLI interface."""

import json
from click.testing import CliRunner
import pytest
from evalguard.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_benchmark(tmp_path):
    runner = CliRunner()
    html_report = str(tmp_path / "bench.html")
    json_report = str(tmp_path / "bench.json")

    result = runner.invoke(main, ["benchmark", "--html", html_report, "--json", json_report])
    assert result.exit_code == 0
    assert "Running EvalGuard AI Benchmark" in result.output
    assert (tmp_path / "bench.html").exists()
    assert (tmp_path / "bench.json").exists()


def test_cli_custom_dataset(tmp_path):
    runner = CliRunner()
    dataset_file = tmp_path / "custom.json"
    dataset_file.write_text(json.dumps([
        {
            "id": "cli_case_1",
            "query": "What is Python?",
            "actual_output": "Python is a high-level programming language.",
            "retrieved_contexts": ["Python is an interpreted, high-level programming language."],
        }
    ]))

    result = runner.invoke(main, ["test", str(dataset_file)])
    assert result.exit_code == 0
    assert "Loaded 1 test cases" in result.output
