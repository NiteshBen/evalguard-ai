"""JSON and machine-readable artifact exporter for EvalGuard AI."""

import json
from typing import Any, Dict
from evalguard.models import SuiteResult


class JSONReporter:
    """Exports test suite evaluation results to standard JSON for CI/CD pipelines."""

    @staticmethod
    def to_dict(suite: SuiteResult) -> Dict[str, Any]:
        """Converts SuiteResult to serializable dictionary including computed summary."""
        data = suite.model_dump()
        data["summary"] = suite.summary.model_dump()
        return data

    @classmethod
    def save(cls, suite: SuiteResult, output_path: str, indent: int = 2) -> None:
        """Serializes and writes SuiteResult to JSON file."""
        data = cls.to_dict(suite)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
