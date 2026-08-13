#!/usr/bin/env python3
"""Fail closed when the selected lesson learns imputation before its split."""

from __future__ import annotations

import ast
import json
from pathlib import Path

SOURCE = Path(
    "cases/case_1_regression/lessons/05_train_test_split_and_first_model/solution.py"
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )


text = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(text, filename=str(SOURCE))
findings: list[dict[str, object]] = []

for loader in ("load_clean_shipments", "load_shipments"):
    node = _function(tree, loader)
    if node and any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "median"
        for call in ast.walk(node)
    ):
        findings.append(
            {
                "rule_id": "CHRONOS-FOLD-LOCAL-IMPUTE-001",
                "path": str(SOURCE),
                "line": node.lineno,
                "reason": "the repository learns an imputation median before the holdout split",
            }
        )

imputer = _function(tree, "impute_driver_experience")
if imputer is None or 'train_df["driver_experience_years"].median()' not in text:
    findings.append(
        {
            "rule_id": "CHRONOS-FOLD-LOCAL-IMPUTE-001",
            "path": str(SOURCE),
            "line": 1,
            "reason": "no supported train-partition-only imputation function was established",
        }
    )

payload = {
    "schema": "chronos.ci-evaluation.v1",
    "policy_id": "chronos-ext011-train-only-imputation",
    "policy_state": "installed",
    "status": "FAIL" if findings else "PASS",
    "blocking_findings": findings,
    "coverage": {
        "mode": "COMPLETE",
        "eligible_supported_files": 1,
        "analyzed_supported_files": 1,
        "missing_supported_files": [],
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
for finding in findings:
    print(
        f"::error file={finding['path']},line={finding['line']}::"
        f"{finding['rule_id']}: {finding['reason']}"
    )
raise SystemExit(2 if findings else 0)
