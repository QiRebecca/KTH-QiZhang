from __future__ import annotations

import os
from pathlib import Path

from nla_codescope.utils import read_jsonl


def test_prepared_split_no_leakage() -> None:
    path = Path(os.environ.get("NLA_TEST_DATA_DIR", "data")) / "prepared_functions.jsonl"
    assert path.exists(), "run scripts/00_prepare_data.py before pytest"
    rows = read_jsonl(path)
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    for field in ("function_id", "code_hash"):
        train = {r[field] for r in by_split["train"]}
        val = {r[field] for r in by_split["val"]}
        test = {r[field] for r in by_split["test"]}
        assert not train & val
        assert not train & test
        assert not val & test


def test_activation_rows_do_not_cross_splits() -> None:
    path = Path(os.environ.get("NLA_TEST_ARTIFACT_DIR", "artifacts")) / "activations.jsonl"
    assert path.exists(), "run scripts/01_extract_activations.py before pytest"
    rows = read_jsonl(path)
    seen: dict[str, str] = {}
    for row in rows:
        fid = row["function_id"]
        split = row["split"]
        if fid in seen:
            assert seen[fid] == split
        seen[fid] = split
