from __future__ import annotations

"""Validate local repository formatting and parser-level reproducibility."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_toml(text: str) -> object:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(text)


def _word_count(path: Path) -> int:
    return len(re.findall(r"\b\S+\b", path.read_text(encoding="utf-8")))


def _run(cmd: list[str], errors: list[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        errors.append(f"{' '.join(cmd)} failed:\n{result.stdout.strip()}")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _check_python(errors: list[str]) -> None:
    _run([sys.executable, "-m", "compileall", "-q", "nla_codescope", "scripts", "tests"], errors)
    for path in [*ROOT.glob("nla_codescope/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        structural_tokens = sum(text.count(tok) for tok in ("def ", "class ", "import "))
        if len(lines) < 3 and structural_tokens:
            errors.append(f"{_rel(path)} looks line-collapsed: {len(lines)} lines with Python structure")
        if text.startswith("from __future__") and "\n" not in text.partition("from __future__")[2]:
            errors.append(f"{_rel(path)} appears to place all Python content on one line")


def _check_shell(errors: list[str]) -> None:
    scripts = sorted(ROOT.glob("scripts/*.sh"))
    if scripts:
        _run(["bash", "-n", *[_rel(p) for p in scripts]], errors)
    for path in scripts:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            errors.append(f"{_rel(path)} is empty")
            continue
        if not lines[0].startswith("#!"):
            errors.append(f"{_rel(path)} is missing a shebang on the first line")
        if len(lines) < 2:
            errors.append(f"{_rel(path)} has no body after the shebang")
        if lines[0].count(";") or " set " in lines[0] or "python " in lines[0]:
            errors.append(f"{_rel(path)} may have shebang and script body collapsed onto one line")
        if path.name == "run_smoke_test.sh":
            body = "\n".join(lines[:8])
            if "artifacts_smoke" not in path.read_text(encoding="utf-8"):
                errors.append("scripts/run_smoke_test.sh does not clearly isolate smoke artifacts")
            if "set -e" not in body:
                errors.append("scripts/run_smoke_test.sh is missing fail-fast shell mode")


def _check_structured_data(errors: list[str]) -> None:
    for path in [ROOT / "pyproject.toml"]:
        try:
            _load_toml(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{_rel(path)} does not parse as TOML: {exc}")

    yaml_paths = [ROOT / "environment.yml", *ROOT.glob("configs/*.yaml"), *ROOT.glob("configs/*.yml"), *ROOT.glob("artifacts/*.yaml"), *ROOT.glob("artifacts/*.yml")]
    for path in yaml_paths:
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{_rel(path)} does not parse as YAML: {exc}")

    for path in [*ROOT.glob("artifacts/*.json"), *ROOT.glob("data/*.json")]:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{_rel(path)} does not parse as JSON: {exc}")

    for path in [*ROOT.glob("artifacts/*.jsonl"), *ROOT.glob("data/*.jsonl")]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        errors.append(f"{_rel(path)}:{lineno} is JSONL but does not contain a JSON object")
        except Exception as exc:
            errors.append(f"{_rel(path)} does not parse as JSONL: {exc}")


def _check_markdown(errors: list[str]) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        errors.append("README.md is missing")
    else:
        text = readme.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) < 50:
            errors.append(f"README.md has only {len(lines)} lines; possible line collapse")
        if not lines or not lines[0].startswith("# "):
            errors.append("README.md does not start with a top-level Markdown heading")
        if "## " not in text:
            errors.append("README.md has no second-level headings")
        if _word_count(readme) > 3000:
            errors.append(f"README.md has {_word_count(readme)} words, above 3000")
        table_lines = [i for i, line in enumerate(lines) if line.startswith("|")]
        if table_lines and not any(re.match(r"^\|[ :\-\|]+\|$", lines[i]) for i in table_lines):
            errors.append("README.md has table-looking rows but no Markdown separator row")
        if not table_lines:
            errors.append("README.md contains no Markdown tables")

    weird_markers = ("", "", "【", "】", "turn0", "ChatGPT prompt", "Codex prompt")
    for path in [readme, *ROOT.glob("docs/*.md")]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if path.name != "README.md" and len(lines) < 2 and len(text) > 200:
            errors.append(f"{_rel(path)} appears collapsed into one Markdown line")
        for lineno, line in enumerate(lines, 1):
            if len(line) > 5000:
                errors.append(f"{_rel(path)}:{lineno} has a line longer than 5000 characters")
        for marker in weird_markers:
            if marker in text:
                errors.append(f"{_rel(path)} contains suspicious generated citation/prompt marker: {marker}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    errors: list[str] = []

    _check_python(errors)
    _check_shell(errors)
    _check_structured_data(errors)
    _check_markdown(errors)

    if errors:
        print("repository format validation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        raise SystemExit(1)
    print("repository format validation passed")


if __name__ == "__main__":
    main()
