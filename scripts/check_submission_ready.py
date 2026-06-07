from __future__ import annotations

"""Fail-fast repository and artifact checks before submission."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nla_codescope.utils import load_config, out_path, repo_root


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _word_count(path: Path) -> int:
    return len(re.findall(r"\b\S+\b", path.read_text(encoding="utf-8")))


def _scan_text_files(root: Path) -> list[str]:
    targets = [
        root / "README.md",
        root / "docs",
        root / "configs",
        root / "scripts",
        root / "nla_codescope",
        root / "tests",
        root / "pyproject.toml",
        root / "environment.yml",
    ]
    files: list[Path] = []
    for target in targets:
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(
                p
                for p in target.rglob("*")
                if p.is_file() and "tests/fixtures/smoke" not in str(p.relative_to(root))
            )
    return [str(p) for p in files]


def _contains_forbidden_abs_path(text: str) -> bool:
    for part in ("root", "Users", "opt", "home"):
        if f"/{part}" in text:
            return True
    return False


def _forbidden_phrases() -> list[str]:
    return [
        "ground-" + "truth explanations",
        "true " + "thoughts",
        "model " + "thoughts",
        "reads the " + "model",
        "faithful " + "explanation",
        "faithful " + "explanations",
        "promising " + "potential",
        "demonstrates " + "the power",
        "paves " + "the way",
        "significant " + "advancement",
        "causal " + "explanation",
    ]


def _repo_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(root).parts
    ]


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _gitattributes_uses_lfs(root: Path) -> bool:
    path = root / ".gitattributes"
    if not path.exists():
        return False
    try:
        return "filter=lfs" in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_qwen25coder_1p5b_L18.yaml")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    cfg = load_config(root / args.config)
    errors: list[str] = []

    readme = root / "README.md"
    if not readme.exists():
        errors.append("README.md is missing")
    else:
        words = _word_count(readme)
        if words > 3000:
            errors.append(f"README.md has {words} words, above 3000")

    required_artifacts = [
        "metrics_main.json",
        "metrics_by_token_role.json",
        "qualitative_examples.jsonl",
        "qualitative_examples_curated.jsonl",
        "nla_meta_main.yaml",
        "metric_audit.json",
        "role_fve_raw_denominator_breakdown.json",
        "per_sample_metrics_main.npz",
        "per_sample_metrics_injection_perturbations.npz",
        "metrics_injection_perturbations.json",
        "paired_delta_ci_all_baselines.json",
        "qualitative_selection_policy.json",
        "manifest.json",
    ]
    for name in required_artifacts:
        if not out_path(cfg, name).exists():
            errors.append(f"missing artifact: {name}")

    required_figures = [
        "architecture.png",
        "main_fve_bar.png",
        "reconstruction_distribution.png",
        "token_role_breakdown.png",
    ]
    for name in required_figures:
        if not (root / "figures" / name).exists():
            errors.append(f"missing figure: {name}")

    audit_path = out_path(cfg, "metric_audit.json")
    if audit_path.exists():
        audit = _read_json(audit_path)
        if audit.get("status") != "complete":
            errors.append("metric_audit.json status is not complete")
        checks = audit.get("checks", {})
        failed = [k for k, v in checks.items() if v is False]
        if failed:
            errors.append(f"metric audit checks failed: {failed}")

    pending_path = out_path(cfg, "gpu_pending.json")
    if pending_path.exists():
        pending = _read_json(pending_path).get("pending_gpu_tasks", [])
        if pending:
            errors.append(f"gpu_pending.json still lists pending tasks: {pending}")

    manifest_path = out_path(cfg, "manifest.json")
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        artifacts = manifest.get("artifacts", {})
        missing_hash = [k for k, v in artifacts.items() if not v.get("sha256")]
        if missing_hash:
            errors.append(f"manifest artifacts missing sha256: {missing_hash}")

    files = _repo_files(root)
    too_large = [p for p in files if p.stat().st_size > 100 * 1024 * 1024]
    if too_large:
        formatted = ", ".join(str(p.relative_to(root)) for p in too_large[:10])
        errors.append(f"files exceed 100 MB and must be omitted or tracked externally/LFS: {formatted}")
    lfs_pointers = [p for p in files if _is_lfs_pointer(p)]
    if lfs_pointers:
        formatted = ", ".join(str(p.relative_to(root)) for p in lfs_pointers[:10])
        errors.append(f"unpulled Git LFS pointer files found in working tree: {formatted}")
    if _gitattributes_uses_lfs(root):
        if shutil.which("git-lfs") is None and shutil.which("git") is not None:
            probe = subprocess.run(["git", "lfs", "version"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if probe.returncode != 0:
                errors.append(".gitattributes uses Git LFS but git-lfs is not installed")

    for file_name in _scan_text_files(root):
        p = Path(file_name)
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = str(p.relative_to(root))
        if _contains_forbidden_abs_path(text):
            errors.append(f"forbidden absolute path in {rel}")
        if ("Pu6" + "L7") in text or ("ssh" + " -p") in text:
            errors.append(f"secret or SSH command in {rel}")
        if rel == "README.md":
            lower = text.lower()
            normalized_lower = " ".join(lower.split())
            for phrase in _forbidden_phrases():
                if phrase in lower:
                    if phrase == ("model " + "thoughts") and ("not as ground-truth model " + "thoughts") in normalized_lower:
                        continue
                    errors.append(f"overclaim phrase in README.md: {phrase}")

    if not args.skip_pytest:
        format_check = subprocess.run([sys.executable, "scripts/validate_repository_format.py"], cwd=root)
        if format_check.returncode != 0:
            errors.append("repository format validation failed")
        consistency = subprocess.run([sys.executable, "scripts/verify_submission_consistency.py"], cwd=root)
        if consistency.returncode != 0:
            errors.append("submission consistency check failed")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root)
        if result.returncode != 0:
            errors.append("pytest failed")

    if errors:
        print("submission readiness check failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        sys.exit(1)
    print("submission readiness check passed")


if __name__ == "__main__":
    main()
