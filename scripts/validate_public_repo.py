from __future__ import annotations

"""Validate the public GitHub clone and raw.githubusercontent.com byte view."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml


RAW_FILES = [
    "README.md",
    "pyproject.toml",
    "environment.yml",
    "nla_codescope/injection.py",
    "scripts/run_smoke_test.sh",
    "scripts/verify_submission_consistency.py",
    "scripts/check_submission_ready.py",
    "configs/main_qwen25coder_1p5b_L18.yaml",
    "artifacts/nla_meta_main.yaml",
    "artifacts/qualitative_examples_curated.jsonl",
]


def _load_toml(text: str) -> object:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(text)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(cmd: list[str], cwd: Path, errors: list[str]) -> None:
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        errors.append(f"{' '.join(cmd)} failed in {cwd}:\n{result.stdout.strip()}")


def _repo_slug(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f"cannot parse GitHub owner/repo from {repo_url}")
    return "/".join(parts[-2:])


def _raw_url(repo_url: str, branch: str, rel: str) -> str:
    return f"https://raw.githubusercontent.com/{_repo_slug(repo_url)}/{branch}/{rel}"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "nla-codescope-public-validator"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read()
    except urllib.error.URLError:
        if shutil.which("curl") is None:
            raise
        result = subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise urllib.error.URLError(result.stderr.decode("utf-8", errors="replace").strip())
        return result.stdout


def _validate_text_kind(rel: str, text: str, tmp_dir: Path, errors: list[str], origin: str) -> None:
    if rel.endswith(".py"):
        p = tmp_dir / rel.replace("/", "__")
        p.write_text(text, encoding="utf-8")
        result = subprocess.run([sys.executable, "-m", "py_compile", str(p)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            errors.append(f"{origin} {rel} does not compile as Python:\n{result.stdout.strip()}")
        if len(text.splitlines()) < 3 and ("def " in text or "class " in text or "import " in text):
            errors.append(f"{origin} {rel} looks line-collapsed")
    elif rel.endswith(".sh"):
        p = tmp_dir / rel.replace("/", "__")
        p.write_text(text, encoding="utf-8")
        result = subprocess.run(["bash", "-n", str(p)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            errors.append(f"{origin} {rel} does not parse as shell:\n{result.stdout.strip()}")
        lines = text.splitlines()
        if not lines or not lines[0].startswith("#!"):
            errors.append(f"{origin} {rel} is missing a shebang on line 1")
        if len(lines) < 5:
            errors.append(f"{origin} {rel} has too few lines; possible line collapse")
    elif rel.endswith(".toml"):
        try:
            _load_toml(text)
        except Exception as exc:
            errors.append(f"{origin} {rel} does not parse as TOML: {exc}")
    elif rel.endswith((".yaml", ".yml")):
        try:
            yaml.safe_load(text)
        except Exception as exc:
            errors.append(f"{origin} {rel} does not parse as YAML: {exc}")
    elif rel.endswith(".jsonl"):
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                errors.append(f"{origin} {rel}:{lineno} does not parse as JSONL: {exc}")
                break
            if not isinstance(obj, dict):
                errors.append(f"{origin} {rel}:{lineno} is not a JSON object")
                break
    elif rel.endswith(".md"):
        lines = text.splitlines()
        if rel == "README.md" and len(lines) < 50:
            errors.append(f"{origin} README.md has only {len(lines)} lines; possible line collapse")
        if rel == "README.md" and (not lines or not lines[0].startswith("# ")):
            errors.append(f"{origin} README.md does not start with a Markdown H1")


def _clone_checks(clone: Path, errors: list[str]) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
    _run([sys.executable, "-m", "compileall", "-q", "nla_codescope", "scripts", "tests"], clone, errors)
    _run(["bash", "-n", "scripts/run_smoke_test.sh", "scripts/run_main_pipeline.sh", "scripts/run_main_a100.sh", "scripts/run_main_autodl.sh", "scripts/run_main_h800.sh"], clone, errors)
    _run([sys.executable, "scripts/validate_repository_format.py"], clone, errors)
    _run([sys.executable, "scripts/verify_submission_consistency.py"], clone, errors)
    _run([sys.executable, "scripts/check_submission_ready.py", "--skip-pytest"], clone, errors)
    return commit


def _clone_public_repo(repo_url: str, branch: str, clone: Path) -> tuple[bool, str]:
    last_output = ""
    for attempt in range(1, 4):
        if clone.exists():
            shutil.rmtree(clone)
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(clone)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
            )
            last_output = result.stdout
            if result.returncode == 0:
                return True, last_output
        except subprocess.TimeoutExpired as exc:
            last_output = (exc.stdout or "") + f"\nclone attempt {attempt} timed out after 180 seconds"
        if attempt < 3:
            time.sleep(5 * attempt)
    return False, last_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/QiRebecca/KTH-QiZhang")
    ap.add_argument("--branch", default="main")
    args = ap.parse_args()

    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="nla-public-validate-") as tmp:
        tmp_path = Path(tmp)
        clone = tmp_path / "repo"
        cloned, clone_output = _clone_public_repo(args.repo_url, args.branch, clone)
        if not cloned:
            print(clone_output, file=sys.stderr)
            raise SystemExit("public repo validation failed: could not clone repository")

        commit = _clone_checks(clone, errors)
        raw_tmp = tmp_path / "raw"
        raw_tmp.mkdir()
        raw_not_verified: list[str] = []
        for rel in RAW_FILES:
            local = clone / rel
            if not local.exists():
                errors.append(f"fresh clone is missing {rel}")
                continue
            local_bytes = local.read_bytes()
            url = _raw_url(args.repo_url, args.branch, rel)
            try:
                raw_bytes = _download(url)
            except (urllib.error.URLError, TimeoutError) as exc:
                raw_not_verified.append(f"{rel}: {url}: {exc}")
                continue

            local_hash = _sha256(local_bytes)
            raw_hash = _sha256(raw_bytes)
            local_lines = local_bytes.decode("utf-8").splitlines()
            raw_text = raw_bytes.decode("utf-8")
            raw_lines = raw_text.splitlines()
            if raw_hash != local_hash:
                errors.append(
                    f"raw and clone disagree for {rel}\n"
                    f"  url: {url}\n"
                    f"  clone_sha256: {local_hash}\n"
                    f"  raw_sha256:   {raw_hash}\n"
                    f"  clone_lines: {len(local_lines)} raw_lines: {len(raw_lines)}"
                )
            if len(raw_lines) != len(local_lines):
                errors.append(f"raw line count differs for {rel}: clone={len(local_lines)} raw={len(raw_lines)} url={url}")
            _validate_text_kind(rel, raw_text, raw_tmp, errors, "raw")

        if raw_not_verified:
            print("public raw validation NOT VERIFIED for:", file=sys.stderr)
            for item in raw_not_verified:
                print(f"- {item}", file=sys.stderr)
            errors.append("public raw validation incomplete because at least one raw URL could not be downloaded")

        if errors:
            print("public repo validation failed:", file=sys.stderr)
            for err in errors:
                print(f"- {err}", file=sys.stderr)
            raise SystemExit(1)
        print(f"public repo validation passed: {args.repo_url} {args.branch} {commit}")


if __name__ == "__main__":
    main()
