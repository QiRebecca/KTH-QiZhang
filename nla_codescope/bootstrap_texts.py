from __future__ import annotations

import re
from typing import Any

from .utils import short_token_count


def syntax_hint(context: str, token_text: str, token_role: str) -> str:
    line = " ".join(context.strip().split())
    line = re.sub(r"[A-Za-z_][A-Za-z0-9_]{24,}", "<long_identifier>", line)
    if len(line) > 90:
        line = line[:87] + "..."
    if token_role == "return_raise_yield_branch":
        return "control flow or returned value near the target token"
    if token_role == "literal_string_or_number":
        return "a literal value participates in local computation"
    if token_role == "function_name_or_identifier":
        return "an identifier is being used or defined"
    return line or f"near token role {token_role}"


def make_bootstrap_text(row: dict[str, Any], max_tokens: int = 60) -> str:
    summary = (row.get("summary_text") or row.get("summary") or "an undocumented Python function").strip()
    role = row["token_role"]
    hint = syntax_hint(row.get("local_context_for_bootstrap_only", ""), row.get("token_text", ""), role)
    text = (
        f"This activation comes from a Python function about: {summary}. "
        f"The local code role is: {role}. "
        f"Nearby syntax suggests: {hint}. "
        f"Likely information: function intent plus the local {role} role."
    )
    words = text.split()
    if len(words) > max_tokens:
        text = " ".join(words[:max_tokens])
    assert short_token_count(text) <= max_tokens
    return text
