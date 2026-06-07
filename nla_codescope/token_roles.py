from __future__ import annotations

import io
import keyword
import re
import tokenize
from dataclasses import dataclass

BRANCH_WORDS = {"return", "raise", "yield", "if", "elif", "else", "for", "while", "try", "except", "with"}
OPERATORS = set("+-*/%@=&|^~<>!:.,;()[]{}")
NUM_RE = re.compile(r"^([0-9]+(\.[0-9]+)?|0x[0-9a-fA-F]+)$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class TokenInfo:
    text: str
    start: int
    end: int
    role: str


def classify_token_text(text: str) -> str:
    stripped = text.strip()
    if text in {"\n", "\r\n"} or stripped == "":
        return "indentation_or_newline"
    if stripped.startswith("#") or (len(stripped) >= 2 and stripped[:1] in {"'", '"'}):
        return "comment_or_docstring"
    if stripped in BRANCH_WORDS:
        return "return_raise_yield_branch"
    if keyword.iskeyword(stripped):
        return "keyword"
    if NUM_RE.match(stripped) or (stripped[:1] in {"'", '"'} and stripped[-1:] == stripped[:1]):
        return "literal_string_or_number"
    if all(ch in OPERATORS for ch in stripped):
        return "operator_or_punctuation"
    if IDENT_RE.match(stripped):
        return "function_name_or_identifier"
    return "other"


def python_tokens(code: str) -> list[TokenInfo]:
    out: list[TokenInfo] = []
    try:
        stream = io.StringIO(code).readline
        for tok in tokenize.generate_tokens(stream):
            text = tok.string
            if not text or tok.type in {tokenize.ENCODING, tokenize.ENDMARKER}:
                continue
            role = classify_token_text(text)
            if tok.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL}:
                role = "indentation_or_newline"
            if tok.type == tokenize.COMMENT:
                role = "comment_or_docstring"
            if tok.type == tokenize.STRING:
                role = "comment_or_docstring" if len(out) < 8 else "literal_string_or_number"
            if tok.type == tokenize.NUMBER:
                role = "literal_string_or_number"
            out.append(TokenInfo(text=text, start=tok.start[1], end=tok.end[1], role=role))
    except tokenize.TokenError:
        for part in re.findall(r"\w+|[^\w\s]|\n", code):
            out.append(TokenInfo(part, 0, 0, classify_token_text(part)))
    return out


def choose_token_positions(tokens: list[TokenInfo], n: int, seed: int) -> list[int]:
    candidates = [i for i, t in enumerate(tokens) if t.text.strip()]
    if not candidates:
        return []
    rng = __import__("random").Random(seed)
    picks: list[int] = []

    def add_first(role: str) -> None:
        for i in candidates:
            if tokens[i].role == role and i not in picks:
                picks.append(i)
                return

    picks.append(rng.choice(candidates))
    add_first("function_name_or_identifier")
    add_first("return_raise_yield_branch")
    for role in ("operator_or_punctuation", "literal_string_or_number", "indentation_or_newline"):
        if len(picks) >= n:
            break
        add_first(role)
    while len(picks) < n:
        i = rng.choice(candidates)
        if i not in picks or len(candidates) < n:
            picks.append(i)
    return picks[:n]
