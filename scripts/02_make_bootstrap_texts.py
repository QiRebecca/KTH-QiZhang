from __future__ import annotations

from common import parse_config
from nla_codescope.bootstrap_texts import make_bootstrap_text
from nla_codescope.utils import ensure_dirs, out_path, read_jsonl, write_jsonl


def main() -> None:
    cfg = parse_config()
    ensure_dirs(cfg)
    rows = read_jsonl(out_path(cfg, "activations.jsonl"))
    max_tokens = int(cfg["bootstrap_text"]["max_tokens"])
    out = []
    for row in rows:
        out.append({"activation_id": row["activation_id"], "split": row["split"], "bootstrap_text": make_bootstrap_text(row, max_tokens)})
    write_jsonl(out_path(cfg, "bootstrap_texts.jsonl"), out)
    print(f"wrote {len(out)} SFT bootstrap texts")


if __name__ == "__main__":
    main()
