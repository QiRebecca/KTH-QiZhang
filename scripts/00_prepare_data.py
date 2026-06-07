from __future__ import annotations

from common import parse_config
from nla_codescope.data import prepare_dataset
from nla_codescope.utils import ensure_dirs, set_seed


def main() -> None:
    cfg = parse_config()
    ensure_dirs(cfg)
    set_seed(int(cfg.get("seed", 17)))
    rows = prepare_dataset(cfg)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"prepared {len(rows)} functions: {counts}")


if __name__ == "__main__":
    main()
