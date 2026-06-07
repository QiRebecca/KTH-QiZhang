from __future__ import annotations

from common import parse_config


def main() -> None:
    cfg = parse_config()
    print(
        "Optional released-checkpoint sanity check is configured for "
        f"{cfg['model']['av_checkpoint']} and {cfg['model']['ar_checkpoint']}. "
        "Run this only after GPU/model cache is available; main results are not based on these checkpoints."
    )


if __name__ == "__main__":
    main()
