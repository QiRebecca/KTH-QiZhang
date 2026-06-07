from __future__ import annotations

import subprocess
from pathlib import Path

from common import parse_config


FILES = (
    "train-00000-of-00003-a17d5a115f8766d8.parquet",
    "train-00001-of-00003-7aa3a623015a4fef.parquet",
    "train-00002-of-00003-667774ef34accc79.parquet",
)


def main() -> None:
    cfg = parse_config()
    data_cfg = cfg["data"]
    repo = data_cfg.get("mirror_dataset_repo", "kejian/codesearchnet-python-raw-457k")
    mirror = data_cfg.get("mirror_endpoint", "https://hf-mirror.com").rstrip("/")
    out_dir = Path(data_cfg["local_parquet_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        dest = out_dir / filename
        if dest.exists() and dest.stat().st_size > 100_000_000:
            print(f"exists {dest} ({dest.stat().st_size} bytes)")
            continue
        url = f"{mirror}/datasets/{repo}/resolve/main/data/{filename}"
        print(f"downloading {url}")
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "10",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--speed-time",
                "120",
                "--speed-limit",
                "1024",
                "-C",
                "-",
                "-o",
                str(dest),
                url,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
