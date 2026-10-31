"""Download the L&T dataset from Kaggle into data/raw/. Needs ~/.kaggle/kaggle.json."""
import sys
from pathlib import Path

DATASET = "mamtadhaker/lt-vehicle-loan-default-prediction"
RAW_DIR = Path("data/raw")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True)
    except Exception as e:  # stop rather than look for another copy (kickoff prompt)
        print(f"Kaggle download failed: {e}", file=sys.stderr)
        return 1
    for f in sorted(RAW_DIR.iterdir()):
        print(f"{f.name}\t{f.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
