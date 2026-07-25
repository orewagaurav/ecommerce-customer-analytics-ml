"""Fetch and prepare the source dataset.

The processed CSV is ~80 MB and is no longer tracked in git — committing it made
every clone and CI checkout carry it forever, for a file that is fully derivable.
This script reproduces it.

    python scripts/get_data.py                 # download raw, then preprocess
    python scripts/get_data.py --skip-download # preprocess an existing raw CSV

Inference does not need any of this: the API and dashboard read the committed
feature store (~1 MB of parquet). Data is only required to retrain or to rebuild
the feature store from scratch.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RAW_CSV = PROJECT_ROOT / "data" / "online_retail_II.csv"
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed_online_retail_II.csv"
KAGGLE_DATASET = "mashlyn/online-retail-ii-uci"


def download_raw(destination: Path) -> bool:
    """Download the raw dataset via the Kaggle CLI, if it is configured."""
    if destination.exists():
        print(f"Raw dataset already present: {destination}")
        return True

    if shutil.which("kaggle") is None:
        print(
            "Kaggle CLI not found.\n"
            "  pip install kaggle, then place your API token at ~/.kaggle/kaggle.json\n"
            f"  or download manually from https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"  and save it as {destination}"
        )
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} ...")
    # check=False: a failed download is reported to the user below, not raised.
    completed = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(destination.parent), "--unzip"],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        print(f"Kaggle download failed:\n{completed.stderr.strip()}")
        return False

    if not destination.exists():
        # The archive sometimes unpacks under a different filename.
        candidates = sorted(destination.parent.glob("*.csv"))
        if candidates:
            candidates[0].rename(destination)

    return destination.exists()


def preprocess(raw_csv: Path, processed_csv: Path) -> None:
    from src.data_preprocessing import run_preprocessing

    print(f"Preprocessing {raw_csv} ...")
    _, summary = run_preprocessing(raw_csv, processed_csv)
    print(summary.to_string(index=False))
    print(f"\nWrote {processed_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and prepare the source dataset")
    parser.add_argument("--raw_csv", type=Path, default=RAW_CSV)
    parser.add_argument("--processed_csv", type=Path, default=PROCESSED_CSV)
    parser.add_argument("--skip-download", action="store_true", dest="skip_download")
    args = parser.parse_args()

    if not args.skip_download and not download_raw(args.raw_csv):
        sys.exit(1)

    if not args.raw_csv.exists():
        print(f"Raw dataset not found at {args.raw_csv}")
        sys.exit(1)

    preprocess(args.raw_csv, args.processed_csv)
    print("\nNext:")
    print("  python project/src/train_models.py")
    print("  python project/src/build_feature_store.py")


if __name__ == "__main__":
    main()
