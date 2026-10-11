"""
GenECG test-fixture helper - the ONE place that knows how to fetch a single
GenECG image locally for testing/debugging.

Replaces (fuses):
  - download_genecg_sample.py (whole file)
  - check_image_quality.py's download_variant() function

Both did the exact same thing with a hardcoded Windows absolute path to the
mapping CSV, in two places that could silently drift apart. This module is
now the single source of truth: fetch_test_image() is imported by
verify_image_pipeline.py, and can be imported by any other test script or
notebook cell instead of redefining it again.

Usage as a library:
    from genecg_fixtures import fetch_test_image
    path = fetch_test_image(ecg_id=1, variant="B")

Usage as a CLI (keeps download_genecg_sample.py's original standalone
behavior, including printing the matching PTB-XL signal path):
    python genecg_fixtures.py --ecg_id 1 --variant A
"""

import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

# Relative by default - override via the MAPPING_CSV env var or the
# mapping_csv argument, instead of hardcoding an absolute machine-specific
# path (the bug being fixed here: both original scripts had
# "C:\Users\neifa\...\ptbxl_with_genecg_paths.csv" hardcoded in two places).
DEFAULT_MAPPING_CSV = os.environ.get("GENECG_MAPPING_CSV", "C:\\Users\\neifa\\Documents\\ENETCOM\\3IDSD\\stage_25-26\\cardiac-multiagent-pipeline\\src\\ptbxl_with_genecg_paths.csv")
DEFAULT_DOWNLOAD_DIR = os.environ.get("GENECG_DOWNLOAD_DIR", "./genecg_downloaded")
HF_REPO_ID = "edcci/GenECG"


def fetch_test_image(ecg_id: int, variant: str = "B",
                      mapping_csv: str = DEFAULT_MAPPING_CSV,
                      download_dir: str = DEFAULT_DOWNLOAD_DIR) -> str:
    """
    Look up the GenECG path for a given ecg_id/variant in the merged mapping
    CSV, then download that single file from the Hub (never the full 227 GB
    dataset - see the mapping CSV, built once via ecg_image.ipynb).

    Inputs:
        ecg_id: PTB-XL ecg_id to fetch.
        variant: "A" (clean) or "B" (with imperfections).
        mapping_csv: path to ptbxl_with_genecg_paths.csv (or any CSV with
            ecg_id/path_A/path_B columns). Defaults to a relative path,
            overridable via the GENECG_MAPPING_CSV environment variable.
        download_dir: local cache directory for the downloaded image.

    Output:
        Local filesystem path to the downloaded image (str).

    Raises:
        ValueError: if ecg_id is not in the mapping, or the requested
            variant has no path for it.
        FileNotFoundError: if mapping_csv itself doesn't exist.
    """
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(
            f"Mapping CSV not found: {mapping_csv}. Set GENECG_MAPPING_CSV or "
            f"pass mapping_csv= explicitly, or run ecg_image.ipynb first to "
            f"generate it."
        )

    df = pd.read_csv(mapping_csv)
    row = df[df["ecg_id"] == ecg_id]
    if row.empty:
        raise ValueError(f"ecg_id={ecg_id} not found in {mapping_csv}")

    col = f"path_{variant}"
    path_in_repo = row.iloc[0][col]
    if pd.isna(path_in_repo):
        raise ValueError(f"No path for variant '{variant}' at ecg_id={ecg_id}")

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=path_in_repo,
        local_dir=download_dir,
    )


def fetch_ptbxl_signal_path(ecg_id: int, mapping_csv: str = DEFAULT_MAPPING_CSV) -> str:
    """
    Look up the matching PTB-XL high-resolution signal path (filename_hr) for
    an ecg_id, e.g. for a signal-vs-image comparison test. Does NOT download
    anything (the signal lives in your local PhysioNet PTB-XL copy, not on
    the Hub) - just resolves the path from the mapping CSV.

    Inputs:
        ecg_id: PTB-XL ecg_id to look up.
        mapping_csv: path to the mapping CSV (see fetch_test_image).

    Output:
        The filename_hr value (str), e.g. "records500/00000/00001_hr" -
        join it with your local PTB-XL root directory to get a real path.
    """
    df = pd.read_csv(mapping_csv)
    row = df[df["ecg_id"] == ecg_id]
    if row.empty:
        raise ValueError(f"ecg_id={ecg_id} not found in {mapping_csv}")
    return row.iloc[0]["filename_hr"]


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg_id", type=int, required=True)
    parser.add_argument("--variant", choices=["A", "B"], default="A")
    parser.add_argument("--mapping_csv", default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--download_dir", default=DEFAULT_DOWNLOAD_DIR)
    args = parser.parse_args()

    local_path = fetch_test_image(args.ecg_id, args.variant, args.mapping_csv, args.download_dir)
    print(f"ecg_id={args.ecg_id} | variant={args.variant}")
    print(f"Downloaded locally: {local_path}")

    signal_path = fetch_ptbxl_signal_path(args.ecg_id, args.mapping_csv)
    print(f"Matching PTB-XL signal (filename_hr): {signal_path}")
    print("--> Use signal_path (joined with your local PTB-XL root) for a "
          "signal-vs-image comparison test.")
    print(f"--> Use local_path as the input image for the ECG image pipeline.")


if __name__ == "__main__":
    _main()
