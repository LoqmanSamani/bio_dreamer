from __future__ import annotations

import argparse
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
import yaml

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config as BotocoreConfig
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# S3 keys for ProteinGym subsets: subset name → (S3 key, local filename)
PROTEINGYM_S3_KEYS: dict[str, tuple[str, str]] = {
    "DMS_substitutions":      ("DMS_substitutions.parquet",       "DMS_substitutions.parquet"),
    "DMS_indels":             ("DMS_indels.parquet",               "DMS_indels.parquet"),
    "clinical_substitutions": ("clinical_substitutions.parquet",   "clinical_substitutions.parquet"),
    "clinical_indels":        ("clinical_indels.parquet",          "clinical_indels.parquet"),
}

# Reference/metadata CSVs from the official ProteinGym GitHub release.
# These contain per-assay metadata including selection_type (Stability/Binding/Activity)
# used by AssayType classification in dataset.py.
PROTEINGYM_REF_URLS: dict[str, str] = {
    "ProteinGym_reference_file_substitutions.csv": (
        "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
        "reference_files/DMS_substitutions/ProteinGym_reference_file_substitutions.csv"
    ),
    "ProteinGym_reference_file_indels.csv": (
        "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
        "reference_files/DMS_indels/ProteinGym_reference_file_indels.csv"
    ),
}

# Zenodo filenames for the Tsuboyama 2023 mega-scale stability dataset
TSUBOYAMA_FILES: dict[str, str] = {
    "processed_datasets": "Processed_K50_dG_datasets.zip",
    "dataset1_csv":       "Tsuboyama2023_Dataset1_20230416.csv",
    "dataset2_3_csv":     "Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
    "single_dms_list":    "Single_DMS_list.csv",
    "double_dms_list":    "Double_DMS_list.csv",
    "triple_dms_list":    "Triple_DMS_list.csv",
}


def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_root(cfg: dict) -> Path:
    root = Path(cfg["data_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _progress_wrap(iterable, *, total: int, desc: str, show: bool):
    if show and HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, unit="B",
                    unit_scale=True, unit_divisor=1024)
    return iterable


def _http_download(
    url: str,
    dest: Path,
    *,
    show_progress: bool = True,
    chunk_size: int = 8192,
    max_retries: int = 3,
    dry_run: bool = False,
) -> None:
    if dry_run:
        logger.info("[DRY-RUN] Would download %s → %s", url, dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                chunks = resp.iter_content(chunk_size=chunk_size)
                wrapped = _progress_wrap(
                    chunks, total=total, desc=dest.name, show=show_progress
                )
                with open(dest, "wb") as fh:
                    for chunk in wrapped:
                        if chunk:
                            fh.write(chunk)
            logger.info("Saved %s", dest)
            return
        except (requests.RequestException, IOError) as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s", attempt, max_retries, url, exc
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts"
                ) from exc


def _s3_download(
    bucket: str,
    key: str,
    dest: Path,
    *,
    region: str = "us-east-2",
    show_progress: bool = True,
    dry_run: bool = False,
) -> None:
    if dry_run:
        logger.info("[DRY-RUN] Would download s3://%s/%s → %s", bucket, key, dest)
        return

    if not HAS_BOTO3:
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        logger.warning("boto3 not installed — falling back to HTTPS: %s", url)
        _http_download(url, dest, show_progress=show_progress)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        region_name=region,
        config=BotocoreConfig(signature_version=UNSIGNED),
    )
    head = s3.head_object(Bucket=bucket, Key=key)
    total_bytes = head["ContentLength"]
    logger.info("Downloading s3://%s/%s  (%.1f MB)", bucket, key, total_bytes / 1e6)

    if show_progress and HAS_TQDM:
        bar = tqdm(total=total_bytes, desc=dest.name, unit="B",
                   unit_scale=True, unit_divisor=1024)
        callback = lambda n: bar.update(n)  # noqa: E731
    else:
        bar = None
        callback = None

    try:
        s3.download_file(bucket, key, str(dest), Callback=callback)
    finally:
        if bar:
            bar.close()

    logger.info("Saved %s", dest)


def _unzip(archive: Path, dest_dir: Path) -> None:
    logger.info("Unzipping %s → %s", archive.name, dest_dir)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extracted %s", archive.name)


def _download_proteingym_hf(
    subset_name: str, dest: Path, dry_run: bool = False
) -> None:
    if dry_run:
        logger.info("[DRY-RUN] Would download HF split %s → %s", subset_name, dest)
        return
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Hugging Face 'datasets' library not installed.\n"
            "Run: pip install datasets"
        )
    logger.info("Streaming ProteinGym/%s from Hugging Face...", subset_name)
    ds = load_dataset("OATML-Markslab/ProteinGym_v1", name=subset_name, split="train")
    import pandas as pd
    df = ds.to_pandas()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info("Saved %s  (%d rows)", dest, len(df))


def download_proteingym(cfg: dict, root: Path, dry_run: bool = False) -> None:
    """Download selected ProteinGym subsets from S3 or Hugging Face."""
    pg_cfg = cfg["proteingym"]
    if not pg_cfg.get("enabled", True):
        logger.info("ProteinGym download disabled — skipping.")
        return

    out_dir = root / pg_cfg.get("output_subdir", "proteingym")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_cfg  = cfg.get("download", {})
    skip    = dl_cfg.get("skip_existing", True)
    prog    = dl_cfg.get("show_progress", True)
    retries = dl_cfg.get("max_retries", 3)
    chunk   = dl_cfg.get("chunk_size", 8192)
    source  = pg_cfg.get("source", "s3")
    bucket  = pg_cfg.get("s3_bucket", "proteingym")
    region  = pg_cfg.get("s3_region", "us-east-2")

    for subset_name, enabled in pg_cfg.get("subsets", {}).items():
        if not enabled:
            continue
        if subset_name not in PROTEINGYM_S3_KEYS:
            logger.warning("Unknown subset '%s' — skipping.", subset_name)
            continue
        s3_key, filename = PROTEINGYM_S3_KEYS[subset_name]
        dest = out_dir / filename
        if skip and dest.exists():
            logger.info("  %s already exists — skipping.", dest.name)
            continue
        logger.info("Downloading ProteinGym / %s", subset_name)
        if source == "s3":
            _s3_download(bucket, s3_key, dest,
                         region=region, show_progress=prog, dry_run=dry_run)
        elif source == "huggingface":
            _download_proteingym_hf(subset_name, dest, dry_run=dry_run)
        else:
            raise ValueError(f"Unknown source '{source}'. Use 's3' or 'huggingface'.")

    if pg_cfg.get("reference_files", True):
        ref_dir = out_dir / "reference_files"
        ref_dir.mkdir(exist_ok=True)
        for fname, url in PROTEINGYM_REF_URLS.items():
            dest = ref_dir / fname
            if skip and dest.exists():
                logger.info("  %s already exists — skipping.", fname)
                continue
            logger.info("Downloading reference file: %s", fname)
            _http_download(url, dest, show_progress=prog,
                           chunk_size=chunk, max_retries=retries, dry_run=dry_run)


def download_tsuboyama(cfg: dict, root: Path, dry_run: bool = False) -> None:
    """Download selected files from the Tsuboyama 2023 Zenodo record."""
    ts_cfg = cfg["tsuboyama"]
    if not ts_cfg.get("enabled", True):
        logger.info("Tsuboyama download disabled — skipping.")
        return

    out_dir = root / ts_cfg.get("output_subdir", "tsuboyama")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_cfg  = cfg.get("download", {})
    skip    = dl_cfg.get("skip_existing", True)
    prog    = dl_cfg.get("show_progress", True)
    retries = dl_cfg.get("max_retries", 3)
    chunk   = dl_cfg.get("chunk_size", 8192)
    base_url = ts_cfg.get("zenodo_base_url", "https://zenodo.org/records/7992926/files")
    do_unzip = ts_cfg.get("unzip", True)

    for key, enabled in ts_cfg.get("files", {}).items():
        if not enabled:
            continue
        if key not in TSUBOYAMA_FILES:
            logger.warning("Unknown Tsuboyama file key '%s' — skipping.", key)
            continue
        filename = TSUBOYAMA_FILES[key]
        dest = out_dir / filename
        url = f"{base_url}/{filename}?download=1"
        if skip and dest.exists():
            logger.info("  %s already exists — skipping.", filename)
        else:
            logger.info("Downloading Tsuboyama / %s", filename)
            _http_download(url, dest, show_progress=prog,
                           chunk_size=chunk, max_retries=retries, dry_run=dry_run)

        if do_unzip and filename.endswith(".zip") and (dry_run or dest.exists()):
            unzip_dir = out_dir / filename.replace(".zip", "")
            if skip and unzip_dir.exists():
                logger.info("  %s already extracted — skipping.", unzip_dir.name)
            elif not dry_run:
                _unzip(dest, unzip_dir)
            else:
                logger.info("[DRY-RUN] Would unzip %s → %s", dest.name, unzip_dir)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download raw data for ProteinDreamer."
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/protein_dreamer/download_config.yaml",
        help="Path to download_config.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only", choices=["proteingym", "tsuboyama"], default=None
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)

    cfg  = _load_config(cfg_path)
    root = _resolve_root(cfg)
    logger.info("Data root: %s", root.resolve())

    if args.only in (None, "proteingym"):
        download_proteingym(cfg, root, dry_run=args.dry_run)
    if args.only in (None, "tsuboyama"):
        download_tsuboyama(cfg, root, dry_run=args.dry_run)

    logger.info("Download complete. Data saved under: %s", root.resolve())


if __name__ == "__main__":
    main()
