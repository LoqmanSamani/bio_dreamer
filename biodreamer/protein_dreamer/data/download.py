"""
downloads raw data for ProteinDreamer from:
  - ProteinGym  : public AWS S3 bucket  (s3://proteingym)
  - Tsuboyama   : Zenodo record 7992926

usage
-----
    # download everything defined in the config:
    python -m protein_dreamer.data.download

    # override config path:
    python -m protein_dreamer.data.download --config path/to/download_config.yaml

    # dry-run (print what would be downloaded, skip all network calls):
    python -m protein_dreamer.data.download --dry-run

    # download only one source:
    python -m protein_dreamer.data.download --only proteingym
    python -m protein_dreamer.data.download --only tsuboyama
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
import yaml

# boto3 for S3 downloads (falls back to HTTPS if not installed)
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




# basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("protein_dreamer.download") # we may change it to "protein_dreamer.data.download" later if we move this file


# s3 keys for ProteinGym subsets
# maps config subset names → (S3 key, local filename)
PROTEINGYM_S3_KEYS: dict[str, tuple[str, str]] = {
    "DMS_substitutions":      ("DMS_substitutions.parquet",  "DMS_substitutions.parquet"),
    "DMS_indels":             ("DMS_indels.parquet",          "DMS_indels.parquet"),
    "clinical_substitutions": ("clinical_substitutions.parquet", "clinical_substitutions.parquet"),
    "clinical_indels":        ("clinical_indels.parquet",     "clinical_indels.parquet"),
}

# reference metadata files stored separately in the official github release
# (also mirrored on s3 – we fall back to github if s3 path missing)
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

# zenodo file names → download slugs
TSUBOYAMA_FILES: dict[str, str] = {
    "processed_datasets": "Processed_K50_dG_datasets.zip",
    "dataset1_csv":       "Tsuboyama2023_Dataset1_20230416.csv",
    "dataset2_3_csv":     "Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
    "single_dms_list":    "Single_DMS_list.csv",
    "double_dms_list":    "Double_DMS_list.csv",
    "triple_dms_list":    "Triple_DMS_list.csv",
}


# helpers
def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def _resolve_root(cfg: dict) -> Path:
    root = Path(cfg["data_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root

def _progress_wrap(iterable, *, total: int, desc: str, show: bool):
    """wrap an iterable with tqdm if available and requested"""
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
    """download a file over https with resume-friendly streaming and retries"""
    if dry_run:
        log.info("[DRY-RUN] Would download %s → %s", url, dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                chunks = resp.iter_content(chunk_size=chunk_size)
                wrapped = _progress_wrap(
                    chunks, total=total,
                    desc=dest.name, show=show_progress
                )
                with open(dest, "wb") as fh:
                    for chunk in wrapped:
                        if chunk:
                            fh.write(chunk)
            log.info("✓  Saved %s", dest)
            return
        except (requests.RequestException, IOError) as exc:
            log.warning("Attempt %d/%d failed for %s: %s",
                        attempt, max_retries, url, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)   # exponential back-off
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
    """download a single object from a public s3 bucket (no credentials)"""
    if dry_run:
        log.info("[DRY-RUN] Would download s3://%s/%s → %s", bucket, key, dest)
        return

    if not HAS_BOTO3:
        # fall back to https public url if boto3 not available
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        log.warning("boto3 not installed – falling back to HTTPS: %s", url)
        _http_download(url, dest, show_progress=show_progress)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        region_name=region,
        config=BotocoreConfig(signature_version=UNSIGNED),
    )

    # get object size for progress bar
    head = s3.head_object(Bucket=bucket, Key=key)
    total_bytes = head["ContentLength"]

    log.info("Downloading s3://%s/%s  (%.1f MB)", bucket, key,
             total_bytes / 1e6)

    if show_progress and HAS_TQDM:
        bar = tqdm(total=total_bytes, desc=dest.name, unit="B",
                   unit_scale=True, unit_divisor=1024)
        callback = lambda bytes_transferred: bar.update(bytes_transferred)  # noqa: e731
    else:
        bar = None
        callback = None

    try:
        s3.download_file(
            bucket, key, str(dest),
            Callback=callback,
        )
    finally:
        if bar:
            bar.close()

    log.info("✓  Saved %s", dest)

def _unzip(archive: Path, dest_dir: Path) -> None:
    log.info("Unzipping %s → %s", archive.name, dest_dir)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)
    log.info("✓  Extracted %s", archive.name)

def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()




# ProteinGym downloader
def download_proteingym(cfg: dict, root: Path, dry_run: bool = False) -> None:
    """download selected ProteinGym subsets from s3 or hugging-face"""
    pg_cfg = cfg["proteingym"]
    if not pg_cfg.get("enabled", True):
        log.info("ProteinGym download disabled – skipping.")
        return

    out_dir = root / pg_cfg.get("output_subdir", "proteingym")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_cfg = cfg.get("download", {})
    skip    = dl_cfg.get("skip_existing", True)
    prog    = dl_cfg.get("show_progress", True)
    retries = dl_cfg.get("max_retries", 3)
    chunk   = dl_cfg.get("chunk_size", 8192)

    source = pg_cfg.get("source", "s3")
    bucket = pg_cfg.get("s3_bucket", "proteingym")
    region = pg_cfg.get("s3_region", "us-east-2")

    subsets: dict = pg_cfg.get("subsets", {})

    for subset_name, enabled in subsets.items():
        if not enabled:
            continue
        if subset_name not in PROTEINGYM_S3_KEYS:
            log.warning("Unknown subset '%s' – skipping.", subset_name)
            continue

        s3_key, filename = PROTEINGYM_S3_KEYS[subset_name]
        dest = out_dir / filename

        if skip and dest.exists():
            log.info("⏭  %s already exists – skipping.", dest.name)
            continue

        log.info("━━  Downloading ProteinGym / %s", subset_name)

        if source == "s3":
            _s3_download(bucket, s3_key, dest,
                         region=region, show_progress=prog, dry_run=dry_run)
        elif source == "huggingface":
            _download_proteingym_hf(subset_name, dest, dry_run=dry_run)
        else:
            raise ValueError(f"Unknown source '{source}'. Use 's3' or 'huggingface'.")

    # reference/metadata files
    if pg_cfg.get("reference_files", True):
        ref_dir = out_dir / "reference_files"
        ref_dir.mkdir(exist_ok=True)
        for fname, url in PROTEINGYM_REF_URLS.items():
            dest = ref_dir / fname
            if skip and dest.exists():
                log.info("⏭  %s already exists – skipping.", fname)
                continue
            log.info("━━  Downloading reference file: %s", fname)
            _http_download(url, dest, show_progress=prog,
                           chunk_size=chunk, max_retries=retries,
                           dry_run=dry_run)


def _download_proteingym_hf(subset_name: str, dest: Path,
                             dry_run: bool = False) -> None:
    """alternative: stream a ProteinGym split from hugging-face and save as parquet"""
    if dry_run:
        log.info("[DRY-RUN] Would download HF split %s → %s", subset_name, dest)
        return
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Hugging Face 'datasets' library not installed.\n"
            "Run: pip install datasets"
        )

    log.info("Streaming ProteinGym/%s from Hugging Face...", subset_name)
    ds = load_dataset("OATML-Markslab/ProteinGym_v1",
                      name=subset_name, split="train")
    df = ds.to_pandas()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    log.info("✓  Saved %s  (%d rows)", dest, len(df))




# Tsuboyama downloader
def download_tsuboyama(cfg: dict, root: Path, dry_run: bool = False) -> None:
    """download selected files from the tsuboyama 2023 zenodo record"""
    ts_cfg = cfg["tsuboyama"]
    if not ts_cfg.get("enabled", True):
        log.info("Tsuboyama download disabled – skipping.")
        return

    out_dir = root / ts_cfg.get("output_subdir", "tsuboyama")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_cfg  = cfg.get("download", {})
    skip    = dl_cfg.get("skip_existing", True)
    prog    = dl_cfg.get("show_progress", True)
    retries = dl_cfg.get("max_retries", 3)
    chunk   = dl_cfg.get("chunk_size", 8192)

    base_url = ts_cfg.get("zenodo_base_url",
                           "https://zenodo.org/records/7992926/files")
    do_unzip = ts_cfg.get("unzip", True)

    files_cfg: dict = ts_cfg.get("files", {})

    for key, enabled in files_cfg.items():
        if not enabled:
            continue
        if key not in TSUBOYAMA_FILES:
            log.warning("Unknown Tsuboyama file key '%s' – skipping.", key)
            continue

        filename = TSUBOYAMA_FILES[key]
        dest     = out_dir / filename
        url      = f"{base_url}/{filename}?download=1"

        if skip and dest.exists():
            log.info("⏭  %s already exists – skipping.", filename)
        else:
            log.info("━━  Downloading Tsuboyama / %s", filename)
            _http_download(url, dest, show_progress=prog,
                           chunk_size=chunk, max_retries=retries,
                           dry_run=dry_run)

        # unzip if it is an archive and unzip is requested
        if do_unzip and filename.endswith(".zip") and (dry_run or dest.exists()):
            unzip_dir = out_dir / filename.replace(".zip", "")
            if skip and unzip_dir.exists():
                log.info("⏭  %s already extracted – skipping.", unzip_dir.name)
            elif not dry_run:
                _unzip(dest, unzip_dir)
            else:
                log.info("[DRY-RUN] Would unzip %s → %s", dest.name, unzip_dir)





def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download raw data for ProteinDreamer."
    )
    parser.add_argument(
        "--config", "-c",
        default="config/protein_dreamer/download_config.yaml",
        help="Path to download_config.yaml (default: configs/protein_dreamer/download_config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without making any network calls.",
    )
    parser.add_argument(
        "--only",
        choices=["proteingym", "tsuboyama"],
        default=None,
        help="Download only one data source.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)

    cfg  = _load_config(cfg_path)
    root = _resolve_root(cfg)
    log.info("Data root: %s", root.resolve())

    if args.dry_run:
        log.info("DRY-RUN mode – no files will be written.")

    if args.only in (None, "proteingym"):
        download_proteingym(cfg, root, dry_run=args.dry_run)

    if args.only in (None, "tsuboyama"):
        download_tsuboyama(cfg, root, dry_run=args.dry_run)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Download complete. Data saved under: %s", root.resolve())


if __name__ == "__main__":
    main()