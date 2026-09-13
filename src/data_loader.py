"""Data loading module for Electrical Grid Stability dataset."""

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config.config import DATASET_CSV_FILENAME, DATASET_RAW_PATH, DATASET_URLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(destination_path: Optional[Path] = None) -> Path:
    """Download the Electrical Grid Stability dataset if not present locally.

    Parameters
    ----------
    destination_path : Path, optional
        Target file path to save the CSV file. Defaults to DATASET_RAW_PATH.

    Returns
    -------
    Path
        Path to the downloaded CSV file.
    """
    dest = Path(destination_path) if destination_path else DATASET_RAW_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Dataset already exists at {dest}")
        return dest

    logger.info(f"Downloading dataset to {dest}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    last_error = None
    for url in DATASET_URLS:
        try:
            logger.info(f"Attempting download from {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()

            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for member in zf.namelist():
                        if member.endswith(".csv"):
                            with zf.open(member) as f_in, open(dest, "wb") as f_out:
                                f_out.write(f_in.read())
                            logger.info(f"Extracted {member} to {dest}")
                            return dest
            else:
                with open(dest, "wb") as f_out:
                    f_out.write(content)
                logger.info(f"Saved dataset directly to {dest}")
                return dest

        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            last_error = e

    raise FileNotFoundError(
        f"Unable to download dataset from known URLs. "
        f"Please manually download '{DATASET_CSV_FILENAME}' from the UCI Machine Learning Repository "
        f"(https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data) "
        f"and place it at: {dest.resolve()}.\nLast error: {last_error}"
    )


def load_raw_data(data_path: Optional[Union[str, Path]] = None, force_download: bool = False) -> pd.DataFrame:
    """Load the raw dataset into a pandas DataFrame.

    Parameters
    ----------
    data_path : Union[str, Path], optional
        Path to the CSV file. If None, uses default path and downloads if missing.
    force_download : bool, default False
        Whether to re-download the dataset even if present.

    Returns
    -------
    pd.DataFrame
        Raw dataset DataFrame.
    """
    target_path = Path(data_path) if data_path else DATASET_RAW_PATH

    if force_download or not target_path.exists():
        download_dataset(target_path)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {target_path}. "
            f"Please ensure '{DATASET_CSV_FILENAME}' is placed in 'data/raw/'."
        )

    logger.info(f"Reading dataset from {target_path}")
    df = pd.read_csv(target_path)
    return df
