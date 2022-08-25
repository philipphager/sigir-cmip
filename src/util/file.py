import hashlib
import logging
import shutil
from pathlib import Path

import pandas as pd
import wget
from sklearn.datasets import load_svmlight_file


def sha256_checksum(path: Path, chunk_size: int = 4 * 1024 * 1024):
    """
    https://github.com/rjagerman/pytorchltr/blob/master/pytorchltr/utils/file.py
    """
    hash_sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def download(url: str, out_path: Path) -> Path:
    if not out_path.exists():
        logging.info(f"Download archived dataset to: {out_path}")
        wget.download(url, str(out_path))

    assert out_path.exists()
    return out_path


def verify_file(path: Path, checksum: str) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if checksum != sha256_checksum(path):
        raise ValueError(f"Checksum verification failed. Wrong or damaged file: {path}")

    return True


def extract(in_path: Path, out_path: Path) -> Path:
    if not out_path.exists():
        logging.info(f"Unpack archived dataset to: {out_path}")
        shutil.unpack_archive(in_path, out_path)

    assert out_path.exists()
    return out_path


def read_svmlight_file(path: Path, load_features: bool) -> pd.DataFrame:
    assert path.exists(), path
    logging.info(f"Parsing dataset with SVMLight format: {path}")
    X, y, queries = load_svmlight_file(str(path), query_id=True)

    if load_features:
        df = pd.DataFrame(X.todense())
        df.columns = df.columns.map(str)
    else:
        df = pd.DataFrame()
        logging.info("Omitting doc features from dataset")

    df["y"] = y
    df["query_id"] = queries
    return df
