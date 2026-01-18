import pandas as pd
import os
from pathlib import Path

# 定数: データフォルダのパス定義
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
BATCH_DIR = DATA_DIR / "batches"
RESULTS_DIR = DATA_DIR / "results"

def load_raw_data(filename="KYOTO2_batch_tagged.csv"):
    """
    元データ (Raw) を読み込む
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    return pd.read_csv(path)

def load_batch_data(filename):
    """
    作業用バッチファイルを読み込む
    """
    path = BATCH_DIR / filename
    return pd.read_csv(path)

def save_batch_data(df, filename):
    """
    作業中のデータを保存する
    """
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BATCH_DIR / filename
    df.to_csv(path, index=False)
    return path

def get_batch_files():
    """
    batchesフォルダにあるCSVファイルリストを取得する
    """
    if not BATCH_DIR.exists():
        return []
    return [f.name for f in BATCH_DIR.glob("*.csv")]

def save_result_data(df, original_filename):
    """
    完了したデータをresultsフォルダに保存する
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_filename = f"result_{original_filename}"
    path = RESULTS_DIR / result_filename

    df.to_csv(path, index=False)
    return path
