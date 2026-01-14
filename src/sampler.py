import pandas as pd
import numpy as np
from .data_handler import load_raw_data, save_batch_data

def create_sample_batch(n_samples, seed, annotator_name, source_file="KYOTO2_batch_merged_tagged.csv"):
    """
    Rawデータからサンプリングを行い，アノテーション用のカラムを追加して保存する
    """
    # 元データを読み込む
    df = load_raw_data(source_file)

    # ランダムサンプリング
    n = min(n_samples, len(df))
    sampled_df = df.sample(n=n, random_state=seed).copy()

    # アノテーション用のカラムを追加
    target_columns = [
        'is_location_related',
        'subjectivity',
        'sentiment_or_noise',
        'user_attribute'
    ]

    for col in target_columns:
        sampled_df[f'human_{col}'] = None # 人手の正解入力欄

    sampled_df['annotator_name'] = annotator_name
    sampled_df['is_completed'] = False    # 完了フラグ
    sampled_df['comments'] = ""           # 備考欄

    # ファイル名の生成
    output_filename = f"batch_seed{seed}_n{n}_{annotator_name}.csv"

    # 保存
    save_path = save_batch_data(sampled_df, output_filename)

    return output_filename, len(sampled_df)