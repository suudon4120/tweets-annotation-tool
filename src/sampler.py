import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # 追加
from .data_handler import load_raw_data, save_batch_data

# 引数に stratify_col を追加 (デフォルトは None)
def create_sample_batch(n_samples, seed, annotator_name, source_file="KYOTO2_batch_tagged.csv", stratify_col=None):
    """
    Rawデータからサンプリングを行い、アノテーション用のカラムを追加して保存する
    stratify_col: 層化抽出の基準となるカラム名。Noneの場合は単純ランダムサンプリング。
    """
    # 元データを読み込む
    df = load_raw_data(source_file)
    
    # データ数より要求サンプル数が多い場合のハンドリング
    n = min(n_samples, len(df))
    
    # サンプリング処理
    if stratify_col:
        # --- 層化抽出モード ---
        try:
            # scikit-learnのtrain_test_splitを使って層化抽出を行う
            # 欠損値があるとエラーになることがあるため、一時的に埋める
            temp_stratify_col = df[stratify_col].astype(str).fillna("Unknown")
            
            # stratify引数を指定して抽出
            sampled_df, _ = train_test_split(
                df, 
                train_size=n, 
                stratify=temp_stratify_col, 
                random_state=seed
            )
            # train_test_splitはシャッフルしてくれるのでそのまま使える
        except ValueError as e:
            # クラスのメンバー数が少なすぎて層化できない場合などのエラー対策
            print(f"Stratification failed (falling back to random): {e}")
            sampled_df = df.sample(n=n, random_state=seed)
    else:
        # --- 単純ランダムサンプリング ---
        sampled_df = df.sample(n=n, random_state=seed)

    # コピーを作成して警告回避
    sampled_df = sampled_df.copy()
    
    # アノテーション用のカラムを追加
    target_columns = [
        'is_location_related', 
        'subjectivity', 
        'sentiment_or_noise', 
        'user_attribute'
    ]
    
    for col in target_columns:
        sampled_df[f'human_{col}'] = None
        
    sampled_df['annotator_name'] = annotator_name
    sampled_df['is_completed'] = False
    sampled_df['comments'] = ""
    
    # ファイル名の生成
    strat_suffix = f"_strat-{stratify_col}" if stratify_col else ""
    output_filename = f"batch_seed{seed}_n{n}{strat_suffix}_{annotator_name}.csv"
    
    # 5. 保存
    save_batch_data(sampled_df, output_filename)
    
    return output_filename, len(sampled_df)