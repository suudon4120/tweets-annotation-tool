import streamlit as st
import pandas as pd
from src.data_handler import get_batch_files, load_batch_data, save_batch_data, save_result_data
from src.sampler import create_sample_batch

# 定数定義
OPTIONS = {
    "is_location_related": [True, False],
    "subjectivity": ["N/A", "主観", "客観"],
    "sentiment_or_noise": ["N/A", "ポジティブ", "ネガティブ", "ノイズ"],
    "user_attribute": ["N/A", "観光客", "住民", "それ以外"]
}

# ページ設定
st.set_page_config(
    page_title="Tweets Annotation Tool",
    page_icon="🏷️",
    layout="wide"
)

# サイドバー & モード選択
st.sidebar.title("🏷️ Annotation Tool")
app_mode = st.sidebar.radio(
    "モードを選択してください",
    ["Annotation", "Sampling"]
)

# Sampling Mode (データセット作成)
if app_mode == "Sampling":
    st.title("📂 新規アノテーションセットの作成")
    st.markdown("""
    元データからツイートを抽出し、作業用ファイルを作成します。
    """)

    col1, col2 = st.columns(2)
    with col1:
        annotator_name = st.text_input("作業者名 (半角英数推奨)", value="user1")
        seed = st.number_input("乱数シード (再現性のため)", value=42, step=1)
    with col2:
        n_samples = st.number_input("抽出件数", value=100, step=10)
    
    st.markdown("---")
    st.subheader("抽出オプション")
    
    # サンプリング手法の選択
    sampling_method = st.radio(
        "サンプリング手法", 
        ["単純ランダム (Simple Random)", "層化抽出 (Stratified)"],
        help="層化抽出は、指定したカラムの比率（分布）を保ったままサンプリングします。"
    )
    
    # 層化抽出の場合のみ、カラム選択を表示
    stratify_col = None
    if sampling_method == "層化抽出 (Stratified)":
        # 選択肢として適切なカラムのみ提示（IDやTextは除外）
        strat_options = [
            'user_attribute', 
            'sentiment_or_noise', 
            'subjectivity', 
            'is_location_related'
        ]
        stratify_col = st.selectbox("どのカラムの比率を維持しますか？", strat_options)

    if st.button("データセットを作成", type="primary"):
        if not annotator_name:
            st.error("作業者名を入力してください。")
        else:
            try:
                # ここで引数を渡す
                filename, count = create_sample_batch(
                    n_samples, 
                    seed, 
                    annotator_name, 
                    stratify_col=stratify_col  # 追加
                )
                st.success(f"✅ 作成完了! ファイル名: {filename} ({count}件)")
                
                if stratify_col:
                    st.info(f"ℹ️ '{stratify_col}' の分布に基づいて層化抽出を行いました。")
                    
                st.info("「Annotation」モードに切り替えて作業を開始してください。")
            except FileNotFoundError:
                st.error("❌ 元データが見つかりません。")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# Annotation Mode (タグ付け作業)
elif app_mode == "Annotation":
    st.title("✏️ アノテーション作業")

    # ファイル選択
    batch_files = get_batch_files()
    if not batch_files:
        st.warning("⚠️ 作業用ファイルが見つかりません．「Sampling」モードでデータを作製してください．")
        st.stop()

    selected_file = st.selectbox("作業ファイルを選択", batch_files)

    # セッション状態の初期化
    if "current_file" not in st.session_state or st.session_state.current_file != selected_file:
        st.session_state.current_file = selected_file
        st.session_state.df = load_batch_data(selected_file)
        # 未完了の最初のインデックスを探す
        df= st.session_state.df
        uncompleted = df[df['is_completed'] != True].index
        if len(uncompleted) > 0:
            st.session_state.current_index = uncompleted[0]
        else:
            st.session_state.current_index = 0

    df = st.session_state.df
    idx = st.session_state.current_index

    # 進捗表示
    total = len(df)
    completed_count = df['is_completed'].sum()
    progress = completed_count / total
    st.progress(progress)
    st.write(f"進捗: {completed_count} / {total} 件完了 (現在のID: {idx})")

    if completed_count >= total:
        st.success("🎉 すべてのデータの確認が完了しました！お疲れ様でした．")
        
        # 結果フォルダへ保存
        save_path = save_result_data(df, selected_file)
        st.info(f"✅️ 結果ファイルを保存しました: {save_path}")

        st.balloons() # 完了のお祝いエフェクト

        if st.button("終了する"):
            st.stop()
        
        st.stop()

    # データ表示
    row = df.iloc[idx]

    st.markdown("---")
    st.subheader("ツイート本文")
    st.info(row['text'])

    def get_default_value(col_name, options):
        # 既に人手入力があればそれを使う
        human_val = row.get(f"human_{col_name}")
        if pd.notna(human_val) and human_val in options:
            return options.index(human_val)
        
        # まだなければLLMの値を使う
        original_val = row.get(col_name)

        if col_name == "is_location_related":
            # CSVから読むと文字列になっている場合があるためキャスト「
            if str(original_val).lower() == "true": original_val = True
            elif str(original_val).lower() == "false": original_val = False
        
        if original_val in options:
            return options.index(original_val)
        
        return 0
    
    # 入力フォーム
    with st.form(key=f"annotation_form_{idx}"):
        col1, col2 = st.columns(2)

        # 場所関連性判定
        with col1:
            st.markdown("### 場所関連性")
            st.caption(f"LLMによるタグ: **{row['is_location_related']}**")
            val_loc = st.radio(
                "正解を選択",
                OPTIONS["is_location_related"],
                index=get_default_value("is_location_related", OPTIONS["is_location_related"]),
                horizontal=True,
                key=f"radio_loc_{idx}"
            )
            # 迷いフラグ
            unc_loc = st.checkbox(
                "迷った (Uncertain)", 
                value=bool(row.get('uncertain_is_location_related', 0)),
                key=f"chk_loc_{idx}"
            )
        
        # 主観客観判定
        with col2:
            st.markdown("### 主観客観判定")
            st.caption(f"LLMによるタグ: **{row['subjectivity']}**")
            val_sub = st.selectbox(
                "正解を選択",
                OPTIONS["subjectivity"],
                index=get_default_value("subjectivity", OPTIONS["subjectivity"]),
                key=f"sel_sub_{idx}"
            )
            unc_sub = st.checkbox(
                "迷った", 
                value=bool(row.get('uncertain_subjectivity', 0)),
                key=f"chk_sub_{idx}"
            )
        
        st.markdown("---")
        col3, col4 = st.columns(2)

        # 感情極性 / ノイズ判定
        with col3:
            st.markdown("### 感情極性 / ノイズ判定")
            st.caption(f"LLMによるタグ: **{row['sentiment_or_noise']}**")
            val_sent = st.selectbox(
                "正解を選択",
                OPTIONS["sentiment_or_noise"],
                index=get_default_value("sentiment_or_noise", OPTIONS["sentiment_or_noise"]),
                key=f"sel_sent_{idx}"
            )
            unc_sent = st.checkbox(
                "迷った", 
                value=bool(row.get('uncertain_sentiment_or_noise', 0)),
                key=f"chk_sent_{idx}"
            )

        # 居住者判定
        with col4:
            st.markdown("### 居住者判定")
            st.caption(f"LLMによるタグ: **{row['user_attribute']}**")
            val_attr = st.selectbox(
                "正解を選択",
                OPTIONS["user_attribute"],
                index=get_default_value("user_attribute", OPTIONS["user_attribute"]),
                key=f"sel_attr_{idx}"
            )
            unc_attr = st.checkbox(
                "迷った", 
                value=bool(row.get('uncertain_user_attribute', 0)),
                key=f"chk_attr_{idx}"
            )
        
        st.markdown("---")
        comments = st.text_input("備考・メモ", value=row.get('comments', "") if pd.notna(row.get('comments', "")) else "")

        # ボタン
        submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 3])
        with submit_col1:
            prev_btn = st.form_submit_button("⬅️ 戻る")
        with submit_col2:
            next_btn = st.form_submit_button("保存して次へ ➡️", type="primary")
        
        # ロジック処理
        if next_btn:
            # データフレームの更新
            st.session_state.df.at[idx, 'human_is_location_related'] = val_loc
            st.session_state.df.at[idx, 'human_subjectivity'] = val_sub
            st.session_state.df.at[idx, 'human_sentiment_or_noise'] = val_sent
            st.session_state.df.at[idx, 'human_user_attribute'] = val_attr
            st.session_state.df.at[idx, 'uncertain_is_location_related'] = 1 if unc_loc else 0
            st.session_state.df.at[idx, 'uncertain_subjectivity'] = 1 if unc_sub else 0
            st.session_state.df.at[idx, 'uncertain_sentiment_or_noise'] = 1 if unc_sent else 0
            st.session_state.df.at[idx, 'uncertain_user_attribute'] = 1 if unc_attr else 0
            st.session_state.df.at[idx, 'comments'] = comments
            st.session_state.df.at[idx, 'is_completed'] = True

            # ファイル保存
            save_batch_data(st.session_state.df, selected_file)

            # インデックスを進める
            if idx < len(df) - 1:
                st.session_state.current_index += 1
                st.rerun()
            else:
                st.success("最後のデータです!")
                st.rerun()
        
        if prev_btn:
            if idx > 0:
                st.session_state.current_index -= 1
                st.rerun()
