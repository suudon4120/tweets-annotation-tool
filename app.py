import streamlit as st
import pandas as pd
from src.data_handler import get_batch_files, load_batch_data, save_batch_data, save_result_data
from src.sampler import create_sample_batch

# 定数定義
OPTIONS = {
    "is_location_related": [True, False],
    "subjectivity": ["N/A", "主観", "客観"],
    "sentiment_or_noise": ["N/A", "ポジティブ", "ネガティブ", "ノイズ(クーポン情報)", "ノイズ(単体場所情報)", "ノイズ(広告・宣伝)", "ノイズ(客観的記述)"],
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
                元データからランダムにツイートを抽出し，作業用ファイルを作成します．
                """)
    
    with st.form("sampling_form"):
        col1, col2 = st.columns(2)
        with col1:
            annotator_name = st.text_input("作業者名 (半角英数推奨)", value="user1")
            seed = st.number_input("乱数シード", value=42, step=1)
        with col2:
            n_samples = st.number_input("抽出件数", value=100, step=10)

        submitted = st.form_submit_button("データセットを作成")

        if submitted:
            if not annotator_name:
                st.error("作業者名を入力してください．")
            else:
                try:
                    filename, count = create_sample_batch(n_samples, seed, annotator_name)
                    st.success(f"✅️ 作成完了! ファイル名: {filename} ({count}件)")
                    st.info("「Annotation」モードに切り替えて作業を開始してください．")
                except FileNotFoundError:
                    st.error("❌️ 元データが見つかりません．data/raw/ フォルダを確認してください．")
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
        # 既に人で入力があればそれを使う
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
                horizontal=True
            )
        
        # 主観客観判定
        with col2:
            st.markdown("### 主観客観判定")
            st.caption(f"LLMによるタグ: **{row['subjectivity']}**")
            val_sub = st.selectbox(
                "正解を選択",
                OPTIONS["subjectivity"],
                index=get_default_value("subjectivity", OPTIONS["subjectivity"]),
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
            )

        # 居住者判定
        with col4:
            st.markdown("### 居住者判定")
            st.caption(f"LLMによるタグ: **{row['user_attribute']}**")
            val_attr = st.selectbox(
                "正解を選択",
                OPTIONS["user_attribute"],
                index=get_default_value("user_attribute", OPTIONS["user_attribute"]),
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
