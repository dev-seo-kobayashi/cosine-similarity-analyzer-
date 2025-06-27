# app.py (trafilatura版)
import streamlit as st
import numpy as np
import trafilatura  # newspaperの代わりにtrafilaturaをインポート
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="コンテンツ関連性スコア分析ツール",
    page_icon="🔍"
)

st.title("🔍 コンテンツ関連性スコア分析ツール")
st.write("検索クエリとページのURLを入力すると、意味的な関連性をAIがスコアリングします。")

@st.cache_resource
def load_model():
    model_name = "intfloat/multilingual-e5-large"
    model = SentenceTransformer(model_name)
    return model

with st.spinner("AIモデルを読み込んでいます...（初回は数分かかります）"):
    model = load_model()

st.header("分析情報の入力")
target_query = st.text_input("【1/2】分析したい検索クエリを入力してください", "コサイン類似度 SEO 活用")
page_url = st.text_input("【2/2】分析したいページのURLを入力してください", "https://www.suzukikenichi.com/blog/cosine-similarity-and-seo/")

if st.button("分析を実行する"):
    if not target_query or not page_url:
        st.error("クエリとURLの両方を入力してください。")
    else:
        content_text = ""
        with st.spinner(f"URLから本文を取得しています...\n{page_url}"):
            try:
                # --- ここからがnewspaperからの変更点 ---
                downloaded = trafilatura.fetch_url(page_url)
                content_text = trafilatura.extract(downloaded)
                # --- ここまでが変更点 ---
                
                if not content_text:
                    st.warning("URLから本文を自動で抽出できませんでした。サイトの構造が複雑な可能性があります。")
            except Exception as e:
                st.error(f"本文の取得に失敗しました: {e}")

        if content_text:
            st.success("本文の取得が完了しました。")
            
            with st.spinner("関連性スコアを計算しています..."):
                query_with_prefix = 'query: ' + target_query
                content_with_prefix = 'passage: ' + content_text
                query_vector = model.encode(query_with_prefix, normalize_embeddings=True)
                content_vector = model.encode(content_with_prefix, normalize_embeddings=True)
                query_vector_2d = np.array([query_vector])
                content_vector_2d = np.array([content_vector])
                similarity_score = cosine_similarity(query_vector_2d, content_vector_2d)
                score = similarity_score[0][0]

            st.header("分析結果")
            st.metric(label="関連性スコア", value=f"{score:.4f}")

            if score >= 0.85:
                st.success("評価: 非常に高い関連性があります。")
            elif score >= 0.80:
                st.info("評価: 高い関連性があります。")
            elif score >= 0.75:
                st.warning("評価: 関連性がありますが、改善の余地がありそうです。")
            else:
                st.error("評価: 関連性が低いです。")

            with st.expander("取得した本文を確認する"):
                st.text(content_text[:1000] + "...")
