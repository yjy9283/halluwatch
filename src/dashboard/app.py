"""
HalluWatch 대시보드.

실행: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.entropy.pipeline import analyze_question

st.set_page_config(page_title="HalluWatch", page_icon="🔦", layout="wide")
st.title("🔦 HalluWatch — LLM 환각 조기경보")

st.markdown(
    "같은 질문을 LLM에 여러 번 물어봐서, 답변들이 의미적으로 얼마나 일치하는지(Semantic Entropy)로 "
    "모델이 실제로 확신하는지 판단합니다. 논문: Farquhar et al. (2024), *Nature*."
)

with st.expander("ℹ️ 이 도구는 무엇을 측정하나요?"):
    st.markdown(
        """
- 같은 질문을 **여러 번, 다르게(temperature를 높여서)** LLM에 물어봅니다.
- 답변들을 의미 기준으로 클러스터링합니다 (표현이 달라도 뜻이 같으면 한 그룹).
- 그룹이 하나로 뭉치면 → 모델이 확신하는 답 (낮은 엔트로피)
- 그룹이 여러 개로 갈리면 → 모델이 헷갈리는 답 (높은 엔트로피, ⚠️ 경고)

**한계**: sentence-transformers 모델을 못 받아오는 환경에서는 TF-IDF(단어 일치 기반)로
자동 대체되는데, 이 경우 패러프레이즈(다른 단어로 같은 뜻)를 잘 못 잡아냅니다. 정확한
측정을 위해서는 로컬 환경에서 sentence-transformers가 정상 동작해야 합니다.
        """
    )

question = st.text_input("질문을 입력하세요", value="세종대왕이 발명한 전자기기는 무엇인가요?")

col1, col2, col3 = st.columns(3)
with col1:
    n_samples = st.slider("샘플 수", 3, 20, 8)
with col2:
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0, 0.1)
with col3:
    uncertainty_threshold = st.slider("불확실 판정 임계값", 0.1, 0.9, 0.4, 0.05)

if st.button("분석 실행"):
    with st.spinner(f"'{question}'에 대해 {n_samples}번 답변 생성 중..."):
        try:
            result = analyze_question(
                question, n_samples=n_samples, temperature=temperature, uncertainty_threshold=uncertainty_threshold
            )
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    if result.embedding_method == "tfidf-fallback":
        st.warning("⚠️ sentence-transformers 로드 실패로 TF-IDF 폴백 사용 중 — 패러프레이즈 인식 정확도가 낮습니다.")

    if result.is_uncertain:
        st.error(f"⚠️ 불확실한 답변입니다 (정규화 엔트로피: {result.normalized_entropy_score:.2f})")
    else:
        st.success(f"✅ 모델이 확신하는 답변으로 보입니다 (정규화 엔트로피: {result.normalized_entropy_score:.2f})")

    m1, m2, m3 = st.columns(3)
    m1.metric("생성된 샘플 수", len(result.samples))
    m2.metric("발견된 의미 클러스터 수", result.n_clusters)
    m3.metric("정규화 엔트로피", f"{result.normalized_entropy_score:.2f}")

    st.markdown("#### 생성된 답변들 (클러스터별)")
    for cluster_id in sorted(set(result.cluster_labels)):
        with st.expander(f"클러스터 {cluster_id} ({result.cluster_labels.count(cluster_id)}개 답변)"):
            for i, (sample, label) in enumerate(zip(result.samples, result.cluster_labels)):
                if label == cluster_id:
                    st.write(f"- {sample}")
