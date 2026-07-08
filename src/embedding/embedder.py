"""
답변 문장들을 벡터(임베딩)로 변환하는 모듈.

기본은 sentence-transformers(all-MiniLM-L6-v2)를 쓰지만, 모델 다운로드가 안 되는
환경(네트워크 제한, 오프라인 등)에서도 죽지 않도록 TF-IDF 기반 폴백을 자동 적용한다.
(DarkShip Hunter 프로젝트에서도 같은 패턴을 썼음 — 외부 의존성은 항상 실패할 수 있다는 전제로
폴백 경로를 설계해두는 게 중요하다는 교훈.)

주의: TF-IDF는 표면적 단어 일치만 보고 의미(paraphrase)를 이해하지 못하므로, 진짜
semantic entropy를 측정하려면 sentence-transformers가 정상 동작하는 환경에서 실행해야 한다.
TF-IDF 폴백은 "코드가 죽지 않고 동작은 한다"는 수준의 안전장치이지, 동등한 성능이 아니다.
"""

from __future__ import annotations
import numpy as np

_MODEL = None
_MODEL_LOAD_FAILED = False


def _try_load_sentence_transformer():
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None or _MODEL_LOAD_FAILED:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _MODEL
    except Exception as e:
        print(f"[경고] sentence-transformers 모델 로드 실패 ({type(e).__name__}) — TF-IDF 폴백으로 전환합니다.")
        _MODEL_LOAD_FAILED = True
        return None


def embed_texts(texts: list[str]) -> tuple[np.ndarray, str]:
    """텍스트 리스트를 임베딩 행렬로 변환한다.

    Returns:
        (embeddings, method) — embeddings는 (N, D) 배열, method는 "sentence-transformers" 또는 "tfidf-fallback"
    """
    if len(texts) == 0:
        return np.zeros((0, 1)), "empty"

    model = _try_load_sentence_transformer()
    if model is not None:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings), "sentence-transformers"

    # 폴백: TF-IDF (표면적 단어 유사도만 반영, 의미 유사도는 못 잡음 — 위 docstring 참고)
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray(), "tfidf-fallback"
    except ValueError:
        # 전부 빈 문자열 등 극단적 케이스
        return np.zeros((len(texts), 1)), "tfidf-fallback"
