"""
임베딩 벡터들을 "의미가 같은 답변끼리" 클러스터링하는 모듈.

원 논문(Farquhar et al., 2024)은 NLI(자연어 추론) 모델로 두 답변이 서로
함의(entailment) 관계인지를 판단해 클러스터링하지만, 이 프로젝트에서는 구현
난이도를 낮추기 위해 문장 임베딩 간 코사인 거리 기반 계층적 클러스터링으로 근사한다.
(README 한계점에 이 근사가 원 논문과 다르다는 걸 명시함)
"""

from __future__ import annotations
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_by_meaning(embeddings: np.ndarray, distance_threshold: float = 0.3) -> np.ndarray:
    """임베딩을 코사인 거리 기반으로 계층적 클러스터링한다.

    Args:
        embeddings: (N, D) 임베딩 행렬
        distance_threshold: 이 거리보다 가까우면 "같은 의미"로 묶는다.
            낮을수록 더 엄격하게(비슷해야만) 묶고, 높을수록 관대하게 묶는다.
            기본값 0.3은 실사용 검증(sentence-transformers 정상 로드된 환경)에서, 0.15로는
            "서울입니다" / "대한민국의 수도는 서울입니다" / "서울이 한국의 수도예요" /
            "한국 수도는 서울" — 명백히 같은 의미인 4문장이 2개 클러스터로 잘못 쪼개지는 걸
            확인하고 상향한 값. 근거: scripts/inspect_embedding_distances.py로 실제 코사인
            거리를 찍어보고 결정할 것 (네트워크가 허용된 환경에서 실행 필요).

    Returns:
        각 샘플의 클러스터 라벨 배열 (길이 N)
    """
    n = len(embeddings)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])

    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average"
    )
    try:
        labels = clustering.fit_predict(embeddings)
    except ValueError:
        # 코사인 거리는 영벡터가 섞여 있으면 계산이 안 됨 (TF-IDF 폴백에서 드물게 발생 가능).
        # 이 경우 유클리드 거리로 대체해 완전히 실패하는 것보다는 근사치라도 반환한다.
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, metric="euclidean", linkage="average"
        )
        labels = clustering.fit_predict(embeddings)
    return labels
