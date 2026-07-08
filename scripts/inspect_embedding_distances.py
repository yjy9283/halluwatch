"""
클러스터링 distance_threshold를 실제 임베딩 거리 기준으로 정밀 캘리브레이션하기 위한
진단 스크립트.

sentence-transformers가 정상 로드되는 환경(네트워크 허용)에서 실행해야 의미가 있다.
TF-IDF 폴백 상태에서 돌리면 이 스크립트 자체가 무의미하다 (표면적 단어 거리만 나옴).

사용법:
    python scripts/inspect_embedding_distances.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from src.embedding.embedder import embed_texts

# 케이스 1: 명백히 같은 의미 (패러프레이즈) — 거리가 다 작아야 함
SAME_MEANING = [
    "서울입니다.",
    "대한민국의 수도는 서울입니다.",
    "서울이 한국의 수도예요.",
    "한국 수도는 서울.",
]

# 케이스 2: 명백히 다른 의미 — 거리가 다 커야 함
DIFFERENT_MEANING = [
    "서울입니다.",
    "정답은 42입니다.",
    "그건 알 수 없습니다.",
    "아마도 파리일 것 같아요.",
]


def print_distance_matrix(label: str, texts: list[str]):
    embeddings, method = embed_texts(texts)
    print(f"\n=== {label} (임베딩 방식: {method}) ===")
    if method == "tfidf-fallback":
        print("⚠️ TF-IDF 폴백 상태 — sentence-transformers가 로드되지 않음. 이 결과는 참고용이 아님.")

    dist_matrix = cosine_distances(embeddings)
    print("문장:")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t}")
    print("코사인 거리 행렬:")
    print(np.round(dist_matrix, 3))
    print(f"평균 거리: {dist_matrix[np.triu_indices(len(texts), k=1)].mean():.3f}")
    print(f"최대 거리: {dist_matrix[np.triu_indices(len(texts), k=1)].max():.3f}")


if __name__ == "__main__":
    print_distance_matrix("같은 의미(패러프레이즈) — 거리가 작아야 함", SAME_MEANING)
    print_distance_matrix("다른 의미 — 거리가 커야 함", DIFFERENT_MEANING)
    print(
        "\n권장: '같은 의미' 그룹의 최대 거리보다 크고, '다른 의미' 그룹의 최소 거리보다 "
        "작은 값을 distance_threshold로 설정하세요 (src/embedding/clustering.py)."
    )
