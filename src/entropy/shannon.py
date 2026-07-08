"""Shannon entropy 계산 — 클러스터 분포로부터 "의미적 불확실성"을 수치화한다."""

from __future__ import annotations
import numpy as np


def shannon_entropy(cluster_labels: np.ndarray) -> float:
    """클러스터 라벨 분포의 Shannon entropy를 계산한다 (자연로그 기준, nats 단위).

    - 답변이 전부 같은 클러스터(모델이 확신) → entropy = 0
    - 답변이 여러 클러스터에 고르게 퍼짐(모델이 헷갈림) → entropy가 커짐 (최댓값 = ln(클러스터 수))
    """
    if len(cluster_labels) == 0:
        return 0.0

    _, counts = np.unique(cluster_labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def normalized_entropy(cluster_labels: np.ndarray) -> float:
    """entropy를 0~1 사이로 정규화한다 (샘플 수와 무관하게 해석 가능하도록).

    최대 엔트로피(모든 샘플이 서로 다른 클러스터)로 나눠서 정규화한다.
    샘플이 1개뿐이거나 전부 같은 클러스터면 0을 반환한다.
    """
    n = len(cluster_labels)
    if n <= 1:
        return 0.0

    raw_entropy = shannon_entropy(cluster_labels)
    max_possible_entropy = np.log(n)  # 모든 샘플이 각자 다른 클러스터인 경우
    if max_possible_entropy == 0:
        return 0.0
    return float(raw_entropy / max_possible_entropy)
