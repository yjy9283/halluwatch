"""전체 파이프라인: 질문 → 샘플 생성 → 임베딩 → 클러스터링 → 엔트로피 → 경고 판정."""

from __future__ import annotations
from dataclasses import dataclass, field

from src.generator.sampler import generate_samples
from src.embedding.embedder import embed_texts
from src.embedding.clustering import cluster_by_meaning
from src.entropy.shannon import shannon_entropy, normalized_entropy


@dataclass
class EntropyResult:
    question: str
    samples: list[str]
    embedding_method: str
    cluster_labels: list[int]
    n_clusters: int
    raw_entropy: float
    normalized_entropy_score: float
    is_uncertain: bool
    threshold: float


def analyze_question(
    question: str,
    n_samples: int = 10,
    temperature: float = 1.0,
    distance_threshold: float = 0.3,
    uncertainty_threshold: float = 0.4,
) -> EntropyResult:
    """질문 하나에 대해 전체 semantic entropy 파이프라인을 실행한다.

    Args:
        question: 분석할 질문
        n_samples: LLM 반복 호출 횟수
        temperature: 답변 다양성 유도용 온도
        distance_threshold: 클러스터링 시 "같은 의미"로 볼 코사인 거리 기준
        uncertainty_threshold: 정규화된 엔트로피가 이 값을 넘으면 "불확실한 답변"으로 경고

    Returns:
        EntropyResult (샘플, 클러스터, 엔트로피, 경고 여부 포함)
    """
    samples = generate_samples(question, n_samples=n_samples, temperature=temperature)

    if len(samples) == 0:
        return EntropyResult(
            question=question,
            samples=[],
            embedding_method="none",
            cluster_labels=[],
            n_clusters=0,
            raw_entropy=0.0,
            normalized_entropy_score=0.0,
            is_uncertain=True,  # 답변 자체를 못 받았으면 당연히 신뢰 불가
            threshold=uncertainty_threshold,
        )

    embeddings, method = embed_texts(samples)
    labels = cluster_by_meaning(embeddings, distance_threshold=distance_threshold)

    raw_h = shannon_entropy(labels)
    norm_h = normalized_entropy(labels)

    return EntropyResult(
        question=question,
        samples=samples,
        embedding_method=method,
        cluster_labels=labels.tolist(),
        n_clusters=len(set(labels.tolist())),
        raw_entropy=raw_h,
        normalized_entropy_score=norm_h,
        is_uncertain=norm_h > uncertainty_threshold,
        threshold=uncertainty_threshold,
    )
