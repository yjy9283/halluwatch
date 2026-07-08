import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.entropy.shannon import shannon_entropy, normalized_entropy
from src.embedding.clustering import cluster_by_meaning


def test_shannon_entropy_is_zero_when_all_same_cluster():
    labels = np.array([0, 0, 0, 0, 0])
    assert shannon_entropy(labels) == 0.0


def test_shannon_entropy_is_positive_when_clusters_differ():
    labels = np.array([0, 0, 1, 1, 2])
    assert shannon_entropy(labels) > 0.0


def test_shannon_entropy_max_when_all_different():
    # 5개 샘플이 전부 다른 클러스터 -> 최대 엔트로피 = ln(5)
    labels = np.array([0, 1, 2, 3, 4])
    assert abs(shannon_entropy(labels) - np.log(5)) < 1e-9


def test_normalized_entropy_range_is_0_to_1():
    all_same = np.array([0, 0, 0, 0])
    all_diff = np.array([0, 1, 2, 3])
    assert normalized_entropy(all_same) == 0.0
    assert abs(normalized_entropy(all_diff) - 1.0) < 1e-9


def test_normalized_entropy_single_sample_is_zero():
    assert normalized_entropy(np.array([0])) == 0.0


def test_cluster_by_meaning_groups_similar_embeddings():
    # 두 그룹: (0,0)근처 3개, (10,10)근처 2개 -> 명확히 분리된 두 클러스터가 나와야 함
    embeddings = np.array(
        [
            [1.0, 1.0],
            [1.01, 1.01],
            [1.02, 0.99],
            [10.0, -10.0],
            [10.01, -9.99],
        ]
    )
    labels = cluster_by_meaning(embeddings, distance_threshold=0.1)
    assert len(set(labels[:3])) == 1  # 앞 3개는 같은 클러스터
    assert len(set(labels[3:])) == 1  # 뒤 2개는 같은 클러스터
    assert labels[0] != labels[3]  # 두 그룹은 서로 다른 클러스터


def test_cluster_by_meaning_handles_single_sample():
    labels = cluster_by_meaning(np.array([[1.0, 2.0]]))
    assert len(labels) == 1
