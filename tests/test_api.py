import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


@patch("src.api.main.analyze_question")
def test_analyze_returns_result_when_samples_present(mock_analyze):
    from src.entropy.pipeline import EntropyResult

    mock_analyze.return_value = EntropyResult(
        question="테스트 질문",
        samples=["답변1", "답변2"],
        embedding_method="sentence-transformers",
        cluster_labels=[0, 0],
        n_clusters=1,
        raw_entropy=0.0,
        normalized_entropy_score=0.0,
        is_uncertain=False,
        threshold=0.4,
    )
    res = client.post("/api/analyze", json={"question": "테스트 질문"})
    assert res.status_code == 200
    data = res.json()
    assert data["n_clusters"] == 1
    assert data["is_uncertain"] is False


@patch("src.api.main.analyze_question")
def test_analyze_returns_502_when_all_samples_fail(mock_analyze):
    """모든 LLM 호출이 실패해 샘플이 0개인 경우, 200(가짜 성공)이 아니라
    502로 명확히 에러 처리되는지 검증 (실사용 테스트 중 발견한 버그의 회귀 방지)."""
    from src.entropy.pipeline import EntropyResult

    mock_analyze.return_value = EntropyResult(
        question="테스트 질문",
        samples=[],
        embedding_method="none",
        cluster_labels=[],
        n_clusters=0,
        raw_entropy=0.0,
        normalized_entropy_score=0.0,
        is_uncertain=True,
        threshold=0.4,
    )
    res = client.post("/api/analyze", json={"question": "테스트 질문"})
    assert res.status_code == 502
    assert "응답을 하나도 받지 못했습니다" in res.json()["detail"]


def test_analyze_validates_request_params():
    # n_samples 범위(3~20) 벗어나면 422 검증 에러
    res = client.post("/api/analyze", json={"question": "질문", "n_samples": 100})
    assert res.status_code == 422


def test_index_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "HalluWatch" in res.text
