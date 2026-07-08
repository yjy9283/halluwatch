"""
HalluWatch API 서버.

프론트엔드(web/)와 API를 같은 서버에서 서빙해 CORS 문제 없이 배포 단순화.

실행: uvicorn src.api.main:app --reload
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.entropy.pipeline import analyze_question

app = FastAPI(title="HalluWatch API")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class AnalyzeRequest(BaseModel):
    question: str
    n_samples: int = Field(default=8, ge=3, le=20)
    temperature: float = Field(default=1.0, ge=0.1, le=1.5)
    uncertainty_threshold: float = Field(default=0.4, ge=0.1, le=0.9)


class AnalyzeResponse(BaseModel):
    question: str
    samples: list[str]
    embedding_method: str
    cluster_labels: list[int]
    n_clusters: int
    raw_entropy: float
    normalized_entropy_score: float
    is_uncertain: bool
    threshold: float


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        result = analyze_question(
            req.question,
            n_samples=req.n_samples,
            temperature=req.temperature,
            uncertainty_threshold=req.uncertainty_threshold,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not result.samples:
        # 모든 샘플 생성이 실패한 경우 (네트워크 차단, API 키 오류, rate limit 등).
        # 이걸 "샘플 0개짜리 성공"으로 응답하면 프론트엔드가 빈 결과를 정상 분석
        # 결과처럼 보여줄 위험이 있어서(실제로 테스트 중 발견함), 명시적으로 에러 처리한다.
        raise HTTPException(
            status_code=502, detail="LLM으로부터 응답을 하나도 받지 못했습니다. 네트워크 상태나 GROQ_API_KEY를 확인하세요."
        )

    return AnalyzeResponse(
        question=result.question,
        samples=result.samples,
        embedding_method=result.embedding_method,
        cluster_labels=result.cluster_labels,
        n_clusters=result.n_clusters,
        raw_entropy=result.raw_entropy,
        normalized_entropy_score=result.normalized_entropy_score,
        is_uncertain=result.is_uncertain,
        threshold=result.threshold,
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
