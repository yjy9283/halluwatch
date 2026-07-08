"""
같은 질문을 LLM에 여러 번 물어봐서 다양한 답변 샘플을 얻는 모듈.

Semantic Entropy 계산의 첫 단계: temperature를 높여 답변의 다양성을 유도하고,
그 다양성 자체가 "모델이 확신하는 정도"를 보여주는 신호가 된다.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def _get_client():
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def generate_samples(
    question: str, n_samples: int = 10, temperature: float = 1.0, model: str = DEFAULT_MODEL, max_tokens: int = 200
) -> list[str]:
    """같은 질문에 대해 n_samples개의 답변을 생성한다.

    Args:
        question: 사용자 질문
        n_samples: 생성할 답변 수 (많을수록 엔트로피 추정이 안정적이지만 API 호출 비용 증가)
        temperature: 높을수록 답변 다양성이 커짐 (Semantic Entropy 측정을 위해서는
            어느 정도 높은 값(0.7~1.2)이 필요 — 너무 낮으면 항상 똑같은 답만 나와 측정 의미 없음)
        model: Groq 모델명
        max_tokens: 답변당 최대 토큰 (짧게 잡아야 여러 번 호출해도 비용/시간 부담이 적음)

    Returns:
        답변 문자열 리스트 (길이 n_samples, API 실패 시 빈 리스트)
    """
    client = _get_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    samples = []
    for _ in range(n_samples):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                samples.append(content.strip())
        except Exception as e:
            # 개별 호출 실패는 건너뛰고 계속 진행 (일부 샘플만으로도 엔트로피 추정 가능)
            print(f"[경고] 샘플 생성 실패: {type(e).__name__}")
            continue

    return samples
