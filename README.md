# 🔦 HalluWatch

> LLM이 "확신 없는 답"을 내놓는 순간을 답변이 나오기 전에 미리 감지하는 환각(Hallucination) 조기경보 시스템

![대시보드](docs/screenshots/dashboard_main.png)

## 문제의식

LLM은 모르는 것도 자신 있는 어조로 답하는 경향이 있습니다. 이 프로젝트는 답변 "내용"을 검증하는
대신, 같은 질문을 여러 번 다르게 물어봤을 때 **답변들이 의미적으로 얼마나 일치하는지**를 측정해서
모델이 실제로 확신하는지 아닌지를 통계적으로 판단합니다.

## 원리 — Semantic Entropy

Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). *"Detecting hallucinations in large language
models using semantic entropy"*, Nature, 630, 625–630의 핵심 아이디어를 직접 구현했습니다.

```
질문 입력
  → LLM에 같은 질문을 N번 반복 호출 (temperature를 높여 답변 다양성 유도)
  → 답변들을 문장 임베딩으로 변환
  → 임베딩 공간에서 클러스터링 (의미적으로 같은 답변끼리 묶음)
  → 클러스터 분포로 Semantic Entropy 계산 (Shannon entropy)
  → 엔트로피가 임계값을 넘으면 "⚠️ 불확실한 답변" 경고
```

핵심 통찰: 문장 표현이 달라도 뜻이 같으면 하나의 클러스터로 묶입니다.
("파리입니다" / "프랑스의 수도는 파리예요" → 같은 클러스터)
반대로 뜻 자체가 갈리면 클러스터가 여러 개로 나뉘고, 엔트로피가 높아집니다.

## 검증 방법

- **알려진 사실 질문** (예: "한국의 수도는?") → 답변이 거의 동일 → 낮은 엔트로피가 나와야 함
- **모델이 모를 수밖에 없는 질문** (예: 최신/가상의 정보) → 답변이 제각각 → 높은 엔트로피가 나와야 함

이 두 극단 케이스로 도구가 실제로 구분해내는지 정량 검증합니다 (README 하단 실행 결과 참고).

## 검증 결과 및 한계 (실측)

실제 sentence-transformers로 검증한 결과 (`scripts/inspect_embedding_distances.py`):

| 케이스 | 코사인 거리 |
|---|---|
| 같은 의미 문장들 사이 최대 거리 | 0.280 |
| 다른 의미 문장들 사이 최소 거리 | 0.219 |

**한계를 정직하게 명시**: 두 값이 **겹칩니다** (0.280 > 0.219) — 하나의 고정 임계값으로
"같은 의미"와 "다른 의미"를 이론적으로 완벽하게 가르는 것은 이 샘플 기준으로는 불가능합니다.
`distance_threshold=0.3`은 이 두 극단 케이스를 기준으로 실용적으로 고른 값이며, 두 거리대가
겹치는 경계 부근의 문장(표현은 비슷한데 실제로는 다른 의미인 경우 등)은 오분류될 여지가
남아있습니다. 이는 코사인 거리 기반 클러스터링이 원 논문의 NLI(자연어 추론) 기반 클러스터링을
근사한 것이기 때문에 생기는 본질적 한계이며, 표본을 늘리거나 임계값을 도메인별로 정교화하면
개선 여지가 있습니다.

## 로컬 실행 방법

```bash
git clone https://github.com/yjy9283/halluwatch.git
cd halluwatch
pip install -r requirements.txt
```

`.env` 파일 생성:
```
GROQ_API_KEY=본인_키_값
```

```bash
streamlit run src/dashboard/app.py
```

테스트:
```bash
pytest tests/ -v
```

## 기술 스택

- LLM: Groq API (gpt-oss-120b)
- 임베딩: sentence-transformers (로컬 실행, 무료)
- 클러스터링: scikit-learn (DBSCAN)
- 통계: Shannon entropy
- 대시보드: Streamlit

## 참고 자료

- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language
  models using semantic entropy. *Nature*, 630, 625–630.
