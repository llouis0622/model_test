# SME Early Warning Cleanup

이 저장소는 `train_full_ensemble.ipynb` 노트북으로 최종 위험 예측 결과물을 생성하고,
`api/` 폴더의 FastAPI 애플리케이션으로 예측 결과와 간단한 규칙 기반 점수를 제공하도록
정리되었습니다.

## 남겨둔 핵심 구성 요소
- `train_full_ensemble.ipynb`: 최신 학습 파이프라인과 최종 예측 산출물을 생성하는 Jupyter 노트북.
- `risk_output_trained.csv`: API가 조회하는 최종 위험 예측 결과 샘플.
- `data/`: 노트북 학습에 필요한 원본 데이터 샘플.
- `api/`: 위험 점수 조회 및 자연어 파싱을 제공하는 FastAPI 서비스 코드.

## 정리하며 제거한 불필요한 항목
아래 파일들은 과거 실험용 파이프라인, 중간 산출물, 혹은 현재 API와 노트북에서 더 이상
사용하지 않는 스크립트라서 삭제했습니다.

- `alerting.py`, `ensemble.py`, `pipeline.py`, `preprocessing.py`, `risk_aggregate.py`, `risk_components.py`, `utils.py`, `viz.py`
- `config.py`, `run.py`, `__main__.py`
- `train_baseline.ipynb`, `train_baseline_fixed.ipynb`, `train_full_ensemble.py`

필요하다면 Git 기록에서 해당 파일들을 언제든 복원할 수 있습니다.

## FastAPI 실행 방법
```bash
uvicorn api.app:app --reload
```

## 예측 조회 예시
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "store_id": "12345",
        "target_month": "2023-06"
      }'
```

## NLP 파싱 엔드포인트 예시
```bash
curl -X POST http://localhost:8000/nlp/parse \
  -H "Content-Type: application/json" \
  -d '{"utterance": "강남구 치킨집인데 최근 3개월 평균 매출은 1,200만원이에요."}'
```

## 주요 코드 변경 설명
- `api/app.py`: FastAPI 엔드포인트 구조를 단순화하고 `quickscore`/`predict_batch` 호출을 명확하게 분리해, 예측 결과가 없는 경우에도 규칙 기반 점수로 자연스럽게 폴백되도록 했습니다. 또한 CORS 설정과 라이프사이클 설명을 위한 모듈 수준 주석을 추가했습니다.
- `api/loader.py`: 환경 변수 기반 경로와 로컬 CSV 파일을 모두 지원하도록 로더를 다듬어, 운영 환경과 로컬 개발 환경에서 동일한 인터페이스로 모델/결과물을 읽어올 수 있게 했습니다.
- `api/service.py`: 배치 예측 로직과 규칙 기반 점수를 함수로 명확히 분리해 재사용성을 높였고, 결과 포맷을 `schemas.PredictResponse`에 맞춰 일관되게 반환하도록 조정했습니다.
- `api/nlp.py`: 기본 정규식과 숫자 파싱 유틸리티를 묶어 자연어 입력을 구조화된 특징으로 변환하는 과정을 단순화했습니다.
- `api/schemas.py`: 요청/응답 스키마에 타입 힌트를 정리해 프론트엔드-백엔드 간 데이터 계약을 명확히 했습니다.
- 불필요한 학습 스크립트, 실험 노트북, IDE 설정 파일은 모두 삭제해 현재 운영에 필요한 노트북과 API 코드만 남겼습니다.
