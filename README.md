# Model Test – Ensemble Training

이 저장소는 `train_full_ensemble.ipynb` 노트북을 기반으로 위험 점수를 추정하고, 결과를 요약한 시각화를 생성하는 최소 코드만 포함합니다.

## 폴더 구조

- `train_full_ensemble.ipynb` – 엔드투엔드 학습 및 추론 노트북
- `preprocessing.py`, `risk_components.py`, `risk_aggregate.py`, `ensemble.py`, `alerting.py`, `pipeline.py`, `config.py`, `utils.py` – 파이프라인을 구성하는 모듈
- `reporting.py` – `risk_output_trained.csv`를 기반으로 시각화 및 요약 테이블 생성
- `data/` – 원본 CSV 데이터 (저장소에는 포함되지 않거나, 로컬에 위치)
- `figures/` – `reporting.py` 또는 노트북 실행 시 생성되는 출력 파일

## 사용 방법

1. **환경 준비**
   ```bash
   pip install -r requirements.txt  # 또는 노트북 첫 셀에서 %pip install 실행
   ```

2. **노트북 실행**
   - `train_full_ensemble.ipynb`를 열어 모든 셀을 순서대로 실행합니다.
   - 기본적으로 현재 작업 디렉터리를 프로젝트 루트로 간주합니다. 필요 시 `MODEL_TEST_HOME` 환경 변수를 설정해 다른 경로를 사용할 수 있습니다.
   - 실행 결과로 `data/preds.csv`, `risk_output_trained.csv`, `figures/` 폴더가 생성됩니다.

3. **시각화만 다시 생성하기**
   ```bash
   python reporting.py risk_output_trained.csv --output figures --top 20
   ```
   - `risk_output_trained.csv` 경로와 출력 폴더를 원하는 값으로 지정할 수 있습니다.

## 결과물

`figures/` 폴더에는 다음 파일들이 생성됩니다.

- `distribution_p_final.png` – `p_final` 분포 히스토그램
- `monthly_mean_p_final.png` – 월별 평균 `p_final` 추이
- `alert_counts.png` – Alert 레벨별 개수 막대그래프
- `top_predictions.csv` – 상위 위험 점수 상점 목록

이러한 산출물은 노트북 실행 시 자동으로 업데이트되며, `reporting.py` 스크립트를 통해 별도로 재생성할 수도 있습니다.
