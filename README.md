# 용용톤 AI 서버 — 월촌교 1시간 뒤 수위 예측 (FastAPI + XGBoost)

## Cloudtype 배포

FastAPI 템플릿에서 이 저장소의 `main`을 선택합니다. Python 3.11, 서비스 포트 `8000`, 시작 명령은 다음과 같습니다.

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

환경변수/시크릿에 `DATA_SOURCE=safetydata`, `SAFETYDATA_WATER_KEY`, `SAFETYDATA_RAIN_KEY`를 설정합니다. 키 값은 Git에 저장하지 않습니다. Health check 경로는 `/health`이며, 배포 후 `/docs`, `/health`, `/live/status`를 확인합니다. `/health`의 HTTP 응답은 수집 지연 중에도 200이며, 본문 `status=degraded`와 `last_water_at`으로 실제 최신성을 판단합니다.

백엔드의 `AI_SERVER_URL`에는 AI 서비스의 접속 주소를 설정합니다. 같은 Cloudtype 배포환경이면 서비스 리소스 탭의 내부 연결 주소를 사용하고, 다른 환경이면 공개 HTTPS 주소를 사용합니다. 로컬용 `run.py`는 Cloudtype 시작 명령으로 사용하지 않습니다.

## macOS / PyCharm에서 실행

1. PyCharm에서 이 폴더를 열고 Python 인터프리터를 `.venv/bin/python`으로 선택합니다.
2. 실행 구성에서 `AI Server (replay)`를 선택해 Run을 누릅니다. 구성 목록에 없으면 `run.py`를 우클릭해 실행하세요.
3. 실행 창에 `Uvicorn running on http://127.0.0.1:8765`가 표시되면 http://127.0.0.1:8765/docs 를 엽니다.
4. 요청을 보내면 같은 실행 창에 HTTP 접근 로그와 예측·상태 로그가 표시됩니다. Stop 버튼으로 종료합니다.

`DATA_SOURCE=replay`는 과거 데이터 시연 모드입니다. `app/live.py`를 직접 실행하면 API 서버가 시작되지 않습니다.

### 실제 관측값 사용

`.env.example`을 `.env`로 복사하고 `SAFETYDATA_WATER_KEY`, `SAFETYDATA_RAIN_KEY`를 입력하세요.
PyCharm에서 `AI Server (live)` 실행 구성을 선택하거나 터미널에서 `DATA_SOURCE=safetydata AI_SERVER_PORT=8766 .venv/bin/python run.py`를 실행합니다. Swagger는 http://127.0.0.1:8766/docs 입니다. `8765`는 기존 재생 모드용입니다.
백엔드의 `AI_SERVER_URL`을 `http://127.0.0.1:8766`으로 설정합니다.
`/health`의 `last_water_at`, `last_rain_at`, `status`와 `/live/status`의 `data_status`를 확인하세요.

SafetyData 수위는 원본 cm 값을 m로 변환합니다(같은 시각의 과거 CSV와 대조 완료). 우량 API는 날짜/관측소 검색을 제공하지 않고 최근 자료가 지연될 수 있어, 오래되거나 빠진 우량을 0으로 대체하지 않습니다. 이때 현재 수위는 표시하지만 1시간·회복 예측은 보류됩니다. `replay` 모드는 계속 시연용입니다.

## 폴더 구조
```
ai-server/
├─ app/
│  ├─ main.py          # API (FastAPI)
│  └─ predictor.py     # 모델 로드 · 특징 계산 · 예측
├─ model/
│  ├─ wolchon_xgb_60min.json   # Colab에서 학습한 모델 (v2: 입력 11개)
│  ├─ features.json            # 입력 특징 순서
│  └─ metrics.json             # 시험 성능
├─ data/
│  └─ yongyongthon_xgboost_training_ready.csv   # 시연(재생)용 과거 데이터
├─ tests/test_api.py
└─ requirements.txt
```

## 실행 (Windows PowerShell)
```powershell
cd ai-server
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```
- API 문서·테스트 화면: http://localhost:8000/docs
- 같은 와이파이의 폰/다른 PC에서 접속하려면 `--host 0.0.0.0` 추가 후 `http://<내 PC IP>:8000`
- 테스트: `.\.venv\Scripts\python.exe -m pytest -q`

## API
| 메서드 | 경로 | 용도 |
|---|---|---|
| GET | `/health` | 서버·모델 상태, 재생 가능 기간 |
| POST | `/predict` | **실서비스용**: 최근 10분 관측값(수위 2시간 + 강수량 3시간)으로 예측 |
| POST | `/predict/features` | 백엔드가 특징 11개를 직접 계산해 보낼 때 |
| GET | `/replay/predict?time=2026-07-09T12:00` | **시연용**: 과거 시각 기준 예측 + 실제 1시간 뒤 수위 |
| GET | `/replay/series?start=2026-07-09T00:00&hours=24` | **시연용**: 기간 그래프 데이터 (실제·예측·강수량) |

### POST /predict 요청 예
기준 시각(마지막 시각)까지 10분 간격, **최소 18개**(강수량 3시간). 수위는 최근 13개(2시간)만 있으면 됩니다.
```json
{
  "observations": [
    {"time": "2026-07-09T09:10", "water_level_m": 0.99, "rain_mm": 1.0},
    "... 10분 간격 ...",
    {"time": "2026-07-09T12:00", "water_level_m": 1.57, "rain_mm": 3.0}
  ]
}
```
빠진 시각이 있으면 422와 함께 어떤 시각 값이 없는지 알려 줍니다.

### 응답 예 (`/replay/predict?time=2026-07-09T12:00`)
```json
{
  "station": {"name": "월촌교", "river": "경안천", "water_station_code": "1016607",
              "rain_station": "용인시(모현면사무소)", "rain_station_code": "10164050"},
  "base_time": "2026-07-09T12:00",
  "target_time": "2026-07-09T13:00",
  "current_level_m": 1.57,
  "predicted_level_m": 2.01,
  "change_m": 0.44,
  "trend": "rising",
  "source": "estimate",
  "model_version": "wolchon_xgb_60min v2 (+water_change_10min_m)",
  "notice": "AI 추정값입니다. 공식 통제 여부와 별개이며, 출입은 용인시 공지를 따르세요.",
  "actual_level_m": 2.68
}
```
- `trend`: 1시간 변화가 +2cm 이상 `rising`, −2cm 이하 `falling`, 그 외 `steady`
- `source: "estimate"`: 앱에서 'AI 추정' 칩으로 표시 (공식 통제 정보와 분리)
- `actual_level_m`: 재생 모드에서만 있는 실제값 (검증·시연용)

## 주의
- 예측값은 AI 추정이며 산책로 이용 가능 여부나 통제 해제를 확정하지 않습니다.
- 급격한 상승 초기에는 예측이 실제보다 늦게 따라가는 경향이 있습니다(시험 그래프 참고).
- 실시간 자료는 재난안전데이터공유플랫폼에서 수집합니다. `/health`에서 최신 관측 시각을 확인하세요.

## v1.1 추가: 실시간(재생) + 회복 예측

### 무엇이 늘었나
- **회복 예측 v0.5** (팀원 학습 모델): 월촌교 수위가 관심 수위 **1.5m 아래로 내려가 2시간 유지**되는 시점
  - 6·12시간 안 확률 + "빨라도 약 N시간" 예상 시각. 24시간 확률은 검증이 안 돼서 제공하지 않아요.
  - 수위가 **하강 중**일 때만 시각을 줘요. 비가 오는 중이면 시각을 숨기고 경고해요. 6시간 넘게 걸릴 것 같으면 "반나절 이상"으로 표시해요.
  - ⚠ 산책로 이용 가능 시각이 아니에요.
- **실시간 수집기**: `DATA_SOURCE=replay`(기본)는 과거 10분 자료를 재생합니다. `DATA_SOURCE=safetydata`는 발급받은 수위·우량 서비스 키로 관측값을 수집합니다.

### 실행 (재생 모드)
```powershell
copy .env.example .env      # 필요하면 REPLAY_START / REPLAY_SPEED 수정
$env:DATA_SOURCE="replay"; $env:REPLAY_START="2026-07-09T06:00"; $env:REPLAY_SPEED="10"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
REPLAY_SPEED=10이면 실제 6분에 데이터 1시간이 흘러요. 서버를 켠 순간이 REPLAY_START예요.

### 실시간 모드
Mac PyCharm에서 `AI Server (live)` 실행 구성을 사용합니다. 설정과 상태 확인 주소는 위의 **실제 관측값 사용** 절을 참고하세요.

### 새 API
| 메서드 | 경로 | 용도 |
|---|---|---|
| GET | `/live/status` | **앱 메인용**: 지금 수위, 공식 단계(관심 1.5 / 주의 4.5 / 경계 5.0 / 심각 6.7m), 1시간 뒤 예측, 회복 예측, 최근 강수 |
| GET | `/live/status?time=2026-07-10T01:00` | (재생 모드) 특정 시각으로 보기 |
| GET | `/live/series?hours=24` | 최근 24시간 수위·강수 그래프용 |
| POST | `/recovery` | 백엔드가 최근 24시간+10분(145개) 관측값을 보내 회복 예측 받기 |

### `/live/status` 응답 예 (2026-07-10 01:00, 재생)
```json
{
  "observed_at": "2026-07-10T01:00",
  "water_level_m": 1.89, "official_stage": "interest", "typical_level_m": 0.82,
  "rain_1h_mm": 0.0,
  "forecast_1h": {"target_time": "2026-07-10T02:00", "level_m": 1.77, "trend": "falling", "source": "estimate"},
  "recovery": {"status": "predicted", "message": "빨라도 약 5시간 뒤 관심 수위(1.5m) 아래로 내려갈 것으로 예상돼요. 비가 다시 오면 더 늦어질 수 있어요",
               "probability_within_6h": 0.54, "probability_within_12h": 0.94, "eta_hours_at_least": 4.7, "eta_time_at_least": "2026-07-10T05:42",
               "meaning": "월촌교 수위가 관심 수위 1.5m 아래로 내려가 2시간 유지되는 시점 (산책로 이용 가능 시각 아님)"},
  "data_status": "ok"
}
```
`recovery.status`: `below_threshold`(1.5m 아래) / `rising`(아직 안 내려감) / `raining`(비 오는 중) / `predicted`(예상 시각 있음) / `insufficient_data`(24시간 자료 부족)
`data_status`: `ok` / `partial`(일부 결측 → 1시간 예측 보류) / `missing`
