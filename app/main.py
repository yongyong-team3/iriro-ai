"""용용톤 AI 서버 — 월촌교 1시간 뒤 수위 예측 (FastAPI)

실행: uvicorn app.main:app --reload --port 8000
문서: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .predictor import BASE, InputError, Prediction, Predictor, ReplayData
from .live import make_source
from .recovery import RecoveryModel, HISTORY_STEPS

STATION = {"name": "월촌교", "river": "경안천", "water_station_code": "1016607",
           "rain_station": "용인시(모현면사무소)", "rain_station_code": "10164050"}
NOTICE = "AI 추정값입니다. 공식 통제 여부와 별개이며, 출입은 용인시 공지를 따르세요."
logger = logging.getLogger("uvicorn.error")

predictor = Predictor()
REPLAY_CSV = Path(os.getenv("REPLAY_CSV", BASE / "data" / "yongyongthon_xgboost_training_ready.csv"))
replay = ReplayData(REPLAY_CSV) if REPLAY_CSV.exists() else None

recovery_model = RecoveryModel()
source = make_source()
OFFICIAL_LEVELS = {"interest": 1.5, "caution": 4.5, "warning": 5.0, "severe": 6.7}  # 한강홍수통제소 월촌교 기준(m)

@asynccontextmanager
async def lifespan(_app):
    """실시간 소스의 관측값을 주기적으로 갱신한다."""
    logger.info("AI 서버 시작 | 데이터 소스=%s | 수위 모델=%s | 회복 모델=recovery-v0.5",
                source.name, predictor.model_version)
    if source.name == "replay":
        logger.warning("REPLAY 모드: 과거 관측 데이터를 재생합니다. 실제 실시간 수위가 아닙니다.")
    if getattr(source, "name", "") in {"hrfco", "safetydata"}:
        import threading
        stop = threading.Event()
        interval = 1800 if source.name == "safetydata" else 600

        def loop():
            while not stop.is_set():
                try:
                    source.refresh()
                    delay = 60 if getattr(source, "last_error", None) else interval
                except Exception as exc:
                    # URL 예외에 서비스 키가 들어갈 수 있으므로 상세 예외를 출력하지 않는다.
                    logger.error("실시간 수집 실패: %s", type(exc).__name__)
                    delay = 60
                stop.wait(delay)
        threading.Thread(target=loop, daemon=True).start()
        try:
            yield
        finally:
            stop.set()
    else:
        yield
    logger.info("AI 서버 종료")


app = FastAPI(title="용용톤 AI 서버", version="1.1.0", lifespan=lifespan,
              description="월촌교 1시간 뒤 수위 예측(v2) + 수위 회복 예측(v0.5) + 실시간(재생) 관측. 모든 예측값은 AI 추정입니다.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── 요청·응답 형식 ─────────────────────────────────────
class Observation(BaseModel):
    time: datetime = Field(..., description="관측 시각(10분 단위, 한국 시간)", examples=["2026-07-09T12:00"])
    water_level_m: float | None = Field(None, description="월촌교 수위(m)")
    rain_mm: float | None = Field(None, description="모현면사무소 10분 강수량(mm)")


class PredictRequest(BaseModel):
    observations: list[Observation] = Field(
        ..., min_length=18,
        description="기준 시각까지의 10분 간격 관측값. 수위는 최근 13개(2시간), 강수량은 최근 18개(3시간) 필요")


class FeaturesRequest(BaseModel):
    base_time: datetime
    water_level_m: float
    water_lag_10min_m: float
    water_lag_30min_m: float
    water_lag_60min_m: float
    water_lag_120min_m: float
    water_change_60min_m: float
    water_change_10min_m: float
    rain_10min_mm: float = Field(ge=0)
    rain_sum_30min_mm: float = Field(ge=0)
    rain_sum_60min_mm: float = Field(ge=0)
    rain_sum_180min_mm: float = Field(ge=0)


class PredictResponse(BaseModel):
    station: dict
    base_time: str = Field(description="기준 시각(현재 관측 시각)")
    target_time: str = Field(description="예측 대상 시각(기준 + 1시간)")
    current_level_m: float
    predicted_level_m: float
    change_m: float
    trend: Literal["rising", "falling", "steady"] = Field(description="2cm 이상 변화 시 rising/falling")
    source: str = "estimate"
    model_version: str
    notice: str = NOTICE
    actual_level_m: float | None = Field(None, description="(재생 모드만) 실제 관측된 1시간 뒤 수위 — 검증용")


def to_response(p: Prediction, actual: float | None = None) -> PredictResponse:
    return PredictResponse(
        station=STATION, base_time=p.base_time.strftime("%Y-%m-%dT%H:%M"),
        target_time=p.target_time.strftime("%Y-%m-%dT%H:%M"),
        current_level_m=p.current_level_m, predicted_level_m=p.predicted_level_m,
        change_m=p.change_m, trend=p.trend, model_version=predictor.model_version,
        actual_level_m=actual)


# ── API ─────────────────────────────────────────────────
@app.get("/health")
def health():
    status = "ok"
    if source.name == "safetydata":
        latest = getattr(source, "last_water_at", None)
        korea_now = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
        status = "ok" if latest and (korea_now - latest).total_seconds() <= 1800 else "degraded"
    info = {"status": status, "model_version": predictor.model_version, "features": predictor.features,
            "recovery_model": "recovery-v0.5", "data_source": source.name, "data_now": source.now().strftime("%Y-%m-%dT%H:%M")}
    if source.name == "safetydata":
        for name, value in (("last_water_at", source.last_water_at), ("last_rain_at", source.last_rain_at),
                            ("last_refresh", source.last_refresh)):
            info[name] = value.strftime("%Y-%m-%dT%H:%M") if value else None
    if getattr(source, "last_error", None):
        info["data_error"] = source.last_error
    if replay:
        lo, hi = replay.range
        info["replay_range"] = [lo.strftime("%Y-%m-%dT%H:%M"), hi.strftime("%Y-%m-%dT%H:%M")]
    return info


@app.post("/predict", response_model=PredictResponse, summary="최근 관측값으로 1시간 뒤 수위 예측")
def predict(req: PredictRequest):
    try:
        p = predictor.predict_observations([o.model_dump() for o in req.observations])
    except InputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    logger.info("1시간 예측 | 기준=%s | 현재=%.2fm | 예측=%.2fm", p.base_time, p.current_level_m, p.predicted_level_m)
    return to_response(p)


@app.post("/predict/features", response_model=PredictResponse, summary="계산된 특징 11개로 바로 예측")
def predict_features(req: FeaturesRequest):
    d = req.model_dump()
    base = d.pop("base_time").replace(tzinfo=None)
    return to_response(predictor.predict_features(base, d))


@app.get("/replay/predict", response_model=PredictResponse, summary="(시연) 과거 시각 기준 예측 + 실제값")
def replay_predict(time: datetime = Query(..., description="예: 2026-07-09T12:00 (10분 단위로 내림)")):
    if not replay:
        raise HTTPException(status_code=503, detail="재생용 CSV가 없습니다.")
    try:
        row = replay.row(time.replace(tzinfo=None))
    except KeyError:
        lo, hi = replay.range
        raise HTTPException(status_code=404, detail=f"해당 시각 자료가 없습니다. 가능 범위 {lo:%Y-%m-%d %H:%M} ~ {hi:%Y-%m-%d %H:%M} (일부 결측 구간 제외)")
    feats = {f: float(row[f]) for f in predictor.features}
    p = predictor.predict_features(row.name.to_pydatetime(), feats)
    logger.info("재생 예측 | 기준=%s | 현재=%.2fm | 예측=%.2fm", p.base_time, p.current_level_m, p.predicted_level_m)
    return to_response(p, actual=round(float(row["target_water_level_60min_m"]), 2))


@app.get("/replay/series", summary="(시연) 기간별 실제 수위·예측 수위·강수량 — 그래프용")
def replay_series(start: datetime = Query(..., examples=["2026-07-09T00:00"]),
                  hours: int = Query(24, ge=1, le=96)):
    if not replay:
        raise HTTPException(status_code=503, detail="재생용 CSV가 없습니다.")
    from datetime import timedelta
    rows = replay.rows(start.replace(tzinfo=None), start.replace(tzinfo=None) + timedelta(hours=hours))
    if rows.empty:
        raise HTTPException(status_code=404, detail="해당 기간 자료가 없습니다.")
    import pandas as pd
    x = pd.DataFrame(rows[predictor.features].astype(float).values, columns=predictor.features)
    pred = (rows["water_level_m"].values + predictor.model.predict(x)).round(2)
    return {
        "station": STATION, "notice": NOTICE, "interval_min": 10,
        "points": [
            {"time": t.strftime("%Y-%m-%dT%H:%M"),
             "water_level_m": round(float(r.water_level_m), 2),
             "rain_mm": float(r.rain_10min_mm),
             "predicted_time": (t + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M"),
             "predicted_level_m": float(pv)}
            for (t, r), pv in zip(rows.iterrows(), pred)
        ],
    }


# ── 실시간 (재생 모드 / 한강홍수통제소) ───────────────────
def official_stage(level: float | None) -> str | None:
    if level is None:
        return None
    for name in ("severe", "warning", "caution", "interest"):
        if level >= OFFICIAL_LEVELS[name]:
            return name
    return "normal"


def _f(v):
    return None if v is None or (isinstance(v, float) and v != v) else round(float(v), 2)


@app.get("/live/status", summary="지금 월촌교 수위 + 1시간 뒤 예측 + 회복 예측 (앱 메인용)")
def live_status(time: datetime | None = Query(None, description="(재생 모드만) 특정 시각으로 보기. 비우면 재생 시계의 '지금'")):
    if time is not None and source.name != "replay":
        raise HTTPException(status_code=400, detail="time 지정은 재생 모드에서만 가능해요.")
    now = time.replace(tzinfo=None) if time else source.now()
    win = source.window(now)
    cur = win.iloc[-1]
    level = _f(cur.water_level_m)
    data_status = "ok" if level is not None else "missing"
    # ① 1시간 뒤 수위 (최근 3시간 필요)
    obs = [{"time": t.to_pydatetime(), "water_level_m": _f(r.water_level_m), "rain_mm": _f(r.rain_10min_mm)}
           for t, r in win.tail(18).iterrows()]
    try:
        p = predictor.predict_observations(obs)
        forecast = {"target_time": p.target_time.strftime("%Y-%m-%dT%H:%M"), "level_m": p.predicted_level_m,
                    "change_m": p.change_m, "trend": p.trend, "source": "estimate", "model_version": predictor.model_version}
    except InputError as e:
        forecast = None
        data_status = "partial" if level is not None else "missing"
        forecast_error = str(e)
    # ② 회복 (최근 24시간 필요)
    rec = recovery_model.assess(win, now).as_dict()
    rain = win.rain_10min_mm
    body = {
        "station": STATION,
        "data_source": source.name,
        "observed_at": now.strftime("%Y-%m-%dT%H:%M"),
        "water_level_m": level,
        "official_stage": official_stage(level),
        "official_levels": OFFICIAL_LEVELS,
        "typical_level_m": 0.82,  # 2023~2026 평균
        "rain_10min_mm": _f(cur.rain_10min_mm),
        "rain_1h_mm": _f(rain.tail(6).sum(min_count=6)),
        "rain_3h_mm": _f(rain.tail(18).sum(min_count=18)),
        "forecast_1h": forecast,
        "recovery": rec,
        "data_status": data_status,
        "notice": NOTICE,
    }
    if forecast is None:
        body["forecast_error"] = forecast_error
    logger.info("상태 조회 | 소스=%s | 관측=%s | 데이터=%s | 수위=%s | 1시간 예측=%s | 회복=%s",
                source.name, body["observed_at"], data_status, level,
                None if forecast is None else forecast["level_m"], rec["status"])
    return body


@app.get("/live/series", summary="최근 N시간 월촌교 수위·모현 강수량 (그래프용)")
def live_series(hours: int = Query(24, ge=1, le=26),
                time: datetime | None = Query(None, description="(재생 모드만) 기준 시각")):
    if time is not None and source.name != "replay":
        raise HTTPException(status_code=400, detail="time 지정은 재생 모드에서만 가능해요.")
    now = time.replace(tzinfo=None) if time else source.now()
    win = source.window(now, hours)
    return {"station": STATION, "data_source": source.name, "interval_min": 10, "official_levels": OFFICIAL_LEVELS,
            "points": [{"time": t.strftime("%Y-%m-%dT%H:%M"), "water_level_m": _f(r.water_level_m),
                        "rain_mm": _f(r.rain_10min_mm)} for t, r in win.iterrows()]}


class RecoveryRequest(BaseModel):
    observations: list[Observation] = Field(..., min_length=HISTORY_STEPS,
        description=f"기준 시각까지 10분 간격 관측값 최소 {HISTORY_STEPS}개(24시간+10분). 수위·강수량 모두 필요")


@app.post("/recovery", summary="최근 24시간 관측값으로 수위 회복(1.5m 아래) 예측 — 백엔드용")
def recovery(req: RecoveryRequest):
    import pandas as pd
    df = pd.DataFrame([{"timestamp": o.time.replace(tzinfo=None, second=0, microsecond=0),
                        "water_level_m": o.water_level_m, "rain_10min_mm": o.rain_mm} for o in req.observations])
    df = df.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index().astype(float)
    if (df.index.minute % 10 != 0).any():
        raise HTTPException(status_code=422, detail="관측 시각은 10분 단위여야 해요.")
    df = df.asfreq("10min")
    return recovery_model.assess(df, df.index.max().to_pydatetime()).as_dict()
