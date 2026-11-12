"""실시간(재생)·회복 API 테스트 — python -m pytest -q"""
import json
from datetime import datetime, timedelta

import pandas as pd
from fastapi.testclient import TestClient

from app import live
from app.main import app

client = TestClient(app)


def test_health_shows_source():
    h = client.get("/health").json()
    assert h["data_source"] == "replay" and h["recovery_model"] == "recovery-v0.5"


def test_live_status_rising_storm():
    r = client.get("/live/status", params={"time": "2026-07-09T12:00"}).json()
    assert r["water_level_m"] == 1.57 and r["official_stage"] == "interest"
    assert r["forecast_1h"]["level_m"] == 2.01 and r["forecast_1h"]["trend"] == "rising"
    assert r["recovery"]["status"] == "raining" and r["recovery"]["eta_hours_at_least"] is None


def test_live_status_falling_gives_eta():
    r = client.get("/live/status", params={"time": "2026-07-10T01:00"}).json()
    assert r["recovery"]["status"] == "predicted"
    assert r["recovery"]["eta_hours_at_least"] > 0 and "산책로 이용 가능 시각 아님" in r["recovery"]["meaning"]


def test_live_status_normal_day():
    r = client.get("/live/status", params={"time": "2026-11-04T12:00"}).json()
    assert r["official_stage"] == "normal" and r["recovery"]["status"] == "below_threshold"
    assert r["data_status"] == "ok"


def test_live_status_missing_rain_is_partial():
    # 2023-11-05 강수량 '-' 결측 구간 → 0으로 채우지 않고 예측 보류
    r = client.get("/live/status", params={"time": "2023-11-05T12:00"}).json()
    assert r["forecast_1h"] is None and r["data_status"] == "partial"


def test_live_series_and_recovery_post():
    s = client.get("/live/series", params={"time": "2026-07-10T01:00", "hours": 24}).json()
    assert len(s["points"]) == 145
    obs = [{"time": p["time"], "water_level_m": p["water_level_m"], "rain_mm": p["rain_mm"]} for p in s["points"]]
    a = client.post("/recovery", json={"observations": obs}).json()
    b = client.get("/live/status", params={"time": "2026-07-10T01:00"}).json()["recovery"]
    assert a == b


def test_hrfco_source_parses_and_keeps_missing(monkeypatch):
    monkeypatch.setenv("HRFCO_API_KEY", "TEST")
    src = live.HrfcoSource()
    end = src.now()
    times = [end - timedelta(minutes=10 * i) for i in range(3)]
    wl = {"content": [{"ymdhm": t.strftime("%Y%m%d%H%M"), "wl": "0.61"} for t in times]}
    rf = {"content": [{"ymdhm": times[0].strftime("%Y%m%d%H%M"), "rf": "-"}, {"ymdhm": times[1].strftime("%Y%m%d%H%M"), "rf": "1.0"}]}
    monkeypatch.setattr(src, "_get", lambda tpl, code, s, e: (wl if "waterlevel" in tpl else rf)["content"])
    src.refresh()
    w = src.window(end, 1)
    assert src.last_error is None
    assert w.water_level_m.iloc[-1] == 0.61 and pd.isna(w.rain_10min_mm.iloc[-1]) and w.rain_10min_mm.iloc[-2] == 1.0
