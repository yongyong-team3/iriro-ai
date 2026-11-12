"""python -m pytest -q  (ai-server 폴더에서)"""
from datetime import datetime, timedelta

import pandas as pd
from fastapi.testclient import TestClient

from app.main import REPLAY_CSV, app

client = TestClient(app)
BASE = datetime(2026, 7, 9, 12, 0)


def observations_from_csv(base: datetime, n: int = 18):
    df = pd.read_csv(REPLAY_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    obs = []
    for k in range(n - 1, -1, -1):
        t = base - timedelta(minutes=10 * k)
        r = df.loc[t]
        obs.append({"time": t.strftime("%Y-%m-%dT%H:%M"),
                    "water_level_m": float(r.water_level_m), "rain_mm": float(r.rain_10min_mm)})
    return obs


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert len(r.json()["features"]) == 11


def test_predict_matches_replay():
    """원시 관측값으로 계산한 예측 = 학습 CSV 특징으로 한 예측 (특징 계산이 학습 때와 같은지)"""
    a = client.post("/predict", json={"observations": observations_from_csv(BASE)})
    b = client.get("/replay/predict", params={"time": BASE.strftime("%Y-%m-%dT%H:%M")})
    assert a.status_code == 200, a.text
    assert b.status_code == 200, b.text
    a, b = a.json(), b.json()
    assert a["predicted_level_m"] == b["predicted_level_m"]
    assert a["target_time"] == "2026-07-09T13:00"
    assert b["actual_level_m"] is not None
    assert a["source"] == "estimate" and "공식 통제" in a["notice"]


def test_predict_missing_history():
    obs = observations_from_csv(BASE, n=19)
    del obs[5]  # 2시간 10분 전 값 하나 빠짐 (강수량 3시간 창 안)
    r = client.post("/predict", json={"observations": obs})
    assert r.status_code == 422 and "값이 없습니다" in r.json()["detail"]


def test_replay_not_found_and_series():
    assert client.get("/replay/predict", params={"time": "2020-01-01T00:00"}).status_code == 404
    s = client.get("/replay/series", params={"start": "2026-07-09T00:00", "hours": 6})
    assert s.status_code == 200 and len(s.json()["points"]) == 37
