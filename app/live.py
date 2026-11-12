"""실시간 관측 수집기 — 월촌교 수위(1016607) + 모현면사무소 강수량(10164050)

DATA_SOURCE 환경변수로 바꿔 끼움
  replay (기본) : 과거 10분 자료(data/observations_10min.csv)를 '지금'처럼 흘려보냄. API 키 없이 시연·개발용
  safetydata    : 재난안전데이터공유플랫폼 수위·우량 API (각 서비스 키 필요)
  hrfco         : 한강홍수통제소 Open API (HRFCO_API_KEY 필요)

재생 시계: 서버가 켜진 순간 = REPLAY_START, 실제 1분마다 REPLAY_SPEED × 1분씩 진행
  예) REPLAY_START=2026-07-09T06:00, REPLAY_SPEED=10 → 실제 6분에 데이터 1시간 진행
"""
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
WATER_CODE = "1016607"
RAIN_CODE = "10164050"
KEEP_HOURS = 26  # 회복 예측에 24시간 + 여유
logger = logging.getLogger("uvicorn.error")


def _floor10(t: datetime) -> datetime:
    return t.replace(second=0, microsecond=0) - timedelta(minutes=t.minute % 10)


class ReplaySource:
    name = "replay"

    def __init__(self, csv_path: Path | None = None, start: str | None = None, speed: float | None = None):
        path = csv_path or Path(os.getenv("REPLAY_OBS_CSV", BASE / "data" / "observations_10min.csv"))
        self.df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        self.start = pd.Timestamp(start or os.getenv("REPLAY_START", "2026-07-09T06:00")).to_pydatetime()
        self.speed = float(speed if speed is not None else os.getenv("REPLAY_SPEED", "10"))
        self.t0 = datetime.now()
        logger.info("데이터 소스: replay | 시작=%s | 속도=%s | 관측 행=%d", self.start, self.speed, len(self.df))

    def now(self) -> datetime:
        elapsed = (datetime.now() - self.t0) * self.speed
        t = _floor10(self.start + elapsed)
        return min(t, self.df.index.max().to_pydatetime())

    def range(self):
        return self.df.index.min().to_pydatetime(), self.df.index.max().to_pydatetime()

    def window(self, end: datetime, hours: int = KEEP_HOURS) -> pd.DataFrame:
        end = _floor10(end)
        idx = pd.date_range(end - timedelta(hours=hours), end, freq="10min")
        return self.df.reindex(idx)[["water_level_m", "rain_10min_mm"]]


class HrfcoSource:
    """한강홍수통제소 Open API.

    ⚠ API 키를 받은 뒤 실제 응답 형식으로 확인·수정이 필요합니다.
    아래 주소 형식과 응답 필드(content[].ymdhm / wl / rf)는 한강홍수통제소 Open API 안내 기준의 가정이에요.
    환경변수 HRFCO_WL_URL, HRFCO_RF_URL 로 주소 형식을 바꿀 수 있어요.
    """
    name = "hrfco"
    WL_URL = "https://api.hrfco.go.kr/{key}/waterlevel/list/10M/{code}/{start}/{end}.json"
    RF_URL = "https://api.hrfco.go.kr/{key}/rainfall/list/10M/{code}/{start}/{end}.json"

    def __init__(self):
        self.key = os.getenv("HRFCO_API_KEY", "")
        if not self.key:
            raise RuntimeError("HRFCO_API_KEY 환경변수가 없어요. .env에 키를 넣거나 DATA_SOURCE=replay 로 실행하세요.")
        self.wl_url = os.getenv("HRFCO_WL_URL", self.WL_URL)
        self.rf_url = os.getenv("HRFCO_RF_URL", self.RF_URL)
        self.cache = pd.DataFrame(columns=["water_level_m", "rain_10min_mm"], dtype=float)
        self.last_error: str | None = None
        self.lock = threading.Lock()
        logger.info("데이터 소스: hrfco | 수위 관측소=%s | 강수 관측소=%s", WATER_CODE, RAIN_CODE)

    def now(self) -> datetime:
        return _floor10(datetime.now())

    def range(self):
        if self.cache.empty:
            return None, None
        return self.cache.index.min().to_pydatetime(), self.cache.index.max().to_pydatetime()

    def _get(self, template: str, code: str, start: datetime, end: datetime) -> list[dict]:
        url = template.format(key=self.key, code=code, start=start.strftime("%Y%m%d%H%M"), end=end.strftime("%Y%m%d%H%M"))
        with urllib.request.urlopen(url, timeout=15) as r:
            body = json.loads(r.read().decode("utf-8"))
        return body.get("content", []) if isinstance(body, dict) else []

    @staticmethod
    def _num(v):
        try:
            x = float(str(v).strip())
            return x if np.isfinite(x) else np.nan
        except (TypeError, ValueError):
            return np.nan  # '-', 빈 값 → 결측 (0으로 채우지 않음)

    def refresh(self) -> None:
        end = self.now()
        start = end - timedelta(hours=KEEP_HOURS)
        try:
            wl = self._get(self.wl_url, WATER_CODE, start, end)
            rf = self._get(self.rf_url, RAIN_CODE, start, end)
            w = {pd.Timestamp(datetime.strptime(str(x["ymdhm"]), "%Y%m%d%H%M")): self._num(x.get("wl")) for x in wl}
            r = {pd.Timestamp(datetime.strptime(str(x["ymdhm"]), "%Y%m%d%H%M")): self._num(x.get("rf")) for x in rf}
            new = pd.DataFrame({"water_level_m": pd.Series(w, dtype=float), "rain_10min_mm": pd.Series(r, dtype=float)})
            with self.lock:
                merged = pd.concat([self.cache, new])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                self.cache = merged[merged.index >= pd.Timestamp(end - timedelta(hours=KEEP_HOURS))]
                water_count = int(self.cache.water_level_m.notna().sum())
                rain_count = int(self.cache.rain_10min_mm.notna().sum())
            self.last_error = None
            logger.info("HRFCO 수집 완료 | 기준=%s | 수위=%d건 | 강수=%d건 | 캐시 수위=%d건, 강수=%d건",
                        end, len(wl), len(rf), water_count, rain_count)
        except Exception as e:  # 네트워크·형식 오류 → 마지막 정상 자료 유지
            # HTTPError 메시지에 URL 속 API 키가 포함될 수 있으므로 로그와 health 응답에서 가린다.
            detail = str(e).replace(self.key, "[REDACTED]")
            self.last_error = f"{type(e).__name__}: {detail}"
            logger.error("HRFCO 수집 실패 | %s", self.last_error)

    def window(self, end: datetime, hours: int = KEEP_HOURS) -> pd.DataFrame:
        end = _floor10(end)
        idx = pd.date_range(end - timedelta(hours=hours), end, freq="10min")
        with self.lock:
            return self.cache.reindex(idx)[["water_level_m", "rain_10min_mm"]]


def make_source():
    kind = os.getenv("DATA_SOURCE", "replay").lower()
    if kind == "safetydata":
        from .safetydata import SafetyDataSource
        return SafetyDataSource()
    if kind == "hrfco":
        return HrfcoSource()
    if kind == "replay":
        return ReplaySource()
    raise ValueError(f"지원하지 않는 DATA_SOURCE: {kind} (replay, safetydata 또는 hrfco 사용)")
