"""재난안전데이터공유플랫폼의 월촌교 수위·모현 10분 우량 수집기.

플랫폼의 수위 API는 날짜 필터 후 관측소 코드 순서로 페이지를 반환한다.
우량 API는 날짜/관측소 필터가 없어 마지막 페이지부터 제한적으로 읽는다.
누락·오래된 우량은 0으로 채우지 않으며, 이 경우 AI 예측은 보류된다.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .live import KEEP_HOURS, RAIN_CODE, WATER_CODE, _floor10

logger = logging.getLogger("uvicorn.error")
PAGE_SIZE = 1000  # 플랫폼이 허용하는 최대 응답 행 수
MAX_RAIN_PAGES = 12  # 호출 한도 보호: 현재 우량 API는 최근 자료가 지연되고 있음
WATER_URL = "https://www.safetydata.go.kr/V2/api/DSSP-IF-00007"
RAIN_URL = "https://www.safetydata.go.kr/V2/api/DSSP-IF-00006"
KOREA = ZoneInfo("Asia/Seoul")


def _kst_now() -> datetime:
    """플랫폼 관측 시각(YMD_HM)과 같은 한국 시각을 naive datetime으로 반환한다."""
    return datetime.now(KOREA).replace(tzinfo=None)


class SafetyDataSource:
    name = "safetydata"

    def __init__(self):
        self.water_key = os.getenv("SAFETYDATA_WATER_KEY", "")
        self.rain_key = os.getenv("SAFETYDATA_RAIN_KEY", "")
        if not self.water_key or not self.rain_key:
            raise RuntimeError("SAFETYDATA_WATER_KEY와 SAFETYDATA_RAIN_KEY가 필요합니다")
        self.cache = pd.DataFrame(columns=["water_level_m", "rain_10min_mm"],
                                  index=pd.DatetimeIndex([]), dtype=float)
        self.lock = threading.Lock()
        self.last_error: str | None = None
        self.last_refresh: datetime | None = None
        self.last_water_at: datetime | None = None
        self.last_rain_at: datetime | None = None
        logger.info("데이터 소스: safetydata | 수위 관측소=%s | 우량 관측소=%s", WATER_CODE, RAIN_CODE)

    def _page(self, kind: str, number: int, date: str | None = None) -> dict:
        key = self.water_key if kind == "water" else self.rain_key
        base = WATER_URL if kind == "water" else RAIN_URL
        params = {"serviceKey": key, "returnType": "json", "pageNo": number, "numOfRows": PAGE_SIZE}
        if kind == "water" and date:
            params["inqDt"] = date
        url = base + "?" + urllib.parse.urlencode(params)
        # urllib 예외 문자열에는 요청 URL과 키가 포함될 수 있다. 호출자는 종류/상태만 기록한다.
        with urllib.request.urlopen(url, timeout=25) as response:
            payload = json.load(response)
        code = str(payload.get("header", {}).get("resultCode", ""))
        if code != "00":
            raise ValueError(f"{kind} API resultCode={code}")
        if not isinstance(payload.get("body"), list):
            raise ValueError(f"{kind} API body 형식 오류")
        return payload

    @staticmethod
    def _timestamp(row: dict) -> datetime | None:
        try:
            return datetime.strptime(str(row["YMD_HM"]), "%Y%m%d%H%M")
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _number(value) -> float | None:
        try:
            number = float(value)
            return number if np.isfinite(number) else None
        except (TypeError, ValueError):
            return None

    def _water_rows(self, start: datetime, end: datetime) -> list[dict]:
        date = start.strftime("%Y%m%d")
        first = self._page("water", 1, date)
        pages = math.ceil(int(first["totalCount"]) / PAGE_SIZE)
        if not pages:
            return []
        fetched = {1: first}

        def rows(page: int) -> list[dict]:
            if page not in fetched:
                fetched[page] = self._page("water", page, date)
            return fetched[page]["body"]

        low, high = 1, pages
        found = None
        while low <= high:
            middle = (low + high) // 2
            page_rows = rows(middle)
            if not page_rows:
                high = middle - 1
                continue
            first_code = str(page_rows[0].get("WALE_OBSVTR_CD", ""))
            last_code = str(page_rows[-1].get("WALE_OBSVTR_CD", ""))
            if WATER_CODE < first_code:
                high = middle - 1
            elif WATER_CODE > last_code:
                low = middle + 1
            else:
                found = middle
                break
        if found is None:
            return []
        selected = list(rows(found))
        page = found - 1
        while page >= 1 and rows(page) and str(rows(page)[-1].get("WALE_OBSVTR_CD", "")) == WATER_CODE:
            selected = list(rows(page)) + selected
            page -= 1
        page = found + 1
        while page <= pages and rows(page) and str(rows(page)[0].get("WALE_OBSVTR_CD", "")) == WATER_CODE:
            selected += list(rows(page))
            page += 1
        return [row for row in selected if str(row.get("WALE_OBSVTR_CD", "")) == WATER_CODE
                and (at := self._timestamp(row)) is not None and start <= at <= end]

    def _rain_rows(self, start: datetime, end: datetime) -> list[dict]:
        first = self._page("rain", 1)
        pages = math.ceil(int(first["totalCount"]) / PAGE_SIZE)
        selected = []
        for page in range(pages, max(0, pages - MAX_RAIN_PAGES), -1):
            rows = first["body"] if page == 1 else self._page("rain", page)["body"]
            dated = [(row, self._timestamp(row)) for row in rows]
            selected.extend(row for row, at in dated if str(row.get("WALE_OBSVTR_CD", "")) == RAIN_CODE
                            and at is not None and start <= at <= end)
            dates = [at for _, at in dated if at is not None]
            if dates and min(dates) <= start:
                break
        return selected

    @staticmethod
    def _safe_error(exc: Exception, key: str) -> str:
        # HTTPError에 포함될 수 있는 API URL 전체를 출력하지 않는다.
        status = getattr(exc, "code", None)
        return f"{type(exc).__name__}" + (f" HTTP {status}" if status is not None else "")

    def refresh(self) -> None:
        end = _floor10(_kst_now())
        start = end - timedelta(hours=KEEP_HOURS + 2)
        errors = []
        water, rain = {}, {}
        try:
            for row in self._water_rows(start, end):
                at, raw = self._timestamp(row), self._number(row.get("WALE_DATA"))
                if at is not None and raw is not None:
                    water[pd.Timestamp(at)] = raw / 100.0  # CSV와 같은 시각의 51 ↔ 0.51m 검증
        except Exception as exc:
            errors.append("water: " + self._safe_error(exc, self.water_key))
        try:
            for row in self._rain_rows(start, end):
                at, raw = self._timestamp(row), self._number(row.get("THEX_DATA"))
                if at is not None and raw is not None and raw >= 0:
                    rain[pd.Timestamp(at)] = raw
        except Exception as exc:
            errors.append("rain: " + self._safe_error(exc, self.rain_key))

        incoming = pd.DataFrame({"water_level_m": pd.Series(water, dtype=float),
                                 "rain_10min_mm": pd.Series(rain, dtype=float)})
        with self.lock:
            combined = self.cache.combine_first(incoming)
            combined.update(incoming)
            self.cache = combined[combined.index >= pd.Timestamp(start)].sort_index()
            w_index = self.cache.water_level_m.last_valid_index() if not self.cache.empty else None
            r_index = self.cache.rain_10min_mm.last_valid_index() if not self.cache.empty else None
            self.last_water_at = w_index.to_pydatetime() if w_index is not None else None
            self.last_rain_at = r_index.to_pydatetime() if r_index is not None else None
        self.last_refresh = _kst_now()
        self.last_error = "; ".join(errors) or None
        logger.info("SafetyData 수집 | 수위=%d건 최신=%s | 우량=%d건 최신=%s%s",
                    len(water), self.last_water_at, len(rain), self.last_rain_at,
                    f" | 오류={self.last_error}" if self.last_error else "")

    def now(self) -> datetime:
        with self.lock:
            return self.last_water_at or _floor10(_kst_now())

    def range(self):
        with self.lock:
            if self.cache.empty:
                return None, None
            return self.cache.index.min().to_pydatetime(), self.cache.index.max().to_pydatetime()

    def window(self, end: datetime, hours: int = KEEP_HOURS) -> pd.DataFrame:
        end = _floor10(end)
        index = pd.date_range(end - timedelta(hours=hours), end, freq="10min")
        with self.lock:
            return self.cache.reindex(index)[["water_level_m", "rain_10min_mm"]]
