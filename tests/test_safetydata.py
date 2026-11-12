from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.safetydata import SafetyDataSource, _kst_now


def test_water_station_page_search_and_centimeter_conversion(monkeypatch):
    monkeypatch.setenv("SAFETYDATA_WATER_KEY", "test-water")
    monkeypatch.setenv("SAFETYDATA_RAIN_KEY", "test-rain")
    source = SafetyDataSource()
    at = _kst_now().replace(second=0, microsecond=0)
    at -= timedelta(minutes=at.minute % 10)
    stamp = at.strftime("%Y%m%d%H%M")
    pages = {
        1: [{"WALE_OBSVTR_CD": "1016606", "YMD_HM": stamp, "WALE_DATA": "70"}],
        2: [{"WALE_OBSVTR_CD": "1016607", "YMD_HM": stamp, "WALE_DATA": "51"}],
        3: [{"WALE_OBSVTR_CD": "1016608", "YMD_HM": stamp, "WALE_DATA": "90"}],
    }
    monkeypatch.setattr(source, "_page", lambda kind, number, date=None: {
        "totalCount": 3000, "body": pages[number] if kind == "water" else []})
    monkeypatch.setattr(source, "_rain_rows", lambda start, end: [])
    source.refresh()
    assert source.now() == at
    assert source.window(at, 1).water_level_m.iloc[-1] == 0.51
    assert source.window(at, 1).rain_10min_mm.isna().all()


def test_rain_missing_does_not_become_zero(monkeypatch):
    monkeypatch.setenv("SAFETYDATA_WATER_KEY", "test-water")
    monkeypatch.setenv("SAFETYDATA_RAIN_KEY", "test-rain")
    source = SafetyDataSource()
    now = _kst_now().replace(second=0, microsecond=0)
    now -= timedelta(minutes=now.minute % 10)
    old = (now - timedelta(days=2)).strftime("%Y%m%d%H%M")
    monkeypatch.setattr(source, "_page", lambda kind, number, date=None: {
        "totalCount": 1000, "body": [{"WALE_OBSVTR_CD": "10164050", "YMD_HM": old, "THEX_DATA": "0"}]})
    assert source._rain_rows(now - timedelta(hours=28), now) == []


def test_upstream_failure_keeps_empty_cache_and_reports_error(monkeypatch):
    monkeypatch.setenv("SAFETYDATA_WATER_KEY", "test-water")
    monkeypatch.setenv("SAFETYDATA_RAIN_KEY", "test-rain")
    source = SafetyDataSource()

    def fail(*_args, **_kwargs):
        raise TimeoutError("request failed")

    monkeypatch.setattr(source, "_page", fail)
    source.refresh()
    assert source.last_water_at is None
    assert source.last_rain_at is None
    assert source.window(_kst_now(), 1).isna().all().all()
    assert source.last_error == "water: TimeoutError; rain: TimeoutError"


def test_refresh_uses_kst_observation_window(monkeypatch):
    monkeypatch.setenv("SAFETYDATA_WATER_KEY", "test-water")
    monkeypatch.setenv("SAFETYDATA_RAIN_KEY", "test-rain")
    source = SafetyDataSource()
    kst = datetime(2026, 11, 12, 22, 10)
    requested = {}
    monkeypatch.setattr("app.safetydata._kst_now", lambda: kst)

    def water_rows(start, end):
        requested.update(start=start, end=end)
        return [{"WALE_OBSVTR_CD": "1016607", "YMD_HM": "202611122200", "WALE_DATA": 54}]

    monkeypatch.setattr(source, "_water_rows", water_rows)
    monkeypatch.setattr(source, "_rain_rows", lambda start, end: [])
    source.refresh()
    assert requested["end"] == kst
    assert source.last_water_at == datetime(2026, 11, 12, 22)
    assert source.last_refresh == kst


def test_successful_stale_rain_page_does_not_invent_zero(monkeypatch):
    monkeypatch.setenv("SAFETYDATA_WATER_KEY", "test-water")
    monkeypatch.setenv("SAFETYDATA_RAIN_KEY", "test-rain")
    source = SafetyDataSource()
    kst = datetime(2026, 11, 12, 22, 10)
    monkeypatch.setattr("app.safetydata._kst_now", lambda: kst)
    monkeypatch.setattr(source, "_water_rows", lambda start, end: [
        {"WALE_OBSVTR_CD": "1016607", "YMD_HM": "202611122200", "WALE_DATA": 54}])
    monkeypatch.setattr(source, "_page", lambda kind, number, date=None: {
        "totalCount": 1000,
        "body": [
            {"WALE_OBSVTR_CD": "10164050", "YMD_HM": "202611100410", "THEX_DATA": 0},
            {"WALE_OBSVTR_CD": "20154070", "YMD_HM": "202611122200", "THEX_DATA": 0},
        ],
    })
    source.refresh()
    assert source.last_error is None
    assert source.last_rain_at is None
    assert source.window(kst, 1).rain_10min_mm.isna().all()


def test_health_staleness_uses_kst_even_if_host_clock_is_utc(monkeypatch):
    from app import main

    fixed_utc = datetime(2026, 11, 12, 13)

    class UtcHostDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            instant = fixed_utc.replace(tzinfo=timezone.utc)
            return instant.astimezone(tz) if tz else fixed_utc

    observed = datetime(2026, 11, 12, 21)
    monkeypatch.setattr(main, "datetime", UtcHostDatetime)
    monkeypatch.setattr(main, "source", SimpleNamespace(
        name="safetydata", last_water_at=observed, last_rain_at=None,
        last_refresh=observed, last_error=None, now=lambda: observed))
    assert main.health()["status"] == "degraded"
