from datetime import datetime, timedelta

from app.safetydata import SafetyDataSource


def test_water_station_page_search_and_centimeter_conversion(monkeypatch):
    monkeypatch.setenv("SAFETYDATA_WATER_KEY", "test-water")
    monkeypatch.setenv("SAFETYDATA_RAIN_KEY", "test-rain")
    source = SafetyDataSource()
    at = datetime.now().replace(second=0, microsecond=0)
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
    now = datetime.now().replace(second=0, microsecond=0)
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
    assert source.window(datetime.now(), 1).isna().all().all()
    assert source.last_error == "water: TimeoutError; rain: TimeoutError"
