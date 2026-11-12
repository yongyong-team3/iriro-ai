"""수위 회복 예측 v0.5 — 서비스용 계산

예측 대상: 월촌교 수위가 1.5m(한강홍수통제소 '관심' 수위) 이하로 내려가 2시간 유지되는 시점
  - 6·12시간 안 회복 확률 (XGBoost 분류)
  - 조건부 예상 시각 (XGBoost 회귀, log1p 학습 → expm1, 0~24h)
  - 24시간 확률은 검증되지 않아 제공하지 않음

⚠ 산책로 이용 가능 시각이 아님. 1.5m는 산책로 침수 기준으로 검증되지 않았음.

표시 규칙 (2026년 시험 결과 분석 반영)
  - 수위가 1.5m 초과 + 최근 1시간 하강 중일 때만 예상 시각 제공 (상승 중엔 오차가 큼)
  - 최근 1시간 비가 오면 예상 시각을 숨기고 경고 (재강우 시 회복을 이르게 예측하는 경향)
  - 예측이 6시간을 넘으면 '반나절 이상 걸릴 수 있음' 문구
  - 시각은 '빨라도 약 N시간' (긴 회복을 짧게 예측하는 경향이 있음)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "model" / "recovery_v05"
THRESHOLD_M = 1.5
HISTORY_STEPS = 24 * 6 + 1  # 24시간 전 값까지 필요 (145개)
FEATURES = [
    "water_level_m", "reference_m", "above_ref_m",
    "change_1h_m", "change_3h_m", "change_6h_m", "change_12h_m", "change_24h_m",
    "rain_past_1h_mm", "rain_past_3h_mm", "rain_past_6h_mm", "rain_past_12h_mm",
    "drop_from_peak_6h_m",
]
FALLING_M_PER_H = -0.01   # 1시간에 1cm 이상 내려가면 '하강 중'
RAIN_HOLD_MM_1H = 1.0     # 최근 1시간 1mm 이상 비 → 예상 시각 숨김
LONG_ETA_H = 6.0


def compute_features(series: pd.DataFrame, threshold_m: float = THRESHOLD_M) -> pd.DataFrame:
    """10분 간격 시계열(water_level_m, rain_10min_mm) → 시점별 13개 특징. 학습 코드(yong_features)와 같은 계산."""
    z = series[["water_level_m", "rain_10min_mm"]].copy()
    z["reference_m"] = float(threshold_m)
    z["above_ref_m"] = z.water_level_m - z.reference_m
    for h in (1, 3, 6, 12, 24):
        z[f"change_{h}h_m"] = z.water_level_m - z.water_level_m.shift(h * 6)
    for h in (1, 3, 6, 12):
        z[f"rain_past_{h}h_mm"] = z.rain_10min_mm.rolling(h * 6, min_periods=h * 6).sum()
    z["drop_from_peak_6h_m"] = z.water_level_m.rolling(36, min_periods=36).max() - z.water_level_m
    return z


@dataclass
class RecoveryResult:
    status: str                  # 'below_threshold' | 'rising' | 'raining' | 'predicted' | 'insufficient_data'
    message: str
    p_6h: float | None = None
    p_12h: float | None = None
    eta_hours: float | None = None
    eta_time: str | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status, "message": self.message,
            "threshold_m": THRESHOLD_M,
            "probability_within_6h": None if self.p_6h is None else round(self.p_6h, 2),
            "probability_within_12h": None if self.p_12h is None else round(self.p_12h, 2),
            "eta_hours_at_least": self.eta_hours,
            "eta_time_at_least": self.eta_time,
            "meaning": "월촌교 수위가 관심 수위 1.5m 아래로 내려가 2시간 유지되는 시점 (산책로 이용 가능 시각 아님)",
            "source": "estimate",
            "model_version": "recovery-v0.5",
        }


class RecoveryModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.m6 = xgb.XGBClassifier(); self.m6.load_model(str(model_dir / "recovery_6h_xgb.json"))
        self.m12 = xgb.XGBClassifier(); self.m12.load_model(str(model_dir / "recovery_12h_xgb.json"))
        self.eta = xgb.XGBRegressor(); self.eta.load_model(str(model_dir / "conditional_eta_xgb.json"))

    def raw(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = feats[FEATURES].astype(float)
        p6 = self.m6.predict_proba(x)[:, 1]
        p12 = np.maximum(p6, self.m12.predict_proba(x)[:, 1])  # 학습 코드와 같이 6h ≤ 12h 보장
        eta = np.clip(np.expm1(self.eta.predict(x)), 0, 24)
        return p6, p12, eta

    def assess(self, series: pd.DataFrame, at: datetime) -> RecoveryResult:
        """series: 10분 간격, at까지의 최근 24시간 이상. at 시점 기준 판단."""
        at = pd.Timestamp(at).floor("10min")
        window = series.loc[at - timedelta(minutes=10 * (HISTORY_STEPS - 1)): at]
        level_now = window.water_level_m.get(at)
        if level_now is None or np.isnan(level_now):
            return RecoveryResult("insufficient_data", "현재 수위 자료가 없어요")
        if level_now <= THRESHOLD_M:
            return RecoveryResult("below_threshold", f"수위가 관심 기준({THRESHOLD_M}m) 아래예요")
        if len(window) < HISTORY_STEPS or window.isna().any().any():
            return RecoveryResult("insufficient_data", "최근 24시간 관측 자료가 부족해 회복 예측을 할 수 없어요")
        f = compute_features(window).iloc[[-1]]
        if f[FEATURES].isna().any(axis=1).iloc[0]:
            return RecoveryResult("insufficient_data", "최근 24시간 관측 자료가 부족해 회복 예측을 할 수 없어요")
        p6, p12, eta = (v[0] for v in self.raw(f))
        change_1h = float(f.change_1h_m.iloc[0]); rain_1h = float(f.rain_past_1h_mm.iloc[0])
        if rain_1h >= RAIN_HOLD_MM_1H:
            return RecoveryResult("raining", "비가 내리고 있어 회복 시점을 예측하기 어려워요", float(p6), float(p12))
        if change_1h > FALLING_M_PER_H:
            return RecoveryResult("rising", "수위가 아직 내려가지 않고 있어요", float(p6), float(p12))
        eta_h = round(float(eta), 1)
        when = (at + timedelta(hours=eta_h)).strftime("%Y-%m-%dT%H:%M")
        if eta_h > LONG_ETA_H:
            msg = "관심 수위 아래로 내려가기까지 반나절 이상 걸릴 수 있어요"
        else:
            msg = f"빨라도 약 {max(1, round(eta_h))}시간 뒤 관심 수위(1.5m) 아래로 내려갈 것으로 예상돼요. 비가 다시 오면 더 늦어질 수 있어요"
        return RecoveryResult("predicted", msg, float(p6), float(p12), eta_h, when)
