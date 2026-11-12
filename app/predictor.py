"""XGBoost 모델 로드 + 특징 계산 + 1시간 뒤 수위 예측.

모델은 '1시간 동안의 수위 변화량'을 예측한다.
1시간 뒤 수위 = round(현재 수위 + 모델 출력, 2)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "model" / "wolchon_xgb_60min.json"
FEATURES_PATH = BASE / "model" / "features.json"
STEP = timedelta(minutes=10)
WATER_HISTORY = 13   # 현재 ~ 120분 전 (10분 간격 13개)
RAIN_HISTORY = 18    # 최근 3시간 (10분 간격 18개)
TREND_THRESHOLD_M = 0.02  # 2cm 이상 변하면 상승/하강


class InputError(ValueError):
    """요청 데이터가 예측에 부족하거나 잘못된 경우"""


@dataclass
class Prediction:
    base_time: datetime
    current_level_m: float
    predicted_level_m: float
    features: dict

    @property
    def target_time(self) -> datetime:
        return self.base_time + timedelta(hours=1)

    @property
    def change_m(self) -> float:
        return round(self.predicted_level_m - self.current_level_m, 2)

    @property
    def trend(self) -> str:
        if self.change_m >= TREND_THRESHOLD_M:
            return "rising"
        if self.change_m <= -TREND_THRESHOLD_M:
            return "falling"
        return "steady"


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH, features_path: Path = FEATURES_PATH):
        meta = json.loads(Path(features_path).read_text(encoding="utf-8"))
        self.features: list[str] = meta["features"]
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        self.model_version = f"{Path(model_path).stem} {meta.get('version', 'v1')}".strip()

    # ── 특징 계산 ─────────────────────────────────────────
    @staticmethod
    def build_features(observations: list[dict]) -> tuple[datetime, dict]:
        """10분 간격 관측값 목록 → 모델 입력 특징.

        observations: [{"time": datetime, "water_level_m": float|None, "rain_mm": float|None}, ...]
        - 마지막 시각이 기준 시각(현재)
        - 수위: 기준 시각 포함 최근 13개(120분 전까지) 필요
        - 강수량: 기준 시각 포함 최근 18개(3시간) 필요
        """
        if not observations:
            raise InputError("관측값이 없습니다.")
        by_time: dict[datetime, dict] = {}
        for o in observations:
            t = o["time"].replace(second=0, microsecond=0, tzinfo=None)
            if t.minute % 10:
                raise InputError(f"관측 시각은 10분 단위여야 합니다: {t:%Y-%m-%d %H:%M}")
            by_time[t] = o
        base = max(by_time)

        def series(key: str, n: int) -> list[float]:
            vals = []
            for k in range(n):
                t = base - STEP * k
                o = by_time.get(t)
                v = None if o is None else o.get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    raise InputError(f"{t:%Y-%m-%d %H:%M}의 {key} 값이 없습니다. "
                                     f"기준 시각 기준 최근 {n}개(10분 간격)가 모두 필요합니다.")
                vals.append(float(v))
            return vals  # [현재, 10분 전, 20분 전, ...]

        w = series("water_level_m", WATER_HISTORY)
        r = series("rain_mm", RAIN_HISTORY)
        if any(x < 0 for x in r):
            raise InputError("강수량은 0 이상이어야 합니다.")
        if any(not (-2.0 <= x <= 20.0) for x in w):
            raise InputError("수위 값이 정상 범위(-2~20m)를 벗어났습니다.")

        feats = {
            "water_level_m": w[0],
            "water_lag_10min_m": w[1],
            "water_lag_30min_m": w[3],
            "water_lag_60min_m": w[6],
            "water_lag_120min_m": w[12],
            "water_change_60min_m": round(w[0] - w[6], 4),
            "water_change_10min_m": round(w[0] - w[1], 4),
            "rain_10min_mm": r[0],
            "rain_sum_30min_mm": sum(r[:3]),
            "rain_sum_60min_mm": sum(r[:6]),
            "rain_sum_180min_mm": sum(r[:18]),
        }
        return base, feats

    # ── 예측 ─────────────────────────────────────────────
    def predict_features(self, base_time: datetime, feats: dict) -> Prediction:
        missing = [f for f in self.features if f not in feats]
        if missing:
            raise InputError(f"특징 값이 빠졌습니다: {missing}")
        x = pd.DataFrame([[float(feats[f]) for f in self.features]], columns=self.features)
        delta = float(self.model.predict(x)[0])
        current = float(feats["water_level_m"])
        return Prediction(base_time=base_time, current_level_m=round(current, 2),
                          predicted_level_m=round(current + delta, 2),
                          features={f: feats[f] for f in self.features})

    def predict_observations(self, observations: list[dict]) -> Prediction:
        base, feats = self.build_features(observations)
        return self.predict_features(base, feats)


class ReplayData:
    """시연용: 과거 학습 데이터 CSV에서 특정 시각의 입력과 실제 1시간 뒤 수위를 꺼냄."""

    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path, parse_dates=["timestamp", "target_time"])
        if "water_change_10min_m" not in df.columns:  # v2 모델 추가 특징
            df["water_change_10min_m"] = df["water_level_m"] - df["water_lag_10min_m"]
        self.df = df.set_index("timestamp").sort_index()

    @property
    def range(self) -> tuple[datetime, datetime]:
        return self.df.index.min().to_pydatetime(), self.df.index.max().to_pydatetime()

    def row(self, t: datetime) -> pd.Series:
        t = pd.Timestamp(t).floor("10min")
        if t not in self.df.index:
            raise KeyError(t)
        return self.df.loc[t]

    def rows(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self.df.loc[pd.Timestamp(start).floor("10min"): pd.Timestamp(end)]
