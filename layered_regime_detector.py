from __future__ import annotations

"""Layered market-regime detector.

Layers
------
1. Risk: VIX percentile + credit-stress percentile
2. Trend: price vs long moving average + normalized regression slope
3. Momentum: multi-horizon absolute momentum
4. Volatility structure: VIX term structure when ^VIX3M is available
5. Composite: structured labels plus an optional defensive score

The detector shifts completed signals by one trading day by default so the
result can be used for today's allocation without same-close look-ahead.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:  # Allows build_regimes_from_data() without yfinance installed.
    yf = None

RiskKey = Literal["stable_risk_on", "fragile", "vol_shock", "crisis"]
TrendKey = Literal["bull", "bear", "sideways", "transition"]
MomentumKey = Literal["strong_positive", "positive", "mixed", "negative"]
VolStructureKey = Literal["contango", "flat", "backwardation", "unavailable"]


def _rolling_last_percentile(values: np.ndarray) -> float:
    """Percentile rank of the last observation within a rolling window."""
    return float(pd.Series(values).rank(pct=True).iloc[-1])


def _annualized_log_slope(values: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized OLS slope of log prices, returned as an approximate return."""
    x = np.arange(len(values), dtype=float)
    y = np.log(np.asarray(values, dtype=float))
    if not np.isfinite(y).all() or len(y) < 2:
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    return float(np.expm1(slope * periods_per_year))


def _dominant_label(labels: pd.Series) -> str:
    clean = labels.dropna().astype(str)
    if clean.empty:
        raise RuntimeError("No valid regime labels in dominance window.")
    counts = clean.value_counts()
    top = counts.max()
    tied = set(counts[counts == top].index)
    last = clean.iloc[-1]
    return last if last in tied else str(next(iter(tied)))


@dataclass
class LayeredRegimeDetector:
    """Build interpretable, layered market regimes from liquid ETF proxies."""

    market_ticker: str = "SPY"
    vix_ticker: str = "^VIX"
    vix3m_ticker: str = "^VIX3M"
    high_yield_ticker: str = "HYG"
    investment_grade_ticker: str = "LQD"

    # Risk layer
    risk_lookback: int = 252
    vix_high_pct: float = 0.70
    credit_high_pct: float = 0.70
    credit_mode: str = "ratio"  # ratio | diff | legacy_diff
    risk_ema_span: int = 10

    # Trend layer
    trend_ma_window: int = 200
    trend_slope_window: int = 63
    bull_slope_threshold: float = 0.05
    bear_slope_threshold: float = -0.05
    sideways_band: float = 0.02

    # Momentum layer
    momentum_windows: Tuple[int, ...] = (21, 63, 126, 252)
    strong_momentum_threshold: float = 0.10

    # Volatility structure layer
    contango_threshold: float = 0.98
    backwardation_threshold: float = 1.02

    # General
    shift_signals_by_one_day: bool = True
    dominance_window: int = 20
    download_padding_bdays: int = 60

    RISK_LABEL_TO_KEY: Dict[str, RiskKey] = field(init=False)
    TREND_LABEL_TO_KEY: Dict[str, TrendKey] = field(init=False)
    MOMENTUM_LABEL_TO_KEY: Dict[str, MomentumKey] = field(init=False)
    VOL_LABEL_TO_KEY: Dict[str, VolStructureKey] = field(init=False)

    def __post_init__(self) -> None:
        if self.credit_mode not in {"ratio", "diff", "legacy_diff"}:
            raise ValueError("credit_mode must be ratio, diff, or legacy_diff")
        if self.risk_lookback < 20:
            raise ValueError("risk_lookback must be at least 20")
        if self.trend_ma_window < 20 or self.trend_slope_window < 5:
            raise ValueError("trend windows are too short")
        if not self.momentum_windows or min(self.momentum_windows) < 2:
            raise ValueError("momentum_windows must contain integers >= 2")
        if self.contango_threshold >= self.backwardation_threshold:
            raise ValueError("contango_threshold must be below backwardation_threshold")

        self.RISK_LABEL_TO_KEY = {
            "Stable Risk-On": "stable_risk_on",
            "Fragile": "fragile",
            "Vol Shock": "vol_shock",
            "Crisis": "crisis",
        }
        self.TREND_LABEL_TO_KEY = {
            "Bull": "bull",
            "Bear": "bear",
            "Sideways": "sideways",
            "Transition": "transition",
        }
        self.MOMENTUM_LABEL_TO_KEY = {
            "Strong Positive": "strong_positive",
            "Positive": "positive",
            "Mixed": "mixed",
            "Negative": "negative",
        }
        self.VOL_LABEL_TO_KEY = {
            "Contango": "contango",
            "Flat": "flat",
            "Backwardation": "backwardation",
            "Unavailable": "unavailable",
        }

    @property
    def tickers(self) -> Tuple[str, ...]:
        return (
            self.market_ticker,
            self.vix_ticker,
            self.vix3m_ticker,
            self.high_yield_ticker,
            self.investment_grade_ticker,
        )

    def fetch_data(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Download adjusted closes. ^VIX3M is optional; core symbols are required."""
        if yf is None:
            raise ImportError(
                "yfinance is required for fetch_data(). Install it with: "
                "pip install yfinance pandas numpy"
            )

        raw = yf.download(
            list(self.tickers),
            start=pd.to_datetime(start_date).date(),
            end=pd.to_datetime(end_date).date(),
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no data for the requested period.")

        if isinstance(raw.columns, pd.MultiIndex):
            field_name = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
            data = raw[field_name].copy()
        else:
            data = raw.copy()

        required = {
            self.market_ticker,
            self.vix_ticker,
            self.high_yield_ticker,
            self.investment_grade_ticker,
        }
        missing = required.difference(data.columns)
        if missing:
            raise RuntimeError(f"Missing required market data: {sorted(missing)}")

        # Do not drop all rows because the optional VIX3M series may have gaps.
        data = data.sort_index().dropna(subset=list(required))
        if data.empty:
            raise RuntimeError("No rows remain after aligning required symbols.")
        return data

    def _credit_stress(self, data: pd.DataFrame) -> pd.Series:
        hyg = data[self.high_yield_ticker]
        lqd = data[self.investment_grade_ticker]
        if self.credit_mode == "ratio":
            return lqd / hyg  # higher = worse credit
        if self.credit_mode == "diff":
            return lqd - hyg  # higher = worse credit
        return hyg - lqd  # retained only for backward compatibility

    def build_regimes_from_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build all layers from an already-downloaded price DataFrame."""
        out = data.copy().sort_index()
        market = out[self.market_ticker]

        # ---------- Risk ----------
        out["CreditStress"] = self._credit_stress(out)
        out["VIXPct"] = out[self.vix_ticker].rolling(self.risk_lookback).apply(
            _rolling_last_percentile, raw=True
        )
        out["CreditStressPct"] = out["CreditStress"].rolling(self.risk_lookback).apply(
            _rolling_last_percentile, raw=True
        )
        out["VIXPctEMA"] = out["VIXPct"].ewm(
            span=self.risk_ema_span, adjust=False, min_periods=self.risk_ema_span
        ).mean()
        out["CreditStressPctEMA"] = out["CreditStressPct"].ewm(
            span=self.risk_ema_span, adjust=False, min_periods=self.risk_ema_span
        ).mean()

        # ---------- Trend ----------
        out["TrendMA"] = market.rolling(self.trend_ma_window).mean()
        out["PriceToTrendMA"] = market / out["TrendMA"] - 1.0
        out["TrendSlopeAnn"] = market.rolling(self.trend_slope_window).apply(
            _annualized_log_slope, raw=True
        )

        # ---------- Momentum ----------
        momentum_columns = []
        for window in self.momentum_windows:
            col = f"Momentum_{window}d"
            out[col] = market.pct_change(window)
            momentum_columns.append(col)
        out["MomentumAverage"] = out[momentum_columns].mean(axis=1)
        out["MomentumPositiveShare"] = (out[momentum_columns] > 0).mean(axis=1)

        # ---------- VIX term structure ----------
        if self.vix3m_ticker in out.columns:
            out["VIXTermRatio"] = out[self.vix_ticker] / out[self.vix3m_ticker]
        else:
            out["VIXTermRatio"] = np.nan

        signal_cols = [
            "VIXPctEMA",
            "CreditStressPctEMA",
            "PriceToTrendMA",
            "TrendSlopeAnn",
            "MomentumAverage",
            "MomentumPositiveShare",
            "VIXTermRatio",
        ]
        sig = out[signal_cols].shift(1) if self.shift_signals_by_one_day else out[signal_cols]

        # Risk labels
        valid_risk = sig[["VIXPctEMA", "CreditStressPctEMA"]].notna().all(axis=1)
        vol_high = sig["VIXPctEMA"] > self.vix_high_pct
        credit_high = sig["CreditStressPctEMA"] > self.credit_high_pct
        out["RiskRegime"] = pd.Series(index=out.index, dtype="object")
        out.loc[valid_risk, "RiskRegime"] = "Stable Risk-On"
        out.loc[valid_risk & ~vol_high & credit_high, "RiskRegime"] = "Fragile"
        out.loc[valid_risk & vol_high & ~credit_high, "RiskRegime"] = "Vol Shock"
        out.loc[valid_risk & vol_high & credit_high, "RiskRegime"] = "Crisis"

        # Trend labels
        valid_trend = sig[["PriceToTrendMA", "TrendSlopeAnn"]].notna().all(axis=1)
        above = sig["PriceToTrendMA"] > self.sideways_band
        below = sig["PriceToTrendMA"] < -self.sideways_band
        slope_bull = sig["TrendSlopeAnn"] >= self.bull_slope_threshold
        slope_bear = sig["TrendSlopeAnn"] <= self.bear_slope_threshold
        near_ma = sig["PriceToTrendMA"].abs() <= self.sideways_band
        weak_slope = sig["TrendSlopeAnn"].abs() < max(
            abs(self.bull_slope_threshold), abs(self.bear_slope_threshold)
        )

        out["TrendRegime"] = pd.Series(index=out.index, dtype="object")
        out.loc[valid_trend, "TrendRegime"] = "Transition"
        out.loc[valid_trend & above & slope_bull, "TrendRegime"] = "Bull"
        out.loc[valid_trend & below & slope_bear, "TrendRegime"] = "Bear"
        out.loc[valid_trend & near_ma & weak_slope, "TrendRegime"] = "Sideways"

        # Momentum labels
        valid_mom = sig[["MomentumAverage", "MomentumPositiveShare"]].notna().all(axis=1)
        avg_mom = sig["MomentumAverage"]
        positive_share = sig["MomentumPositiveShare"]
        out["MomentumRegime"] = pd.Series(index=out.index, dtype="object")
        out.loc[valid_mom, "MomentumRegime"] = "Mixed"
        out.loc[valid_mom & (avg_mom > 0) & (positive_share >= 0.75), "MomentumRegime"] = "Positive"
        out.loc[
            valid_mom
            & (avg_mom >= self.strong_momentum_threshold)
            & (positive_share == 1.0),
            "MomentumRegime",
        ] = "Strong Positive"
        out.loc[valid_mom & (avg_mom < 0) & (positive_share <= 0.25), "MomentumRegime"] = "Negative"

        # Volatility-structure labels
        ratio = sig["VIXTermRatio"]
        out["VolStructureRegime"] = "Unavailable"
        available = ratio.notna()
        out.loc[available, "VolStructureRegime"] = "Flat"
        out.loc[available & (ratio < self.contango_threshold), "VolStructureRegime"] = "Contango"
        out.loc[available & (ratio > self.backwardation_threshold), "VolStructureRegime"] = "Backwardation"

        # Keys and composite representation
        out["RiskKey"] = out["RiskRegime"].map(self.RISK_LABEL_TO_KEY)
        out["TrendKey"] = out["TrendRegime"].map(self.TREND_LABEL_TO_KEY)
        out["MomentumKey"] = out["MomentumRegime"].map(self.MOMENTUM_LABEL_TO_KEY)
        out["VolStructureKey"] = out["VolStructureRegime"].map(self.VOL_LABEL_TO_KEY)

        required_labels = ["RiskRegime", "TrendRegime", "MomentumRegime"]
        valid_composite = out[required_labels].notna().all(axis=1)
        out["CompositeRegime"] = pd.Series(index=out.index, dtype="object")
        out.loc[valid_composite, "CompositeRegime"] = (
            out.loc[valid_composite, "RiskRegime"].astype(str)
            + " | "
            + out.loc[valid_composite, "TrendRegime"].astype(str)
            + " | "
            + out.loc[valid_composite, "MomentumRegime"].astype(str)
            + " | "
            + out.loc[valid_composite, "VolStructureRegime"].astype(str)
        )

        # 0 = most risk-on, 100 = most defensive. This is an allocation aid,
        # not a calibrated probability.
        risk_points = out["RiskRegime"].map(
            {"Stable Risk-On": 0, "Fragile": 35, "Vol Shock": 55, "Crisis": 75}
        )
        trend_points = out["TrendRegime"].map(
            {"Bull": 0, "Sideways": 10, "Transition": 20, "Bear": 35}
        )
        momentum_points = out["MomentumRegime"].map(
            {"Strong Positive": 0, "Positive": 5, "Mixed": 15, "Negative": 25}
        )
        vol_points = out["VolStructureRegime"].map(
            {"Contango": 0, "Flat": 5, "Backwardation": 15, "Unavailable": 5}
        )
        out["DefensiveScore"] = (risk_points + trend_points + momentum_points + vol_points).clip(0, 100)

        return out

    def build_regimes(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        data = self.fetch_data(start_date=start_date, end_date=end_date)
        return self.build_regimes_from_data(data)

    def current_regime(
        self,
        *,
        as_of: Optional[str | pd.Timestamp] = None,
        return_diagnostics: bool = True,
    ) -> Dict[str, Any] | str:
        """Return the latest fully formed layered regime."""
        as_of_ts = (
            pd.Timestamp.today().normalize()
            if as_of is None
            else pd.to_datetime(as_of).tz_localize(None).normalize()
        )
        longest_window = max(
            self.risk_lookback + self.risk_ema_span,
            self.trend_ma_window,
            max(self.momentum_windows),
        )
        history_bdays = longest_window + self.download_padding_bdays
        start = as_of_ts - pd.tseries.offsets.BDay(history_bdays)
        end = as_of_ts + pd.tseries.offsets.BDay(1)
        regimes = self.build_regimes(start_date=start, end_date=end)
        regimes = regimes.loc[regimes.index <= as_of_ts]
        valid = regimes.dropna(subset=["CompositeRegime"])
        if valid.empty:
            raise RuntimeError("No fully formed layered regime is available.")
        row = valid.iloc[-1]

        if not return_diagnostics:
            return str(row["CompositeRegime"])

        return {
            "as_of": str(as_of_ts.date()),
            "last_date_in_data": str(valid.index[-1].date()),
            "risk": {"label": row["RiskRegime"], "key": row["RiskKey"]},
            "trend": {"label": row["TrendRegime"], "key": row["TrendKey"]},
            "momentum": {"label": row["MomentumRegime"], "key": row["MomentumKey"]},
            "volatility_structure": {
                "label": row["VolStructureRegime"],
                "key": row["VolStructureKey"],
            },
            "composite_regime": row["CompositeRegime"],
            "defensive_score": float(row["DefensiveScore"]),
            "signals": {
                "vix_percentile_ema": float(row["VIXPctEMA"]),
                "credit_stress_percentile_ema": float(row["CreditStressPctEMA"]),
                "price_to_trend_ma": float(row["PriceToTrendMA"]),
                "annualized_trend_slope": float(row["TrendSlopeAnn"]),
                "average_momentum": float(row["MomentumAverage"]),
                "positive_momentum_share": float(row["MomentumPositiveShare"]),
                "vix_term_ratio": None
                if pd.isna(row["VIXTermRatio"])
                else float(row["VIXTermRatio"]),
            },
        }

    def dominant_regime(
        self,
        *,
        as_of: Optional[str | pd.Timestamp] = None,
        dominance_window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return modal labels over a recent window, with latest-label tie breaks."""
        as_of_ts = (
            pd.Timestamp.today().normalize()
            if as_of is None
            else pd.to_datetime(as_of).tz_localize(None).normalize()
        )
        window_size = int(dominance_window or self.dominance_window)
        longest_window = max(
            self.risk_lookback + self.risk_ema_span,
            self.trend_ma_window,
            max(self.momentum_windows),
        )
        start = as_of_ts - pd.tseries.offsets.BDay(
            longest_window + window_size + self.download_padding_bdays
        )
        end = as_of_ts + pd.tseries.offsets.BDay(1)
        frame = self.build_regimes(start_date=start, end_date=end)
        frame = frame.loc[frame.index <= as_of_ts].dropna(subset=["CompositeRegime"])
        window = frame.tail(window_size)
        if window.empty:
            raise RuntimeError("No valid rows in dominance window.")

        labels = {
            "risk": _dominant_label(window["RiskRegime"]),
            "trend": _dominant_label(window["TrendRegime"]),
            "momentum": _dominant_label(window["MomentumRegime"]),
            "volatility_structure": _dominant_label(window["VolStructureRegime"]),
        }
        composite = " | ".join(labels.values())
        return {
            "as_of": str(as_of_ts.date()),
            "last_date_in_data": str(window.index[-1].date()),
            "dominance_window": window_size,
            "labels": labels,
            "composite_regime": composite,
            "median_defensive_score": float(window["DefensiveScore"].median()),
            "counts": {
                col: window[col].value_counts().to_dict()
                for col in [
                    "RiskRegime",
                    "TrendRegime",
                    "MomentumRegime",
                    "VolStructureRegime",
                ]
            },
        }

    def recent_regimes(
        self,
        *,
        n_days: int = 60,
        as_of: Optional[str | pd.Timestamp] = None,
    ) -> pd.DataFrame:
        as_of_ts = (
            pd.Timestamp.today().normalize()
            if as_of is None
            else pd.to_datetime(as_of).tz_localize(None).normalize()
        )
        longest_window = max(
            self.risk_lookback + self.risk_ema_span,
            self.trend_ma_window,
            max(self.momentum_windows),
        )
        start = as_of_ts - pd.tseries.offsets.BDay(
            longest_window + int(n_days) + self.download_padding_bdays
        )
        end = as_of_ts + pd.tseries.offsets.BDay(1)
        frame = self.build_regimes(start_date=start, end_date=end)
        columns = [
            "RiskRegime",
            "TrendRegime",
            "MomentumRegime",
            "VolStructureRegime",
            "CompositeRegime",
            "DefensiveScore",
            "VIXPctEMA",
            "CreditStressPctEMA",
            "PriceToTrendMA",
            "TrendSlopeAnn",
            "MomentumAverage",
            "MomentumPositiveShare",
            "VIXTermRatio",
        ]
        return frame.loc[frame.index <= as_of_ts, columns].tail(int(n_days)).copy()


if __name__ == "__main__":
    detector = LayeredRegimeDetector()
    print(detector.current_regime())
