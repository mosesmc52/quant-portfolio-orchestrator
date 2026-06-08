from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

RegimeKey = Literal["stable_risk_on", "fragile", "vol_shock", "crisis"]


def _rolling_last_percentile(x: np.ndarray) -> float:
    # Percentile of the last element within the rolling window
    s = pd.Series(x)
    return float(s.rank(pct=True).iloc[-1])


@dataclass
class RegimeDetector:
    """
    Regime detector using:
      - Rolling percentile of VIX level
      - Rolling percentile of CreditProxy (HYG/LQD or HYG-LQD)
    and then smoothing BOTH percentiles with an EMA (EWMA) before thresholding.

    Output regimes (keys):
      - stable_risk_on
      - fragile
      - vol_shock
      - crisis

    Internal labels:
      - Stable Risk-On
      - Fragile
      - Vol Shock
      - Crisis
    """

    vix_high_pct: float = 0.70
    spread_wide_pct: float = 0.70
    lookback: int = 252
    credit_mode: str = "ratio"  # "ratio" | "diff"
    dominance_window: int = 20  # mode over last N days
    shift_regime_by_one_day: bool = True  # align with backtest (lookahead fix)

    # EMA smoothing config (focus of this class)
    ema_span: int = 10  # higher => smoother, more lag
    ema_min_periods: Optional[int] = None  # if None, defaults to ema_span

    tickers: Tuple[str, str, str] = ("^VIX", "HYG", "LQD")

    LABEL_TO_KEY: Dict[str, RegimeKey] = None  # set in __post_init__

    def __post_init__(self):
        if self.credit_mode not in ("ratio", "diff"):
            raise ValueError("credit_mode must be one of: 'ratio', 'diff'")
        if not (0.0 < self.vix_high_pct < 1.0):
            raise ValueError("vix_high_pct must be in (0, 1)")
        if not (0.0 < self.spread_wide_pct < 1.0):
            raise ValueError("spread_wide_pct must be in (0, 1)")
        if self.lookback < 20:
            raise ValueError("lookback too small; use 126 or 252")
        if self.dominance_window < 1:
            raise ValueError("dominance_window must be >= 1")
        if self.ema_span < 1:
            raise ValueError("ema_span must be >= 1")

        self.LABEL_TO_KEY = {
            "Stable Risk-On": "stable_risk_on",
            "Fragile": "fragile",
            "Vol Shock": "vol_shock",
            "Crisis": "crisis",
        }

    # ----------------------------
    # Data fetch
    # ----------------------------
    def fetch_data(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()

        data = yf.download(
            list(self.tickers),
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )["Adj Close"].dropna()

        if data.empty:
            raise RuntimeError(
                "yfinance returned no data for ^VIX/HYG/LQD in the requested range."
            )
        return data

    # ----------------------------
    # Regime building (EMA-smoothed)
    # ----------------------------
    def build_regimes(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Returns DataFrame indexed by Date with:
          RegimeLabel
          VIXPct, CreditStressPct
          VIXPctEMA, CreditStressPctEMA

        Design:
          - Higher CreditStress always means worse credit.
          - No regime label is assigned until both EMA signals are valid.
          - If self.shift_regime_by_one_day=True, regime labels are based on
            yesterday's completed signals so the output is directly tradable
            for today's allocation.

        Backward-compatible aliases are also returned:
          - SpreadPct = CreditStressPct
          - SpreadPctEMA = CreditStressPctEMA
        """
        data = self.fetch_data(start_date=start_date, end_date=end_date).copy()

        required_cols = {"^VIX", "HYG", "LQD"}
        missing = required_cols.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns from fetched data: {sorted(missing)}")

        # -----------------------------
        # Credit stress proxy
        # Higher must always mean worse credit.
        # -----------------------------
        if self.credit_mode == "ratio":
            # HYG/LQD rising generally means credit improving/risk-on.
            # Invert to LQD/HYG so higher means more credit stress.
            data["CreditStress"] = data["LQD"] / data["HYG"]
        elif self.credit_mode == "diff":
            # HYG - LQD falling generally means credit worsening.
            # Invert to LQD - HYG so higher means more credit stress.
            data["CreditStress"] = data["LQD"] - data["HYG"]
        else:
            raise ValueError("credit_mode must be either 'ratio' or 'diff'.")

        # -----------------------------
        # Rolling percentiles
        # Higher percentile = more stress.
        # -----------------------------
        data["VIXPct"] = (
            data["^VIX"]
            .rolling(self.lookback)
            .apply(_rolling_last_percentile, raw=True)
        )

        data["CreditStressPct"] = (
            data["CreditStress"]
            .rolling(self.lookback)
            .apply(_rolling_last_percentile, raw=True)
        )

        # Backward-compatible aliases for older downstream code.
        data["SpreadPct"] = data["CreditStressPct"]

        # -----------------------------
        # EMA smoothing
        # -----------------------------
        minp = (
            int(self.ema_min_periods)
            if self.ema_min_periods is not None
            else int(self.ema_span)
        )

        data["VIXPctEMA"] = (
            data["VIXPct"]
            .ewm(span=self.ema_span, adjust=False, min_periods=minp)
            .mean()
        )

        data["CreditStressPctEMA"] = (
            data["CreditStressPct"]
            .ewm(span=self.ema_span, adjust=False, min_periods=minp)
            .mean()
        )

        # Backward-compatible alias for older downstream code.
        data["SpreadPctEMA"] = data["CreditStressPctEMA"]

        # -----------------------------
        # Signal alignment
        # -----------------------------
        if self.shift_regime_by_one_day:
            # Use yesterday's completed signals for today's allocation.
            vix_sig = data["VIXPctEMA"].shift(1)
            credit_sig = data["CreditStressPctEMA"].shift(1)
        else:
            # Research/diagnostic mode only; not directly executable same-day.
            vix_sig = data["VIXPctEMA"]
            credit_sig = data["CreditStressPctEMA"]

        valid = np.isfinite(vix_sig) & np.isfinite(credit_sig)

        vol_high = vix_sig > self.vix_high_pct
        credit_stressed = credit_sig > self.spread_wide_pct

        # -----------------------------
        # Regime labels
        # -----------------------------
        data["RegimeLabel"] = pd.Series(index=data.index, dtype="object")

        data.loc[valid, "RegimeLabel"] = "Stable Risk-On"
        data.loc[valid & (~vol_high) & credit_stressed, "RegimeLabel"] = "Fragile"
        data.loc[valid & vol_high & (~credit_stressed), "RegimeLabel"] = "Vol Shock"
        data.loc[valid & vol_high & credit_stressed, "RegimeLabel"] = "Crisis"

        return data[
            [
                "RegimeLabel",
                "VIXPct",
                "CreditStressPct",
                "SpreadPct",
                "VIXPctEMA",
                "CreditStressPctEMA",
                "SpreadPctEMA",
            ]
        ].copy()

    # ----------------------------
    # Dominance logic (mode w/ tie-break)
    # ----------------------------
    @staticmethod
    def _dominant_label(window_labels: pd.Series) -> str:
        """
        Dominant = mode over window; tie-breaker = most recent label.
        """
        counts = window_labels.value_counts()
        top = counts.max()
        tied = counts[counts == top].index.tolist()
        last = str(window_labels.iloc[-1])
        return last if last in tied else str(tied[0])

    def dominant_regime(
        self,
        *,
        as_of: Optional[str | pd.Timestamp] = None,
        dominance_window: Optional[int] = None,
        return_diagnostics: bool = True,
    ) -> RegimeKey | Dict[str, Any]:
        """
        Returns dominant regime key as of `as_of` (defaults to today).

        Dominant regime is computed over the last `dominance_window` trading days
        (defaults to self.dominance_window).

        build_regimes() already handles signal shifting when
        shift_regime_by_one_day=True, so no additional shift is applied here.
        """
        dom_w = int(dominance_window or self.dominance_window)

        as_of_ts = (
            pd.Timestamp.today().normalize()
            if as_of is None
            else pd.to_datetime(as_of).tz_localize(None).normalize()
        )

        # Pull enough history for rolling lookback + EMA warmup + dominance window
        padding_days = 40  # holidays / gaps
        history_days = self.lookback + dom_w + max(self.ema_span, 5) + padding_days
        start_ts = as_of_ts - pd.tseries.offsets.BDay(history_days)
        end_ts = as_of_ts + pd.tseries.offsets.BDay(1)

        regimes = self.build_regimes(start_date=start_ts, end_date=end_ts)
        regimes = regimes.loc[regimes.index <= as_of_ts].copy()
        if regimes.empty:
            raise RuntimeError(
                "No regime rows available up to as_of date (check yfinance availability)."
            )

        window = regimes.tail(dom_w)
        if window.empty:
            raise RuntimeError(
                "dominance_window produced empty window; increase padding or lower dominance_window."
            )

        dominant_label = self._dominant_label(window["RegimeLabel"])
        dominant_key: RegimeKey = self.LABEL_TO_KEY[dominant_label]

        if not return_diagnostics:
            return dominant_key

        last_label = str(window["RegimeLabel"].iloc[-1])
        counts = window["RegimeLabel"].value_counts()

        return {
            "dominant_regime": dominant_key,
            "dominant_label": dominant_label,
            "last_regime": self.LABEL_TO_KEY[last_label],
            "last_label": last_label,
            "as_of": str(as_of_ts.date()),
            "last_date_in_data": str(window.index[-1].date()),
            "dominance_window": dom_w,
            "counts": counts.to_dict(),
            "ema_span": int(self.ema_span),
            "ema_min_periods": (
                int(self.ema_min_periods)
                if self.ema_min_periods is not None
                else int(self.ema_span)
            ),
            "params": {
                "vix_high_pct": float(self.vix_high_pct),
                "spread_wide_pct": float(self.spread_wide_pct),
                "lookback": int(self.lookback),
                "credit_mode": str(self.credit_mode),
                "shift_regime_by_one_day": bool(self.shift_regime_by_one_day),
                "tickers": list(self.tickers),
            },
        }

    # ----------------------------
    # Notebook convenience
    # ----------------------------
    def recent_regimes(
        self,
        *,
        n_days: int = 60,
        as_of: Optional[str | pd.Timestamp] = None,
        include_key: bool = True,
    ) -> pd.DataFrame:
        """
        Returns last n_days of regimes. build_regimes() already handles signal shifting when configured.
        """
        as_of_ts = (
            pd.Timestamp.today().normalize()
            if as_of is None
            else pd.to_datetime(as_of).tz_localize(None).normalize()
        )

        padding_days = 40
        history_days = self.lookback + n_days + max(self.ema_span, 5) + padding_days
        start_ts = as_of_ts - pd.tseries.offsets.BDay(history_days)
        end_ts = as_of_ts + pd.tseries.offsets.BDay(1)

        regimes = self.build_regimes(start_date=start_ts, end_date=end_ts)
        regimes = regimes.loc[regimes.index <= as_of_ts].copy()

        tail = regimes.tail(int(n_days)).copy()
        if include_key:
            tail["RegimeKey"] = tail["RegimeLabel"].map(self.LABEL_TO_KEY)
        return tail
