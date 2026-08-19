# alpaca_py_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence, Union

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"
ALPACA_MARKET_DATA_BASE_URL = "https://data.alpaca.markets"


@dataclass
class AlpacaAPI:
    """
    Thin compatibility adapter so your existing code can keep calling:
      - get_account()
      - list_positions()
      - get_position(symbol)
      - submit_order(...)
      - get_bars(symbol, timeframe, start, end, adjustment="all", feed="iex")
    while internally using alpaca-py TradingClient + StockHistoricalDataClient.
    """

    trading: TradingClient
    data: StockHistoricalDataClient

    @classmethod
    def from_env(
        cls,
        api_key: str,
        secret_key: str,
        *,
        paper: bool,
    ) -> "AlpacaAPI":
        base_url = ALPACA_PAPER_BASE_URL if paper else ALPACA_LIVE_BASE_URL
        trading = TradingClient(
            api_key,
            secret_key,
            paper=paper,
            url_override=base_url,
        )
        data = StockHistoricalDataClient(
            api_key,
            secret_key,
            # Market data is served from a separate API host. The paper/live
            # distinction applies to trading, not to the historical data API.
            url_override=ALPACA_MARKET_DATA_BASE_URL,
        )
        return cls(trading=trading, data=data)

    # ---- Trading compatibility ----
    def get_account(self):
        return self.trading.get_account()

    def list_positions(self):
        # alpaca-py naming
        return self.trading.get_all_positions()

    def get_position(self, symbol: str):
        # alpaca-py naming; raises APIError if not found
        return self.trading.get_open_position(symbol)

    def submit_order(
        self,
        *,
        symbol: str,
        time_in_force: str,
        side: str,
        type: str,
        qty: Union[int, float, str],
    ):
        """
        Minimal subset your code uses: market orders only.
        """
        if str(type).lower() != "market":
            raise ValueError(
                "Adapter currently supports only market orders (type='market')."
            )

        side_enum = OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL

        tif = str(time_in_force).lower()
        if tif != "day":
            raise ValueError("Adapter currently supports only time_in_force='day'.")

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=TimeInForce.DAY,
        )
        return self.trading.submit_order(order_data=req)

    # ---- Market data compatibility ----
    def get_bars(
        self,
        symbol: Union[str, Sequence[str]],
        timeframe: TimeFrame,
        start: Union[str, datetime],
        end: Union[str, datetime],
        adjustment: str = "all",
        feed: str = "iex",
    ):
        """
        Returns a BarSet (alpaca-py). Your existing _bars_to_series_close()
        already handles objects with .df, but note BarSet.df is MultiIndex.
        """
        feed_enum = DataFeed.IEX if str(feed).lower() == "iex" else DataFeed.SIP

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=adjustment,
            feed=feed_enum,
        )
        return self.data.get_stock_bars(req)
