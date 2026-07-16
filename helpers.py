import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import boto3
from alpaca.common.exceptions import APIError
from alpaca.data.timeframe import TimeFrame
from log import log


def str2bool(value):
    valid = {
        "true": True,
        "t": True,
        "1": True,
        "on": True,
        "false": False,
        "f": False,
        "0": False,
    }

    if isinstance(value, bool):
        return value

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


def getenv_float(name: str, default: float) -> float:
    """
    Read an environment variable as a float.

    - Returns `default` if the variable is missing
    - Returns `default` if conversion fails
    """
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _position_side_multiplier(position) -> int:
    return -1 if str(getattr(position, "side", "")).lower() == "short" else 1


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return str2bool(value)
        except ValueError:
            return default
    return bool(value)


def _latest_price_for_symbol(api, symbol: str, open_positions_by_symbol: dict) -> float:
    existing_position = open_positions_by_symbol.get(symbol)
    if existing_position is not None:
        current_price = _safe_float(getattr(existing_position, "current_price", None))
        if current_price > 0:
            return current_price

    end = datetime.utcnow()
    start = end - timedelta(days=5)
    bars = api.get_bars(
        symbol=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="iex",
    )
    bars_df = bars.df
    if bars_df.empty:
        raise ValueError(f"No market data returned for symbol '{symbol}'")

    if "symbol" in bars_df.index.names:
        symbol_bars = bars_df.xs(symbol, level="symbol")
    else:
        symbol_bars = bars_df

    close_price = _safe_float(symbol_bars.iloc[-1]["close"])
    if close_price <= 0:
        raise ValueError(f"Invalid close price returned for symbol '{symbol}'")
    return close_price


def _whole_share_target(quantity: float) -> int:
    return math.trunc(quantity)


def _whole_share_order_qty(delta_quantity: float) -> int:
    return int(math.floor(abs(delta_quantity) + 0.5))


def build_layered_regime_key(
    risk_key: str,
    trend_key: str,
    momentum_key: str,
    vol_structure_key: str,
) -> str:
    return "__".join((risk_key, trend_key, momentum_key, vol_structure_key))


def _normalize_combine_mode(combine_mode: str | None) -> str:
    mode = str(combine_mode or "and").strip().lower()
    if mode not in {"and", "or"}:
        raise ValueError("combine_mode must be 'and' or 'or'")
    return mode


def resolve_regime_allocations(
    weights_by_regime: dict,
    active_regime_keys: list[str],
    combine_mode: str = "and",
) -> tuple[list[str], dict]:
    mode = _normalize_combine_mode(combine_mode)
    active_regime_set = {key for key in active_regime_keys if key}
    matched_regimes = []
    combined_allocations = defaultdict(float)

    for regime_key, regime_allocations in weights_by_regime.items():
        selector_parts = [part for part in str(regime_key).split("__") if part]
        if not selector_parts:
            continue

        if mode == "and":
            matched = all(part in active_regime_set for part in selector_parts)
        else:
            matched = any(part in active_regime_set for part in selector_parts)

        if not matched:
            continue

        matched_regimes.append(str(regime_key))
        for strategy_name, weight in regime_allocations.items():
            combined_allocations[str(strategy_name)] += _safe_float(weight)

    if not matched_regimes:
        raise ValueError(
            "No regime weights configured for active regimes "
            f"{sorted(active_regime_set)} using combine_mode='{mode}'"
        )

    return matched_regimes, dict(combined_allocations)


def run_portfolio_regime_iteration(
    strategy_weights_path,
    dominant_regime,
    weights_by_regime,
    active_regime_keys,
    account,
    api,
    is_paper,
    is_live_trade,
    equity_fraction=1.0,
    combine_mode="and",
):
    weights_path = Path(strategy_weights_path).expanduser()
    matched_regimes, regime_allocations = resolve_regime_allocations(
        weights_by_regime=weights_by_regime,
        active_regime_keys=active_regime_keys,
        combine_mode=combine_mode,
    )

    if not weights_path.exists():
        raise FileNotFoundError(f"Strategy weights path does not exist: {weights_path}")

    strategy_files = sorted(weights_path.glob("*.json"))
    if not strategy_files:
        raise FileNotFoundError(f"No strategy weight files found in: {weights_path}")

    equity = _safe_float(getattr(account, "equity", None))
    if equity <= 0:
        raise ValueError("Account equity must be positive to size positions")

    equity_fraction = _safe_float(equity_fraction, default=-1.0)
    if not 0.0 <= equity_fraction <= 1.0:
        raise ValueError("equity_fraction must be between 0 and 1 inclusive")

    log(
        f"Running regime iteration for detected regime '{dominant_regime}' "
        f"using combine_mode='{_normalize_combine_mode(combine_mode)}' and matched weights "
        f"{matched_regimes} with {len(strategy_files)} strategy files "
        f"on {'paper' if is_paper else 'live'} account with equity {equity:.2f} "
        f"and trading allocation {equity_fraction:.2%} "
        f"(live_trade={'on' if is_live_trade else 'off'})",
        "info",
    )

    active_target_weights_by_symbol = defaultdict(float)
    protected_target_weights_by_symbol = defaultdict(float)
    liquidating_symbols = set()
    loaded_strategies = 0

    for strategy_file in strategy_files:
        payload = json.loads(strategy_file.read_text())
        strategy_name = payload.get("strategy")
        trade_today = _safe_bool(payload.get("trade_today", True), True)
        liquidate_when_inactive = _safe_bool(
            payload.get("liquidate_when_inactive", False), False
        )
        regime_weight = _safe_float(regime_allocations.get(strategy_name))
        if regime_weight == 0:
            log(
                f"Skipping strategy '{strategy_name}' from '{strategy_file.name}' because "
                f"regime weight is 0 for '{dominant_regime}'",
                "info",
            )
            continue

        if not trade_today:
            if liquidate_when_inactive:
                log(
                    f"Strategy '{strategy_name}' from '{strategy_file.name}' is paused and "
                    "will be liquidated (trade_today=false, liquidate_when_inactive=true)",
                    "warning",
                )
                for position in payload.get("positions", []):
                    symbol = str(position.get("symbol", "")).strip()
                    if symbol:
                        liquidating_symbols.add(symbol)
            else:
                log(
                    f"Strategy '{strategy_name}' from '{strategy_file.name}' is paused and "
                    "its allocations will be preserved without rebalancing (trade_today=false)",
                    "warning",
                )
                capital_requested = _safe_float(
                    payload.get("capital_requested", 1.0), 1.0
                )
                strategy_multiplier = regime_weight * capital_requested
                for position in payload.get("positions", []):
                    symbol = str(position.get("symbol", "")).strip()
                    if symbol:
                        raw_weight = _safe_float(position.get("target_weight"))
                        protected_target_weights_by_symbol[symbol] += (
                            raw_weight * strategy_multiplier
                        )
            continue

        capital_requested = _safe_float(payload.get("capital_requested", 1.0), 1.0)
        strategy_multiplier = regime_weight * capital_requested
        loaded_strategies += 1
        log(
            f"Including strategy '{strategy_name}' from '{strategy_file.name}' with multiplier "
            f"{strategy_multiplier:.4f}",
            "info",
        )

        for position in payload.get("positions", []):
            symbol = position["symbol"]
            raw_weight = _safe_float(position.get("target_weight"))
            active_target_weights_by_symbol[symbol] += raw_weight * strategy_multiplier

    scaled_active_target_weights_by_symbol = {
        symbol: weight * equity_fraction
        for symbol, weight in active_target_weights_by_symbol.items()
    }
    scaled_protected_target_weights_by_symbol = {
        symbol: weight * equity_fraction
        for symbol, weight in protected_target_weights_by_symbol.items()
    }
    target_weights_by_symbol = defaultdict(float)
    all_target_symbols = set(scaled_active_target_weights_by_symbol) | set(
        scaled_protected_target_weights_by_symbol
    )
    for symbol in all_target_symbols:
        target_weights_by_symbol[symbol] = (
            scaled_active_target_weights_by_symbol.get(symbol, 0.0)
            + scaled_protected_target_weights_by_symbol.get(symbol, 0.0)
        )

    protected_only_symbols = set(protected_target_weights_by_symbol) - set(
        active_target_weights_by_symbol
    )

    if not target_weights_by_symbol:
        log(
            f"No target allocations were produced for regime '{dominant_regime}'. "
            "Proceeding with an empty target portfolio.",
            "warning",
        )

    open_positions = api.list_positions()
    open_positions_by_symbol = {
        position.symbol: position for position in open_positions
    }
    current_weights_by_symbol = {}
    tradable_symbols = set(active_target_weights_by_symbol) | liquidating_symbols

    for position in open_positions:
        symbol = position.symbol
        if symbol not in protected_only_symbols or symbol in liquidating_symbols:
            tradable_symbols.add(symbol)
        market_value = _safe_float(getattr(position, "market_value", None))
        current_weights_by_symbol[symbol] = market_value / equity

    order_candidates = []
    for symbol in sorted(tradable_symbols):
        target_weight = target_weights_by_symbol.get(symbol, 0.0)
        active_target_weight = scaled_active_target_weights_by_symbol.get(symbol, 0.0)
        protected_target_weight = scaled_protected_target_weights_by_symbol.get(
            symbol, 0.0
        )
        current_weight = current_weights_by_symbol.get(symbol, 0.0)

        price = _latest_price_for_symbol(api, symbol, open_positions_by_symbol)

        current_position = open_positions_by_symbol.get(symbol)
        if current_position is not None:
            current_qty = _safe_float(getattr(current_position, "qty", None))
            current_qty *= _position_side_multiplier(current_position)
        else:
            current_qty = 0.0

        if protected_target_weight > 0:
            rebalance_current_weight = max(
                current_weight - protected_target_weight, 0.0
            )
            rebalance_target_weight = active_target_weight
            protected_qty = _whole_share_target(
                (protected_target_weight * equity) / price
            )
            rebalance_current_qty = max(current_qty - protected_qty, 0.0)
        else:
            rebalance_current_weight = current_weight
            rebalance_target_weight = target_weight
            rebalance_current_qty = current_qty

        delta_weight = rebalance_target_weight - rebalance_current_weight
        if abs(delta_weight) < 0.0025:
            continue

        target_qty = (rebalance_target_weight * equity) / price
        target_qty_whole = _whole_share_target(target_qty)
        delta_qty = target_qty_whole - rebalance_current_qty
        order_qty = _whole_share_order_qty(delta_qty)
        if order_qty < 1:
            continue

        order_candidates.append(
            {
                "symbol": symbol,
                "price": price,
                "target_weight": target_weight,
                "current_weight": current_weight,
                "protected_target_weight": protected_target_weight,
                "rebalance_target_weight": rebalance_target_weight,
                "rebalance_current_weight": rebalance_current_weight,
                "delta_weight": delta_weight,
                "current_qty": current_qty,
                "target_qty": target_qty_whole,
                "delta_qty": delta_qty,
                "order_qty": order_qty,
            }
        )

    if not order_candidates:
        log("No rebalance orders required for current regime targets", "success")
        return {
            "dominant_regime": dominant_regime,
            "active_regime_keys": list(active_regime_keys),
            "orders_submitted": 0,
            "matched_regimes": matched_regimes,
            "combine_mode": _normalize_combine_mode(combine_mode),
            "target_weights": dict(target_weights_by_symbol),
            "equity_fraction": equity_fraction,
            "is_live_trade": is_live_trade,
        }

    order_candidates.sort(key=lambda item: item["delta_qty"])
    submitted_orders = []

    for candidate in order_candidates:
        side = "buy" if candidate["delta_qty"] > 0 else "sell"
        qty = candidate["order_qty"]
        if qty <= 0:
            continue

        log(
            f"{side.upper()} {candidate['symbol']} qty={qty} "
            f"target_w={candidate['target_weight']:.4f} current_w={candidate['current_weight']:.4f} "
            f"protected_w={candidate['protected_target_weight']:.4f} "
            f"px={candidate['price']:.2f}",
            "info",
        )

        if not is_live_trade:
            log(
                f"Dry run only: skipped {side} order for {candidate['symbol']} qty={qty}",
                "warning",
            )
            continue

        try:
            order = api.submit_order(
                symbol=candidate["symbol"],
                time_in_force="day",
                side=side,
                type="market",
                qty=qty,
            )
        except APIError as exc:
            log(
                f"Order failed for {candidate['symbol']} ({side} {qty}): {exc}",
                "error",
            )
            continue

        submitted_orders.append(
            {
                "symbol": candidate["symbol"],
                "side": side,
                "qty": qty,
                "target_weight": candidate["target_weight"],
                "current_weight": candidate["current_weight"],
                "order_id": getattr(order, "id", None),
            }
        )
        log(
            f"Submitted {side} order for {candidate['symbol']} qty={qty}",
            "success",
        )

    log(
        f"Regime iteration complete: {len(submitted_orders)} orders submitted from {loaded_strategies} strategy files",
        "success",
    )
    return {
        "dominant_regime": dominant_regime,
        "active_regime_keys": list(active_regime_keys),
        "matched_regimes": matched_regimes,
        "combine_mode": _normalize_combine_mode(combine_mode),
        "orders_submitted": len(submitted_orders),
        "orders": submitted_orders,
        "target_weights": dict(target_weights_by_symbol),
        "equity_fraction": equity_fraction,
        "is_live_trade": is_live_trade,
    }


def print_orders_table(result: dict) -> str:
    buf = StringIO()
    orders = result.get("orders", [])

    buf.write(f"Dominant regime: {result.get('dominant_regime', 'unknown')}\n")
    buf.write(
        f"Live trade: {'on' if result.get('is_live_trade') else 'off'} | "
        f"Equity fraction: {float(result.get('equity_fraction', 1.0)):.2%} | "
        f"Orders submitted: {result.get('orders_submitted', 0)}\n"
    )

    if not orders:
        buf.write("No orders.\n")
        return buf.getvalue()

    headers = ("Symbol", "Side", "Qty", "Target W", "Current W", "Order ID")
    rows = []
    for order in orders:
        rows.append(
            (
                str(order.get("symbol", "")),
                str(order.get("side", "")),
                str(int(order.get("qty", 0))),
                f"{float(order.get('target_weight', 0)):.4f}",
                f"{float(order.get('current_weight', 0)):.4f}",
                str(order.get("order_id", "")),
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def _write_row(values):
        buf.write(
            " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))
        )
        buf.write("\n")

    _write_row(headers)
    buf.write("-+-".join("-" * width for width in widths))
    buf.write("\n")
    for row in rows:
        _write_row(row)

    return buf.getvalue()


def download_file_from_digitalocean_spaces(
    file_path: str,
    *,
    bucket_name: str,
    region: str,
    object_key: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    content_type: str = "application/json",
    acl: str | None = None,
) -> dict:
    destination = Path(file_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    resolved_object_key = object_key or destination.name
    endpoint_url = f"https://{region}.digitaloceanspaces.com"

    client_kwargs = {
        "service_name": "s3",
        "region_name": region,
        "endpoint_url": endpoint_url,
    }
    if access_key is not None:
        client_kwargs["aws_access_key_id"] = access_key
    if secret_key is not None:
        client_kwargs["aws_secret_access_key"] = secret_key

    client = boto3.client(**client_kwargs)
    client.download_file(bucket_name, resolved_object_key, str(destination))

    return {
        "bucket_name": bucket_name,
        "region": region,
        "object_key": resolved_object_key,
        "file_path": str(destination),
        "endpoint_url": endpoint_url,
        # Kept for signature compatibility with the matching upload helper shape.
        "content_type": content_type,
        "acl": acl,
    }
