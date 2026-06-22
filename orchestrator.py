import json
import os
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from alpaca_adapter import AlpacaAPI
from dotenv import find_dotenv, load_dotenv
from helpers import (
    download_file_from_digitalocean_spaces,
    getenv_float,
    run_portfolio_regime_iteration,
    str2bool,
)
from log import log
from regime_detector import RegimeDetector
from SES import AmazonSES

load_dotenv(find_dotenv())

ZILLAIQ_API_BASE_URL = os.getenv(
    "ZILLAIQ_API_BASE_URL", "http://localhost:8000"
).rstrip("/")

weights_by_regime = {
    "stable_risk_on": {
        "trend": 1.0,
        "pairs-coint": 0.0,
        # "vol-harvest": 0.0,
        "etf-trend-regime-vol-shock": 0.0,
        "etf-trend-regime-crisis": 0.0,
    },
    "fragile": {
        "trend": 0.70,
        "pairs-coint": 0.30,
        # "vol-harvest": 0.00,
        "etf-trend-regime-vol-shock": 0.0,
        "etf-trend-regime-crisis": 0.0,
    },
    "vol_shock": {
        "trend": 0.0,
        "pairs-coint": 0.0,
        # "vol-harvest": 1.0,
        "etf-trend-regime-vol-shock": 1.0,
        "etf-trend-regime-crisis": 0.0,
    },
    "crisis": {
        "trend": 0.0,
        "pairs-coint": 0.0,
        # "vol-harvest": 0.0,
        "etf-trend-regime-vol-shock": 0.0,
        "etf-trend-regime-crisis": 1.0,
    },
}

remote_files = (
    "etf-pairs-cointegration.json",
    "etf-trend-regime-crisis.json",
    "etf-trend-regime-vol-shock.json",
    # "etf-volatility-harvest.json",
    "etf-trend-rp-vt.json",
)

output_path = Path("./strategy_targets")
output_path.mkdir(parents=True, exist_ok=True)
spaces_region = os.environ.get("SPACES_REGION")
spaces_bucket = os.environ.get("SPACES_BUCKET")
spaces_access_key = os.environ.get("SPACES_KEY")
spaces_secret_key = os.environ.get("SPACES_SECRET")
spaces_object_key_prefix = os.environ.get("SPACES_OBJECT_KEY_PATH", "").strip("/")

log(
    f"Downloading {len(remote_files)} strategy files from Spaces bucket "
    f"'{spaces_bucket}' into '{output_path.resolve()}'",
    "info",
)

for filename in remote_files:
    local_path = output_path / filename
    object_key = (
        f"{spaces_object_key_prefix}/{filename}"
        if spaces_object_key_prefix
        else filename
    )

    log(f"Downloading '{object_key}' -> '{local_path}'", "info")

    download_file_from_digitalocean_spaces(
        file_path=str(local_path),
        region=spaces_region,
        object_key=object_key,
        bucket_name=spaces_bucket,
        access_key=spaces_access_key,
        secret_key=spaces_secret_key,
    )

log(
    f"Downloaded {len(remote_files)} strategy files to '{output_path.resolve()}'",
    "info",
)


detector = RegimeDetector(
    ema_span=60,
    lookback=252,
    vix_high_pct=0.70,
    spread_wide_pct=0.70,
    credit_mode="legacy_diff",
    shift_regime_by_one_day=True,
)
as_of = datetime.now()
result = detector.dominant_regime(as_of=as_of)
dominant_regime = result["dominant_regime"]
log(f"Regime Detected: {dominant_regime}", "info")


is_live_trade = str2bool(os.getenv("LIVE_TRADE", False))
equity_fraction = getenv_float("EQUITY_FRACTION", 1.0)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def request_json(
    method: str, url: str, payload: dict | None = None, token: str | None = None
):
    headers = {"Accept": "application/json"}
    body = None

    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url=url, data=body, headers=headers, method=method)

    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {url} failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def fetch_active_alpaca_credentials() -> list[dict]:
    alpaca_key_id = str(os.getenv("ALPACA_KEY_ID", "")).strip()
    alpaca_secret_key = str(os.getenv("ALPACA_SECRET_KEY", "")).strip()

    if alpaca_key_id and alpaca_secret_key:
        log(
            "Using ALPACA credentials from environment variables",
            "info",
        )
        return [
            {
                "id": "env",
                "client_name": "environment",
                "fund_strategy_code": "environment",
                "environment": os.getenv("ALPACA_ENVIRONMENT", "paper"),
                "broker": "ALPACA",
                "key_id": alpaca_key_id,
                "secret_key": alpaca_secret_key,
            }
        ]

    username = require_env("ZILLAIQ_USERNAME")
    password = require_env("ZILLAIQ_PASSWORD")

    token_response = request_json(
        "POST",
        f"{ZILLAIQ_API_BASE_URL}/api/token/",
        payload={"username": username, "password": password},
    )
    access_token = token_response.get("access")
    if not access_token:
        raise RuntimeError("Token response did not include an access token.")

    credentials = request_json(
        "GET",
        f"{ZILLAIQ_API_BASE_URL}/api/credentials/active",
        token=access_token,
    )
    if not isinstance(credentials, list) or not credentials:
        raise RuntimeError("No active credentials were returned by the API.")

    alpaca_credentials = [
        item
        for item in credentials
        if str(item.get("broker", "")).upper() == "ALPACA"
        and item.get("key_id")
        and item.get("secret_key")
    ]
    if not alpaca_credentials:
        raise RuntimeError("No active ALPACA credentials were returned by the API.")

    log(
        f"Using {len(alpaca_credentials)} active ALPACA credential(s) from the API",
        "info",
    )
    return alpaca_credentials


def build_strategy_allocations(
    strategy_weights_path: Path,
    dominant_regime: str,
    weights_by_regime: dict[str, dict[str, float]],
    equity_fraction: float,
) -> list[dict]:
    regime_allocations = weights_by_regime.get(dominant_regime, {})
    strategy_allocations = []

    for strategy_file in sorted(Path(strategy_weights_path).glob("*.json")):
        payload = json.loads(strategy_file.read_text())
        strategy_name = str(payload.get("strategy", strategy_file.stem))
        trade_today = str2bool(payload.get("trade_today", True))
        liquidate_when_inactive = str2bool(
            payload.get("liquidate_when_inactive", False)
        )
        regime_weight = float(regime_allocations.get(strategy_name, 0.0))

        if regime_weight == 0:
            continue

        if not trade_today and liquidate_when_inactive:
            continue

        capital_requested = float(payload.get("capital_requested", 1.0) or 1.0)
        strategy_weight = regime_weight * capital_requested * equity_fraction
        positions = []

        for position in payload.get("positions", []):
            symbol = str(position.get("symbol", "")).strip()
            if not symbol:
                continue

            target_weight = float(position.get("target_weight", 0.0) or 0.0)
            positions.append(
                {
                    "symbol": symbol,
                    "target_weight": target_weight * strategy_weight,
                }
            )

        positions.sort(key=lambda item: item["symbol"])
        strategy_allocations.append(
            {
                "strategy": strategy_name,
                "target_weight": strategy_weight,
                "positions": positions,
            }
        )

    return strategy_allocations


def format_strategy_allocations_plain(strategy_allocations: list[dict]) -> str:
    if not strategy_allocations:
        return "No strategy allocations."

    lines = []
    for index, strategy in enumerate(strategy_allocations):
        lines.append(f"{strategy['strategy']}: {float(strategy['target_weight']):.2%}")
        positions = strategy.get("positions", [])
        if not positions:
            lines.append("  No ticker allocations.")
        else:
            for position in positions:
                lines.append(
                    f"  {position['symbol']}: {float(position['target_weight']):.2%}"
                )
        if index < len(strategy_allocations) - 1:
            lines.append("-" * 32)

    return "\n".join(lines)


def format_strategy_allocations_html(strategy_allocations: list[dict]) -> str:
    if not strategy_allocations:
        return "No strategy allocations."

    sections = []
    for strategy in strategy_allocations:
        lines = [f"{strategy['strategy']}: {float(strategy['target_weight']):.2%}"]
        positions = strategy.get("positions", [])
        if not positions:
            lines.append("&nbsp;&nbsp;No ticker allocations.")
        else:
            for position in positions:
                lines.append(
                    f"&nbsp;&nbsp;{position['symbol']}: {float(position['target_weight']):.2%}"
                )
        lines.append("--------------------------------")
        sections.append("<br>".join(lines))

    return "<br>".join(sections)


credential_reports = []

for credential in fetch_active_alpaca_credentials():

    alpaca_key = credential["key_id"]
    alpaca_secret = credential["secret_key"]
    environment = str(credential.get("environment", "")).strip().lower()
    is_paper = environment in {"paper", "sandbox"}

    log(
        "Running active ALPACA credential "
        f"id={credential.get('id')} client={credential.get('client_name')} "
        f"strategy={credential.get('fund_strategy_code')} environment={environment or 'unknown'}",
        "info",
    )

    api = AlpacaAPI.from_env(
        api_key=alpaca_key,
        secret_key=alpaca_secret,
        paper=is_paper,
    )

    account = api.get_account()

    portfolio = run_portfolio_regime_iteration(
        strategy_weights_path=output_path,
        dominant_regime=dominant_regime,
        weights_by_regime=weights_by_regime,
        account=account,
        equity_fraction=equity_fraction,
        api=api,
        is_paper=is_paper,
        is_live_trade=is_live_trade,
    )

    credential_reports.append(
        {
            "credential": credential,
            "environment": environment,
            "portfolio": portfolio,
        }
    )

# Email Positions
EMAIL_POSITIONS = str2bool(os.getenv("EMAIL_POSITIONS", False))

message_sections_plain = []
message_sections_html = []

for report in credential_reports:
    credential = report["credential"]
    total_portfolios_updated = len(credential_reports)
    strategy_allocations = build_strategy_allocations(
        strategy_weights_path=output_path,
        dominant_regime=report["portfolio"]["dominant_regime"],
        weights_by_regime=weights_by_regime,
        equity_fraction=float(report["portfolio"].get("equity_fraction", 1.0)),
    )
    strategy_allocations_plain = format_strategy_allocations_plain(strategy_allocations)
    strategy_allocations_html = format_strategy_allocations_html(strategy_allocations)
    header = (
        f"Client: {credential.get('client_name', 'unknown')} | "
        f"Strategy: {credential.get('fund_strategy_code', 'unknown')} | "
        f"Environment: {report['environment'] or 'unknown'} | "
        f"Credential ID: {credential.get('id', 'unknown')}"
    )
    message_sections_plain.append(
        f"{header}\n"
        f"Total Portfolios Updated: {total_portfolios_updated}\n"
        f"Regime: {report['portfolio']['dominant_regime']}\n"
        f"{strategy_allocations_plain}"
    )

    message_sections_html.append(
        f"{header}<br><br>"
        f"Total Portfolios Updated: {total_portfolios_updated}<br>"
        f"Regime: {report['portfolio']['dominant_regime']}<br>"
        f"<pre>{strategy_allocations_html}</pre>"
    )

message_body_plain = "\n\n".join(message_sections_plain)
message_body_html = "<br><br>".join(message_sections_html)

print("---------------------------------------------------\n")
print(message_body_plain)

if EMAIL_POSITIONS:
    TO_ADDRESSES = [
        a.strip() for a in os.getenv("TO_ADDRESSES", "").split(",") if a.strip()
    ]
    FROM_ADDRESS = os.getenv("FROM_ADDRESS", "")

    ses = AmazonSES(
        region=os.environ.get("AWS_SES_REGION_NAME"),
        access_key=os.environ.get("AWS_SES_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SES_SECRET_ACCESS_KEY"),
        from_address=FROM_ADDRESS,
    )

    status = "Live" if is_live_trade else "Test"
    today = datetime.utcnow().strftime("%Y-%m-%d")
    subject = f"Quant Portfolio Orchestrator Report - {status} - {today}"

    for to_address in TO_ADDRESSES:
        ses.send_html_email(
            to_address=to_address,
            subject=subject,
            content=message_body_html,
        )

print("---------------------------------------------------\n")
print(message_body_plain)
