import os
from datetime import datetime
from pathlib import Path

from alpaca_adapter import AlpacaAPI
from dotenv import find_dotenv, load_dotenv
from helpers import (
    download_file_from_digitalocean_spaces,
    print_orders_table,
    run_portfolio_regime_iteration,
    str2bool,
)
from log import log
from regime_detector import RegimeDetector
from SES import AmazonSES

load_dotenv(find_dotenv())

weights_by_regime = {
    "stable_risk_on": {
        "trend": 1.0,
        "triple_coint": 0.0,
        "triple_vol": 0.0,
    },
    "fragile": {
        "trend": 0.30,
        "triple_coint": 0.50,
        "triple_vol": 0.20,
    },
    "vol_shock": {
        "trend": 0.0,
        "triple_coint": 0.0,
        "futures_trend": 1.0,
    },
    "crisis": {
        "trend": 0.0,
        "triple_coint": 1.00,
        "triple_vol": 0.0,
    },
}

remote_files = (
    "etf-triple-pairs.json",
    "etf-volatility-harvest.json",
    "etf-trend-rp-vt.json",
)

output_path = Path("./strategy_weights")
output_path.mkdir(parents=True, exist_ok=True)
# spaces_region = os.environ.get("SPACES_REGION")
# spaces_bucket = os.environ.get("SPACES_BUCKET")
# spaces_access_key = os.environ.get("SPACES_KEY")
# spaces_secret_key = os.environ.get("SPACES_SECRET")
# spaces_object_key_prefix = os.environ.get("SPACES_OBJECT_KEY_PATH", "").strip("/")

# log(
#     f"Downloading {len(remote_files)} strategy files from Spaces bucket "
#     f"'{spaces_bucket}' into '{output_path.resolve()}'",
#     "info",
# )
#
# for filename in remote_files:
#     local_path = output_path / filename
#     object_key = (
#         f"{spaces_object_key_prefix}/{filename}"
#         if spaces_object_key_prefix
#         else filename
#     )
#
#     log(f"Downloading '{object_key}' -> '{local_path}'", "info")
#
#     download_file_from_digitalocean_spaces(
#         file_path=str(local_path),
#         region=spaces_region,
#         object_key=object_key,
#         bucket_name=spaces_bucket,
#         access_key=spaces_access_key,
#         secret_key=spaces_secret_key,
#     )
#
# log(
#     f"Downloaded {len(remote_files)} strategy files to '{output_path.resolve()}'",
#     "info",
# )


detector = RegimeDetector(
    ema_span=60,
    lookback=252,
    vix_high_pct=0.70,
    spread_wide_pct=0.70,
    credit_mode="ratio",
    shift_regime_by_one_day=True,
)
as_of = datetime.now()
result = detector.dominant_regime(as_of=as_of)
dominant_regime = result["dominant_regime"]
log(f"Regime Detected: {dominant_regime}", "info")


alpaca_key = os.getenv("ALPACA_KEY_ID")
alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
base_url = (os.getenv("ALPACA_BASE_URL") or "").lower()

# Simple heuristic: treat "paper" URLs as paper trading
is_paper = ("paper" in base_url) or str2bool(os.getenv("ALPACA_PAPER", True))
is_live_trade = str2bool(os.getenv("LIVE_TRADE", False))

api = AlpacaAPI.from_env(
    api_key=alpaca_key,
    secret_key=alpaca_secret,
    paper=is_paper,
)

account = api.get_account()
portfolio_value = round(float(account.equity), 3)

portfolio = run_portfolio_regime_iteration(
    strategy_weights_path=output_path,
    dominant_regime=dominant_regime,
    weights_by_regime=weights_by_regime,
    account=account,
    api=api,
    is_paper=is_paper,
    is_live_trade=is_live_trade,
)

# Email Positions
EMAIL_POSITIONS = str2bool(os.getenv("EMAIL_POSITIONS", False))

orders_table = print_orders_table(portfolio)
orders_table_html = orders_table.replace("\n", "<br>")
message_body_html = (
    f"Portfolio Value: {portfolio_value}<br><pre>{orders_table_html}</pre>"
)
message_body_plain = f"Portfolio Value: {portfolio_value}\n{orders_table}"

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
    subject = f"Quant Portfolio Orchestrator Report - {status} - {today}| Regime: {dominant_regime}"

    for to_address in TO_ADDRESSES:
        ses.send_html_email(
            to_address=to_address,
            subject=subject,
            content=message_body_html,
        )

print("---------------------------------------------------\n")
print(message_body_plain)
