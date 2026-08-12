# quant-portfolio-orchestrator
Portfolio Orchestrator for all Quant Strategies

Strategy targets may set `initialize_portfolio` to `true` when an empty account
should receive that strategy's target positions on its first deployment. This
is an explicit bootstrap permission: it only applies when the strategy is
selected by the active regime, the account has no open positions, and
`liquidate_when_inactive` is `false`. It does not change the meaning of
`trade_today: false` for accounts that already have positions.
