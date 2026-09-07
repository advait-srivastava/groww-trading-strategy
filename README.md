# groww-trading-strategy

Systematic trading strategies for NSE equities and NIFTY options, built on the
[Groww API](https://groww.in/trade-api/docs), with real backtests and a
dry-run-first execution model.

> **⚠️ Disclaimer:** This is a personal project, not financial advice. It
> trades real money if you pass `--confirm` to the live scripts. Past
> backtested performance does not guarantee future results. Options trading
> involves substantial risk of loss. Use at your own risk.

## What's here

**Equity: momentum + trend rotation** (`strategy/backtest.py`, `strategy/rebalance.py`)
Monthly-rebalanced, long-only strategy over the Nifty 200: rank stocks trading
above their 200-day average by 12-month momentum, size the top 18 by inverse
volatility, with a market-breadth/volatility overlay that raises the cash
buffer when conditions turn risk-off.

Backtested 2022-01-03 to 2026-08-28 against real Groww historical data.
Signals are taken from each rebalance date's close and filled one session
later, so no result depends on trading at a price the signal was derived from.

| | CAGR | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|
| Strategy (breadth/vol overlay, default) | 24.18% | 0.88 | -21.05% | 1.15 |
| Strategy (no overlay) | 25.49% | 0.89 | -25.27% | 1.01 |
| NIFTY 50 buy & hold | 7.03% | 0.04 | -16.47% | 0.43 |

The overlay's case is drawdown, not risk-adjusted return: it costs 1.3 points
of CAGR and leaves Sharpe flat (0.89 -> 0.88), but cuts max drawdown by 4.2
points and lifts Calmar from 1.01 to 1.15. The Alpha Vantage macro overlay
(`--macro-regime`) stays off by default -- it backtests worse than doing
nothing (Sharpe 0.82, max drawdown -26.05% used alone).

**These figures are survivorship-biased and read high.** NSE publishes only
the *current* Nifty 200 list and no dated archive, so history is tested
against today's constituents: names dropped from the index are absent, and
names added are present partly because they rose. `strategy/universe.py` keeps
a dated snapshot archive, and `--point-in-time` restricts each rebalance to
the membership known as of that date; every run prints whether its universe is
point-in-time or biased. Against a synthetic earlier snapshot, restricting
membership moved CAGR to 16.72% and Sharpe to 0.53 -- index composition
matters more here than any other modelling choice. The archive only covers
dates recorded from 2026-08-30 onward.

**Options: weekly NIFTY bull put credit spread** (`strategy/options_backtest.py`, `strategy/options_rebalance.py`)
Sells a defined-risk put spread each week when the same regime signal reads
risk-on, sized to a fixed budget. Groww has no historical option data, so
this is validated with a Black-Scholes approximation (clearly caveated in
the code) plus forward paper-tracking — not a real historical backtest.

## Architecture

```
client.py                  Groww auth (API key+secret or TOTP)
place_order.py              One-off manual order placement (dry-run by default)
portfolio.py                Print current holdings/positions
check_connection.py         Sanity-check API connectivity

strategy/
  universe.py                Live Nifty 200 constituent list (from NSE)
  data.py                     Historical OHLCV fetch/cache from Groww
  factors.py                  Core signal logic: trend filter, momentum, inverse-vol weighting
  regime.py                   Market breadth + volatility regime signal (validated overlay)
  macro.py                    Alpha Vantage macro overlay (USD/INR, US 10Y yield) — off by
                               default; backtested worse than no overlay, kept for reference
  backtest.py                 Equity strategy backtest engine
  rebalance.py                Live equity rebalance (dry-run by default; --confirm to trade)
  risk_check.py               Daily stop-loss circuit breaker (sells only)

  options_pricing.py          Black-Scholes helpers
  options_strategy.py         Shared credit-spread rules (backtest + live)
  options_backtest.py         Synthetic options backtest (see caveats in its docstring)
  options_rebalance.py        Live options entry/exit (dry-run by default; --confirm to trade)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Generate credentials at the [Groww API Keys page](https://groww.in/trade-api/api-keys)
and fill in `.env`. Two auth methods are supported:

- **API key + secret** — needs daily re-approval on the Groww dashboard.
- **TOTP** — longer-lived, no daily approval, recommended for anything you'll run more than once a day.

An [Alpha Vantage](https://www.alphavantage.co/support/#api-key) key (free
tier) is optional, only needed if you re-enable the macro overlay in
`strategy/backtest.py --macro-regime`.

## Usage

All live-trading scripts print a dry-run plan by default and require an
explicit `--confirm` flag to place real orders.

```bash
# Equity strategy
python -m strategy.backtest                    # backtest with current defaults
python -m strategy.rebalance                    # see this month's target portfolio + trades
python -m strategy.rebalance --confirm          # actually place the trades
python -m strategy.risk_check                   # check for stop-loss breaches (run daily)

# Options strategy
python -m strategy.options_backtest             # synthetic backtest
python -m strategy.options_rebalance entry      # see this week's spread trade
python -m strategy.options_rebalance entry --confirm
python -m strategy.options_rebalance exit       # check/close an open position (run daily)
```

## Known limitations

- **Survivorship bias**: the equity backtest ranks *today's* Nifty 200 list
  against history — index constituents removed over 2022–2026 aren't
  included, which can flatter results vs. a true point-in-time process.
- **Data window**: Groww's historical API only reaches back to ~2020, so
  every backtest here covers one market regime, not a full multi-cycle history.
- **No historical options data**: Groww's instrument master only carries
  live contracts. The options backtest is a Black-Scholes approximation on
  real NIFTY prices, not a backtest on real traded option prices.
- **Costs**: equity backtest assumes 10bps per unit of turnover (stress-tested
  to 25bps); real slippage on rebalance day, especially on names breaking
  trend alongside the rest of the market, could exceed this.
