"""Trade ledger for the live books.

    python -m syslib.ledger --data data/merged --out results/book

Each book is a registered strategy run on CAPITAL of notional USD. Positions
are the engine's buffered positions rounded to whole contracts. From
BOOK_START on, every change in a contract position is a trade event:

  open    flat -> long/short
  close   long/short -> flat
  flip    long <-> short (a close and an open on the same day)
  resize  same direction, size changed by at least RESIZE_MIN of the size at
          the last reported event

The ledger is rebuilt from the full position history on every run, so it is
always consistent with the current data. Outputs:

  positions.csv  open trades: entry date and price, contracts, P&L to date
  closed.csv     closed trades with exit and realised P&L
  events.csv     every trade event since BOOK_START
  blotter.md     what changed since the previous run, then the open book
  state.json     date of the last run's data, used to find "new since last run"

P&L is gross of costs: contracts x change in back-adjusted price x point
size x FX, so roll gaps are not P&L.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import s183  # noqa: F401
from .data import load_panel
from .engine import backtest
from .signals import Ctx
from .strategies import REGISTRY

BOOKS = ["trend_carry", "multi_7", "s183_replica"]
CAPITAL = 100_000_000
BOOK_START = "2026-10-26"
RESIZE_MIN = 0.5


def book_positions(panel, name):
    s = REGISTRY[name]
    c = Ctx(panel.subset(s.universe))
    r = backtest(c, s.build(c), name, **s.engine)
    p = c.p
    fx = p.fx_for(list(r.positions.columns)).reindex(r.positions.index).ffill()
    ps = p.meta["pointsize"].reindex(r.positions.columns)
    contracts = (r.positions * CAPITAL / (fx * ps)).round().fillna(0) + 0.0
    pnl = (contracts.shift(1) * c.adj.diff().reindex(contracts.index) * ps * fx).fillna(0)
    return dict(contracts=contracts, pnl=pnl, price=p.price.reindex(contracts.index).ffill(),
                forecast=r.forecasts, meta=p.meta)


def trades_for(name, bp, start):
    con, pnl, px, fc = bp["contracts"], bp["pnl"], bp["price"], bp["forecast"]
    dates = con.index
    start = min(pd.Timestamp(start), dates[-1])
    i0 = int(dates.searchsorted(start))
    events, open_trades, closed = [], [], []
    for inst in con.columns:
        q = con[inst].values
        cur = 0.0
        entry = None
        for t in range(i0, len(dates)):
            new = q[t]
            d = dates[t]
            price = px[inst].iloc[t]
            if t == i0:
                if new != 0:
                    entry = dict(entry_date=d, entry_price=price, contracts=new, t0=t)
                    events.append(dict(date=d, strategy=name, instrument=inst, action="open",
                                       contracts_before=0, contracts_after=new, price=price))
                cur = new
                continue
            if new == cur:
                continue
            flip = cur != 0 and new != 0 and np.sign(new) != np.sign(cur)
            if cur != 0 and (new == 0 or flip):
                realised = pnl[inst].iloc[entry["t0"] + 1:t + 1].sum()
                closed.append(dict(strategy=name, instrument=inst, direction="long" if cur > 0 else "short",
                                   contracts=cur, entry_date=entry["entry_date"],
                                   entry_price=entry["entry_price"], exit_date=d, exit_price=price,
                                   pnl_usd=realised))
                events.append(dict(date=d, strategy=name, instrument=inst,
                                   action="flip" if flip else "close", contracts_before=cur,
                                   contracts_after=new, price=price))
                entry = None
                if flip:
                    entry = dict(entry_date=d, entry_price=price, contracts=new, t0=t)
            elif cur == 0:
                entry = dict(entry_date=d, entry_price=price, contracts=new, t0=t)
                events.append(dict(date=d, strategy=name, instrument=inst, action="open",
                                   contracts_before=0, contracts_after=new, price=price))
            else:
                ref = entry["contracts"]
                if abs(new - ref) >= max(1, RESIZE_MIN * abs(ref)):
                    events.append(dict(date=d, strategy=name, instrument=inst, action="resize",
                                       contracts_before=ref, contracts_after=new, price=price))
                    entry["contracts"] = new
            cur = new
        if entry is not None and cur != 0:
            sgn = np.sign(fc[inst].fillna(0))
            flips = sgn.ne(sgn.shift()).loc[:dates[entry["t0"]]]
            since = flips[flips].index.max() if flips.any() else None
            open_trades.append(dict(
                strategy=name, instrument=inst, direction="long" if cur > 0 else "short",
                contracts=int(cur), forecast=round(float(fc[inst].iloc[-1]), 1),
                entry_date=entry["entry_date"].date(), entry_price=entry["entry_price"],
                last_date=dates[-1].date(), last_price=px[inst].iloc[-1],
                pnl_usd=round(float(pnl[inst].iloc[entry["t0"] + 1:].sum())),
                signal_since=since.date() if since is not None else None))
    return events, open_trades, closed


def _table(df, cols):
    if df.empty:
        return ["_none_", ""]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(r[c]) else
                                     (f"{r[c]:,.0f}" if isinstance(r[c], (int, float, np.floating, np.integer)) and c in ("pnl_usd",)
                                      else (f"{r[c]:,.6g}" if isinstance(r[c], (float, np.floating)) else str(r[c])))
                                     for c in cols) + " |")
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/merged")
    ap.add_argument("--out", default="results/book")
    ap.add_argument("--start", default=BOOK_START)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state_p = out / "state.json"
    prev_asof = json.loads(state_p.read_text())["as_of"] if state_p.exists() else None

    panel = load_panel(a.data)
    ev, op, cl = [], [], []
    desc = panel.meta["description"]
    for name in BOOKS:
        bp = book_positions(panel, name)
        e, o, c = trades_for(name, bp, a.start)
        ev += e
        op += o
        cl += c
        print(f"{name}: {len(o)} open, {len(c)} closed, {len(e)} events")
    events = pd.DataFrame(ev)
    if not events.empty:
        events = events.sort_values(["date", "strategy", "instrument"])
    pos = pd.DataFrame(op)
    closed = pd.DataFrame(cl)
    as_of = panel.adj.dropna(how="all").index.max().date().isoformat()
    for df in (events, pos, closed):
        if not df.empty:
            df.insert(2, "description", df["instrument"].map(desc))
    events.to_csv(out / "events.csv", index=False)
    pos.to_csv(out / "positions.csv", index=False)
    closed.to_csv(out / "closed.csv", index=False)

    new = events
    if prev_asof and not events.empty:
        new = events[pd.to_datetime(events["date"]) > pd.Timestamp(prev_asof)]
    lines = [f"# Trade blotter, data to {as_of}", "",
             f"Books: {', '.join(BOOKS)}, each sized at ${CAPITAL / 1e6:,.0f}m notional, "
             f"20% vol target, live since {a.start}. P&L gross of costs, USD.", "",
             f"## Changes since {'last run (' + prev_asof + ')' if prev_asof else 'inception'}", ""]
    if not new.empty:
        new = new.assign(date=pd.to_datetime(new["date"]).dt.date)
    lines += _table(new, ["date", "strategy", "instrument", "description", "action",
                          "contracts_before", "contracts_after", "price"]) if not new.empty else ["No trades.", ""]
    for name in BOOKS:
        sub = pos[pos.strategy == name] if not pos.empty else pos
        tot = sub["pnl_usd"].sum() if not sub.empty else 0
        lines += [f"## {name}: {len(sub)} open, open P&L ${tot:,.0f}", ""]
        if not sub.empty:
            lines += _table(sub.sort_values(["direction", "instrument"]),
                            ["instrument", "description", "direction", "contracts", "forecast",
                             "entry_date", "entry_price", "last_price", "pnl_usd", "signal_since"])
    (out / "blotter.md").write_text("\n".join(lines))
    state_p.write_text(json.dumps({"as_of": as_of, "n_new_events": int(len(new))}))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
