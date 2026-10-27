import numpy as np
import pandas as pd

from syslib.data import Panel
from syslib.engine import backtest
from syslib.metrics import haircut_sharpe, sharpe
from syslib.signals import Ctx, scale_forecast


def _panel(n=3000, k=4, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    cols = [f"I{i}" for i in range(k)]
    steps = rng.normal(drift, 1.0, (n, k))
    adj = pd.DataFrame(100 + steps.cumsum(0), idx, cols)
    price = adj + 1000
    ym = pd.DataFrame(2000.0, idx, cols)
    meta = pd.DataFrame({"group": ["A", "A", "B", "B"][:k], "pointsize": 1.0,
                         "commission": 0.0, "spread": 0.0}, index=cols)
    return Panel(adj, price, price, ym, ym + 0.25, meta)


def test_forecast_scaling_targets_10():
    p = _panel()
    raw = Ctx(p).norm_price.diff(20)
    f = scale_forecast(raw)
    assert abs(f.iloc[1000:].abs().mean().mean() - 10) < 2
    assert f.abs().max().max() <= 20


def test_vol_target_hit_with_constant_forecast():
    p = _panel(n=4000)
    c = Ctx(p)
    fc = (c.adj * 0 + 10).where(c.adj.notna())
    r = backtest(c, fc, idm=1.0, buffer=0.0)
    vol = r.returns.iloc[300:].std() * 16
    # 4 uncorrelated instruments at IDM 1 -> 20% / 2 = 10%
    assert 0.08 < vol < 0.12


def test_no_lookahead_random_walk_has_no_edge():
    p = _panel(n=5000, k=4, seed=3)
    c = Ctx(p)
    from syslib.signals import ewmac
    fc = scale_forecast(ewmac(c, 16, 64))
    r = backtest(c, fc)
    assert abs(sharpe(r.returns)) < 0.5


def test_perfect_foresight_is_detected():
    # a forecast that peeks at tomorrow must produce a huge Sharpe: sanity check
    # that the engine aligns forecast t with the move from t to t+1
    p = _panel(n=2000, seed=4)
    c = Ctx(p)
    fc = np.sign(c.diff.shift(-1)) * 10
    r = backtest(c, fc, buffer=0.0)
    assert sharpe(r.returns) > 5


def test_costs_reduce_returns():
    p = _panel(seed=5)
    p.meta["spread"] = 0.5
    c = Ctx(p)
    from syslib.signals import ewmac
    fc = scale_forecast(ewmac(c, 4, 16))
    r = backtest(c, fc)
    assert r.costs.sum() > 0
    assert (r.gross - r.costs).sum() < r.gross.sum()


def test_haircut_shrinks():
    assert haircut_sharpe(1.0, 3.0, 20) < 1.0


def test_ledger_events_open_flip_close():
    from syslib.ledger import trades_for
    idx = pd.bdate_range("2026-01-05", periods=6)
    con = pd.DataFrame({"X": [0, 2, 2, -3, -3, 0]}, index=idx, dtype=float)
    bp = dict(contracts=con, pnl=con * 0 + 1.0, price=con * 0 + 100.0,
              forecast=con * 5.0)
    ev, op, cl = trades_for("b", bp, "2026-01-05")
    assert [e["action"] for e in ev] == ["open", "flip", "close"]
    assert len(cl) == 2 and not op
    assert cl[0]["pnl_usd"] == 2.0   # held from day 1 to day 3
