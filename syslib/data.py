"""Daily price panels built from pysystemtrade's public futures CSVs.

build_panels() turns the (intraday-stamped) adjusted and multiple-price CSVs
into daily wide panels (dates x instruments) and writes them as parquet.
load_panel() reads them back as a Panel.

Panels:
  adj        back-adjusted (Panama) price, used for returns and trend
  price      price of the currently held contract, unadjusted
  carry      price of the carry contract
  price_ym   held contract as fractional years (yyyy + (mm-1)/12)
  carry_ym   carry contract as fractional years
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import BROAD, GROUP_OF

PANELS = ["adj", "price", "carry", "price_ym", "carry_ym"]


def _daily_last(df):
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df[df.index.dayofweek < 5]


def _contract_to_years(c):
    c = pd.to_numeric(c, errors="coerce")
    yyyy = np.floor(c / 10000)
    mm = np.floor((c % 10000) / 100)
    return yyyy + (mm - 1) / 12.0


def build_panels(pst_data_dir, out_dir, instruments=BROAD):
    """Build daily panels from a pysystemtrade checkout's data/futures directory."""
    pst = Path(pst_data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = {k: {} for k in PANELS}
    for inst in instruments:
        f_adj = pst / "adjusted_prices_csv" / f"{inst}.csv"
        f_mul = pst / "multiple_prices_csv" / f"{inst}.csv"
        if not f_adj.exists() or not f_mul.exists():
            print(f"  skip {inst}: no data")
            continue
        adj = _daily_last(pd.read_csv(f_adj, index_col=0).iloc[:, 0])
        mul = _daily_last(pd.read_csv(f_mul, index_col=0))
        cols["adj"][inst] = adj
        cols["price"][inst] = mul["PRICE"]
        cols["carry"][inst] = mul["CARRY"]
        cols["price_ym"][inst] = _contract_to_years(mul["PRICE_CONTRACT"])
        cols["carry_ym"][inst] = _contract_to_years(mul["CARRY_CONTRACT"])
    for k, d in cols.items():
        pd.DataFrame(d).sort_index().to_parquet(out / f"{k}.parquet")
    cfg = pd.read_csv(pst / "csvconfig" / "instrumentconfig.csv").set_index("Instrument")
    sc = pd.read_csv(pst / "csvconfig" / "spreadcosts.csv").set_index("Instrument")
    insts = sorted(cols["adj"])
    meta = pd.DataFrame({
        "group": [GROUP_OF.get(i, "Other") for i in insts],
        "description": cfg.Description.reindex(insts).values,
        "currency": cfg.Currency.reindex(insts).values,
        "pointsize": cfg.Pointsize.reindex(insts).values,
        "commission": cfg.PerBlock.reindex(insts).fillna(0).values,
        "spread": sc.SpreadCost.reindex(insts).values,
    }, index=pd.Index(insts, name="instrument"))
    meta.to_csv(out / "meta.csv")
    print(f"Built panels for {len(insts)} instruments -> {out}")
    return meta


@dataclass
class Panel:
    adj: pd.DataFrame
    price: pd.DataFrame
    carry: pd.DataFrame
    price_ym: pd.DataFrame
    carry_ym: pd.DataFrame
    meta: pd.DataFrame

    @property
    def instruments(self):
        return list(self.adj.columns)

    def subset(self, instruments=None, start=None, end=None):
        insts = [i for i in (instruments or self.instruments) if i in self.adj.columns]
        sl = slice(start, end)
        return Panel(*(getattr(self, k).loc[sl, insts] for k in PANELS),
                     meta=self.meta.loc[insts])

    @property
    def groups(self):
        return self.meta["group"]


def load_panel(data_dir, instruments=None, start=None, end=None):
    d = Path(data_dir)
    frames = {k: pd.read_parquet(d / f"{k}.parquet") for k in PANELS}
    idx = frames["adj"].index
    frames = {k: v.reindex(idx) for k, v in frames.items()}
    meta = pd.read_csv(d / "meta.csv", index_col=0)
    p = Panel(**frames, meta=meta)
    if instruments is not None or start is not None or end is not None:
        p = p.subset(instruments, start, end)
    return p
