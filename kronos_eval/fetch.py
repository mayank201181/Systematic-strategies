"""Fetch Kronos weights and Yahoo daily OHLCV for the futures universe.

Runs in GitHub Actions (needs Hugging Face and Yahoo). Output goes to an
orphan `kronos-assets` branch so the evaluation can run anywhere.
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from huggingface_hub import snapshot_download

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for repo in ["NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"]:
    snapshot_download(repo, local_dir=out / repo.split("/")[1])

MONTHLY = QUARTERLY = None
YAHOO = {
    "SP500": ("ES", "CME", QUARTERLY), "NASDAQ": ("NQ", "CME", QUARTERLY),
    "RUSSELL_mini": ("RTY", "CME", QUARTERLY), "SP400": ("EMD", "CME", QUARTERLY),
    "NIKKEI": ("NIY", "CME", QUARTERLY),
    "US2": ("ZT", "CBT", QUARTERLY), "US5": ("ZF", "CBT", QUARTERLY),
    "US10": ("ZN", "CBT", QUARTERLY), "US20": ("ZB", "CBT", QUARTERLY),
    "US30": ("UB", "CBT", QUARTERLY),
    "AUD": ("6A", "CME", QUARTERLY), "GBP": ("6B", "CME", QUARTERLY),
    "CAD": ("6C", "CME", QUARTERLY), "EUR": ("6E", "CME", QUARTERLY),
    "JPY": ("6J", "CME", QUARTERLY), "NZD": ("6N", "CME", QUARTERLY),
    "CHF": ("6S", "CME", QUARTERLY), "MXP": ("6M", "CME", QUARTERLY),
    "BRE": ("6L", "CME", MONTHLY), "DX": ("DX", "NYB", QUARTERLY),
    "CRUDE_W": ("CL", "NYM", MONTHLY), "BRENT_W": ("BZ", "NYM", MONTHLY),
    "GASOILINE": ("RB", "NYM", MONTHLY), "HEATOIL": ("HO", "NYM", MONTHLY),
    "GAS_US": ("NG", "NYM", MONTHLY),
    "GOLD": ("GC", "CMX", "GJMQVZ"), "SILVER": ("SI", "CMX", "HKNUZ"),
    "COPPER": ("HG", "CMX", "HKNUZ"), "PLAT": ("PL", "NYM", "FJNV"),
    "PALLAD": ("PA", "NYM", "HMUZ"), "ALUMINIUM": ("ALI", "CMX", MONTHLY),
    "CORN": ("ZC", "CBT", "HKNUZ"), "SOYBEAN": ("ZS", "CBT", "FHKNQUX"),
    "SOYOIL": ("ZL", "CBT", "FHKNQUVZ"), "SOYMEAL": ("ZM", "CBT", "FHKNQUVZ"),
    "WHEAT": ("ZW", "CBT", "HKNUZ"), "REDWHEAT": ("KE", "CBT", "HKNUZ"),
    "OATIES": ("ZO", "CBT", "HKNUZ"), "RICE": ("ZR", "CBT", "FHKNUX"),
    "COCOA": ("CC", "NYB", "HKNUZ"), "COFFEE": ("KC", "NYB", "HKNUZ"),
    "SUGAR11": ("SB", "NYB", "HKNV"), "COTTON2": ("CT", "NYB", "HKNVZ"),
    "OJ": ("OJ", "NYB", "FHKNUX"),
    "LIVECOW": ("LE", "CME", "GJMQVZ"), "LEANHOG": ("HE", "CME", "GJKMNQVZ"),
    "FEEDCOW": ("GF", "CME", "FHJKQUVX"),
    "BITCOIN": ("BTC", "CME", MONTHLY), "ETHEREUM": ("ETH", "CME", MONTHLY),
}

tick = {f"{r}=F": i for i, (r, _, _) in YAHOO.items()}
rows = []
for t, inst in tick.items():
    df = yf.download(t, start="2012-01-01", auto_adjust=False, progress=False)
    if df.empty:
        print("no data", t)
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df["instrument"] = inst
    rows.append(df.reset_index(names="date"))
    print(t, inst, len(df), df.index.min().date(), df.index.max().date())
pd.concat(rows).to_parquet(out / "ohlcv.parquet")
