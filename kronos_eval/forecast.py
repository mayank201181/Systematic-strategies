"""Zero-shot Kronos-small forecasts for every instrument and every OOS day.

Runs in GitHub Actions with the upstream Kronos repo on sys.path. For each
date t (from 2024-07-01, after the stated June-2024 pre-training cutoff) and
each instrument, it feeds the last LOOKBACK daily bars up to and including t
and draws SAMPLES paths of HORIZON bars. It keeps the sample mean and spread
of the predicted close at each horizon, never any bar after t.

    python forecast.py DATA OUT CHUNK NCHUNKS
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps

LOOKBACK, HORIZON, SAMPLES, START = 96, 5, 8, "2024-07-01"
COLS = ["open", "high", "low", "close", "volume", "amount"]


def sample_paths(tok, model, x, xs, ys, n):
    """Kronos auto_regressive_inference, but return every sample (no averaging)."""
    from model.kronos import sample_from_logits
    with torch.no_grad():
        B, L, _ = x.shape
        x = x.clip(-5, 5).repeat_interleave(n, 0)
        stamp = torch.cat([xs, ys], 1).repeat_interleave(n, 0)
        pre, post = tok.encode(x, half=True)
        for i in range(ys.shape[1]):
            s1, ctx = model.decode_s1(pre, post, stamp[:, : pre.shape[1]])
            a = sample_from_logits(s1[:, -1], temperature=1.0, top_k=0, top_p=0.9)
            s2 = model.decode_s2(ctx, a)
            b = sample_from_logits(s2[:, -1], temperature=1.0, top_k=0, top_p=0.9)
            pre, post = torch.cat([pre, a], 1), torch.cat([post, b], 1)
        z = tok.decode([pre, post], half=True)[:, -ys.shape[1]:]
        return z.reshape(B, n, ys.shape[1], -1).numpy()


def main():
    data, out, chunk, nchunks = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    torch.manual_seed(chunk)
    df = pd.read_parquet(data)
    df["amount"] = df.volume * df[["open", "high", "low", "close"]].mean(axis=1)
    panel = {i: g.set_index("date").sort_index() for i, g in df.groupby("instrument")}
    days = sorted(set(df.date[df.date >= START]))
    days = np.array_split(np.array(days[:-1]), nchunks)[chunk]   # need t+1 to score
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
    rows = []
    for k, t in enumerate(days):
        t0 = time.time()
        xs, stamps, ystamps, meta = [], [], [], []
        for inst, g in panel.items():
            if t not in g.index:
                continue
            h = g.loc[:t].iloc[-LOOKBACK:]
            if len(h) < LOOKBACK:
                continue
            x = h[COLS].to_numpy(np.float32)
            mu, sd = x.mean(0), x.std(0)
            xs.append((x - mu) / (sd + 1e-5))
            stamps.append(calc_time_stamps(h.index.to_series()).to_numpy(np.float32))
            fut = pd.bdate_range(t, periods=HORIZON + 1)[1:].to_series()
            ystamps.append(calc_time_stamps(fut).to_numpy(np.float32))
            meta.append((inst, h.close.iloc[-1], mu[3], sd[3]))
        if not xs:
            continue
        z = sample_paths(tok, model, torch.tensor(np.stack(xs)), torch.tensor(np.stack(stamps)),
                         torch.tensor(np.stack(ystamps)), SAMPLES)
        for j, (inst, c0, mu, sd) in enumerate(meta):
            close = z[j, :, :, 3] * (sd + 1e-5) + mu          # samples x horizon
            r = {"date": t, "instrument": inst, "close": c0}
            for hh in range(HORIZON):
                r[f"mean_{hh+1}"] = close[:, hh].mean()
                r[f"std_{hh+1}"] = close[:, hh].std()
                r[f"up_{hh+1}"] = (close[:, hh] > c0).mean()
            rows.append(r)
        print(f"{k+1}/{len(days)} {pd.Timestamp(t).date()} n={len(meta)} {time.time()-t0:.1f}s", flush=True)
    pd.DataFrame(rows).to_parquet(f"{out}/forecast_{chunk:02d}.parquet")


if __name__ == "__main__":
    main()
