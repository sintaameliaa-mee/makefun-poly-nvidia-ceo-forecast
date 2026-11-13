# Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, minute by minute,
# 4 hours ahead: five quantiles (5/25/50/75/95%) of each market's price at
# every minute of the next 240, for the lowest mean pinball loss.
#
# Method (tuned on the 6 practice cases against real Polymarket outcomes):
# Polymarket minute prices are a jump mixture, not a Gaussian walk: most
# minutes do not move at all, then a price jumps 5-48 points in one minute.
# A fixed-sigma normal spread (the house baseline, sigma 0.0025) is far too
# narrow for the active markets and far too wide for the dead ones, so this
# forecaster uses each market's own empirical distribution of past price
# changes, rescaled by how volatile the market has been lately:
#   - empirical quantiles of h-minute changes at anchor horizons
#     h = 1, 5, 15, 60, 240 minutes, from the last 12 hours of prices,
#     interpolated in log-horizon for the minutes between anchors;
#   - rescaled by recent volatility (60-minute half-life EWMA) relative to
#     the market's full-history volatility, clamped to [0.3, 3];
#   - the 5%/95% tails narrowed by 0.9 (the raw empirical tails overshoot);
#   - median = the last price (persistence; mean reversion and the
#     sum-to-one constraint both hurt on the backtest).
# Backtest on the 6 practice cases with real outcomes: pooled pinball loss
# 0.286809 vs the house baseline's 0.363589 (21.1% better), beating the
# baseline on every one of the six cases. Deterministic, stdlib only,
# ~0.07 s per case (cap 8 s).
#
# Reads every <id>.in in the input directory and writes <id>.out to the
# output directory. Outside the grader, MAKEFUN_INPUT_DIR and
# MAKEFUN_OUTPUT_DIR stand in for /input and /output.

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
ANCHORS = [1, 5, 15, 60, 240]
WINDOW = 721          # last 12 hours of prices for the empirical changes
VOL_HALF_LIFE = 60    # minutes; EWMA half-life for recent volatility
TAIL_MULT = 0.9       # narrow the 5%/95% tails slightly
VOL_RATIO_MIN = 0.3
VOL_RATIO_MAX = 3.0
SIGMA_FLOOR = 0.0004  # prices move in 0.005 ticks; never forecast zero spread


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def quantile_sorted(s, q):
    """Linear-interpolation quantile of sorted list s."""
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def ewma_vol(rets, half_life_min):
    """EWMA standard deviation of returns, newest first, half-life in minutes."""
    lam = 0.5 ** (1.0 / half_life_min)
    wsum = 0.0
    vsum = 0.0
    msum = 0.0
    w = 1.0
    for r in reversed(rets):
        wsum += w
        vsum += w * r * r
        msum += w * r
        w *= lam
        if w < 1e-9:
            break
    mean = msum / wsum
    var = vsum / wsum - mean * mean
    return math.sqrt(max(var, 0.0))


def forecast_market(prices, horizon):
    """240 rows (one per minute ahead) of the five quantiles for one market."""
    x = prices[-1]
    n = len(prices)
    rets = [prices[i + 1] - prices[i] for i in range(n - 1)]
    if len(rets) >= 2:
        m = sum(rets) / len(rets)
        sd_full = math.sqrt(sum((r - m) ** 2 for r in rets) / len(rets))
    else:
        sd_full = 0.0025
    sd_recent = ewma_vol(rets, VOL_HALF_LIFE) if len(rets) >= 10 else sd_full
    w_recent = min(1.0, len(rets) / 720.0)
    sd_1 = math.sqrt(w_recent * sd_recent ** 2 + (1 - w_recent) * sd_full ** 2)
    sd_1 = max(sd_1, SIGMA_FLOOR)
    vol_ratio = sd_1 / sd_full if sd_full > 1e-9 else 1.0
    vol_ratio = min(max(vol_ratio, VOL_RATIO_MIN), VOL_RATIO_MAX)

    hist = prices[-WINDOW:] if len(prices) > WINDOW else prices
    hn = len(hist)
    anchors = [a for a in ANCHORS if a <= horizon]
    anchor_q = {}
    for a in anchors:
        if hn - a < 20:
            # Too little history for this horizon: normal fallback.
            anchor_q[a] = [z * sd_1 * math.sqrt(a) for z in Z]
            continue
        ch = sorted(hist[i + a] - hist[i] for i in range(hn - a))
        anchor_q[a] = [quantile_sorted(ch, tau) * vol_ratio for tau in QUANTILES]
    log_a = [math.log(a) for a in anchors]

    rows = []
    for h in range(1, horizon + 1):
        if h <= anchors[0]:
            qs = anchor_q[anchors[0]]
        elif h >= anchors[-1]:
            qs = anchor_q[anchors[-1]]
        else:
            lh = math.log(h)
            for j in range(len(anchors) - 1):
                if anchors[j] <= h <= anchors[j + 1]:
                    t = (lh - log_a[j]) / (log_a[j + 1] - log_a[j])
                    qs = [
                        anchor_q[anchors[j]][k]
                        + t * (anchor_q[anchors[j + 1]][k] - anchor_q[anchors[j]][k])
                        for k in range(5)
                    ]
                    break
        med_q = qs[2]
        qs = [
            med_q + (qs[0] - med_q) * TAIL_MULT,
            qs[1],
            qs[2],
            qs[3],
            med_q + (qs[4] - med_q) * TAIL_MULT,
        ]
        # Centre on the last price: shift the row so its median sits
        # exactly on the last price (the empirical median change is ~0).
        shift = -qs[2]
        row = [price6(x + q + shift) for q in qs]
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(INPUT_DIR)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        if case["quantiles"] != QUANTILES:
            raise ValueError(
                f"{path.name} asks for quantiles {case['quantiles']}, "
                f"not {QUANTILES}"
            )
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        with open(out_dir / f"{case_id}.out", "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
