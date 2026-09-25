"""Empirical price-action study on the bundled 1-min NIFTY / SENSEX data."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/jaysathvara/VPIN"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

DATA_TAG = "12m"   # override with PA_DATA_TAG=3y to use the Dhan 3-year download

def load(sym, tag=None):
    tag = tag or __import__("os").environ.get("PA_DATA_TAG", DATA_TAG)
    df = pd.read_csv(f"{ROOT}/algo_vpin_v2/data/{sym}_{tag}_1min.csv.gz")
    df["timestamp"] = pd.to_datetime(df["timestamp"]); df = df.sort_values("timestamp")
    t = df.timestamp.dt.time
    df = df[(t >= pd.Timestamp("09:15").time()) & (t < pd.Timestamp("15:30").time())].copy()
    df["date"] = df.timestamp.dt.date
    full = df.groupby("date").size(); df = df[df.date.isin(full[full >= 370].index)]
    df["min"] = (df.timestamp.dt.hour - 9) * 60 + df.timestamp.dt.minute - 15  # 0..374
    return df.reset_index(drop=True)

def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan

def study(sym):
    df = load(sym); out = {}
    days = [g.reset_index(drop=True) for _, g in df.groupby("date", sort=True)]
    dates = [g.date.iloc[0] for g in days]
    print(f"\n{'='*30} {sym.upper()} : {len(days)} sessions {'='*30}")

    # ---------- A. time-of-day profile ----------
    df["b5"] = df["min"] // 5
    rng5 = df.groupby(["date", "b5"]).agg(h=("high", "max"), l=("low", "min"), c=("close", "last"), o=("open", "first"))
    rng5["r"] = rng5.h - rng5.l
    prof = rng5.groupby("b5").r.median()
    labels = {b: f"{9 + (15 + 5*b)//60:02d}:{(15 + 5*b)%60:02d}" for b in prof.index}
    print("\nA. median 5-min range (pts) by time of day:")
    print("  " + "  ".join(f"{labels[b]}={prof[b]:.1f}" for b in prof.index if b % 3 == 0))
    dayrng = df.groupby("date").agg(h=("high", "max"), l=("low", "min")); dayrng["r"] = dayrng.h - dayrng.l
    pct_by = {}
    for m in (15, 30, 60, 120, 180):
        part = df[df["min"] < m].groupby("date").agg(h=("high", "max"), l=("low", "min"))
        pct_by[m] = float(((part.h - part.l) / dayrng.r).median())
    print(f"  median day range {dayrng.r.median():.0f} pts; fraction of day range formed by minute:", {k: round(v, 2) for k, v in pct_by.items()})
    noise = {k: float(df.groupby("date").apply(lambda g: (g.close.diff(k)).abs().median()).median()) for k in (1, 3, 5, 15)}
    print("  median absolute close-to-close move over k minutes (noise floor):", {k: round(v, 1) for k, v in noise.items()})
    out["tod_profile"] = {labels[b]: round(float(prof[b]), 2) for b in prof.index}; out["day_range_median"] = float(dayrng.r.median()); out["range_fraction_by_min"] = pct_by; out["noise"] = noise

    # ---------- B. serial correlation of k-min returns ----------
    print("\nB. lag-1 autocorrelation of non-overlapping k-minute log returns (all session / first hour / 10:15-14:30 / last hour):")
    rows = []
    for k in (1, 3, 5, 15, 30):
        recs = {"k": k}
        for name, lo, hi in (("all", 0, 375), ("first60", 0, 60), ("mid", 60, 315), ("last60", 315, 375)):
            r1, r2 = [], []
            for g in days:
                c = g.close.values
                idx = np.arange(lo, hi - k, k)
                rr = np.log(c[np.minimum(idx + k, 374)] / c[idx])
                r1.extend(rr[:-1]); r2.extend(rr[1:])
            r1, r2 = np.array(r1), np.array(r2)
            ac = np.corrcoef(r1, r2)[0, 1]; n = len(r1)
            recs[name] = f"{ac:+.3f} (t={ac*np.sqrt(n):+.1f})"
        rows.append(recs)
    print(pd.DataFrame(rows).to_string(index=False))
    # conditional continuation after a large k-min move
    print("  after a k-min move in the top decile of |move|: mean next-k-min move in the same direction (pts), hit-rate of same sign:")
    cont = {}
    for k in (1, 5, 15):
        prev, nxt = [], []
        for g in days:
            c = g.close.values
            for i in range(k, 375 - k, k):
                prev.append(c[i] - c[i - k]); nxt.append(c[i + k] - c[i])
        prev, nxt = np.array(prev), np.array(nxt)
        thr = np.quantile(np.abs(prev), 0.9); m = np.abs(prev) >= thr
        same = nxt[m] * np.sign(prev[m])
        cont[k] = (float(same.mean()), float((same > 0).mean()), float(tstat(same)), int(m.sum()))
        print(f"    k={k:2d}: mean {same.mean():+.2f} pts, same-sign {100*(same>0).mean():.1f}%, t={tstat(same):+.2f}, n={m.sum()}")
    out["continuation_top_decile"] = cont

    # ---------- C. intraday momentum (first 30 min -> last 30 min) ----------
    f30, l30, f60, rest = [], [], [], []
    for g in days:
        c = g.close.values; o = g.open.values[0]
        f30.append(c[29] / o - 1); l30.append(c[-1] / c[344] - 1); f60.append(c[59] / o - 1); rest.append(c[-1] / c[59] - 1)
    f30, l30, f60, rest = map(np.array, (f30, l30, f60, rest))
    b = np.polyfit(f30, l30, 1)[0]; agree = (np.sign(f30) == np.sign(l30)).mean()
    b2 = np.polyfit(f60, rest, 1)[0]; agree2 = (np.sign(f60) == np.sign(rest)).mean()
    print(f"\nC. intraday momentum: first30->last30 slope {b:+.3f}, sign agreement {100*agree:.1f}% (n={len(f30)}), corr {np.corrcoef(f30,l30)[0,1]:+.3f}")
    print(f"   first60->rest-of-day slope {b2:+.3f}, sign agreement {100*agree2:.1f}%, corr {np.corrcoef(f60,rest)[0,1]:+.3f}")
    out["intraday_momentum"] = {"f30_l30_agree": float(agree), "f30_l30_corr": float(np.corrcoef(f30, l30)[0, 1]), "f60_rest_agree": float(agree2), "f60_rest_corr": float(np.corrcoef(f60, rest)[0, 1])}

    # ---------- D. opening range breakout ----------
    print("\nD. opening-range breakout (first break after OR ends, before 12:00):")
    orb = {}
    for orm in (5, 15, 30):
        recs = []
        for g in days:
            h, l, c = g.high.values, g.low.values, g.close.values
            orh, orl = h[:orm].max(), l[:orm].min(); orr = orh - orl; mid = (orh + orl) / 2
            brk = None
            for i in range(orm, 165):
                if c[i] > orh: brk = (i, +1); break
                if c[i] < orl: brk = (i, -1); break
            if brk is None: recs.append({"orr": orr, "broke": 0}); continue
            i, d = brk; e = c[i]
            fut_c = c[i + 1:]; fut_h = h[i + 1:]; fut_l = l[i + 1:]
            # did it reach +1 OR (target) before returning to OR mid (stop)?
            tgt = e + d * orr; stp = mid
            hit_t = np.where((fut_h >= tgt) if d > 0 else (fut_l <= tgt))[0]; hit_s = np.where((fut_l <= stp) if d > 0 else (fut_h >= stp))[0]
            t_i = hit_t[0] if len(hit_t) else 10**6; s_i = hit_s[0] if len(hit_s) else 10**6
            win = t_i < s_i
            # fakeout: close back inside OR within 15 bars
            fake = any((fut_c[:15] < orh) if d > 0 else (fut_c[:15] > orl))
            mfe = (fut_h[:60].max() - e) * d if d > 0 else (e - fut_l[:60].min())
            mae = (e - fut_l[:60].min()) if d > 0 else (fut_h[:60].max() - e)
            close_dir = (c[-1] - e) * d
            # both sides broken?
            both = (c[i:].min() < orl) if d > 0 else (c[i:].max() > orh)
            recs.append({"orr": orr, "broke": 1, "dir": d, "bar": i, "win_1R_before_mid": win, "fake15": fake, "mfe60": mfe, "mae60": mae, "to_close": close_dir, "both": both, "no_res": t_i == s_i == 10**6})
        r = pd.DataFrame(recs); rb = r[r.broke == 1].copy()
        for col in ("win_1R_before_mid", "fake15", "both", "no_res"): rb[col] = rb[col].astype(bool)
        print(f"  OR{orm:2d}: median OR range {r.orr.median():.0f} pts | broke {100*r.broke.mean():.0f}% of days | 1xOR target before mid-OR stop: {100*rb.win_1R_before_mid.mean():.0f}% (unresolved {100*rb.no_res.mean():.0f}%) | closed back inside within 15m: {100*rb.fake15.mean():.0f}% | later broke other side: {100*rb.both.mean():.0f}%")
        print(f"        entry->close in break direction: mean {rb.to_close.mean():+.1f} pts, median {rb.to_close.median():+.1f}, >0 on {100*(rb.to_close>0).mean():.0f}% (t={tstat(rb.to_close):+.2f}) | 60-min MFE med {rb.mfe60.median():.0f}, MAE med {rb.mae60.median():.0f}")
        orb[orm] = {"or_med": float(r.orr.median()), "broke": float(r.broke.mean()), "win1R": float(rb.win_1R_before_mid.mean()), "fake15": float(rb.fake15.mean()), "to_close_mean": float(rb.to_close.mean()), "to_close_pos": float((rb.to_close > 0).mean()), "t": float(tstat(rb.to_close)), "n": int(len(rb))}
    out["orb"] = orb

    # ---------- E. previous-day high / low ----------
    print("\nE. previous-day high/low: first touch (close within 0.05%) -> what happens next:")
    pd_recs = []
    for j in range(1, len(days)):
        if (dates[j] - dates[j - 1]).days > 5: continue
        p, g = days[j - 1], days[j]
        pdh, pdl, pdc = p.high.max(), p.low.min(), p.close.iloc[-1]
        c, h, l = g.close.values, g.high.values, g.low.values
        for lvl, name, d in ((pdh, "PDH", +1), (pdl, "PDL", -1)):
            if d > 0 and c[0] > lvl: continue
            if d < 0 and c[0] < lvl: continue
            touch = np.where((h >= lvl) if d > 0 else (l <= lvl))[0]
            touch = touch[(touch >= 5) & (touch < 345)]
            if not len(touch): continue
            i = touch[0]; e = c[i]
            broke = (c[i] > lvl) if d > 0 else (c[i] < lvl)
            f15 = (c[min(i + 15, 374)] - e) * d; f30 = (c[min(i + 30, 374)] - e) * d
            held = ((c[min(i + 15, 374)] > lvl) if d > 0 else (c[min(i + 15, 374)] < lvl))
            pd_recs.append({"lvl": name, "bar": i, "broke_on_touch": broke, "f15": f15, "f30": f30, "held15": held, "to_close": (c[-1] - e) * d})
    r = pd.DataFrame(pd_recs)
    for name in ("PDH", "PDL"):
        x = r[r.lvl == name]
        print(f"  {name}: touches {len(x)} | closed beyond on touch bar {100*x.broke_on_touch.mean():.0f}% | beyond level 15m later {100*x.held15.mean():.0f}% | next 15m {x.f15.mean():+.1f} pts (t={tstat(x.f15):+.2f}), 30m {x.f30.mean():+.1f} (t={tstat(x.f30):+.2f}) | to close {x.to_close.mean():+.1f} (t={tstat(x.to_close):+.2f})")
        xb = x[x.broke_on_touch]; xr = x[~x.broke_on_touch]
        print(f"       if closed beyond (n={len(xb)}): next 30m {xb.f30.mean():+.1f} (t={tstat(xb.f30):+.2f}), to close {xb.to_close.mean():+.1f} | if rejected (n={len(xr)}): next 30m {xr.f30.mean():+.1f} (t={tstat(xr.f30):+.2f}), to close {xr.to_close.mean():+.1f}")
    out["pdhl"] = r.groupby("lvl").agg(n=("f30", "size"), f30=("f30", "mean"), held15=("held15", "mean"), to_close=("to_close", "mean")).to_dict("index")

    # ---------- F. gaps ----------
    print("\nF. opening gaps vs previous close:")
    gr = []
    for j in range(1, len(days)):
        if (dates[j] - dates[j - 1]).days > 5: continue
        p, g = days[j - 1], days[j]; pdc = p.close.iloc[-1]; o = g.open.iloc[0]; c = g.close.values; h, l = g.high.values, g.low.values
        gap = (o - pdc) / pdc * 100; d = np.sign(gap)
        filled = ((l.min() <= pdc) if d > 0 else (h.max() >= pdc)) if d != 0 else True
        fill_bar = int(np.where((l <= pdc) if d > 0 else (h >= pdc))[0][0]) if (filled and d != 0) else None
        gr.append({"gap": gap, "absgap": abs(gap), "filled": filled, "fill_bar": fill_bar, "first60_with_gap": (c[59] - o) * d, "day_with_gap": (c[-1] - o) * d})
    r = pd.DataFrame(gr); r["bucket"] = pd.cut(r.absgap, [0, 0.2, 0.5, 1.0, 10], labels=["<0.2%", "0.2-0.5%", "0.5-1%", ">1%"])
    print(r.groupby("bucket").agg(n=("gap", "size"), filled=("filled", "mean"), med_fill_min=("fill_bar", "median"), first60_in_gap_dir=("first60_with_gap", "mean"), day_in_gap_dir=("day_with_gap", "mean")).round(2).to_string())
    out["gaps"] = r.groupby("bucket").agg(n=("gap", "size"), filled=("filled", "mean"), first60=("first60_with_gap", "mean"), day=("day_with_gap", "mean")).to_dict("index")

    # ---------- G. candlestick patterns on 1-min and 5-min ----------
    print("\nG. candlestick patterns -> forward move (pts) in pattern direction; compared with unconditional |move|:")
    def candles(g, k):
        if k == 1: return g[["open", "high", "low", "close"]].values
        o = g.open.values[::k]; c = g.close.values[k - 1::k]; n = min(len(o), len(c))
        h = np.array([g.high.values[i*k:(i+1)*k].max() for i in range(n)]); l = np.array([g.low.values[i*k:(i+1)*k].min() for i in range(n)])
        return np.column_stack([o[:n], h, l, c[:n]])
    pat = {}
    for k in (1, 5):
        recs = []
        for g in days:
            cd = candles(g, k); n = len(cd)
            for i in range(2, n - 4):
                o, h, l, c = cd[i]; rngc = h - l
                if rngc <= 0: continue
                body = abs(c - o); up_w = h - max(o, c); dn_w = min(o, c) - l
                po, ph, pl, pc = cd[i - 1]
                fwd1 = cd[i + 1][3] - c; fwd3 = cd[i + 3][3] - c
                sig = None
                if dn_w >= 0.6 * rngc and body <= 0.3 * rngc: sig = ("hammer", +1)
                elif up_w >= 0.6 * rngc and body <= 0.3 * rngc: sig = ("shooting_star", -1)
                elif c > o and pc < po and c >= ph and o <= pl and body > 0.5 * rngc: sig = ("bull_engulf", +1)
                elif c < o and pc > po and c <= pl and o >= ph and body > 0.5 * rngc: sig = ("bear_engulf", -1)
                # context: after a 3-candle decline for hammer / rise for star
                if sig is None: continue
                trend_ok = (cd[i - 1][3] < cd[i - 3][3]) if sig[1] > 0 else (cd[i - 1][3] > cd[i - 3][3])
                recs.append({"pat": sig[0], "d": sig[1], "ctx": trend_ok, "f1": fwd1 * sig[1], "f3": fwd3 * sig[1]})
        r = pd.DataFrame(recs)
        uncond = np.median(np.abs(np.diff(np.concatenate([candles(g, k)[:, 3] for g in days]))))
        print(f"  {k}-min candles (unconditional median |1-candle move| {uncond:.1f} pts):")
        for p_, x in r.groupby("pat"):
            xc = x[x.ctx]
            print(f"    {p_:14s} n={len(x):5d} next1 {x.f1.mean():+.2f} (t={tstat(x.f1):+.2f}) next3 {x.f3.mean():+.2f} (t={tstat(x.f3):+.2f}) | with prior trend context n={len(xc)} next3 {xc.f3.mean():+.2f} (t={tstat(xc.f3):+.2f}) hit {100*(xc.f3>0).mean():.0f}%")
            pat[f"{k}m_{p_}"] = {"n": int(len(x)), "f3": float(x.f3.mean()), "t3": float(tstat(x.f3)), "ctx_n": int(len(xc)), "ctx_f3": float(xc.f3.mean()), "ctx_t3": float(tstat(xc.f3))}
    out["candles"] = pat

    # ---------- H. structure: 5-min swing regime ----------
    print("\nH. 5-min market structure: after a confirmed higher-high + higher-low (or LH+LL) on 5-min swings, next 15/30-min move in trend direction:")
    recs = []
    for g in days:
        cd = candles(g, 5); h, l, c = cd[:, 1], cd[:, 2], cd[:, 3]; n = len(cd)
        sh = [i for i in range(2, n - 2) if h[i] == h[i-2:i+3].max()]; sl = [i for i in range(2, n - 2) if l[i] == l[i-2:i+3].min()]
        for i in range(8, n - 6):
            hs = [x for x in sh if x <= i - 2][-2:]; ls = [x for x in sl if x <= i - 2][-2:]
            if len(hs) < 2 or len(ls) < 2: continue
            up = h[hs[1]] > h[hs[0]] and l[ls[1]] > l[ls[0]]; dn = h[hs[1]] < h[hs[0]] and l[ls[1]] < l[ls[0]]
            if not (up or dn): continue
            d = 1 if up else -1
            recs.append({"d": d, "f3": (c[i + 3] - c[i]) * d, "f6": (c[i + 6] - c[i]) * d, "pullback": (c[i] < c[i - 1]) if up else (c[i] > c[i - 1])})
    r = pd.DataFrame(recs)
    print(f"  n={len(r)} | next 15m {r.f3.mean():+.2f} pts (t={tstat(r.f3):+.2f}, hit {100*(r.f3>0).mean():.0f}%) | next 30m {r.f6.mean():+.2f} (t={tstat(r.f6):+.2f})")
    rp = r[r.pullback]
    print(f"  entering only on a pullback candle (n={len(rp)}): next 15m {rp.f3.mean():+.2f} (t={tstat(rp.f3):+.2f}), next 30m {rp.f6.mean():+.2f} (t={tstat(rp.f6):+.2f}), hit {100*(rp.f6>0).mean():.0f}%")
    out["structure"] = {"n": int(len(r)), "f15": float(r.f3.mean()), "t15": float(tstat(r.f3)), "f30": float(r.f6.mean()), "t30": float(tstat(r.f6)), "pb_n": int(len(rp)), "pb_f30": float(rp.f6.mean()), "pb_t30": float(tstat(rp.f6))}

    # ---------- I. trend day detection ----------
    print("\nI. trend-day detection: at time T, price beyond OR15 by >= 0.5x OR range -> move from T to close in that direction:")
    out["trend_day"] = {}
    for T, lab in ((60, "10:15"), (75, "10:30"), (105, "11:00")):
        recs = []
        for g in days:
            h, l, c = g.high.values, g.low.values, g.close.values
            orh, orl = h[:15].max(), l[:15].min(); orr = orh - orl; p = c[T]
            if p >= orh + 0.5 * orr: d = 1
            elif p <= orl - 0.5 * orr: d = -1
            else: continue
            recs.append({"d": d, "to_close": (c[-1] - p) * d, "mae": (p - l[T:].min()) if d > 0 else (h[T:].max() - p)})
        r = pd.DataFrame(recs)
        print(f"  {lab}: qualifying {len(r):3d}/{len(days)} | to close: mean {r.to_close.mean():+.1f} pts, median {r.to_close.median():+.1f}, >0 on {100*(r.to_close>0).mean():.0f}% (t={tstat(r.to_close):+.2f}) | median MAE after {r.mae.median():.0f} pts")
        out["trend_day"][lab] = {"n": int(len(r)), "to_close": float(r.to_close.mean()), "pos": float((r.to_close > 0).mean()), "t": float(tstat(r.to_close))}

    # ---------- J. mean-reversion / fade tests ----------
    print("\nJ. fade tests (spot pts, no costs):")
    noise15 = noise[15]
    # J1: fade the first 30-min move from 09:45 to 15:15 / 11:15
    recs = []
    for g in days:
        c, o = g.close.values, g.open.values[0]; mv = c[29] - o
        d = -np.sign(mv)
        if d == 0: continue
        recs.append({"absmv_pct": abs(mv) / o * 100, "to_1115": (c[120] - c[30]) * d, "to_1515": (c[360] - c[30]) * d, "mae_1515": ((c[30] - g.low.values[31:361].min()) if d > 0 else (g.high.values[31:361].max() - c[30]))})
    r = pd.DataFrame(recs); out["fade30"] = {}
    for thr in (0.0, 0.15, 0.3, 0.5):
        x = r[r.absmv_pct >= thr]
        print(f"  J1 fade first-30m move (|move|>={thr}% , n={len(x):3d}): to 11:15 {x.to_1115.mean():+.1f} pts (hit {100*(x.to_1115>0).mean():.0f}%, t={tstat(x.to_1115):+.2f}) | to 15:15 {x.to_1515.mean():+.1f} (hit {100*(x.to_1515>0).mean():.0f}%, t={tstat(x.to_1515):+.2f}) | median MAE {x.mae_1515.median():.0f}")
        out["fade30"][thr] = {"n": int(len(x)), "to_1115": float(x.to_1115.mean()), "t_1115": float(tstat(x.to_1115)), "to_1515": float(x.to_1515.mean()), "t_1515": float(tstat(x.to_1515)), "hit_1515": float((x.to_1515 > 0).mean())}
    # J2: fade the gap (>= thr) from 09:20 toward previous close, exit 15:15 or when filled
    recs = []
    for j in range(1, len(days)):
        if (dates[j] - dates[j - 1]).days > 5: continue
        p, g = days[j - 1], days[j]; pdc = p.close.iloc[-1]; c, h, l = g.close.values, g.high.values, g.low.values
        gap = (g.open.values[0] - pdc) / pdc * 100; d = -np.sign(gap)
        if d == 0: continue
        e = c[5]
        fill = np.where((h[6:361] >= pdc) if d > 0 else (l[6:361] <= pdc))[0]
        exit_p = pdc if len(fill) else c[360]
        recs.append({"absgap": abs(gap), "pnl": (exit_p - e) * d, "filled": len(fill) > 0, "mae": ((e - l[6:361].min()) if d > 0 else (h[6:361].max() - e))})
    r = pd.DataFrame(recs); out["gapfade"] = {}
    for thr in (0.2, 0.3, 0.5, 0.75):
        x = r[r.absgap >= thr]
        print(f"  J2 fade gap >= {thr}% (n={len(x):3d}): pnl to fill-or-15:15 {x.pnl.mean():+.1f} pts (hit {100*(x.pnl>0).mean():.0f}%, filled {100*x.filled.mean():.0f}%, t={tstat(x.pnl):+.2f}) | median MAE {x.mae.median():.0f}")
        out["gapfade"][thr] = {"n": int(len(x)), "pnl": float(x.pnl.mean()), "t": float(tstat(x.pnl)), "hit": float((x.pnl > 0).mean())}
    # J3: first-hour 15-min reversal: bars 15..60, 15-min move >= k * noise15 -> opposite for next 15 min (first qualifying per day)
    out["rev15"] = {}
    for k in (1.5, 2.0, 3.0):
        recs = []
        for g in days:
            c = g.close.values
            for i in range(15, 61):
                mv = c[i] - c[i - 15]
                if abs(mv) >= k * noise15:
                    d = -np.sign(mv); recs.append({"f15": (c[i + 15] - c[i]) * d, "f30": (c[i + 30] - c[i]) * d}); break
        r = pd.DataFrame(recs)
        print(f"  J3 first-hour 15m reversal (|15m move| >= {k}x{noise15:.0f} pts, n={len(r):3d}): next 15m {r.f15.mean():+.1f} (hit {100*(r.f15>0).mean():.0f}%, t={tstat(r.f15):+.2f}) | next 30m {r.f30.mean():+.1f} (t={tstat(r.f30):+.2f})")
        out["rev15"][k] = {"n": int(len(r)), "f15": float(r.f15.mean()), "t15": float(tstat(r.f15)), "f30": float(r.f30.mean()), "t30": float(tstat(r.f30))}
    # J4: sub-period robustness of the two headline effects
    print("  J4 sub-period check (first60 -> rest-of-day corr | fade-30 to 15:15 mean pts | gap>=0.5% fade mean pts):")
    for lab, mask in (("2025-11", np.array([d_.year == 2025 for d_ in dates])), ("2026", np.array([d_.year == 2026 for d_ in dates]))):
        idx = np.where(mask)[0]
        f60 = np.array([days[i].close.values[59] / days[i].open.values[0] - 1 for i in idx]); rest = np.array([days[i].close.values[-1] / days[i].close.values[59] - 1 for i in idx])
        f30 = np.array([(days[i].close.values[360] - days[i].close.values[30]) * -np.sign(days[i].close.values[29] - days[i].open.values[0]) for i in idx])
        print(f"     {lab}: n={len(idx)} corr {np.corrcoef(f60, rest)[0,1]:+.3f} | fade30 {f30.mean():+.1f} pts (t={tstat(f30):+.2f}, hit {100*(f30>0).mean():.0f}%)")
    return out

if __name__ == "__main__":
    res = {s: study(s) for s in ("nifty", "sensex")}
    json.dump(res, open(str(__import__("pathlib").Path(__file__).resolve().parent / "pa_study.json"), "w"), indent=1, default=str)
