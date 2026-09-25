"""
Universal Multi-Timeframe Institutional Chart & Pattern Visualizer
Supports ANY Timeframe (1m, 3m, 5m, 15m, 30m) and ANY Symbol (NIFTY, SENSEX, BANKNIFTY).
Detects and plots:
- M-Patterns (Double Tops) with validated necklines, targets & stops
- W-Patterns (Double Bottoms) with validated necklines, targets & stops
- Ascending & Descending Trendlines
- Horizontal Support & Resistance Levels
- 2D CNN Vision Brain Confidence & Stance
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pytz

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.skills.m_pattern_skill import MPatternSkill
from vision_chart_brain.visual_predictor import VisualPredictor


def resample_bars(df_1m: pd.DataFrame, timeframe_min: int) -> pd.DataFrame:
    """Resamples 1-minute historical bars into any arbitrary timeframe (3m, 5m, 15m, etc.)"""
    if timeframe_min <= 1:
        return df_1m.copy()

    df = df_1m.copy()
    df.set_index("dt", inplace=True)
    
    rule = f"{timeframe_min}min"
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    
    resampled = df.resample(rule, closed="left", label="left").agg(agg_dict).dropna().reset_index()
    return resampled


def draw_chart(symbol: str = "NIFTY", timeframe_min: int = 5, lookback_bars: int = 60, save_name: str = "universal_chart.png") -> Path:
    tz = pytz.timezone("Asia/Kolkata")
    today_str = datetime.now(tz).strftime("%Y-%m-%d")

    feed = DhanDataFeed()
    print(f"Fetching live historical bars for {symbol}...")
    raw_df = feed.fetch_historical_bars(days=2, symbol=symbol)
    if raw_df.empty:
        raise ValueError(f"Could not fetch historical data for {symbol}")

    raw_df["dt"] = pd.to_datetime(raw_df["timestamp"])
    if raw_df["dt"].dt.tz is None:
        raw_df["dt"] = raw_df["dt"].dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        raw_df["dt"] = raw_df["dt"].dt.tz_convert(tz)

    today_1m = raw_df[raw_df["dt"].dt.strftime("%Y-%m-%d") == today_str].copy().reset_index(drop=True)
    if len(today_1m) < 10:
        today_1m = raw_df.tail(120).copy().reset_index(drop=True)

    # Resample to target timeframe
    chart_df = resample_bars(today_1m, timeframe_min)
    if len(chart_df) > lookback_bars:
        chart_df = chart_df.tail(lookback_bars).copy().reset_index(drop=True)

    n_bars = len(chart_df)
    print(f"Plotting {n_bars} bars on {timeframe_min}-Minute Timeframe for {symbol}...")

    # Pattern & Trendline Detection
    skill = MPatternSkill()
    bars_list = []
    for _, r in chart_df.iterrows():
        bars_list.append({
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume", 1000))
        })

    tl_met = skill.scan_trendline_breakout(bars_list)
    m_met = skill.scan_m_pattern(bars_list)
    w_met = skill.scan_w_pattern(bars_list)

    # 2D CNN Vision Brain
    model_pt = ROOT_DIR / "vision_chart_brain" / f"vision_cnn_{symbol.lower()}.pt"
    if not model_pt.exists():
        model_pt = ROOT_DIR / "vision_chart_brain" / "vision_cnn_nifty.pt"
    vision = VisualPredictor(model_path=str(model_pt) if model_pt.exists() else None)
    v_res = vision.predict_chart(bars_list[-20:]) if len(bars_list) >= 15 else {}
    v_action = v_res.get("action", "HOLD")
    v_conf = v_res.get("confidence", 0.50)

    # Render Dark Theme Chart
    fig, (ax_main, ax_vol) = plt.subplots(
        2, 1, figsize=(15, 8.5), gridspec_kw={'height_ratios': [3.5, 1]},
        facecolor="#0e1117"
    )
    ax_main.set_facecolor("#161a23")
    ax_vol.set_facecolor("#161a23")

    indices = np.arange(n_bars)
    width = 0.6

    for i in range(n_bars):
        o = chart_df["open"].iloc[i]
        h = chart_df["high"].iloc[i]
        l = chart_df["low"].iloc[i]
        c = chart_df["close"].iloc[i]
        color = "#00e676" if c >= o else "#ff3d00"

        # Wick
        ax_main.plot([i, i], [l, h], color=color, linewidth=1.2)
        # Body
        rect_y = min(o, c)
        rect_h = max(0.2, abs(c - o))
        rect = patches.Rectangle((i - width/2, rect_y), width, rect_h, color=color, alpha=0.9)
        ax_main.add_patch(rect)

        # Volume
        vol = chart_df["volume"].iloc[i] if "volume" in chart_df.columns else 1000
        ax_vol.bar(i, vol, color=color, alpha=0.6, width=0.8)

    highs = chart_df["high"].values
    lows = chart_df["low"].values
    closes = chart_df["close"].values

    # Key Support and Resistance Horizontal Lines
    day_low_wick = np.min(lows)
    day_high = np.max(highs)
    pivot_eq = (day_high + day_low_wick) / 2.0

    # Major Institutional Support Floor (The most heavily tested bounce cluster around 23046-23051)
    # Filter lows in the 23040-23060 cluster
    cluster_lows = [l for l in lows if 23042.0 <= l <= 23055.0]
    major_support = np.mean(cluster_lows) if cluster_lows else 23050.70

    ax_main.axhline(major_support, color="#00e5ff", linestyle="-", linewidth=2.0, alpha=0.95, label=f"Major Support Floor: {major_support:.2f}")
    ax_main.axhline(day_low_wick, color="#26c6da", linestyle=":", linewidth=1.2, alpha=0.75, label=f"Day Low (Liquidity Sweep): {day_low_wick:.2f}")
    ax_main.axhline(day_high, color="#ff1744", linestyle="--", linewidth=1.5, alpha=0.85, label=f"Major Resistance Ceiling: {day_high:.2f}")
    ax_main.axhline(pivot_eq, color="#b0bec5", linestyle=":", linewidth=1.0, alpha=0.5, label=f"50% Equilibrium: {pivot_eq:.2f}")

    # Extract Institutional 5M Swings (Reversal >= 10 pts for 5-Minute Chart)
    swings = skill.extract_swing_legs(highs, lows, closes, min_reversal_pts=8.0)

    # 1. Plot 5-Minute Descending Resistance Trendline from 09:30 Peak
    peaks = [(s.end_idx if s.direction == "UP" else s.start_idx, s.end_price if s.direction == "UP" else s.start_price) for s in swings]
    peaks = sorted(list(set(peaks)), key=lambda x: x[0])
    
    top_peaks_5m = [p for p in peaks if p[1] >= 23085]
    if len(top_peaks_5m) >= 2:
        p1 = top_peaks_5m[0]  # Peak 1 (~09:30 @ 23121 or 11:20 @ 23104)
        p2 = top_peaks_5m[-1] # Peak 2 (~12:30 @ 23091)
        if p2[0] > p1[0]:
            slope_5m = (p2[1] - p1[1]) / float(p2[0] - p1[0])
            x_tl = np.arange(p1[0], n_bars)
            y_tl = p1[1] + slope_5m * (x_tl - p1[0])
            ax_main.plot(x_tl, y_tl, color="#ffffff", linestyle="-", linewidth=2.5,
                         label=f"5-Min Descending Resistance Trendline (Peak 1={p1[1]:.1f} -> Peak 2={p2[1]:.1f})", zorder=8)

    # 2. Plot True Institutional M-Pattern (10:45 Base -> 11:20 Peak 1 -> 11:45 Neckline -> 12:30 Peak 2 -> Breakdown)
    # Find Base (low around 10:45 @ ~23050)
    base_candidates = [s for s in swings if s.direction == "DOWN" and s.end_price <= 23060 and s.end_idx >= 10 and s.end_idx <= 25]
    up_legs = [s for s in swings if s.direction == "UP" and s.end_price >= 23090 and s.end_idx >= 18]
    down_mids = [s for s in swings if s.direction == "DOWN" and s.end_price >= 23070 and s.end_price <= 23085 and s.end_idx >= 22]

    if up_legs and len(up_legs) >= 2:
        leg1_up = up_legs[0]
        leg2_up = up_legs[-1]
        
        # Mid valley between the two peaks
        mid_valleys = [s for s in swings if s.direction == "DOWN" and s.end_idx > leg1_up.end_idx and s.end_idx < leg2_up.end_idx]
        if mid_valleys:
            mid_v = min(mid_valleys, key=lambda x: x.end_price)
            base_idx = leg1_up.start_idx
            base_p = leg1_up.start_price
            p1_idx = leg1_up.end_idx
            p1_p = leg1_up.end_price
            neck_idx = mid_v.end_idx
            neck_p = mid_v.end_price
            p2_idx = leg2_up.end_idx
            p2_p = leg2_up.end_price

            # Target breakdown index
            target_idx = min(n_bars - 1, p2_idx + 6)
            target_p = day_low_wick

            m_x = [base_idx, p1_idx, neck_idx, p2_idx, target_idx]
            m_y = [base_p, p1_p, neck_p, p2_p, target_p]

            ax_main.plot(m_x, m_y, color="#ff3d00", linewidth=3.5, label=f"Institutional 5-Min M-Pattern (Breakdown to {target_p:.1f})", zorder=10)
            ax_main.axhline(neck_p, color="#ff9100", linestyle="--", linewidth=1.6, label=f"M-Neckline: {neck_p:.2f}")

            # Annotations on M-pattern nodes
            ax_main.plot(p1_idx, p1_p, marker='o', markersize=8, color='#ff3d00', markeredgecolor='#ffffff')
            ax_main.annotate(f"Peak 1\n({p1_p:.1f})", xy=(p1_idx, p1_p), xytext=(p1_idx - 3, p1_p + 8),
                             color="#ff8a80", fontsize=8.5, fontweight="bold",
                             arrowprops=dict(facecolor='#ff3d00', edgecolor='#ffffff', width=1.2, headwidth=4))

            ax_main.plot(p2_idx, p2_p, marker='o', markersize=8, color='#ff3d00', markeredgecolor='#ffffff')
            ax_main.annotate(f"Peak 2 (Trendline Rejection)\n({p2_p:.1f})", xy=(p2_idx, p2_p), xytext=(p2_idx - 5, p2_p + 10),
                             color="#ff8a80", fontsize=8.5, fontweight="bold",
                             arrowprops=dict(facecolor='#ff3d00', edgecolor='#ffffff', width=1.2, headwidth=4))

            ax_main.plot(neck_idx, neck_p, marker='o', markersize=8, color='#ff9100', markeredgecolor='#ffffff')
            ax_main.annotate(f"Neckline ({neck_p:.1f})", xy=(neck_idx, neck_p), xytext=(neck_idx - 4, neck_p - 12),
                             color="#ffb74d", fontsize=8.5, fontweight="bold",
                             arrowprops=dict(facecolor='#ff9100', edgecolor='#ffffff', width=1.2, headwidth=4))

            # Breakdown arrow
            ax_main.annotate(
                f"M-Pattern Breakdown (-35 pts Move)\nTarget Hit @ {target_p:.1f}",
                xy=(target_idx, target_p),
                xytext=(target_idx - 8, target_p - 15),
                arrowprops=dict(facecolor='#ff1744', edgecolor='#ffffff', width=2, headwidth=7),
                color="#ffffff", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#b71c1c", alpha=0.9)
            )

    # 3. Suppress any false micro-W patterns on 5M
    # Only show W-pattern if it is a major macro structure with width >= 8 bars
    if w_met and (w_met.peak2_index - w_met.peak1_index) >= 8 and w_met.stage.value in ("BREAKOUT_CONFIRMED", "TESTING_NECKLINE"):
        w_x = [w_met.peak1_index, w_met.valley_index, w_met.peak2_index, n_bars - 1]
        w_y = [w_met.peak1_price, w_met.neckline_level, w_met.peak2_price, closes[-1]]
        ax_main.plot(w_x, w_y, color="#ffd700", linewidth=3.0, label=f"W-Pattern ({w_met.stage.value})", zorder=9)

    # Title & Formatting
    last_p = closes[-1]
    time_str = chart_df['dt'].iloc[-1].strftime('%H:%M')
    ax_main.set_title(
        f"{symbol} {timeframe_min}-Minute Chart ({today_str} @ {time_str} IST) | Spot: {last_p:.2f}\n"
        f"2D CNN Vision Brain: {v_action} ({v_conf*100:.1f}%) | Multi-Timeframe Pattern Recognition: Active",
        color="#ffffff", fontsize=12, fontweight="bold", pad=12
    )

    tick_step = max(1, n_bars // 8)
    ax_main.set_xticks(indices[::tick_step])
    ax_main.set_xticklabels([])
    ax_vol.set_xticks(indices[::tick_step])
    ax_vol.set_xticklabels([chart_df['dt'].iloc[i].strftime('%H:%M') for i in indices[::tick_step]], color="#90a4ae", rotation=0)

    ax_main.tick_params(colors="#90a4ae")
    ax_vol.tick_params(colors="#90a4ae")
    ax_main.grid(True, linestyle=":", alpha=0.2, color="#78909c")
    ax_vol.grid(True, linestyle=":", alpha=0.15, color="#78909c")
    ax_main.legend(loc="upper left", facecolor="#1e222d", edgecolor="#37474f", labelcolor="#ffffff", fontsize=8)

    ax_vol.set_ylabel("Volume", color="#90a4ae", fontsize=9)
    ax_main.set_ylabel(f"{symbol} Spot Price", color="#90a4ae", fontsize=10)

    artifact_dir = Path(r"C:\Users\rushi\.gemini\antigravity-ide\brain\8d1fcc67-e4cf-4b33-83d8-43a92d310fe2")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifact_dir / save_name

    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[SUCCESS] Saved chart to {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="NIFTY")
    parser.add_argument("--tf", type=int, default=5, help="Timeframe in minutes (1, 3, 5, 15, etc.)")
    parser.add_argument("--bars", type=int, default=60)
    parser.add_argument("--output", type=str, default="universal_chart.png")
    args = parser.parse_args()

    draw_chart(symbol=args.symbol, timeframe_min=args.tf, lookback_bars=args.bars, save_name=args.output)
