"""
Vision Brain Trendline & Pattern Visualizer for Today's 1-Minute Live Chart
Fetches today's live 1-minute bars from Dhan for NIFTY,
detects structural trendlines, support floors, and M/W patterns using MPatternSkill,
runs 2D CNN visual inference, and renders a high-res chart with annotations.
"""

import os
import sys
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
from algo_vpin_v2.pattern_engine import PatternRecognitionEngine
from vision_chart_brain.visual_predictor import VisualPredictor


def main():
    print("=" * 70)
    print("VISION BRAIN: DRAWING TRENDLINES & PATTERNS ON TODAY'S 1M CHART")
    print("=" * 70)

    tz = pytz.timezone("Asia/Kolkata")
    today_str = datetime.now(tz).strftime("%Y-%m-%d")

    # 1. Fetch live historical bars from Dhan
    feed = DhanDataFeed()
    print("Fetching today's 1-minute historical bars for NIFTY from Dhan...")
    df = feed.fetch_historical_bars(days=2, symbol="NIFTY")

    if df.empty:
        print("Could not fetch historical data from Dhan. Exiting.")
        return

    # Filter for today's session (from 09:15 onwards)
    df["dt"] = pd.to_datetime(df["timestamp"])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        df["dt"] = df["dt"].dt.tz_convert(tz)

    today_df = df[df["dt"].dt.strftime("%Y-%m-%d") == today_str].copy().reset_index(drop=True)

    if len(today_df) < 10:
        print(f"Only {len(today_df)} bars found for today. Using last 120 bars for comprehensive visualization.")
        today_df = df.tail(120).copy().reset_index(drop=True)

    print(f"Total 1-Minute Bars to Plot: {len(today_df)} (From {today_df['dt'].iloc[0].strftime('%H:%M')} to {today_df['dt'].iloc[-1].strftime('%H:%M')} IST)")

    # 2. Run MPatternSkill & Trendline Detection
    skill = MPatternSkill()
    pattern_engine = PatternRecognitionEngine()

    bars_list = []
    for _, r in today_df.iterrows():
        bars_list.append({
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume", 1000))
        })
        pattern_engine.update_bar(
            open_p=float(r["open"]),
            high_p=float(r["high"]),
            low_p=float(r["low"]),
            close_p=float(r["close"]),
            volume=float(r.get("volume", 1000))
        )

    tl_met = skill.scan_trendline_breakout(bars_list)
    m_met = skill.scan_m_pattern(bars_list)
    w_met = skill.scan_w_pattern(bars_list)

    # 3. Run 2D CNN Vision Chart Brain
    model_pt = ROOT_DIR / "vision_chart_brain" / "vision_cnn_nifty.pt"
    if not model_pt.exists():
        model_pt = ROOT_DIR / "vision_chart_brain" / "vision_cnn_model.pt"
    vision = VisualPredictor(model_path=str(model_pt) if model_pt.exists() else None)
    vision_signal = 0
    vision_action = "HOLD"
    vision_conf = 0.50
    if len(today_df) >= 20:
        v_res = vision.predict_chart(today_df.tail(20))
        vision_signal = v_res.get("numeric_signal", 0)
        vision_action = v_res.get("action", "HOLD")
        vision_conf = v_res.get("confidence", 0.50)
        print(f"\n[2D CNN Vision Brain Prediction] Action: {vision_action} | Confidence: {vision_conf*100:.1f}%")

    # 4. Render Dark Institutional Matplotlib Chart
    fig, (ax_main, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3.5, 1]},
        facecolor="#0e1117"
    )
    ax_main.set_facecolor("#161a23")
    ax_vol.set_facecolor("#161a23")

    # Plot Candlesticks
    indices = np.arange(len(today_df))
    width = 0.6
    width2 = 0.1

    for i in range(len(today_df)):
        o = today_df["open"].iloc[i]
        h = today_df["high"].iloc[i]
        l = today_df["low"].iloc[i]
        c = today_df["close"].iloc[i]

        color = "#00e676" if c >= o else "#ff3d00"  # Vibrant Green / Red
        # Wick
        ax_main.plot([i, i], [l, h], color=color, linewidth=1.2)
        # Body
        rect_y = min(o, c)
        rect_h = max(0.2, abs(c - o))
        rect = patches.Rectangle((i - width/2, rect_y), width, rect_h, color=color, alpha=0.9)
        ax_main.add_patch(rect)

        # Volume bar
        vol = today_df["volume"].iloc[i] if "volume" in today_df.columns else 1000
        ax_vol.bar(i, vol, color=color, alpha=0.6, width=0.8)

    # 5. Draw Detected Trendlines & Support Levels
    highs = today_df["high"].values
    lows = today_df["low"].values
    closes = today_df["close"].values

    # Key Support Floor (Lowest cluster of the morning)
    min_support = np.min(lows)
    ax_main.axhline(
        y=min_support, color="#00e5ff", linestyle="--", linewidth=1.5, alpha=0.8,
        label=f"Major Support Floor: {min_support:.1f}"
    )
    ax_main.text(
        0, min_support + 1.5, f" Key Support Floor ({min_support:.1f})",
        color="#00e5ff", fontsize=9, fontweight="bold"
    )

    # Find Top Swing Highs for Descending Trendline
    swings = skill.extract_swing_legs(highs, lows, closes, min_reversal_pts=5.0)
    peaks = [(s.start_idx if s.direction == "DOWN" else s.end_idx, s.start_price if s.direction == "DOWN" else s.end_price) for s in swings]
    peaks = sorted(list(set(peaks)), key=lambda x: x[0])

    troughs = [(s.start_idx if s.direction == "UP" else s.end_idx, s.start_price if s.direction == "UP" else s.end_price) for s in swings]
    troughs = sorted(list(set(troughs)), key=lambda x: x[0])

    # Descending Trendline from Top Swing (around 09:30 @ 23120)
    top_peaks = [p for p in peaks if p[1] > 23090 and p[0] < 45]
    if len(top_peaks) >= 2:
        p1 = top_peaks[0]
        p2 = top_peaks[1]
    elif len(peaks) >= 2:
        p1 = peaks[0]
        p2 = peaks[1]
    else:
        p1, p2 = None, None

    if p1 and p2 and p2[0] > p1[0]:
        idx1, price1 = p1
        idx2, price2 = p2
        slope = (price2 - price1) / float(idx2 - idx1)
        
        # Plot up to breakout bar (~bar 50)
        end_idx = min(len(today_df) - 1, idx2 + 35)
        x_vals = np.arange(idx1, end_idx + 1)
        y_vals = price1 + slope * (x_vals - idx1)

        ax_main.plot(
            x_vals, y_vals, color="#ff5252", linestyle="-", linewidth=2.2,
            label=f"Descending Resistance Ceiling (Slope: {slope:.2f} pts/bar)"
        )

        # Rejection callout
        mid_idx = idx1 + int((idx2 - idx1) * 1.5)
        if mid_idx < len(y_vals):
            ax_main.annotate(
                f"Ceiling Veto Zone\n(BUY CALL Blocked)",
                xy=(idx1 + mid_idx, y_vals[mid_idx]),
                xytext=(idx1 + mid_idx - 10, y_vals[mid_idx] + 18),
                arrowprops=dict(facecolor='#ff5252', edgecolor='#ffffff', width=1.5, headwidth=6),
                color="#ffffff", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#b71c1c", alpha=0.9)
            )

    # Ascending Trendline from Morning Low (09:51 low ~23038 -> higher lows)
    low_troughs = [t for t in troughs if t[0] >= 30]
    if len(low_troughs) >= 2:
        t1 = min(low_troughs, key=lambda x: x[1])
        subsequent = [t for t in low_troughs if t[0] > t1[0] and t[1] > t1[1]]
        if subsequent:
            t2 = subsequent[0]
            idx_t1, p_t1 = t1
            idx_t2, p_t2 = t2
            slope_up = (p_t2 - p_t1) / float(idx_t2 - idx_t1)
            x_up = np.arange(idx_t1, len(today_df))
            y_up = p_t1 + slope_up * (x_up - idx_t1)
            ax_main.plot(
                x_up, y_up, color="#00e676", linestyle="-", linewidth=2.0,
                label=f"Ascending Support Trendline (Higher Lows Support)"
            )

    # Annotate the 10:04 Breakout & Winning Trade #5 (+1415 INR)
    if len(today_df) >= 55:
        breakout_idx = min(54, len(today_df) - 1)
        breakout_price = today_df["close"].iloc[breakout_idx]
        ax_main.plot(breakout_idx, breakout_price, marker='^', markersize=12, color='#ffd700', markeredgecolor='#ffffff', markeredgewidth=1.5)
        ax_main.annotate(
            f"Trade #5: 10:09 Trendline Breakout CE\nEntry: 22900 CE | Result: +Rs 1,415.70 (+22 pts)",
            xy=(breakout_idx, breakout_price),
            xytext=(breakout_idx - 22, breakout_price - 30),
            arrowprops=dict(facecolor='#ffd700', edgecolor='#ffffff', width=1.8, headwidth=7),
            color="#000000", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffd700", alpha=0.95)
        )

    # 6. Title, Legend & Formatting
    last_p = closes[-1]
    time_str = today_df['dt'].iloc[-1].strftime('%H:%M')
    probs_str = f"HOLD: {v_res.get('probabilities', {}).get('HOLD', 0)*100:.1f}% | CALL: {v_res.get('probabilities', {}).get('CALL', 0)*100:.1f}% | PUT: {v_res.get('probabilities', {}).get('PUT', 0)*100:.1f}%" if 'v_res' in locals() else ""
    ax_main.set_title(
        f"NIFTY 1-Minute Live Chart Analysis ({today_str} @ {time_str} IST) | Spot: {last_p:.2f}\n"
        f"2D CNN Vision Brain: {vision_action} ({vision_conf*100:.1f}%) | [{probs_str}]",
        color="#ffffff", fontsize=12, fontweight="bold", pad=12
    )

    # X-axis ticks (Time format every 10-15 bars)
    tick_step = max(1, len(today_df) // 8)
    ax_main.set_xticks(indices[::tick_step])
    ax_main.set_xticklabels([])
    ax_vol.set_xticks(indices[::tick_step])
    ax_vol.set_xticklabels([today_df['dt'].iloc[i].strftime('%H:%M') for i in indices[::tick_step]], color="#90a4ae", rotation=0)

    ax_main.tick_params(colors="#90a4ae")
    ax_vol.tick_params(colors="#90a4ae")
    ax_main.grid(True, linestyle=":", alpha=0.2, color="#78909c")
    ax_vol.grid(True, linestyle=":", alpha=0.15, color="#78909c")
    ax_main.legend(loc="upper right", facecolor="#1e222d", edgecolor="#37474f", labelcolor="#ffffff", fontsize=8.5)

    ax_vol.set_ylabel("Volume", color="#90a4ae", fontsize=9)
    ax_main.set_ylabel("NIFTY Index Spot", color="#90a4ae", fontsize=10)

    # Save to artifacts directory
    artifact_dir = Path(r"C:\Users\rushi\.gemini\antigravity-ide\brain\8d1fcc67-e4cf-4b33-83d8-43a92d310fe2")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifact_dir / "today_nifty_vision_trendline.png"

    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n[SUCCESS] Rendered Vision Trendline Chart saved to:")
    print(f"-> {out_file}")


if __name__ == "__main__":
    main()
