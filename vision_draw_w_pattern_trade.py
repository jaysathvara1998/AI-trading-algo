"""
Vision Brain & W-Pattern Visualizer for the 12:52 IST Trade
Plots the exact 4-leg W-Pattern (Trough 1, Neckline Mid-Peak, Trough 2, Leg 4 Entry)
detected by MPatternSkill on today's NIFTY 1-minute chart.
"""

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


def main():
    print("=" * 70)
    print("PLOTTING W-PATTERN DETECTED BY BRAIN AT 12:52 IST")
    print("=" * 70)

    tz = pytz.timezone("Asia/Kolkata")
    today_str = datetime.now(tz).strftime("%Y-%m-%d")

    feed = DhanDataFeed()
    df = feed.fetch_historical_bars(days=2, symbol="NIFTY")
    if df.empty:
        print("Could not fetch historical data.")
        return

    df["dt"] = pd.to_datetime(df["timestamp"])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        df["dt"] = df["dt"].dt.tz_convert(tz)

    today_df = df[df["dt"].dt.strftime("%Y-%m-%d") == today_str].copy().reset_index(drop=True)

    if len(today_df) < 10:
        print("Not enough bars.")
        return

    # Focus on the relevant window (e.g. from 11:45 onwards to highlight the W formation)
    zoom_df = today_df[today_df["dt"].dt.strftime("%H:%M") >= "11:45"].copy().reset_index(drop=True)
    if len(zoom_df) < 20:
        zoom_df = today_df.tail(70).copy().reset_index(drop=True)

    skill = MPatternSkill()
    bars_list = []
    for _, r in zoom_df.iterrows():
        bars_list.append({
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume", 1000))
        })

    # Scan W-pattern
    w_met = skill.scan_w_pattern(bars_list)
    
    # Render Chart
    fig, (ax_main, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3.5, 1]},
        facecolor="#0e1117"
    )
    ax_main.set_facecolor("#161a23")
    ax_vol.set_facecolor("#161a23")

    indices = np.arange(len(zoom_df))
    width = 0.6

    for i in range(len(zoom_df)):
        o = zoom_df["open"].iloc[i]
        h = zoom_df["high"].iloc[i]
        l = zoom_df["low"].iloc[i]
        c = zoom_df["close"].iloc[i]
        color = "#00e676" if c >= o else "#ff3d00"

        ax_main.plot([i, i], [l, h], color=color, linewidth=1.2)
        rect_y = min(o, c)
        rect_h = max(0.2, abs(c - o))
        rect = patches.Rectangle((i - width/2, rect_y), width, rect_h, color=color, alpha=0.9)
        ax_main.add_patch(rect)

        vol = zoom_df["volume"].iloc[i] if "volume" in zoom_df.columns else 1000
        ax_vol.bar(i, vol, color=color, alpha=0.6, width=0.8)

    # Highlight W Pattern Structure if found or by swings
    highs = zoom_df["high"].values
    lows = zoom_df["low"].values
    closes = zoom_df["close"].values
    swings = skill.extract_swing_legs(highs, lows, closes, min_reversal_pts=4.0)

    # Plot ZigZag Swings in Cyan to show the Wave structure clearly
    for s in swings:
        color = "#00e5ff"
        ax_main.plot([s.start_idx, s.end_idx], [s.start_price, s.end_price], color=color, linestyle="--", linewidth=1.5, alpha=0.7)

    if w_met:
        p1_idx = w_met.peak1_index
        p1_price = w_met.peak1_price
        neck_idx = w_met.valley_index
        neck_price = w_met.neckline_level
        p2_idx = w_met.peak2_index
        p2_price = w_met.peak2_price

        # Draw thick W overlay in bright Gold
        w_x = [p1_idx, neck_idx, p2_idx, len(zoom_df) - 1]
        w_y = [p1_price, neck_price, p2_price, closes[-1]]
        ax_main.plot(w_x, w_y, color="#ffd700", linewidth=3.5, label="Double Bottom W Structure", zorder=10)

        # Neckline Horizontal Line
        ax_main.axhline(neck_price, color="#ff9100", linestyle=":", linewidth=2, label=f"W-Neckline: {neck_price:.2f}")

        # Trough 1 marker
        ax_main.plot(p1_idx, p1_price, marker='o', markersize=9, color='#ffd700', markeredgecolor='#ffffff')
        ax_main.annotate(f"Trough 1\n({p1_price:.1f})", xy=(p1_idx, p1_price), xytext=(p1_idx - 5, p1_price - 15),
                         color="#ffd700", fontsize=9, fontweight="bold",
                         arrowprops=dict(facecolor='#ffd700', edgecolor='#ffffff', width=1.5, headwidth=5))

        # Mid-Peak Neckline marker
        ax_main.plot(neck_idx, neck_price, marker='o', markersize=9, color='#ff9100', markeredgecolor='#ffffff')
        ax_main.annotate(f"Mid-Peak Neckline\n({neck_price:.1f})", xy=(neck_idx, neck_price), xytext=(neck_idx - 6, neck_price + 15),
                         color="#ff9100", fontsize=9, fontweight="bold",
                         arrowprops=dict(facecolor='#ff9100', edgecolor='#ffffff', width=1.5, headwidth=5))

        # Trough 2 marker
        ax_main.plot(p2_idx, p2_price, marker='o', markersize=9, color='#ffd700', markeredgecolor='#ffffff')
        ax_main.annotate(f"Trough 2 (Support Hold)\n({p2_price:.1f})", xy=(p2_idx, p2_price), xytext=(p2_idx - 8, p2_price - 18),
                         color="#ffd700", fontsize=9, fontweight="bold",
                         arrowprops=dict(facecolor='#ffd700', edgecolor='#ffffff', width=1.5, headwidth=5))

    # Mark 12:52 Trade Entry
    entry_indices = [i for i, dt in enumerate(zoom_df["dt"]) if dt.strftime("%H:%M") == "12:52"]
    if entry_indices:
        e_idx = entry_indices[0]
        e_price = zoom_df["close"].iloc[e_idx]
        ax_main.plot(e_idx, e_price, marker='^', markersize=14, color='#00e676', markeredgecolor='#ffffff', markeredgewidth=2, zorder=12)
        ax_main.annotate(
            "TRADE ENTRY @ 12:52:13 IST\nBUY 22900 CALL @ INR 230.73\n(W-Reversal Leg 4 + 99% XGBoost)",
            xy=(e_idx, e_price),
            xytext=(e_idx - 15, e_price + 25),
            arrowprops=dict(facecolor='#00e676', edgecolor='#ffffff', width=2, headwidth=8),
            color="#000000", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#00e676", alpha=0.95)
        )

    last_p = closes[-1]
    time_str = zoom_df['dt'].iloc[-1].strftime('%H:%M')
    ax_main.set_title(
        f"NIFTY 1-Minute Live Chart ({today_str} @ {time_str} IST) | Spot: {last_p:.2f}\n"
        f"Brain's Visual Eye: W-Pattern (Double Bottom) Reversal Structure at Institutional Support",
        color="#ffffff", fontsize=12, fontweight="bold", pad=12
    )

    tick_step = max(1, len(zoom_df) // 8)
    ax_main.set_xticks(indices[::tick_step])
    ax_main.set_xticklabels([])
    ax_vol.set_xticks(indices[::tick_step])
    ax_vol.set_xticklabels([zoom_df['dt'].iloc[i].strftime('%H:%M') for i in indices[::tick_step]], color="#90a4ae", rotation=0)

    ax_main.tick_params(colors="#90a4ae")
    ax_vol.tick_params(colors="#90a4ae")
    ax_main.grid(True, linestyle=":", alpha=0.2, color="#78909c")
    ax_vol.grid(True, linestyle=":", alpha=0.15, color="#78909c")
    ax_main.legend(loc="upper left", facecolor="#1e222d", edgecolor="#37474f", labelcolor="#ffffff", fontsize=8.5)

    ax_vol.set_ylabel("Volume", color="#90a4ae", fontsize=9)
    ax_main.set_ylabel("NIFTY Index Spot", color="#90a4ae", fontsize=10)

    artifact_dir = Path(r"C:\Users\rushi\.gemini\antigravity-ide\brain\8d1fcc67-e4cf-4b33-83d8-43a92d310fe2")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifact_dir / "today_w_pattern_trade_1252.png"

    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[SUCCESS] Saved W-Pattern Trade chart to: {out_file}")


if __name__ == "__main__":
    main()
