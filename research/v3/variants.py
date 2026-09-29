"""Signal-measurement variants: raw underlying points, no rupee cap, so the question is only 'is there drift'."""
import sys; sys.path.insert(0, ".")
import pandas as pd
from pa_engine.backtest import experiment
from pa_engine.candles import load_sessions
from pa_engine.config import EngineConfig
sym, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = EngineConfig().for_symbol(sym).override(**{"cost.instrument": "points", "risk.max_risk_inr": 1e9})
V = {
 "pts_baseline_5_3": {},
 "pts_retest_close_5_3": {"confirm.mode": "retest_close"},
 "pts_tf_15_5": {"tf.context": 15, "tf.setup": 5},
 "pts_retest_close_15_5": {"tf.context": 15, "tf.setup": 5, "confirm.mode": "retest_close"},
 "pts_tf_30_5": {"tf.context": 30, "tf.setup": 5},
 "pts_tf_15_3": {"tf.context": 15, "tf.setup": 3},
 "pts_tf_30_15": {"tf.context": 30, "tf.setup": 15},
 "pts_retest_close_30_15": {"tf.context": 30, "tf.setup": 15, "confirm.mode": "retest_close"},
 "pts_hold_only_5_3": {"exit.tp1_fraction": 0.0, "exit.breakeven_after_r": 0.0, "exit.trail": "none", "exit.tp2_fraction": 1.0, "exit.time_exit_bars": 0, "exit.invalidation_exit": False},
 "pts_hold_only_15_5": {"tf.context": 15, "tf.setup": 5, "exit.tp1_fraction": 0.0, "exit.breakeven_after_r": 0.0, "exit.trail": "none", "exit.tp2_fraction": 1.0, "exit.time_exit_bars": 0, "exit.invalidation_exit": False},
 "pts_no_rr_filter_5_3": {"risk.min_rr": 0.0, "risk.max_stop_atr": 99},
 "pts_no_rr_retest_close_5_3": {"risk.min_rr": 0.0, "risk.max_stop_atr": 99, "confirm.mode": "retest_close"},
}
ss = load_sessions(data)
df = experiment(base, ss, V, out_dir=out)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(df.to_string(index=False)); df.to_csv(f"{out}/variants.csv", index=False)
