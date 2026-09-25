"""
Multi-Asset (NIFTY & SENSEX) Machine Learning Training Pipeline for Algo VPIN v2.0
Ingests up to 1-2 years of real historical 1-minute exchange bars from Dhan,
calculates VPIN toxicity, GARCH(1,1) volatility, and multi-timeframe macro features,
and trains:
1. SVM RBF Kernel
2. XGBoost Gradient Boosted Trees
3. Deep Artificial Neural Network (ANN) Multi-Class Perceptron
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pytz
import numpy as np
import pandas as pd
import joblib

# Add project root directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_vpin_v2.config import CONFIG, AppConfig
from algo_vpin_v2.data_feed import DhanDataFeed, BarOHLCV
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.svm_filter import SVMTradeFilter
from algo_vpin_v2.xgboost_brain import XGBoostBrain
from algo_vpin_v2.ann_brain import ANNBrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("algo_vpin_v2.train_multi_asset")


INDEX_META = {
    "NIFTY": {"sec_id": "13", "step_thresh": 7.0, "start_price": 24000.0, "svm_thresh": 1.0},
    "SENSEX": {"sec_id": "51", "step_thresh": 22.0, "start_price": 80000.0, "svm_thresh": 3.0},
    "BANKNIFTY": {"sec_id": "25", "step_thresh": 16.0, "start_price": 52000.0, "svm_thresh": 2.5},
}


class MultiAssetBrainTrainer:
    def __init__(self, config: AppConfig = CONFIG):
        self.config = config
        self.data_dir = ROOT_DIR / "algo_vpin_v2" / "data"
        self.models_dir = ROOT_DIR / "algo_vpin_v2" / "models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feed = DhanDataFeed(self.config)
        self.tz = pytz.timezone("Asia/Kolkata")

    def fetch_or_load_historical_bars(self, symbol: str = "NIFTY", months: int = 12) -> pd.DataFrame:
        """Fetches multi-month historical bars from Dhan with local compressed CSV caching"""
        sym_clean = symbol.strip().upper()
        meta = INDEX_META.get(sym_clean, INDEX_META["NIFTY"])
        sec_id = meta["sec_id"]

        cache_file = self.data_dir / f"{sym_clean.lower()}_{months}m_1min.csv.gz"
        if cache_file.exists():
            logger.info(f"Loading cached {sym_clean} historical dataset from {cache_file}...")
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded {len(df):,} bars for {sym_clean} (from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']})")
            return df

        logger.info(f"Fetching {months} months of 1-minute historical data for {sym_clean} from Dhan API (SecID: {sec_id}, Segment: IDX_I)...")
        dhan = self.feed.tradehull_client.Dhan if self.feed.tradehull_client else None

        if dhan is None:
            logger.warning("Tradehull Dhan client unavailable. Generating synthetic dataset.")
            return self.feed.generate_synthetic_bars(n_bars=months * 22 * 375, start_price=meta["start_price"])

        all_bars = []
        end_date = datetime.now(self.tz)

        for m in range(months):
            start_chunk = end_date - timedelta(days=28)
            f_str = start_chunk.strftime('%Y-%m-%d')
            t_str = end_date.strftime('%Y-%m-%d')
            try:
                r = dhan.intraday_minute_data(
                    security_id=str(sec_id),
                    exchange_segment='IDX_I',
                    instrument_type='INDEX',
                    from_date=f_str,
                    to_date=t_str
                )
                if isinstance(r, dict) and r.get('status') == 'success' and 'data' in r:
                    chunk_df = pd.DataFrame(r['data'])
                    if not chunk_df.empty and 'timestamp' in chunk_df.columns:
                        all_bars.append(chunk_df)
                        logger.info(f"[{sym_clean}] Chunk {m+1}/{months} ({f_str} to {t_str}): {len(chunk_df):,} bars")
            except Exception as e:
                logger.warning(f"Error fetching chunk {f_str} to {t_str}: {e}")

            end_date = start_chunk - timedelta(days=1)

        if not all_bars:
            logger.warning(f"No exchange bars returned for {sym_clean}. Generating synthetic dataset.")
            return self.feed.generate_synthetic_bars(n_bars=months * 22 * 375, start_price=meta["start_price"])

        final_df = pd.concat(all_bars, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], unit='s', utc=True).dt.tz_convert(self.tz)
        
        # Save to compressed CSV
        final_df.to_csv(cache_file, index=False, compression='gzip')
        logger.info(f"Saved {len(final_df):,} bars for {sym_clean} to {cache_file}")
        return final_df

    def process_and_train(self, symbol: str = "NIFTY", months: int = 12):
        logger.info("=" * 80)
        logger.info(f"  STARTING MULTI-ASSET BRAIN TRAINING FOR: {symbol.upper()} ({months} Months)")
        logger.info("=" * 80)

        df = self.fetch_or_load_historical_bars(symbol=symbol, months=months)

        vpin_calc = VPINCalculator(self.config.vpin)
        garch_eng = GARCHEngine(self.config.garch)
        macro_eng = MacroFeatureEngine()
        from algo_vpin_v2.smc_engine import SMCEngine, StructureTrend, FVGType
        smc_eng = SMCEngine()

        svm_brain = SVMTradeFilter()
        xgb_brain = XGBoostBrain(self.config.ensemble)
        ann_brain = ANNBrain(self.config.ensemble, hidden_layer_sizes=(128, 64, 32))

        # Calibrate adaptive VPIN volume
        total_vol = float(df['volume'].sum())
        days_approx = max(1.0, len(df) / 375.0)
        avg_daily_vol = total_vol / days_approx
        vpin_calc.update_bucket_volume(avg_daily_vol)
        logger.info(f"Calibrated VPIN Bucket Volume V = {vpin_calc.bucket_volume:,.1f} (ADV: {avg_daily_vol:,.0f})")

        # Initialize Macro state
        macro_eng.initialize_from_history(df.head(min(1000, len(df))))

        logger.info(f"Extracting quantitative & Smart Money Concepts (SMC) features across {len(df):,} historical bars...")
        
        svm_X, svm_y = [], []
        xgb_X, xgb_y = [], []
        ann_X, ann_y = [], []

        # We need forward returns for ground-truth labeling
        closes = df['close'].values
        opens = df['open'].values if 'open' in df.columns else closes
        highs = df['high'].values if 'high' in df.columns else closes
        lows = df['low'].values if 'low' in df.columns else closes
        volumes = df['volume'].values
        timestamps = df['timestamp'].values

        sym_clean = symbol.strip().upper()
        meta = INDEX_META.get(sym_clean, INDEX_META["NIFTY"])
        step_thresh = meta["step_thresh"]
        svm_thresh = meta["svm_thresh"]

        n_bars = len(df)
        forward_window = 8  # 8-minute forward horizon calibrated for 100-150 pt daily regime

        logger.info(f"Generating training labels with 8-min horizon (Threshold: ±{step_thresh} pts for {sym_clean})...")

        for i in range(1, n_bars - forward_window):
            o_p = float(opens[i])
            c_p = float(closes[i])
            h_p = float(highs[i])
            l_p = float(lows[i])
            vol = float(volumes[i])
            ts = timestamps[i]

            macro_st = macro_eng.update_1min_bar(c_p, h_p, l_p)
            vpin_res = vpin_calc.process_bar(c_p, vol, pd.Timestamp(ts))
            garch_res = garch_eng.add_bar(c_p, pd.Timestamp(ts))
            smc_st = smc_eng.update_bar(o_p, h_p, l_p, c_p, vol)

            if garch_res is None:
                continue

            prev_c = float(closes[i-1])
            price_delta = c_p - prev_c
            rolling_vol = garch_res.sigma_next
            garch_forecast = garch_res.mu_next

            # Forward return target for predictive classification
            future_c = float(closes[i + forward_window])
            future_ret_pts = future_c - c_p

            # SMC Features
            smc_t_val = 1 if smc_st.trend == StructureTrend.BULLISH else (-1 if smc_st.trend == StructureTrend.BEARISH else 0)
            fvg_bias = 0
            if smc_st.active_fvgs:
                last_fvg = smc_st.active_fvgs[-1]
                fvg_bias = 1 if last_fvg.gap_type == FVGType.BULLISH else -1
            
            sweep_bias = 0
            if smc_st.liquidity_sweep:
                sweep_bias = 1 if "BULLISH" in smc_st.liquidity_sweep else -1

            # Chinmay Scalping Skills: Triple-Barrier Evaluation across forward window
            # Bull Target: +step_thresh (1.5R), Bull SL: -step_thresh / 1.5 (1.0R)
            # Bear Target: -step_thresh (1.5R), Bear SL: +step_thresh / 1.5 (1.0R)
            bull_target = c_p + step_thresh
            bull_sl = c_p - (step_thresh / 1.5)
            bear_target = c_p - step_thresh
            bear_sl = c_p + (step_thresh / 1.5)

            is_bull_win = False
            is_bear_win = False

            # Scan forward bars to see which barrier is reached first
            for f_idx in range(i + 1, min(i + forward_window + 1, n_bars)):
                f_h = float(highs[f_idx])
                f_l = float(lows[f_idx])

                # Bull Evaluation
                if not is_bull_win:
                    if f_h >= bull_target and f_l > bull_sl:
                        is_bull_win = True
                    elif f_l <= bull_sl:
                        is_bull_win = False

                # Bear Evaluation
                if not is_bear_win:
                    if f_l <= bear_target and f_h < bear_sl:
                        is_bear_win = True
                    elif f_h >= bear_sl:
                        is_bear_win = False

                if is_bull_win or is_bear_win:
                    break

            # 1. SVM Features (3-dim micro features)
            s_vec = svm_brain.build_feature_vector(price_delta, rolling_vol, garch_forecast)
            if is_bull_win and not is_bear_win:
                svm_target = 1
            elif is_bear_win and not is_bull_win:
                svm_target = -1
            else:
                future_c = float(closes[min(i + forward_window, n_bars - 1)])
                svm_target = 1 if future_c >= c_p else -1
            svm_X.append(s_vec)
            svm_y.append(svm_target)

            # 2. XGBoost Features (14-dim with SMC)
            x_vec = xgb_brain.build_feature_vector(
                price_delta=price_delta,
                rolling_vol=rolling_vol,
                garch_forecast=garch_forecast,
                vpin=vpin_res.vpin,
                dist_pdh=macro_st.dist_pdh_pct,
                dist_pdl=macro_st.dist_pdl_pct,
                week_pos=macro_st.week_range_position,
                trend_15m=macro_st.trend_15m_bias,
                smc_trend=smc_t_val,
                smc_bos=smc_st.bos_direction,
                smc_choch=smc_st.choch_direction,
                smc_range_pos=smc_st.dealing_range_pos,
                smc_fvg_bias=fvg_bias,
                smc_sweep_bias=sweep_bias
            )
            xgb_target = 1 if (is_bull_win and not is_bear_win) else 0
            xgb_X.append(x_vec)
            xgb_y.append(xgb_target)

            # 3. ANN 3-Class Target (0: Bear Scalp Win, 1: Chop/Invalid, 2: Bull Scalp Win)
            if is_bull_win and not is_bear_win:
                ann_target = 2
            elif is_bear_win and not is_bull_win:
                ann_target = 0
            else:
                ann_target = 1
            ann_X.append(x_vec)
            ann_y.append(ann_target)

        ann_y_arr = np.array(ann_y)
        n_bull = int(np.sum(ann_y_arr == 2))
        n_chop = int(np.sum(ann_y_arr == 1))
        n_bear = int(np.sum(ann_y_arr == 0))
        logger.info(
            f"Dataset built for {sym_clean}: {len(svm_X):,} samples. "
            f"ANN Distribution -> BULL(2): {n_bull:,} ({n_bull/len(ann_y)*100:.1f}%), "
            f"CHOP(1): {n_chop:,} ({n_chop/len(ann_y)*100:.1f}%), "
            f"BEAR(0): {n_bear:,} ({n_bear/len(ann_y)*100:.1f}%)"
        )

        # Train SVM on representative high-density window (8,000 samples for fast RBF convergence)
        logger.info("[1/3] Training SVM RBF Kernel Model on 8,000 recent regime samples...")
        svm_brain.feature_history = svm_X[-8000:]
        svm_brain.target_history = svm_y[-8000:]
        svm_ok = svm_brain.train_model()
        svm_path = self.models_dir / f"svm_model_{sym_clean.lower()}.joblib"
        svm_brain.model_path = svm_path
        svm_brain.save_model()
        logger.info(f"SVM RBF Model trained & saved to {svm_path} (Status: {svm_ok})")

        # Train XGBoost on full sample history
        logger.info(f"[2/3] Training XGBoost Gradient Boosted Decision Forest on {len(xgb_X):,} samples...")
        xgb_brain.feature_history = xgb_X
        xgb_brain.target_history = xgb_y
        xgb_ok = xgb_brain.train_model()
        xgb_path = self.models_dir / f"xgb_model_{sym_clean.lower()}.joblib"
        xgb_brain.model_path = xgb_path
        xgb_brain.save_model()
        logger.info(f"XGBoost Model trained & saved to {xgb_path} (Status: {xgb_ok})")

        # Train Deep ANN on full sample history
        logger.info(f"[3/3] Training Deep Artificial Neural Network (ANN) on {len(ann_X):,} samples...")
        ann_brain.feature_history = ann_X
        ann_brain.target_history = ann_y
        ann_path = self.models_dir / f"ann_model_{sym_clean.lower()}.joblib"
        ann_brain.model_path = ann_path
        ann_ok = ann_brain.train()
        logger.info(f"Deep ANN Model trained & saved to {ann_path} (Status: {ann_ok})")

        # Also save default active models if NIFTY
        if sym_clean == "NIFTY":
            joblib.dump(svm_brain.model, ROOT_DIR / "svm_model.joblib")
            joblib.dump(xgb_brain.model, ROOT_DIR / "xgb_model.joblib")
            joblib.dump(ann_brain.model, ROOT_DIR / "ann_model.joblib")
            logger.info("Synchronized primary production models (svm_model.joblib, xgb_model.joblib, ann_model.joblib).")

        logger.info("=" * 80)
        logger.info(f"  COMPLETED TRAINING FOR {sym_clean}: All 3 Brains Ready & Saved!")
        logger.info("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Asset Brain Trainer for NIFTY, SENSEX, and BANKNIFTY")
    parser.add_argument("--symbols", type=str, default="NIFTY,SENSEX,BANKNIFTY", help="Comma-separated symbols to train")
    parser.add_argument("--months", type=int, default=12, help="Number of months of 1-minute history to ingest")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    trainer = MultiAssetBrainTrainer()
    for sym in symbols:
        try:
            trainer.process_and_train(symbol=sym, months=args.months)
        except Exception as e:
            logger.exception(f"Failed to train brains for {sym}: {e}")


if __name__ == "__main__":
    main()
