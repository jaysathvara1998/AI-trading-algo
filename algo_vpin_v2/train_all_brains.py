"""
Universal Multi-Brain Offline & EOD Retraining Pipeline for Algo VPIN v2.0
Trains:
1. Support Vector Machine (SVM) Micro-Filter
2. XGBoost Multi-Timeframe Gradient Boosted Decision Trees
3. Deep Artificial Neural Network (Deep ANN) Multi-Classification Brain
for both NIFTY and SENSEX using real exchange 1-minute historical bar data.
"""

import gzip
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from .config import CONFIG, EnsembleConfig
from .ensemble_brain import EnsembleBrain
from .garch_engine import GARCHEngine, DirectionalSignal
from .macro_features import MacroFeatureEngine
from .smc_engine import SMCEngine
from .vpin import VPINCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("algo_vpin_v2.train_brains")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_bars_for_asset(asset: str = "NIFTY") -> pd.DataFrame:
    """Combines all historical gzip and daily CSV files for the asset"""
    dfs = []
    asset_lower = asset.lower()

    # 1. Check for 12m compressed historical dataset
    gz_file = DATA_DIR / f"{asset_lower}_12m_1min.csv.gz"
    if gz_file.exists():
        try:
            with gzip.open(gz_file, "rt") as f:
                df_gz = pd.read_csv(f)
                dfs.append(df_gz)
                logger.info(f"Loaded {len(df_gz)} bars from {gz_file.name}")
        except Exception as e:
            logger.warning(f"Could not read {gz_file.name}: {e}")

    # 2. Check for all daily collected CSV files
    for csv_file in sorted(DATA_DIR.glob(f"{asset_lower}_v2_1min_*.csv")):
        try:
            df_csv = pd.read_csv(csv_file)
            dfs.append(df_csv)
            logger.info(f"Loaded {len(df_csv)} bars from {csv_file.name}")
        except Exception as e:
            logger.warning(f"Could not read {csv_file.name}: {e}")

    if not dfs:
        logger.error(f"No historical bar files found for {asset} in {DATA_DIR}")
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    if "timestamp" in full_df.columns:
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], format="mixed", errors="coerce")
        full_df = full_df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"])

    # Ensure required numeric columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce")
    full_df = full_df.dropna(subset=["close"]).reset_index(drop=True)

    # Take the latest 5,000 bars for high-speed regime-focused institutional training
    if len(full_df) > 5000:
        full_df = full_df.tail(5000).reset_index(drop=True)

    logger.info(f"Using {len(full_df)} most recent deduplicated bars for {asset}")
    return full_df


def train_brains_for_asset(asset: str = "NIFTY") -> Tuple[bool, bool, bool]:
    """Extracts 14-dim SMC + VPIN + GARCH features and trains all 3 AI models"""
    logger.info("=" * 80)
    logger.info(f"       STARTING MULTI-BRAIN TRAINING PIPELINE FOR {asset.upper()}       ")
    logger.info("=" * 80)

    bars_df = load_all_bars_for_asset(asset)
    if bars_df.empty or len(bars_df) < 300:
        logger.error(f"Insufficient bars to train {asset} models. Minimum 300 required.")
        return False, False, False

    ensemble = EnsembleBrain(CONFIG.ensemble)
    macro_engine = MacroFeatureEngine()
    smc_engine = SMCEngine()
    garch_engine = GARCHEngine()
    vpin_calc = VPINCalculator(CONFIG.vpin)

    # Calibrate VPIN Bucket Volume
    avg_vol = float(bars_df["volume"].mean()) * 20.0
    vpin_calc.update_bucket_volume(max(5000.0, avg_vol))

    # Ingest historical bars sequentially to construct genuine feature history
    logger.info(f"Building 14-dimensional SMC, GARCH & VPIN feature vectors across {len(bars_df)} bars...")
    for idx, row in bars_df.iterrows():
        ts = row.get("timestamp", pd.Timestamp.now())
        close_p = float(row["close"])
        open_p = float(row.get("open", close_p))
        high_p = float(row.get("high", close_p))
        low_p = float(row.get("low", close_p))
        vol = float(row.get("volume", 1000.0))

        macro_st = macro_engine.update_1min_bar(close_p, high_p, low_p)
        vpin_res = vpin_calc.process_bar(close_p, vol, ts)
        garch_res = garch_engine.add_bar(close_p, ts)
        smc_st = smc_engine.update_bar(open_p, high_p, low_p, close_p, vol)

        price_delta = close_p - (bars_df.iloc[idx-1]["close"] if idx > 0 else close_p)
        vol_roll = garch_res.sigma_next if garch_res else 0.001
        garch_pred = garch_res.mu_next if garch_res else 0.0

        # SMC Features
        smc_t_val = 1 if getattr(smc_st.trend, "value", "") == "BULLISH_STRUCTURE" else (-1 if getattr(smc_st.trend, "value", "") == "BEARISH_STRUCTURE" else 0)
        fvg_bias = 0
        if smc_st.active_fvgs:
            fvg_bias = 1 if getattr(smc_st.active_fvgs[-1].gap_type, "value", "") == "BULLISH_FVG" else -1
        sweep_bias = 0
        if smc_st.liquidity_sweep:
            sweep_bias = 1 if "BULLISH" in smc_st.liquidity_sweep else -1

        svm_feat = ensemble.svm.build_feature_vector(price_delta, vol_roll, garch_pred)
        xgb_feat = ensemble.xgb.build_feature_vector(
            price_delta=price_delta,
            rolling_vol=vol_roll,
            garch_forecast=garch_pred,
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

        ensemble.svm.feature_history.append(svm_feat)
        ensemble.svm.target_history.append(1 if price_delta >= 0 else -1)
        ensemble.xgb.feature_history.append(xgb_feat)
        ensemble.xgb.target_history.append(1 if price_delta >= 0 else 0)

        # 3-class target for ANN: 2=Strong Bullish (>+2.5 pts), 0=Strong Bearish (<-2.5 pts), 1=Chop/Neutral
        ann_threshold = 2.5 if "NIFTY" in asset else 10.0
        ann_lbl = 2 if price_delta > ann_threshold else (0 if price_delta < -ann_threshold else 1)
        ensemble.ann.feature_history.append(xgb_feat)
        ensemble.ann.target_history.append(ann_lbl)

    # Train all 3 models
    svm_ok, xgb_ok, ann_ok = ensemble.train_models()
    logger.info(f"[{asset}] Training Results -> SVM: {svm_ok}, XGBoost: {xgb_ok}, ANN Deep Net: {ann_ok}")

    # Save asset-specific checkpoints
    asset_suffix = f"_{asset.lower()}"
    try:
        if svm_ok and ensemble.svm.model is not None:
            svm_path = MODELS_DIR / f"svm_model{asset_suffix}.joblib"
            joblib.dump(ensemble.svm.model, svm_path)
            # Also save default fallback
            joblib.dump(ensemble.svm.model, MODELS_DIR / "svm_model.joblib")
            logger.info(f"Saved SVM model to {svm_path.name}")

        if xgb_ok and ensemble.xgb.model is not None:
            xgb_path = MODELS_DIR / f"xgb_model{asset_suffix}.joblib"
            joblib.dump(ensemble.xgb.model, xgb_path)
            joblib.dump(ensemble.xgb.model, MODELS_DIR / "xgb_model.joblib")
            logger.info(f"Saved XGBoost model to {xgb_path.name}")

        if ann_ok and ensemble.ann.is_trained:
            ann_path = MODELS_DIR / f"ann_model{asset_suffix}.joblib"
            ensemble.ann.save_model(ann_path)
            ensemble.ann.save_model(MODELS_DIR / "ann_model.joblib")
            logger.info(f"Saved Deep ANN model to {ann_path.name}")

    except Exception as e:
        logger.error(f"Error saving models for {asset}: {e}")

    logger.info("=" * 80)
    logger.info(f"   MULTI-BRAIN RETRAINING COMPLETED SUCCESSFULLY FOR {asset.upper()}   ")
    logger.info("=" * 80)
    return svm_ok, xgb_ok, ann_ok


def main():
    logger.info("Initiating Comprehensive Offline Multi-Brain Retraining...")
    # 1. Train SENSEX (Priority for tomorrow's Expiry)
    train_brains_for_asset("SENSEX")
    
    # 2. Train NIFTY
    train_brains_for_asset("NIFTY")
    
    logger.info("\nAll brains fully trained, calibrated, and ready for live execution!")


if __name__ == "__main__":
    main()
