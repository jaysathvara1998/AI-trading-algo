"""
Layer 2: Rolling GARCH(1,1) Volatility & Momentum Forecaster (v2.0)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np
import pandas as pd
from arch import arch_model
import logging

from .config import GARCHConfig

logger = logging.getLogger("algo_vpin_v2.garch_engine")


class DirectionalSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class GARCHForecastResult:
    sigma_next: float
    annualized_vol: float
    mu_next: float
    signal: DirectionalSignal
    omega: float
    alpha: float
    beta: float
    is_fitted: bool
    timestamp: pd.Timestamp


class GARCHEngine:
    def __init__(self, config: Optional[GARCHConfig] = None):
        self.config = config or GARCHConfig()
        self.prices: List[float] = []
        self.log_returns: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        
        self.last_omega: float = 1e-6
        self.last_alpha: float = 0.05
        self.last_beta: float = 0.90
        self.last_mu: float = 0.0
        self.last_variance: float = 1e-4
        self.bars_since_fit: int = 0
        self.is_warmed_up: bool = False

    def add_bar(self, close_price: float, timestamp: pd.Timestamp) -> Optional[GARCHForecastResult]:
        self.prices.append(close_price)
        self.timestamps.append(timestamp)

        if len(self.prices) < 2:
            return None

        ret = np.log(close_price / self.prices[-2])
        self.log_returns.append(ret)

        if len(self.log_returns) > 1000:
            self.log_returns = self.log_returns[-1000:]
            self.prices = self.prices[-1000:]
            self.timestamps = self.timestamps[-1000:]

        if len(self.log_returns) < self.config.min_obs_for_fit:
            rolling_std = float(np.std(self.log_returns, ddof=1)) if len(self.log_returns) > 2 else 0.001
            rolling_mean = float(np.mean(self.log_returns)) if len(self.log_returns) > 2 else 0.0
            return GARCHForecastResult(
                sigma_next=rolling_std,
                annualized_vol=rolling_std * np.sqrt(252 * 375),
                mu_next=rolling_mean,
                signal=DirectionalSignal.HOLD,
                omega=self.last_omega,
                alpha=self.last_alpha,
                beta=self.last_beta,
                is_fitted=False,
                timestamp=timestamp
            )

        self.bars_since_fit += 1
        if not self.is_warmed_up or self.bars_since_fit >= self.config.refit_interval:
            self._fit_garch()

        sigma_next, mu_next = self._forecast_1step()
        annualized_vol = sigma_next * np.sqrt(252 * 375)

        if mu_next >= self.config.buy_threshold_ret:
            signal = DirectionalSignal.BUY
        elif mu_next <= self.config.sell_threshold_ret:
            signal = DirectionalSignal.SELL
        else:
            signal = DirectionalSignal.HOLD

        return GARCHForecastResult(
            sigma_next=sigma_next,
            annualized_vol=annualized_vol,
            mu_next=mu_next,
            signal=signal,
            omega=self.last_omega,
            alpha=self.last_alpha,
            beta=self.last_beta,
            is_fitted=self.is_warmed_up,
            timestamp=timestamp
        )

    def _fit_garch(self):
        rets = np.array(self.log_returns, dtype=np.float64) * 100.0
        try:
            am = arch_model(
                rets,
                mean="AR",
                lags=1,
                vol="GARCH",
                p=self.config.p_order,
                q=self.config.q_order,
                dist=self.config.dist,
                rescale=False
            )
            res = am.fit(disp="off", show_warning=False)
            params = res.params
            self.last_mu = float(params.get("Const", 0.0)) / 100.0
            self.last_omega = float(params.get("omega", 1e-4)) / 10000.0
            self.last_alpha = float(params.get("alpha[1]", 0.05))
            self.last_beta = float(params.get("beta[1]", 0.90))

            cond_vol = res.conditional_volatility
            if len(cond_vol) > 0:
                self.last_variance = float((cond_vol[-1] / 100.0) ** 2)

            self.is_warmed_up = True
            self.bars_since_fit = 0
        except Exception as e:
            logger.debug(f"GARCH MLE optimization note: {e}. Utilizing rolling moments.")
            self.last_variance = float(np.var(self.log_returns[-50:]))
            self.last_mu = float(np.mean(self.log_returns[-20:]))

    def _forecast_1step(self) -> tuple:
        last_ret = self.log_returns[-1]
        last_resid = last_ret - self.last_mu
        h_next = self.last_omega + (self.last_alpha * (last_resid ** 2)) + (self.last_beta * self.last_variance)
        h_next = max(1e-8, min(0.01, h_next))
        self.last_variance = h_next
        sigma_next = float(np.sqrt(h_next))
        mu_next = float(np.mean(self.log_returns[-15:])) if len(self.log_returns) >= 15 else self.last_mu
        return sigma_next, mu_next
