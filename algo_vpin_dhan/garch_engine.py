"""
Layer 2: GARCH(1,1) Volatility Forecasting Engine
Computes log returns, fits rolling GARCH(1,1) via MLE using arch package,
and produces 1-step-ahead conditional return direction and variance forecasts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings
from arch import arch_model

from .config import GARCHConfig


class DirectionalSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class GARCHForecastResult:
    """Output metrics from 1-step ahead GARCH(1,1) forecast"""
    mu_next: float              # 1-step-ahead conditional mean return mu_{t+1}
    h_next: float               # 1-step-ahead conditional variance h_{t+1}
    sigma_next: float           # 1-step-ahead conditional std deviation sqrt(h_{t+1})
    annualized_vol: float       # Annualized volatility percentage
    signal: DirectionalSignal   # BUY / SELL / HOLD based on delta_1 threshold
    omega: float                # GARCH omega parameter
    alpha: float                # GARCH alpha parameter (ARCH term)
    beta: float                 # GARCH beta parameter (GARCH term)
    is_converged: bool          # Whether MLE solver converged


class GARCHEngine:
    """
    Fits rolling GARCH(1,1) model and generates forward conditional forecasts.
    """

    def __init__(self, config: Optional[GARCHConfig] = None):
        self.config = config or GARCHConfig()
        self.p = self.config.p
        self.q = self.config.q
        self.mean_model = self.config.mean_model
        self.dist = self.config.dist
        self.delta_1 = self.config.delta_1
        self.rolling_window = self.config.rolling_window
        self.min_fit_samples = self.config.min_fit_samples

        # Internal price and return buffers
        self.prices: List[float] = []
        self.returns: List[float] = []
        self.timestamps: List[pd.Timestamp] = []

        # Last successful model parameters
        self._last_omega = 1e-6
        self._last_alpha = 0.05
        self._last_beta = 0.90

    def add_bar(self, close_price: float, timestamp: Optional[pd.Timestamp] = None) -> Optional[GARCHForecastResult]:
        """
        Adds a new close price, updates log returns, and computes 1-step forecast.
        """
        if len(self.prices) > 0:
            prev_price = self.prices[-1]
            if prev_price > 0 and close_price > 0:
                ret = float(np.log(close_price / prev_price))
            else:
                ret = 0.0
            self.returns.append(ret)
            if timestamp is not None:
                self.timestamps.append(timestamp)

        self.prices.append(close_price)

        # Maintain rolling window capacity
        if len(self.returns) > self.rolling_window * 2:
            self.returns = self.returns[-self.rolling_window:]
            self.prices = self.prices[-(self.rolling_window + 1):]
            if self.timestamps:
                self.timestamps = self.timestamps[-self.rolling_window:]

        if len(self.returns) < self.min_fit_samples:
            return None

        return self.forecast_next_step()

    def forecast_next_step(self) -> GARCHForecastResult:
        """
        Fits GARCH(1,1) on trailing returns and generates 1-step ahead forecast.
        """
        window_returns = np.array(self.returns[-self.rolling_window:])

        # Guard against zero variance
        sample_std = float(np.std(window_returns))
        if sample_std < 1e-7:
            sample_std = 1e-5
            window_returns = window_returns + np.random.normal(0, 1e-6, size=len(window_returns))

        # Scale returns by 100 for numerical stability during MLE optimization
        scale_factor = 100.0
        scaled_returns = window_returns * scale_factor

        is_converged = False
        mu_scaled = 0.0
        h_scaled = (sample_std * scale_factor) ** 2
        omega = self._last_omega
        alpha = self._last_alpha
        beta = self._last_beta

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Specify AR(1) or Constant Mean GARCH(1,1) model
                am = arch_model(
                    scaled_returns,
                    mean=self.mean_model,
                    lags=self.config.ar_lags if self.mean_model == "AR" else 0,
                    vol="Garch",
                    p=self.p,
                    q=self.q,
                    dist=self.dist,
                    rescale=False
                )
                res = am.fit(disp="off", show_warning=False)

                if res.convergence_flag == 0:
                    is_converged = True
                    forecast = res.forecast(horizon=1, reindex=False)
                    mu_scaled = float(forecast.mean.iloc[-1, 0])
                    h_scaled = float(forecast.variance.iloc[-1, 0])

                    # Cache parameters
                    omega = float(res.params.get("omega", self._last_omega))
                    alpha = float(res.params.get("alpha[1]", self._last_alpha))
                    beta = float(res.params.get("beta[1]", self._last_beta))
                    self._last_omega, self._last_alpha, self._last_beta = omega, alpha, beta
            except Exception:
                # Fallback to EWMA / previous step estimate if MLE fails
                is_converged = False

        if not is_converged:
            # Fallback formulation: EWMA variance forecast
            mu_scaled = float(np.mean(scaled_returns[-10:]))
            prev_ret_scaled = scaled_returns[-1]
            h_scaled = omega + alpha * (prev_ret_scaled ** 2) + beta * h_scaled

        # Unscale back to raw return units
        mu_next = mu_scaled / scale_factor
        h_next = max(1e-10, h_scaled / (scale_factor ** 2))
        sigma_next = float(np.sqrt(h_next))

        # Annualized volatility (assuming 375 1-minute bars per day, 252 days)
        bars_per_year = 375 * 252
        annualized_vol = sigma_next * np.sqrt(bars_per_year)

        # Directional Trigger:
        # If mu_{t+1} > delta_1 -> BUY
        # If mu_{t+1} < -delta_1 -> SELL
        # Otherwise -> HOLD
        if mu_next > self.delta_1:
            signal = DirectionalSignal.BUY
        elif mu_next < -self.delta_1:
            signal = DirectionalSignal.SELL
        else:
            signal = DirectionalSignal.HOLD

        return GARCHForecastResult(
            mu_next=mu_next,
            h_next=h_next,
            sigma_next=sigma_next,
            annualized_vol=annualized_vol,
            signal=signal,
            omega=omega,
            alpha=alpha,
            beta=beta,
            is_converged=is_converged
        )

    def reset(self):
        """Resets engine state"""
        self.prices.clear()
        self.returns.clear()
        self.timestamps.clear()
