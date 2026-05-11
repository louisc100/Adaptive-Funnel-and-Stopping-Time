"""
Interactive Trading Algorithm — Phase 6 Walk-Forward Backtest Version
-----------------------------------------------------
A PyQt5 app with:
  - Sliders for sigma, true mu, adaptive drift strength, k, delta, transaction cost, speed
  - Generate / Play / Pause / Reset buttons
  - Time scrubber to seek anywhere in the simulation
  - Three live-updating panels:
      1. Price path with predictive funnel boundaries
         (buy ^ green, sell v red)
      2. Z deviation signal  (+/- k dashed thresholds)
      3. Wealth curves:
           realized wealth       (amber staircase)
           mark-to-market wealth (purple dashed)
           buy-and-hold benchmark (teal dotted)
  - Live stats sidebar (t, price, wealth, return, drawdown, benchmark gap)

Phase 6 includes:
  1. Explicit profit threshold h = log((1+c_buy)/(1-c_sell)).
  2. Sell condition requires log_return > h.
  3. Transaction-cost-adjusted realized wealth update.
  4. Buy-and-hold benchmark added.
  5. Holding state is recorded after same-step decisions.
  6. Gaussian shocks are no longer clipped by default.
  7. Completed trade records, drawdown, exposure, and benchmark-gap metrics.
  8. Visible center, upper, and lower funnel paths on the price chart.
  9. Kalman-filtered drift estimates used by the predictive funnel.
 10. GARCH-style conditional volatility forecasts used by the funnel width.
 11. Walk-forward parameter selection with out-of-sample test folds.

Dependencies:
    pip install numpy matplotlib PyQt5
"""

import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QGridLayout, QGroupBox, QSizePolicy,
    QMessageBox, QInputDialog, QDialog, QListWidget, QListWidgetItem,
    QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = "#0f1117"
PANEL_BG = "#1a1d27"
GRID_C   = "#262736"
TEXT_C   = "#ccccdd"
BLUE     = "#378ADD"
GREEN    = "#1D9E75"
RED      = "#D85A30"
AMBER    = "#EF9F27"
PURPLE   = "#7F77DD"
TEAL     = "#9FE1CB"
F_UPPER  = "#F2B84B"
F_LOWER  = "#6CC6FF"
DEFAULT_REAL_SYMBOL = "SPY"
REAL_DATA_CHOICES = [
    ("SPY", "S&P 500 ETF"),
    ("NVDA", "Nvidia"),
    ("TSLA", "Tesla"),
    ("MSFT", "Microsoft"),
    ("AAPL", "Apple"),
    ("AMZN", "Amazon"),
    ("GOOGL", "Alphabet"),
    ("META", "Meta"),
    ("Custom ticker...", "Type another Yahoo symbol"),
]
REAL_PERIOD_CHOICES = [
    ("5 years", 5 * 365.25, "1d"),
    ("2 years", 2 * 365.25, "1d"),
    ("1 year", 365.25, "1d"),
    ("1 month", 31, "1h"),
    ("1 week", 7, "15m"),
]
DEFAULT_RANDOM_STOP_SLOPE = 4.0
DEFAULT_RANDOM_STOP_INTERCEPT = 0.0
DEFAULT_RANDOM_STOP_SEED = 17


def real_period_label(days, interval=None):
    """Return the UI label for a configured real-data lookback."""
    if days is None:
        choice_label, _, choice_interval = REAL_PERIOD_CHOICES[0]
        label = choice_label
        interval = interval or choice_interval
    else:
        label = f"{float(days):.0f} days"
        for choice_label, choice_days, choice_interval in REAL_PERIOD_CHOICES:
            if abs(float(days) - float(choice_days)) < 1e-9:
                label = choice_label
                if interval is None:
                    interval = choice_interval
                break
    return f"{label}, {interval} bars" if interval else label


def periods_per_year_for_interval(interval):
    """Approximate U.S. equity bar count per year for annualized metrics."""
    return {
        "1d": 252,
        "1wk": 52,
        "1h": 252 * 6.5,
        "30m": 252 * 13,
        "15m": 252 * 26,
        "5m": 252 * 78,
    }.get(interval, 252)


# ── Phase 1 utilities ─────────────────────────────────────────────────────────
def profit_threshold(c_buy=0.0, c_sell=0.0):
    """
    Minimum log-return needed to break even after proportional transaction costs.

    h = log((1+c_buy)/(1-c_sell))
    """
    if c_buy < 0:
        raise ValueError("c_buy must be nonnegative.")
    if not (0 <= c_sell < 1):
        raise ValueError("c_sell must satisfy 0 <= c_sell < 1.")
    return np.log((1.0 + c_buy) / (1.0 - c_sell))


def net_liquidation_multiplier(price_now, price_buy, c_buy=0.0, c_sell=0.0):
    """
    Net wealth multiplier if one enters at price_buy and liquidates at price_now.
    Uses full-wealth investment convention.
    """
    return ((1.0 - c_sell) * price_now) / ((1.0 + c_buy) * price_buy)


def sigmoid(x):
    """Numerically stable logistic map used by randomized stopping."""
    x = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + np.exp(-x))


def max_drawdown(values):
    """Return the worst peak-to-trough drawdown as a negative decimal."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1.0
    return float(np.min(drawdowns))


def summarize_simulation(data, t=None):
    """Phase 2 summary metrics for strategy, mark-to-market, and benchmark."""
    if data is None:
        return {}
    if t is None:
        t = data["N"]
    t = max(0, min(int(t), data["N"]))
    sl = slice(0, t + 1)

    realized = data["realized_w"][t]
    mtm = data["portfolio_w"][t]
    buy_hold = data["buy_hold_w"][t]
    holding = data["holding"][sl]

    return {
        "realized_return": realized / 1000.0 - 1.0,
        "mtm_return": mtm / 1000.0 - 1.0,
        "buy_hold_return": buy_hold / 1000.0 - 1.0,
        "benchmark_gap": mtm / buy_hold - 1.0 if buy_hold != 0 else np.nan,
        "max_mtm_drawdown": max_drawdown(data["portfolio_w"][sl]),
        "max_buy_hold_drawdown": max_drawdown(data["buy_hold_w"][sl]),
        "exposure": float(np.mean(holding)) if holding.size else 0.0,
        "completed_trades": len([tr for tr in data["trade_log"] if tr["sell_t"] <= t]),
    }


def annualized_sharpe(values, periods_per_year=252):
    """Annualized Sharpe ratio from a wealth curve, using zero risk-free rate."""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return np.nan
    returns = np.diff(values) / values[:-1]
    vol = np.std(returns, ddof=1)
    if vol <= 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * np.mean(returns) / vol)


def calmar_ratio(values, periods_per_year=252):
    """Annualized return divided by absolute maximum drawdown."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or values[0] <= 0:
        return np.nan
    years = max((values.size - 1) / periods_per_year, 1e-12)
    annual_return = (values[-1] / values[0]) ** (1.0 / years) - 1.0
    drawdown = abs(max_drawdown(values))
    if drawdown <= 0:
        return np.nan
    return float(annual_return / drawdown)


def kalman_drift_estimates(log_returns, init_mu, obs_var, process_var):
    """
    Estimate one-step drift with a local-level Kalman filter.

    mu_prior[s] is the drift estimate used to predict return s before seeing it.
    mu_filtered[t] is the posterior estimate after observing returns through t.
    """
    n_steps = len(log_returns)
    obs_var = max(float(obs_var), 1e-12)
    process_var = max(float(process_var), 0.0)

    mu_prior = np.zeros(n_steps + 1)
    mu_filtered = np.zeros(n_steps + 1)
    post_mean = float(init_mu)
    post_var = obs_var
    mu_prior[0] = post_mean
    mu_filtered[0] = post_mean

    for step, r in enumerate(log_returns, start=1):
        pred_mean = post_mean
        pred_var = post_var + process_var
        mu_prior[step] = pred_mean

        kalman_gain = pred_var / (pred_var + obs_var)
        post_mean = pred_mean + kalman_gain * (r - pred_mean)
        post_var = (1.0 - kalman_gain) * pred_var
        mu_filtered[step] = post_mean

    return mu_prior, mu_filtered


def garch_volatility_estimates(
    log_returns,
    mu_prior,
    init_var,
    alpha=0.06,
    beta=0.90,
):
    """
    Estimate one-step conditional variance with a GARCH(1,1)-style recursion.

    var_prior[s] is the variance forecast used for return s before seeing it.
    The update uses the squared innovation after return s is observed.
    """
    n_steps = len(log_returns)
    init_var = max(float(init_var), 1e-12)
    alpha = min(max(float(alpha), 0.0), 0.98)
    beta = min(max(float(beta), 0.0), 0.98)
    if alpha + beta >= 0.999:
        scale = 0.999 / (alpha + beta)
        alpha *= scale
        beta *= scale

    omega = (1.0 - alpha - beta) * init_var
    var_prior = np.zeros(n_steps + 1)
    sigma_hat = np.zeros(n_steps + 1)
    next_var = init_var
    var_prior[0] = init_var
    sigma_hat[0] = np.sqrt(init_var)

    for step, r in enumerate(log_returns, start=1):
        forecast_var = max(next_var, 1e-12)
        var_prior[step] = forecast_var
        sigma_hat[step] = np.sqrt(forecast_var)

        innovation = r - mu_prior[step]
        next_var = omega + alpha * innovation * innovation + beta * forecast_var

    return var_prior, sigma_hat


def gaussian_likelihood(x, means, variances):
    """Numerically safe Gaussian density values for HMM filtering."""
    variances = np.maximum(np.asarray(variances, dtype=float), 1e-12)
    means = np.asarray(means, dtype=float)
    z = (x - means) / np.sqrt(variances)
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi * variances)


def regime_hmm_estimates(log_returns, n_states=3, init_window=60, stay_prob=0.94):
    """
    Lightweight Gaussian HMM filter for regime-aware funnel forecasts.

    The emission parameters are initialized from an early training window by
    sorting returns into volatility buckets. Forward filtering is then causal:
    the forecast for step s uses the predicted regime distribution before
    observing return s, while the displayed state uses the posterior after it.
    """
    r = np.asarray(log_returns, dtype=float)
    n_steps = len(r)
    n_states = int(np.clip(n_states, 2, 5))
    init_n = int(np.clip(init_window, n_states * 5, max(n_states * 5, n_steps)))
    train = r[:init_n] if init_n > 0 else r
    if train.size < n_states:
        train = r if r.size else np.zeros(n_states)

    center = float(np.mean(train)) if train.size else 0.0
    vol_score = np.abs(train - center)
    order = np.argsort(vol_score)
    buckets = np.array_split(train[order], n_states)

    means = np.array([
        float(np.mean(bucket)) if bucket.size else center
        for bucket in buckets
    ])
    variances = np.array([
        float(np.var(bucket, ddof=1)) if bucket.size > 1 else float(np.var(train))
        for bucket in buckets
    ])
    global_var = float(np.var(train, ddof=1)) if train.size > 1 else 1e-4
    variances = np.maximum(variances, max(global_var * 0.05, 1e-8))

    # Label states from calm to stress by increasing volatility.
    state_order = np.argsort(variances)
    means = means[state_order]
    variances = variances[state_order]

    off_diag = (1.0 - stay_prob) / max(n_states - 1, 1)
    trans = np.full((n_states, n_states), off_diag)
    np.fill_diagonal(trans, stay_prob)

    prior = np.full(n_states, 1.0 / n_states)
    probs = np.zeros((n_steps + 1, n_states))
    mu_prior = np.zeros(n_steps + 1)
    mu_hat = np.zeros(n_steps + 1)
    var_prior = np.zeros(n_steps + 1)
    sigma_hat = np.zeros(n_steps + 1)
    regime_state = np.zeros(n_steps + 1, dtype=int)
    probs[0] = prior
    mu_prior[0] = float(np.dot(prior, means))
    second_moment = np.dot(prior, variances + means * means)
    var_prior[0] = max(second_moment - mu_prior[0] ** 2, 1e-12)
    sigma_hat[0] = np.sqrt(var_prior[0])

    post = prior
    for step, ret in enumerate(r, start=1):
        pred = post @ trans
        pred = pred / max(np.sum(pred), 1e-12)
        mu_pred = float(np.dot(pred, means))
        second_moment = float(np.dot(pred, variances + means * means))
        var_pred = max(second_moment - mu_pred * mu_pred, 1e-12)
        mu_prior[step] = mu_pred
        var_prior[step] = var_pred
        sigma_hat[step] = np.sqrt(var_pred)

        likelihood = gaussian_likelihood(ret, means, variances)
        post = pred * likelihood
        post = post / max(np.sum(post), 1e-12)
        probs[step] = post
        regime_state[step] = int(np.argmax(post))
        mu_hat[step] = float(np.dot(post, means))

    return {
        "mu_prior": mu_prior,
        "mu_hat": mu_hat,
        "var_prior": var_prior,
        "sigma_hat": sigma_hat,
        "regime_probs": probs,
        "regime_state": regime_state,
        "regime_means": means,
        "regime_variances": variances,
        "transition_matrix": trans,
        "regime_labels": ["calm", "mixed", "stress"][:n_states],
    }


def run_strategy_on_log_prices(
    lp,
    k,
    delta,
    sigma_seed,
    c_buy=0.0,
    c_sell=0.0,
    drift_process_var=1e-7,
    drift_init=0.0,
    garch_alpha=0.06,
    garch_beta=0.90,
    max_funnel_lookback=60,
    trailing_stop=0.04,
    trend_entry_z=0.35,
    generalized_momentum_c=None,
    use_hmm_regime=False,
    hmm_states=3,
    randomized_stopping=False,
    random_stop_slope=DEFAULT_RANDOM_STOP_SLOPE,
    random_stop_intercept=DEFAULT_RANDOM_STOP_INTERCEPT,
    random_stop_seed=DEFAULT_RANDOM_STOP_SEED,
    mode="Simulation",
    extra=None,
):
    """Run the adaptive funnel rule on an existing log-price path."""
    lp = np.asarray(lp, dtype=float)
    if lp.size < 2:
        raise ValueError("A price path must contain at least two observations.")
    prices = np.exp(lp)
    log_returns = np.diff(lp)

    n_steps = lp.size - 1
    N = lp.size
    sigma_seed = max(float(sigma_seed), 1e-6)
    hmm_info = None
    if use_hmm_regime:
        hmm_info = regime_hmm_estimates(log_returns, n_states=hmm_states)
        mu_step = hmm_info["mu_prior"]
        mu_hat = hmm_info["mu_hat"]
        var_step = hmm_info["var_prior"]
        sigma_hat = hmm_info["sigma_hat"]
    else:
        mu_step, mu_hat = kalman_drift_estimates(
            log_returns,
            init_mu=drift_init,
            obs_var=sigma_seed * sigma_seed,
            process_var=drift_process_var,
        )
        var_step, sigma_hat = garch_volatility_estimates(
            log_returns,
            mu_prior=mu_step,
            init_var=sigma_seed * sigma_seed,
            alpha=garch_alpha,
            beta=garch_beta,
        )
    cumulative_drift = np.zeros(N)
    cumulative_var = np.zeros(N)
    cumulative_drift[1:] = np.cumsum(mu_step[1:])
    cumulative_var[1:] = np.cumsum(var_step[1:])

    stop_rng = np.random.default_rng(random_stop_seed) if randomized_stopping else None

    realized_w  = np.full(N, 1000.0)
    portfolio_w = np.full(N, 1000.0)
    buy_hold_w  = np.full(N, 1000.0)
    Zsig        = np.full(N, np.nan)
    funnel_mid  = np.full(N, np.nan)
    funnel_up   = np.full(N, np.nan)
    funnel_low  = np.full(N, np.nan)
    holding     = np.zeros(N, dtype=bool)
    buy_times   = []
    sell_times  = []
    trade_log   = []

    h = profit_threshold(c_buy, c_sell)

    wealth = 1000.0
    in_pos = True
    buy_t  = 1
    buy_lp = lp[1]
    buy_price = prices[1]
    peak_lp = buy_lp
    buy_times.append(1)

    # Buy-and-hold benchmark starts from the same initial buy time.
    bh_buy_t = 1
    bh_buy_price = prices[bh_buy_t]

    for t in range(1, N):
        M = lp[t] - lp[t - delta] if t >= delta else np.nan
        el = t - buy_t if buy_t is not None else 0
        ref_t = buy_t
        ref_lp = buy_lp
        if max_funnel_lookback is not None and el > max_funnel_lookback:
            ref_t = t - max_funnel_lookback
            ref_lp = lp[ref_t]
        if el >= 0:
            drift_since_ref = cumulative_drift[t] - cumulative_drift[ref_t]
            var_since_ref = cumulative_var[t] - cumulative_var[ref_t]
            center = ref_lp + drift_since_ref
            width = k * np.sqrt(var_since_ref)
            funnel_mid[t] = np.exp(center)
            funnel_up[t] = np.exp(center + width)
            funnel_low[t] = np.exp(center - width)

        if in_pos:
            peak_lp = max(peak_lp, lp[t])
            if el >= 1:
                denom = np.sqrt(var_since_ref)
                Z = (lp[t] - ref_lp - drift_since_ref) / denom if denom > 0 else np.nan
                Zsig[t] = Z
                log_ret = lp[t] - buy_lp
                trail_drawdown = peak_lp - lp[t]
                momentum_threshold = 0.0
                if generalized_momentum_c is not None and el > 0:
                    momentum_threshold = -float(generalized_momentum_c) / np.sqrt(el)
                funnel_exit = (
                    el >= delta + 1
                    and not np.isnan(Z)
                    and Z > k
                    and not np.isnan(M)
                    and M <= momentum_threshold
                    and log_ret > h
                )
                trailing_exit = (
                    trailing_stop is not None
                    and log_ret > h
                    and trail_drawdown >= trailing_stop
                )

                # Phase 1 corrected sell rule:
                # funnel exit or trailing-profit exit, both above the cost threshold.
                if funnel_exit or trailing_exit:
                    exit_reason = "trailing" if trailing_exit else "funnel"
                    stop_probability = 1.0
                    stop_draw = np.nan
                    if randomized_stopping:
                        strengths = []
                        if funnel_exit:
                            strengths.append(float(Z - k))
                        if trailing_exit:
                            scale = max(float(trailing_stop), 1e-12)
                            strengths.append(float((trail_drawdown - trailing_stop) / scale))
                        exit_strength = max(strengths) if strengths else 0.0
                        stop_probability = sigmoid(
                            random_stop_intercept
                            + random_stop_slope * max(exit_strength, 0.0)
                        )
                        stop_draw = float(stop_rng.random())
                        if stop_draw >= stop_probability:
                            holding[t] = in_pos
                            realized_w[t] = wealth
                            portfolio_w[t] = wealth * net_liquidation_multiplier(
                                prices[t], buy_price, c_buy=c_buy, c_sell=c_sell
                            )
                            if t >= bh_buy_t:
                                buy_hold_w[t] = 1000.0 * net_liquidation_multiplier(
                                    prices[t], bh_buy_price, c_buy=c_buy, c_sell=c_sell
                                )
                            continue
                        exit_reason = f"randomized-{exit_reason}"
                    net_mult = net_liquidation_multiplier(
                        prices[t], buy_price, c_buy=c_buy, c_sell=c_sell
                    )
                    trade_log.append({
                        "buy_t": buy_t,
                        "sell_t": t,
                        "gross_log_return": log_ret,
                        "net_multiplier": net_mult,
                        "exit_reason": exit_reason,
                        "stop_probability": stop_probability,
                        "stop_draw": stop_draw,
                    })
                    wealth *= net_mult
                    in_pos = False
                    sell_times.append(t)
                    # Last sell time becomes reference point for the symmetric re-entry rule.
                    buy_lp = lp[t]
                    buy_t  = t
                    buy_price = prices[t]
        else:
            # Re-entry rule:
            #   1. bargain buy after a lower-funnel deviation and momentum recovery;
            #   2. trend buy when price is above the adaptive center with positive momentum.
            if buy_t is not None and t >= delta + 1 and not np.isnan(M):
                if el >= 1:
                    denom = np.sqrt(var_since_ref)
                    Zd = (lp[t] - buy_lp - drift_since_ref) / denom if denom > 0 else np.nan
                    bargain_entry = not np.isnan(Zd) and Zd < -k and M >= 0
                    trend_entry = (
                        trend_entry_z is not None
                        and not np.isnan(Zd)
                        and Zd > trend_entry_z
                        and M > 0
                    )
                    if bargain_entry or trend_entry:
                        in_pos = True
                        buy_t  = t
                        buy_lp = lp[t]
                        buy_price = prices[t]
                        peak_lp = buy_lp
                        buy_times.append(t)

        # Record state AFTER same-step buy/sell decisions.
        holding[t] = in_pos
        realized_w[t] = wealth

        # Mark-to-market wealth is interpreted as after-cost liquidation value.
        if in_pos and buy_t is not None:
            portfolio_w[t] = wealth * net_liquidation_multiplier(
                prices[t], buy_price, c_buy=c_buy, c_sell=c_sell
            )
        else:
            portfolio_w[t] = wealth

        # Buy-and-hold benchmark, also shown as after-cost liquidation value.
        if t >= bh_buy_t:
            buy_hold_w[t] = 1000.0 * net_liquidation_multiplier(
                prices[t], bh_buy_price, c_buy=c_buy, c_sell=c_sell
            )

    hmm_extra = {}
    if hmm_info is not None:
        hmm_extra = {
            key: hmm_info[key]
            for key in (
                "regime_probs",
                "regime_state",
                "regime_means",
                "regime_variances",
                "transition_matrix",
                "regime_labels",
            )
        }

    return dict(
        prices=prices, lp=lp,
        realized_w=realized_w, portfolio_w=portfolio_w, buy_hold_w=buy_hold_w,
        funnel_mid=funnel_mid, funnel_up=funnel_up, funnel_low=funnel_low,
        Zsig=Zsig, holding=holding,
        buy_times=set(buy_times), sell_times=set(sell_times),
        trade_log=trade_log,
        mu_step=mu_step, mu_hat=mu_hat,
        var_step=var_step, sigma_hat=sigma_hat,
        N=n_steps, k=k, h=h, c_buy=c_buy, c_sell=c_sell,
        mode=mode, drift_process_var=drift_process_var,
        drift_init=drift_init, garch_alpha=garch_alpha, garch_beta=garch_beta,
        max_funnel_lookback=max_funnel_lookback,
        trailing_stop=trailing_stop,
        trend_entry_z=trend_entry_z,
        generalized_momentum_c=generalized_momentum_c,
        use_hmm_regime=use_hmm_regime,
        hmm_states=hmm_states,
        randomized_stopping=randomized_stopping,
        random_stop_slope=random_stop_slope,
        random_stop_intercept=random_stop_intercept,
        random_stop_seed=random_stop_seed,
        **hmm_extra,
        **(extra or {}),
    )


# ── Simulation and walk-forward backtest ──────────────────────────────────────
def run_simulation(
    mu,
    sigma,
    k,
    delta,
    n_steps=500,
    seed=None,
    c_buy=0.0,
    c_sell=0.0,
    clip_noise=False,
    drift_process_var=1e-7,
    drift_init=0.0,
    garch_alpha=0.06,
    garch_beta=0.90,
    max_funnel_lookback=60,
    trailing_stop=0.04,
    trend_entry_z=0.35,
    generalized_momentum_c=None,
    randomized_stopping=False,
    random_stop_slope=DEFAULT_RANDOM_STOP_SLOPE,
    random_stop_intercept=DEFAULT_RANDOM_STOP_INTERCEPT,
    random_stop_seed=DEFAULT_RANDOM_STOP_SEED,
):
    """
    Simulate the adaptive funnel rule.

    The exact paper model uses Gaussian shocks. Set clip_noise=True only for
    smoother visualization, not for theory validation.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_steps)
    if clip_noise:
        noise = np.clip(noise, -2.0, 2.0)

    lp = np.empty(n_steps + 1)
    lp[0] = np.log(100.0)
    for t in range(n_steps):
        lp[t + 1] = lp[t] + mu + sigma * noise[t]

    return run_strategy_on_log_prices(
        lp,
        k=k,
        delta=delta,
        sigma_seed=sigma,
        c_buy=c_buy,
        c_sell=c_sell,
        drift_process_var=drift_process_var,
        drift_init=drift_init,
        garch_alpha=garch_alpha,
        garch_beta=garch_beta,
        max_funnel_lookback=max_funnel_lookback,
        trailing_stop=trailing_stop,
        trend_entry_z=trend_entry_z,
        generalized_momentum_c=generalized_momentum_c,
        randomized_stopping=randomized_stopping,
        random_stop_slope=random_stop_slope,
        random_stop_intercept=random_stop_intercept,
        random_stop_seed=random_stop_seed,
        mode="Simulation",
        extra={"clip_noise": clip_noise},
    )


def generate_regime_price_path(n_steps=756, seed=7):
    """Create a real-data-like path with changing drift and volatility regimes."""
    rng = np.random.default_rng(seed)
    lp = np.empty(n_steps + 1)
    lp[0] = np.log(100.0)
    regimes = [
        (0.00035, 0.010),
        (-0.00020, 0.018),
        (0.00055, 0.012),
        (-0.00005, 0.024),
    ]
    for t in range(n_steps):
        mu_r, sigma_r = regimes[(t // 126) % len(regimes)]
        shock = rng.standard_normal()
        if rng.random() < 0.025:
            shock += rng.normal(0.0, 2.5)
        lp[t + 1] = lp[t] + mu_r + sigma_r * shock
    return lp


def estimate_sigma_seed(log_prices):
    """Use only past returns to initialize adaptive volatility."""
    returns = np.diff(np.asarray(log_prices, dtype=float))
    if returns.size < 2:
        return 0.015
    sigma = float(np.std(returns, ddof=1))
    return max(sigma, 1e-4)


def load_price_csv(path, price_columns=("Adj Close", "Close", "Price")):
    """Load adjusted-close style prices from a CSV file without extra packages."""
    prices = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")
        price_col = next((c for c in price_columns if c in reader.fieldnames), None)
        if price_col is None:
            raise ValueError(
                "CSV must contain one of these columns: "
                + ", ".join(price_columns)
            )
        for row in reader:
            raw = row.get(price_col, "")
            if raw:
                prices.append(float(raw.replace(",", "")))
    prices = np.asarray(prices, dtype=float)
    if prices.size < 2 or np.any(prices <= 0):
        raise ValueError("CSV must contain at least two positive prices.")
    return np.log(prices)


def parse_daily_price_rows(rows, price_columns=("Adj Close", "Close", "Price")):
    """Convert API/CSV rows into log prices, preserving chronological order."""
    if not rows:
        raise ValueError("No price rows were returned.")
    fieldnames = rows[0].keys()
    price_col = next((c for c in price_columns if c in fieldnames), None)
    if price_col is None:
        raise ValueError(
            "Price data must contain one of these columns: "
            + ", ".join(price_columns)
        )

    cleaned = []
    for row in rows:
        raw_price = row.get(price_col, "")
        raw_date = row.get("Date", "")
        if raw_price in ("", "null", "None"):
            continue
        cleaned.append((raw_date, float(raw_price.replace(",", ""))))
    cleaned.sort(key=lambda item: item[0])

    prices = np.asarray([price for _, price in cleaned], dtype=float)
    if prices.size < 2 or np.any(prices <= 0):
        raise ValueError("Price data must contain at least two positive prices.")
    return np.log(prices)


def fetch_yahoo_log_prices(symbol=DEFAULT_REAL_SYMBOL, years=5, days=None, interval="1d"):
    """
    Fetch adjusted/close price bars from Yahoo Finance's public chart endpoint.

    Example symbol: SPY. This endpoint needs no API key.
    """
    period2 = int(time.time())
    lookback_days = float(days) if days is not None else float(years) * 365.25
    period1 = period2 - int(lookback_days * 24 * 60 * 60)
    query_symbol = urllib.parse.quote(symbol.upper())
    query_interval = urllib.parse.quote(interval)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{query_symbol}"
        f"?period1={period1}&period2={period2}"
        f"&interval={query_interval}&events=history&includeAdjustedClose=true"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not fetch Yahoo data for {symbol}: {exc}") from exc

    payload = json.loads(text)
    error = payload.get("chart", {}).get("error")
    if error:
        raise ValueError(f"Yahoo returned an error for {symbol}: {error}")
    results = payload.get("chart", {}).get("result") or []
    if not results:
        raise ValueError(f"Yahoo returned no data for symbol {symbol}.")

    indicators = results[0].get("indicators", {})
    adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")
    if not adjclose:
        quote_close = indicators.get("quote", [{}])[0].get("close")
        adjclose = quote_close
    prices = np.asarray([p for p in adjclose if p is not None], dtype=float)
    if prices.size < 2 or np.any(prices <= 0):
        raise ValueError(f"Yahoo returned insufficient positive prices for {symbol}.")
    return np.log(prices)


def fetch_yahoo_daily_log_prices(symbol=DEFAULT_REAL_SYMBOL, years=5, days=None):
    """Backward-compatible daily Yahoo fetch wrapper."""
    return fetch_yahoo_log_prices(symbol=symbol, years=years, days=days, interval="1d")


def select_walk_forward_params(
    train_lp,
    k_grid,
    delta_grid,
    c_buy,
    c_sell,
    drift_process_var,
    garch_alpha,
    garch_beta,
):
    """Choose k and delta on the training window only."""
    sigma_seed = estimate_sigma_seed(train_lp)
    best = None
    for k_candidate in k_grid:
        for delta_candidate in delta_grid:
            if delta_candidate >= len(train_lp) - 2:
                continue
            data = run_strategy_on_log_prices(
                train_lp,
                k=k_candidate,
                delta=delta_candidate,
                sigma_seed=sigma_seed,
                c_buy=c_buy,
                c_sell=c_sell,
                drift_process_var=drift_process_var,
                garch_alpha=garch_alpha,
                garch_beta=garch_beta,
                mode="Training fold",
            )
            metrics = summarize_simulation(data)
            score = (
                metrics["mtm_return"]
                + metrics["max_mtm_drawdown"]
                + 0.15 * metrics["benchmark_gap"]
            )
            if best is None or score > best["score"]:
                best = {
                    "k": k_candidate,
                    "delta": delta_candidate,
                    "score": score,
                    "sigma_seed": sigma_seed,
                }
    return best


PLAY_LEARN_GRIDS = {
    "k": [0.8, 1.1, 1.4, 1.8, 2.2],
    "delta": [1, 2, 3, 5, 8, 13, 21],
    "q": [0.0, 5e-8, 1e-7, 2e-7, 5e-7],
    "L": [20, 40, 60, 90, 126],
    "a": [0.02, 0.04, 0.06, 0.08],
    "z_trend": [0.0, 0.25, 0.35, 0.50, 0.75],
    "rho": [0.10, 0.25, 0.40, 0.60],
}


def select_online_learning_params(
    train_lp,
    optimize_params,
    current_params,
    c_buy,
    c_sell,
    garch_alpha,
    garch_beta,
):
    """Choose selected Play-Learn parameters on the recent training window."""
    sigma_seed = estimate_sigma_seed(train_lp)
    train_safe_delta = max(1, len(train_lp) - 3)
    fixed_current = dict(current_params)
    if "delta" not in optimize_params:
        fixed_current["delta"] = min(int(fixed_current["delta"]), train_safe_delta)
    if "L" not in optimize_params:
        fixed_current["L"] = int(fixed_current["L"])
    candidate_lists = [
        PLAY_LEARN_GRIDS[name] if name in optimize_params else [fixed_current[name]]
        for name in PLAY_LEARN_GRIDS
    ]
    best = None
    names = list(PLAY_LEARN_GRIDS)
    for values in np.array(np.meshgrid(*candidate_lists, indexing="ij")).T.reshape(-1, len(names)):
        candidate = dict(zip(names, values))
        candidate["delta"] = int(candidate["delta"])
        candidate["L"] = int(candidate["L"])
        if candidate["delta"] >= len(train_lp) - 2:
            continue
        data = run_strategy_on_log_prices(
            train_lp,
            k=candidate["k"],
            delta=candidate["delta"],
            sigma_seed=sigma_seed,
            c_buy=c_buy,
            c_sell=c_sell,
            drift_process_var=candidate["q"],
            garch_alpha=garch_alpha,
            garch_beta=garch_beta,
            max_funnel_lookback=candidate["L"],
            trailing_stop=candidate["a"],
            trend_entry_z=candidate["z_trend"],
            mode="Play-Learn training fold",
        )
        metrics = summarize_simulation(data)
        score = (
            metrics["mtm_return"]
            + metrics["max_mtm_drawdown"]
            + 0.15 * metrics["benchmark_gap"]
        )
        if best is None or score > best["score"]:
            best = {**candidate, "score": score, "sigma_seed": sigma_seed}
    return best


def stitch_walk_forward_tests(fold_results, folds, source, mode):
    """Build one animated out-of-sample path from sequential test folds."""
    if not fold_results:
        raise RuntimeError("No walk-forward test folds were produced.")

    concat = {
        "prices": [],
        "lp": [],
        "realized_w": [],
        "portfolio_w": [],
        "funnel_mid": [],
        "funnel_up": [],
        "funnel_low": [],
        "Zsig": [],
        "holding": [],
        "mu_step": [],
        "mu_hat": [],
        "var_step": [],
        "sigma_hat": [],
    }
    buy_times = set()
    sell_times = set()
    trade_log = []
    mtm_capital = 1.0
    realized_capital = 1.0
    h = fold_results[0]["h"]
    c_buy = fold_results[0]["c_buy"]
    c_sell = fold_results[0]["c_sell"]
    offset = 0

    for fold_idx, data in enumerate(fold_results):
        start = 0 if fold_idx == 0 else 1
        local_len = len(data["prices"]) - start

        for key in ("prices", "lp", "funnel_mid", "funnel_up", "funnel_low",
                    "Zsig", "holding", "mu_step", "mu_hat", "var_step", "sigma_hat"):
            concat[key].extend(data[key][start:])
        concat["realized_w"].extend(data["realized_w"][start:] * realized_capital)
        concat["portfolio_w"].extend(data["portfolio_w"][start:] * mtm_capital)

        for buy_t in data["buy_times"]:
            if buy_t >= start:
                buy_times.add(offset + buy_t - start)
        for sell_t in data["sell_times"]:
            if sell_t >= start:
                sell_times.add(offset + sell_t - start)
        for trade in data["trade_log"]:
            trade_log.append({
                **trade,
                "buy_t": offset + trade["buy_t"] - start,
                "sell_t": offset + trade["sell_t"] - start,
                "fold": fold_idx + 1,
            })

        realized_capital *= data["realized_w"][-1] / 1000.0
        mtm_capital *= data["portfolio_w"][-1] / 1000.0
        offset += local_len

    prices = np.asarray(concat["prices"], dtype=float)
    buy_hold_w = np.full(prices.size, 1000.0)
    if prices.size > 1:
        bh_buy_price = prices[1]
        for t in range(1, prices.size):
            buy_hold_w[t] = 1000.0 * net_liquidation_multiplier(
                prices[t], bh_buy_price, c_buy=c_buy, c_sell=c_sell
            )

    summary = {
        "folds": folds,
        "oos_return": mtm_capital - 1.0,
        "avg_fold_return": float(np.mean([f["mtm_return"] for f in folds])),
        "avg_benchmark_gap": float(np.mean([f["benchmark_gap"] for f in folds])),
        "avg_sharpe": float(np.nanmean([f["sharpe"] for f in folds])),
        "avg_calmar": float(np.nanmean([f["calmar"] for f in folds])),
    }
    last_fold = folds[-1]
    return dict(
        prices=prices,
        lp=np.asarray(concat["lp"], dtype=float),
        realized_w=np.asarray(concat["realized_w"], dtype=float),
        portfolio_w=np.asarray(concat["portfolio_w"], dtype=float),
        buy_hold_w=buy_hold_w,
        funnel_mid=np.asarray(concat["funnel_mid"], dtype=float),
        funnel_up=np.asarray(concat["funnel_up"], dtype=float),
        funnel_low=np.asarray(concat["funnel_low"], dtype=float),
        Zsig=np.asarray(concat["Zsig"], dtype=float),
        holding=np.asarray(concat["holding"], dtype=bool),
        buy_times=buy_times,
        sell_times=sell_times,
        trade_log=trade_log,
        mu_step=np.asarray(concat["mu_step"], dtype=float),
        mu_hat=np.asarray(concat["mu_hat"], dtype=float),
        var_step=np.asarray(concat["var_step"], dtype=float),
        sigma_hat=np.asarray(concat["sigma_hat"], dtype=float),
        N=prices.size - 1,
        k=last_fold["k"],
        h=h,
        c_buy=c_buy,
        c_sell=c_sell,
        mode=mode,
        source=source,
        walk_forward_summary=summary,
        selected_params={"k": last_fold["k"], "delta": last_fold["delta"]},
    )


def run_walk_forward_demo(
    full_lp=None,
    c_buy=0.0,
    c_sell=0.0,
    drift_process_var=1e-7,
    garch_alpha=0.06,
    garch_beta=0.90,
    seed=7,
    source="Offline regime path",
    train_size=252,
    test_size=63,
    k_grid=None,
    delta_grid=None,
    stitch_tests=True,
):
    """
    Demonstrate Phase 6 walk-forward logic on a regime-changing offline path.

    Real-data Phase 6 uses the same protocol after loading adjusted close prices.
    This demo is intentionally local so the app works without internet access.
    """
    if full_lp is None:
        full_lp = generate_regime_price_path(seed=seed)
    else:
        full_lp = np.asarray(full_lp, dtype=float)
    if k_grid is None:
        k_grid = [0.8, 1.1, 1.4, 1.8, 2.2]
    if delta_grid is None:
        delta_grid = [5, 8, 13, 21]
    folds = []
    fold_results = []
    oos_multiplier = 1.0
    last_data = None

    for train_start in range(0, len(full_lp) - train_size - test_size, test_size):
        train_end = train_start + train_size
        test_end = train_end + test_size
        train_lp = full_lp[train_start:train_end + 1]
        test_lp = full_lp[train_end:test_end + 1]
        chosen = select_walk_forward_params(
            train_lp,
            k_grid=k_grid,
            delta_grid=delta_grid,
            c_buy=c_buy,
            c_sell=c_sell,
            drift_process_var=drift_process_var,
            garch_alpha=garch_alpha,
            garch_beta=garch_beta,
        )
        if chosen is None:
            continue

        test_data = run_strategy_on_log_prices(
            test_lp,
            k=chosen["k"],
            delta=chosen["delta"],
            sigma_seed=chosen["sigma_seed"],
            c_buy=c_buy,
            c_sell=c_sell,
            drift_process_var=drift_process_var,
            garch_alpha=garch_alpha,
            garch_beta=garch_beta,
            mode="Walk-forward OOS fold",
        )
        metrics = summarize_simulation(test_data)
        fold_multiplier = test_data["portfolio_w"][-1] / 1000.0
        oos_multiplier *= fold_multiplier
        folds.append({
            "train": (train_start, train_end),
            "test": (train_end, test_end),
            "k": chosen["k"],
            "delta": chosen["delta"],
            "mtm_return": metrics["mtm_return"],
            "benchmark_gap": metrics["benchmark_gap"],
            "max_drawdown": metrics["max_mtm_drawdown"],
            "sharpe": annualized_sharpe(test_data["portfolio_w"]),
            "calmar": calmar_ratio(test_data["portfolio_w"]),
        })
        fold_results.append(test_data)
        last_data = test_data

    if last_data is None:
        raise RuntimeError("Walk-forward demo did not produce any folds.")

    if stitch_tests:
        return stitch_walk_forward_tests(
            fold_results,
            folds,
            source=source,
            mode="Walk-forward stitched OOS",
        )

    summary = {
        "folds": folds,
        "oos_return": oos_multiplier - 1.0,
        "avg_fold_return": float(np.mean([f["mtm_return"] for f in folds])),
        "avg_benchmark_gap": float(np.mean([f["benchmark_gap"] for f in folds])),
        "avg_sharpe": float(np.nanmean([f["sharpe"] for f in folds])),
        "avg_calmar": float(np.nanmean([f["calmar"] for f in folds])),
    }
    last_fold = folds[-1]
    last_data["mode"] = "Walk-forward OOS fold"
    last_data["source"] = source
    last_data["walk_forward_summary"] = summary
    last_data["selected_params"] = {
        "k": last_fold["k"],
        "delta": last_fold["delta"],
    }
    return last_data


def run_online_adaptive_strategy(
    lp,
    c_buy=0.0,
    c_sell=0.0,
    drift_process_var=1e-7,
    garch_alpha=0.06,
    garch_beta=0.90,
    update_window=5,
    learning_rate=0.25,
    k_init=1.2,
    delta_init=3,
    max_funnel_lookback=60,
    trailing_stop=0.04,
    trend_entry_z=0.35,
    k_grid=None,
    delta_grid=None,
    optimize_params=None,
    source="Real daily prices",
):
    """
    Trade one continuous account while updating selected parameters from recent data.

    At the start of each new week, the previous week is used to propose
    parameters. The live parameters are then blended toward that proposal.
    """
    lp = np.asarray(lp, dtype=float)
    if lp.size < update_window * 2 + 2:
        raise ValueError("Online adaptation needs at least two update windows.")
    if k_grid is None:
        k_grid = [0.8, 1.1, 1.4, 1.8, 2.2]
    if delta_grid is None:
        delta_grid = [1, 2, 3]
    if optimize_params is None:
        optimize_params = ["k", "delta"]

    prices = np.exp(lp)
    log_returns = np.diff(lp)
    n_steps = lp.size - 1
    N = lp.size
    sigma_seed = estimate_sigma_seed(lp[:update_window + 1])
    mu_step, mu_hat = kalman_drift_estimates(
        log_returns,
        init_mu=0.0,
        obs_var=sigma_seed * sigma_seed,
        process_var=drift_process_var,
    )
    var_step, sigma_hat = garch_volatility_estimates(
        log_returns,
        mu_prior=mu_step,
        init_var=sigma_seed * sigma_seed,
        alpha=garch_alpha,
        beta=garch_beta,
    )
    cumulative_drift = np.zeros(N)
    cumulative_var = np.zeros(N)
    cumulative_drift[1:] = np.cumsum(mu_step[1:])
    cumulative_var[1:] = np.cumsum(var_step[1:])

    realized_w  = np.full(N, 1000.0)
    portfolio_w = np.full(N, 1000.0)
    buy_hold_w  = np.full(N, 1000.0)
    Zsig        = np.full(N, np.nan)
    funnel_mid  = np.full(N, np.nan)
    funnel_up   = np.full(N, np.nan)
    funnel_low  = np.full(N, np.nan)
    holding     = np.zeros(N, dtype=bool)
    k_path      = np.full(N, float(k_init))
    delta_path  = np.full(N, int(delta_init), dtype=int)
    buy_times   = []
    sell_times  = []
    trade_log   = []
    parameter_log = []

    h = profit_threshold(c_buy, c_sell)
    wealth = 1000.0
    in_pos = True
    buy_t = 1
    buy_lp = lp[1]
    buy_price = prices[1]
    peak_lp = buy_lp
    buy_times.append(1)
    bh_buy_t = 1
    bh_buy_price = prices[bh_buy_t]

    current_k = float(k_init)
    current_delta = int(delta_init)
    current_q = float(drift_process_var)
    current_L = int(max_funnel_lookback) if max_funnel_lookback is not None else None
    current_a = trailing_stop
    current_z = trend_entry_z
    current_rho = float(learning_rate)

    for t in range(1, N):
        # Update parameters at the start of each new week using only past data.
        if t > update_window and (t - 1) % update_window == 0:
            train_start = max(0, t - 1 - update_window)
            train_lp = lp[train_start:t]
            if set(optimize_params) == {"k", "delta"}:
                chosen = select_walk_forward_params(
                    train_lp,
                    k_grid=k_grid,
                    delta_grid=delta_grid,
                    c_buy=c_buy,
                    c_sell=c_sell,
                    drift_process_var=current_q,
                    garch_alpha=garch_alpha,
                    garch_beta=garch_beta,
                )
                if chosen is not None:
                    chosen = {
                        "k": chosen["k"],
                        "delta": chosen["delta"],
                        "q": current_q,
                        "L": current_L,
                        "a": current_a,
                        "z_trend": current_z,
                        "rho": current_rho,
                    }
            else:
                chosen = select_online_learning_params(
                    train_lp,
                    optimize_params=optimize_params,
                    current_params={
                        "k": current_k,
                        "delta": current_delta,
                        "q": current_q,
                        "L": current_L,
                        "a": current_a,
                        "z_trend": current_z,
                        "rho": current_rho,
                    },
                    c_buy=c_buy,
                    c_sell=c_sell,
                    garch_alpha=garch_alpha,
                    garch_beta=garch_beta,
                )
            if chosen is not None:
                old_k = current_k
                old_delta = current_delta
                old_q = current_q
                old_L = current_L
                old_a = current_a
                old_z = current_z
                old_rho = current_rho
                target_rho = chosen["rho"] if "rho" in optimize_params else current_rho
                current_rho = float(target_rho)
                current_k = (
                    (1.0 - current_rho) * current_k
                    + current_rho * chosen["k"]
                )
                blended_delta = (
                    (1.0 - current_rho) * current_delta
                    + current_rho * chosen["delta"]
                )
                current_delta = int(np.clip(
                    round(blended_delta),
                    min(PLAY_LEARN_GRIDS["delta"]),
                    max(PLAY_LEARN_GRIDS["delta"]),
                ))
                if "q" in optimize_params:
                    current_q = (
                        (1.0 - current_rho) * current_q
                        + current_rho * chosen["q"]
                    )
                if "L" in optimize_params:
                    blended_L = (
                        (1.0 - current_rho) * current_L
                        + current_rho * chosen["L"]
                    )
                    current_L = int(np.clip(
                        round(blended_L),
                        min(PLAY_LEARN_GRIDS["L"]),
                        max(PLAY_LEARN_GRIDS["L"]),
                    ))
                if "a" in optimize_params:
                    current_a = (
                        (1.0 - current_rho) * current_a
                        + current_rho * chosen["a"]
                    )
                if "z_trend" in optimize_params:
                    current_z = (
                        (1.0 - current_rho) * current_z
                        + current_rho * chosen["z_trend"]
                    )
                parameter_log.append({
                    "t": t,
                    "train": (train_start, t - 1),
                    "suggested_k": chosen["k"],
                    "suggested_delta": chosen["delta"],
                    "suggested_q": chosen["q"],
                    "suggested_L": chosen["L"],
                    "suggested_a": chosen["a"],
                    "suggested_z_trend": chosen["z_trend"],
                    "suggested_rho": chosen["rho"],
                    "old_k": old_k,
                    "old_delta": old_delta,
                    "old_q": old_q,
                    "old_L": old_L,
                    "old_a": old_a,
                    "old_z_trend": old_z,
                    "old_rho": old_rho,
                    "new_k": current_k,
                    "new_delta": current_delta,
                    "new_q": current_q,
                    "new_L": current_L,
                    "new_a": current_a,
                    "new_z_trend": current_z,
                    "new_rho": current_rho,
                })

        k_path[t] = current_k
        delta_path[t] = current_delta
        M = lp[t] - lp[t - current_delta] if t >= current_delta else np.nan
        el = t - buy_t if buy_t is not None else 0
        ref_t = buy_t
        ref_lp = buy_lp
        if current_L is not None and el > current_L:
            ref_t = t - current_L
            ref_lp = lp[ref_t]
        if el >= 0:
            drift_since_ref = cumulative_drift[t] - cumulative_drift[ref_t]
            var_since_ref = cumulative_var[t] - cumulative_var[ref_t]
            center = ref_lp + drift_since_ref
            width = current_k * np.sqrt(var_since_ref)
            funnel_mid[t] = np.exp(center)
            funnel_up[t] = np.exp(center + width)
            funnel_low[t] = np.exp(center - width)

        if in_pos:
            peak_lp = max(peak_lp, lp[t])
            if el >= 1:
                denom = np.sqrt(var_since_ref)
                Z = (lp[t] - ref_lp - drift_since_ref) / denom if denom > 0 else np.nan
                Zsig[t] = Z
                log_ret = lp[t] - buy_lp
                trail_drawdown = peak_lp - lp[t]
                funnel_exit = (
                    el >= current_delta + 1
                    and not np.isnan(Z)
                    and Z > current_k
                    and not np.isnan(M)
                    and M <= 0
                    and log_ret > h
                )
                trailing_exit = (
                    current_a is not None
                    and log_ret > h
                    and trail_drawdown >= current_a
                )
                if funnel_exit or trailing_exit:
                    net_mult = net_liquidation_multiplier(
                        prices[t], buy_price, c_buy=c_buy, c_sell=c_sell
                    )
                    trade_log.append({
                        "buy_t": buy_t,
                        "sell_t": t,
                        "gross_log_return": log_ret,
                        "net_multiplier": net_mult,
                        "exit_reason": "trailing" if trailing_exit else "funnel",
                    })
                    wealth *= net_mult
                    in_pos = False
                    sell_times.append(t)
                    buy_lp = lp[t]
                    buy_t = t
                    buy_price = prices[t]
        else:
            if buy_t is not None and t >= current_delta + 1 and not np.isnan(M):
                if el >= 1:
                    denom = np.sqrt(var_since_ref)
                    Zd = (lp[t] - buy_lp - drift_since_ref) / denom if denom > 0 else np.nan
                    bargain_entry = not np.isnan(Zd) and Zd < -current_k and M >= 0
                    trend_entry = (
                        trend_entry_z is not None
                        and current_z is not None
                        and not np.isnan(Zd)
                        and Zd > current_z
                        and M > 0
                    )
                    if bargain_entry or trend_entry:
                        in_pos = True
                        buy_t = t
                        buy_lp = lp[t]
                        buy_price = prices[t]
                        peak_lp = buy_lp
                        buy_times.append(t)

        holding[t] = in_pos
        realized_w[t] = wealth
        if in_pos and buy_t is not None:
            portfolio_w[t] = wealth * net_liquidation_multiplier(
                prices[t], buy_price, c_buy=c_buy, c_sell=c_sell
            )
        else:
            portfolio_w[t] = wealth
        if t >= bh_buy_t:
            buy_hold_w[t] = 1000.0 * net_liquidation_multiplier(
                prices[t], bh_buy_price, c_buy=c_buy, c_sell=c_sell
            )

    return dict(
        prices=prices, lp=lp,
        realized_w=realized_w, portfolio_w=portfolio_w, buy_hold_w=buy_hold_w,
        funnel_mid=funnel_mid, funnel_up=funnel_up, funnel_low=funnel_low,
        Zsig=Zsig, holding=holding,
        buy_times=set(buy_times), sell_times=set(sell_times),
        trade_log=trade_log,
        mu_step=mu_step, mu_hat=mu_hat,
        var_step=var_step, sigma_hat=sigma_hat,
        k_path=k_path, delta_path=delta_path,
        N=n_steps, k=current_k, h=h, c_buy=c_buy, c_sell=c_sell,
        mode="Real data online adaptive",
        source=source,
        parameter_log=parameter_log,
        selected_params={
            "k": current_k,
            "delta": current_delta,
            "q": current_q,
            "L": current_L,
            "a": current_a,
            "z_trend": current_z,
            "rho": current_rho,
        },
        online_summary={
            "updates": len(parameter_log),
            "learning_rate": current_rho,
            "update_window": update_window,
            "max_funnel_lookback": current_L,
            "trailing_stop": current_a,
            "trend_entry_z": current_z,
            "optimized": list(optimize_params),
        },
    )


def run_real_price_walk_forward(
    symbol=DEFAULT_REAL_SYMBOL,
    lookback_days=None,
    interval="1d",
    c_buy=0.0,
    c_sell=0.0,
    drift_process_var=1e-7,
    garch_alpha=0.06,
    garch_beta=0.90,
    learning_rate=0.25,
    k_init=1.2,
    delta_init=3,
    trailing_stop=0.04,
    trend_entry_z=0.35,
):
    """Fetch real market bars and run one continuous online adaptive test."""
    full_lp = fetch_yahoo_log_prices(
        symbol=symbol,
        days=lookback_days,
        interval=interval,
    )
    data = run_online_adaptive_strategy(
        full_lp,
        c_buy=c_buy,
        c_sell=c_sell,
        drift_process_var=drift_process_var,
        garch_alpha=garch_alpha,
        garch_beta=garch_beta,
        learning_rate=learning_rate,
        k_init=k_init,
        delta_init=delta_init,
        trailing_stop=trailing_stop,
        trend_entry_z=trend_entry_z,
        source=f"Yahoo adjusted {symbol.upper()}, {real_period_label(lookback_days, interval)}",
    )
    data["symbol"] = symbol.upper()
    data["lookback_days"] = lookback_days
    data["interval"] = interval
    data["periods_per_year"] = periods_per_year_for_interval(interval)
    data["mode"] = f"Real {symbol.upper()} online adaptive"
    return data


# ── Matplotlib canvas ─────────────────────────────────────────────────────────
class TradingCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 6.6), facecolor=BG, tight_layout=False)
        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.97,
                                  bottom=0.07, hspace=0.45)
        gs = gridspec.GridSpec(3, 1, figure=self.fig,
                                height_ratios=[2.8, 1.25, 1.8])
        self.ax_p = self.fig.add_subplot(gs[0])
        self.ax_z = self.fig.add_subplot(gs[1])
        self.ax_w = self.fig.add_subplot(gs[2])
        for ax in (self.ax_p, self.ax_z, self.ax_w):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_C, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_C)
            ax.grid(color=GRID_C, linewidth=0.5)

        self.ax_p.set_ylabel("Price ($)",   color=TEXT_C, fontsize=8)
        self.ax_z.set_ylabel("Z statistic", color=TEXT_C, fontsize=8)
        self.ax_w.set_ylabel("Wealth ($)",  color=TEXT_C, fontsize=8)
        self.ax_w.set_xlabel("Time step",   color=TEXT_C, fontsize=8)

        self._init_artists()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _init_artists(self):
        ax_p, ax_z, ax_w = self.ax_p, self.ax_z, self.ax_w

        self.ln_price,  = ax_p.plot([], [], color=BLUE,  lw=1.4, zorder=2)
        self.ln_fu,     = ax_p.plot([], [], color=F_UPPER, lw=0.9, ls="--",
                                     alpha=0.72, zorder=1)
        self.ln_fl,     = ax_p.plot([], [], color=F_LOWER, lw=0.9, ls="--",
                                     alpha=0.72, zorder=1)
        self.ln_fm,     = ax_p.plot([], [], color=TEXT_C, lw=0.75, ls=":",
                                     alpha=0.45, zorder=1)
        self.sc_buy     = ax_p.scatter([], [], marker="^", color=GREEN, s=70, zorder=5)
        self.sc_sell    = ax_p.scatter([], [], marker="v", color=RED,   s=70, zorder=5)
        self.vl_p       = ax_p.axvline(0, color="white", lw=0.8, alpha=0.4, ls="--")

        self.ln_z,  = ax_z.plot([], [], color=TEAL,  lw=1.1, zorder=2)
        self.ln_kp, = ax_z.plot([], [], color=RED,   lw=0.8, ls="--", alpha=0.7)
        self.ln_kn, = ax_z.plot([], [], color=GREEN, lw=0.8, ls="--", alpha=0.7)
        self.vl_z   = ax_z.axvline(0, color="white", lw=0.8, alpha=0.4, ls="--")

        self.ln_rw, = ax_w.plot([], [], color=AMBER,  lw=2.0,
                                 drawstyle="steps-post", zorder=4)
        self.ln_pw, = ax_w.plot([], [], color=PURPLE, lw=1.0,
                                 ls="--", zorder=3)
        self.ln_bh, = ax_w.plot([], [], color=TEAL,   lw=1.0,
                                 ls=":", zorder=2)
        self.vl_w   = ax_w.axvline(0, color="white", lw=0.8, alpha=0.4, ls="--")

    def render_frame(self, data, t, autoscale=True):
        if data is None:
            return
        end    = t + 1
        xs     = np.arange(end)
        prices = data["prices"]
        fu     = data["funnel_up"]
        fl     = data["funnel_low"]
        fm     = data["funnel_mid"]
        rw     = data["realized_w"]
        pw     = data["portfolio_w"]
        bh     = data["buy_hold_w"]
        Zsig   = data["Zsig"]
        k_path = data.get("k_path")
        if k_path is None:
            k_path = np.full_like(prices, data["k"], dtype=float)

        # ── price panel ──
        self.ln_price.set_data(xs, prices[:end])
        self.ln_fu.set_data(xs, fu[:end])
        self.ln_fl.set_data(xs, fl[:end])
        self.ln_fm.set_data(xs, fm[:end])
        bxs = [b for b in data["buy_times"]  if b < end]
        sxs = [s for s in data["sell_times"] if s < end]
        self.sc_buy.set_offsets(
            np.c_[bxs, prices[bxs]] if bxs else np.empty((0, 2)))
        self.sc_sell.set_offsets(
            np.c_[sxs, prices[sxs]] if sxs else np.empty((0, 2)))
        self.vl_p.set_xdata([t, t])
        finite_funnels = np.concatenate([
            fu[:end][np.isfinite(fu[:end])],
            fl[:end][np.isfinite(fl[:end])],
            fm[:end][np.isfinite(fm[:end])],
        ])
        py = np.concatenate([prices[:end], finite_funnels])
        if autoscale:
            self.ax_p.set_xlim(0, max(end, 10))
            self.ax_p.set_ylim(py.min() * 0.995, py.max() * 1.005)

        # ── Z panel ──
        zv = Zsig[:end].copy()
        self.ln_z.set_data(xs, zv)
        self.ln_kp.set_data(xs, k_path[:end])
        self.ln_kn.set_data(xs, -k_path[:end])
        self.vl_z.set_xdata([t, t])
        valid = zv[~np.isnan(zv)]
        k_now = k_path[:end]
        zlo = min(valid.min() if len(valid) else -1, -np.nanmax(k_now)) - 0.3
        zhi = max(valid.max() if len(valid) else  1,  np.nanmax(k_now)) + 0.3
        if autoscale:
            self.ax_z.set_xlim(0, max(end, 10))
            self.ax_z.set_ylim(zlo, zhi)

        # ── wealth panel ──
        self.ln_rw.set_data(xs, rw[:end])
        self.ln_pw.set_data(xs, pw[:end])
        self.ln_bh.set_data(xs, bh[:end])
        self.vl_w.set_xdata([t, t])
        wv  = np.concatenate([rw[:end], pw[:end], bh[:end]])
        wlo = wv.min(); whi = wv.max()
        wsp = max(whi - wlo, 10)
        if autoscale:
            self.ax_w.set_xlim(0, max(end, 10))
            self.ax_w.set_ylim(wlo - wsp * 0.05, whi + wsp * 0.08)

        self.draw_idle()


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    SPEEDS = [1, 2, 3, 5, 10, 20]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Algorithm — Interactive Phase 6")
        self.resize(1280, 760)
        self._apply_dark_palette()

        self.sim_data  = None
        self.fixed_sim_data = None
        self.synthetic_lp = None
        self.current_t = 0
        self.playing   = False
        self.speed_idx = 2

        self.timer = QTimer()
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._tick)

        # ── central widget ────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ── left: chart ───────────────────────────────────────────────────────
        chart_panel = QWidget()
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(4)
        self.canvas = TradingCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(
            f"QToolBar {{ background: {PANEL_BG}; border: 0; }}"
            f"QToolButton {{ color: {TEXT_C}; background: transparent; }}"
            "QToolButton:hover { background: #2e3148; }"
        )
        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(self.canvas, stretch=1)
        root.addWidget(chart_panel, stretch=1)

        # ── right: controls ───────────────────────────────────────────────────
        ctrl_panel = QWidget()
        ctrl_panel.setMinimumWidth(620)
        ctrl_panel.setMaximumWidth(760)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(4)
        root.addWidget(ctrl_panel)

        # Parameters group
        param_box = QGroupBox("Parameters")
        param_box.setStyleSheet(self._group_style())
        param_grid = QGridLayout(param_box)
        param_grid.setHorizontalSpacing(10)
        param_grid.setVerticalSpacing(2)

        self.sliders = {}
        defs = [
            ("sigma  (volatility)", "sigma", 5,  40,  15, 1000),
            ("true mu  (market drift)", "mu",  -20,  30,   5, 10000),
            ("drift adapt  (Q x1e-8)", "drift_q", 0, 100, 10, 1),
            ("k  (threshold)",      "k",     5,  25,  12, 10),
            ("delta  (window)",     "delta", 1,  20,   8, 1),
            ("L  (funnel lookback)", "lookback_L", 20, 160, 60, 1),
            ("trail a",             "trail_a", 0, 12,   4, 100),
            ("z trend",             "z_trend", 0, 100, 35, 100),
            ("rho  (learn rate)",   "rho",   5,  80,  25, 100),
            ("mom c  (gen)",        "mom_c", 0,  8,   2, 100),
            ("cost/side",           "cost",  0,  50,   0, 10000),
        ]
        for idx, (lbl, key, mn, mx, val, scale) in enumerate(defs):
            block_row = (idx // 3) * 2
            block_col = (idx % 3) * 3
            label = QLabel(lbl)
            label.setStyleSheet(f"color:{TEXT_C};font-size:10px;")
            val_lbl = QLabel(self._fmt(key, val / scale))
            val_lbl.setStyleSheet(
                "color:white;font-size:10px;font-weight:bold;min-width:46px;")
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            sl = QSlider(Qt.Horizontal)
            sl.setMinimum(mn); sl.setMaximum(mx); sl.setValue(val)
            sl.setStyleSheet(self._slider_style())
            sl.valueChanged.connect(
                lambda v, k=key, s=scale, vl=val_lbl:
                vl.setText(self._fmt(k, v / s)))
            self.sliders[key] = (sl, scale)
            param_grid.addWidget(label,   block_row,     block_col, 1, 3)
            param_grid.addWidget(sl,      block_row + 1, block_col, 1, 2)
            param_grid.addWidget(val_lbl, block_row + 1, block_col + 2)
        param_grid.setColumnStretch(1, 1)
        param_grid.setColumnStretch(4, 1)
        param_grid.setColumnStretch(7, 1)

        ctrl_layout.addWidget(param_box)

        # Playback group
        play_box = QGroupBox("Playback")
        play_box.setStyleSheet(self._group_style())
        play_layout = QVBoxLayout(play_box)
        play_layout.setSpacing(3)

        # Speed slider
        sp_row = QHBoxLayout()
        sp_lbl = QLabel("Speed")
        sp_lbl.setStyleSheet(f"color:{TEXT_C};font-size:12px;")
        self.speed_val_lbl = QLabel(f"{self.SPEEDS[self.speed_idx]}x")
        self.speed_val_lbl.setStyleSheet(
            "color:white;font-size:12px;font-weight:bold;min-width:30px;")
        self.speed_val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.speed_sl = QSlider(Qt.Horizontal)
        self.speed_sl.setMinimum(0)
        self.speed_sl.setMaximum(len(self.SPEEDS) - 1)
        self.speed_sl.setValue(self.speed_idx)
        self.speed_sl.setStyleSheet(self._slider_style())
        self.speed_sl.valueChanged.connect(self._on_speed_change)
        sp_row.addWidget(sp_lbl)
        sp_row.addWidget(self.speed_sl)
        sp_row.addWidget(self.speed_val_lbl)
        play_layout.addLayout(sp_row)

        # Scrubber
        sc_row = QHBoxLayout()
        sc_lbl = QLabel("t =")
        sc_lbl.setStyleSheet(f"color:{TEXT_C};font-size:12px;")
        self.scrub_lbl = QLabel("0 / 0")
        self.scrub_lbl.setStyleSheet(
            "color:white;font-size:12px;font-weight:bold;min-width:54px;")
        self.scrub_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setMinimum(0)
        self.scrubber.setMaximum(500)
        self.scrubber.setValue(0)
        self.scrubber.setEnabled(False)
        self.scrubber.setStyleSheet(self._slider_style())
        self.scrubber.sliderPressed.connect(self._on_scrub_press)
        self.scrubber.valueChanged.connect(self._on_scrub)
        sc_row.addWidget(sc_lbl)
        sc_row.addWidget(self.scrubber)
        sc_row.addWidget(self.scrub_lbl)
        play_layout.addLayout(sc_row)

        # Buttons
        btn_grid = QGridLayout()
        btn_grid.setSpacing(5)
        self.btn_gen   = self._btn("Simulate",      self._generate)
        self.btn_learn = self._btn("Play-Learn",    self._play_learn, enabled=False)
        self.btn_fixed = self._btn("Fixed Path",    self._fixed_path, enabled=False)
        self.btn_ext   = self._btn("Extensions",    self._extensions, enabled=False)
        self.btn_wf    = self._btn("Walk-forward",  self._walk_forward)
        self.btn_real  = self._btn("Real Data",     self._real_data)
        self.btn_play  = self._btn("▶  Play",       self._toggle_play, enabled=False)
        self.btn_reset = self._btn("↺  Reset",      self._reset,       enabled=False)
        self.btn_autoscale = self._btn("Auto-scale ON", self._toggle_autoscale)
        self.btn_autoscale.setCheckable(True)
        self.btn_autoscale.setChecked(True)
        btn_grid.addWidget(self.btn_gen,   0, 0)
        btn_grid.addWidget(self.btn_learn, 0, 1)
        btn_grid.addWidget(self.btn_fixed, 1, 0)
        btn_grid.addWidget(self.btn_ext,   1, 1)
        btn_grid.addWidget(self.btn_wf,    2, 0)
        btn_grid.addWidget(self.btn_real,  2, 1)
        btn_grid.addWidget(self.btn_play,  3, 0)
        btn_grid.addWidget(self.btn_reset, 3, 1)
        btn_grid.addWidget(self.btn_autoscale, 4, 0, 1, 2)
        play_layout.addLayout(btn_grid)
        ctrl_layout.addWidget(play_box)

        # Legend
        legend_box = QGroupBox("Legend")
        legend_box.setStyleSheet(self._group_style())
        leg_layout = QGridLayout(legend_box)
        leg_layout.setHorizontalSpacing(8)
        leg_layout.setVerticalSpacing(1)
        legend_items = [
            (BLUE,   "Price path"),
            (F_UPPER, "Upper funnel"),
            (F_LOWER, "Lower funnel"),
            (GREEN,  "Buy signal"),
            (RED,    "Sell signal"),
            (TEAL,   "Z statistic / buy-hold"),
            (AMBER,  "Realized wealth"),
            (PURPLE, "Mark-to-market value"),
        ]
        for idx, (color, text) in enumerate(legend_items):
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color};font-size:14px;")
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{TEXT_C};font-size:10px;")
            row.addWidget(dot); row.addWidget(lbl); row.addStretch()
            leg_layout.addLayout(row, idx % 4, idx // 4)
        ctrl_layout.addWidget(legend_box)

        # Live stats group
        stats_box = QGroupBox("Live stats")
        stats_box.setStyleSheet(self._group_style())
        stats_layout = QGridLayout(stats_box)
        stats_layout.setHorizontalSpacing(8)
        stats_layout.setVerticalSpacing(2)
        stats_layout.setContentsMargins(6, 10, 6, 6)

        self.stat_labels = {}
        stat_defs = [
            ("mode",     "Mode"),
            ("source",   "Source"),
            ("t",        "t"),
            ("price",    "Price"),
            ("pos",      "Position"),
            ("regime",   "Regime"),
            ("z",        "Z"),
            ("mu_hat",   "mu hat"),
            ("sigma_hat", "sigma hat"),
            ("wealth",   "Realized W"),
            ("mtm",      "MtM W"),
            ("bh",       "Buy-hold W"),
            ("ret",      "Realized"),
            ("mtm_ret",  "MtM ret"),
            ("bh_ret",   "BH ret"),
            ("gap",      "MtM vs BH"),
            ("dd",       "Max DD"),
            ("trades",   "Trades"),
            ("exposure", "Exposure"),
            ("sharpe",   "Sharpe"),
            ("calmar",   "Calmar"),
            ("wf_ret",   "Learn/OOS"),
            ("h",        "h"),
        ]
        for idx, (key, caption) in enumerate(stat_defs):
            row = idx // 2
            col = (idx % 2) * 2
            cap_lbl = QLabel(caption)
            cap_lbl.setStyleSheet(f"color:{TEXT_C};font-size:10px;")
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet(
                "color:white;font-size:11px;font-weight:bold;")
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setMinimumWidth(68)
            stats_layout.addWidget(cap_lbl, row, col)
            stats_layout.addWidget(val_lbl, row, col + 1)
            self.stat_labels[key] = val_lbl
        stats_layout.setColumnStretch(1, 1)
        stats_layout.setColumnStretch(3, 1)

        ctrl_layout.addWidget(stats_box)
        ctrl_layout.addStretch()

        # Auto-generate on launch
        self._generate()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _fmt(self, key, v):
        if key == "sigma": return f"{v:.3f}"
        if key == "mu":    return f"{v:.4f}"
        if key == "drift_q": return f"{v:.0f}"
        if key == "k":     return f"{v:.1f}"
        if key == "delta": return f"{int(v)}"
        if key == "lookback_L": return f"{int(v)}"
        if key == "trail_a": return f"{v:.2f}"
        if key == "z_trend": return f"{v:.2f}"
        if key == "rho": return f"{v:.2f}"
        if key == "mom_c": return f"{v:.2f}"
        if key == "cost":  return f"{v * 10000:.0f} bp"
        return str(v)

    def _btn(self, text, slot, enabled=True):
        b = QPushButton(text)
        b.setEnabled(enabled)
        b.setStyleSheet(self._btn_style())
        b.clicked.connect(slot)
        return b

    def _toggle_autoscale(self):
        enabled = self.btn_autoscale.isChecked()
        self.btn_autoscale.setText("Auto-scale ON" if enabled else "Auto-scale OFF")
        if enabled and self.sim_data is not None:
            self._set_frame(self.current_t)

    def _tick(self):
        if self.sim_data is None:
            return
        nxt = min(self.current_t + self.SPEEDS[self.speed_idx],
                  self.sim_data["N"])
        self._set_frame(nxt)
        if nxt >= self.sim_data["N"]:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("▶  Play")

    def _generate(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶  Play")
        p = {k: sl.value() / scale for k, (sl, scale) in self.sliders.items()}
        data = run_simulation(
            mu=p["mu"], sigma=p["sigma"],
            k=p["k"], delta=int(p["delta"]),
            c_buy=p["cost"], c_sell=p["cost"],
            clip_noise=False,
            drift_process_var=p["drift_q"] * 1e-8,
            max_funnel_lookback=int(p["lookback_L"]),
            trailing_stop=p["trail_a"],
            trend_entry_z=p["z_trend"],
        )
        self.sim_data = data
        self.fixed_sim_data = data
        self.synthetic_lp = data["lp"].copy()
        self.scrubber.setMaximum(data["N"])
        self.scrubber.setEnabled(True)
        self.btn_learn.setEnabled(True)
        self.btn_fixed.setEnabled(True)
        self.btn_ext.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._set_frame(0)

    def _fixed_path(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶  Play")
        if self.fixed_sim_data is None:
            QMessageBox.information(
                self,
                "No fixed simulation",
                "Click Simulate first to create a fixed run.",
            )
            return
        self.sim_data = self.fixed_sim_data
        self.scrubber.setMaximum(self.sim_data["N"])
        self.scrubber.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._set_frame(0)

    def _extensions(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶  Play")
        if self.synthetic_lp is None:
            QMessageBox.information(
                self,
                "No simulated path",
                "Click Simulate first so Extensions can reuse the same path.",
            )
            return
        extension = self._choose_extension()
        if extension is None:
            return
        p = {k: sl.value() / scale for k, (sl, scale) in self.sliders.items()}
        if extension == "generalized_momentum":
            sigma_seed = estimate_sigma_seed(self.synthetic_lp)
            data = run_strategy_on_log_prices(
                self.synthetic_lp,
                k=p["k"],
                delta=int(p["delta"]),
                sigma_seed=sigma_seed,
                c_buy=p["cost"],
                c_sell=p["cost"],
                drift_process_var=p["drift_q"] * 1e-8,
                max_funnel_lookback=int(p["lookback_L"]),
                trailing_stop=p["trail_a"],
                trend_entry_z=p["z_trend"],
                generalized_momentum_c=p["mom_c"],
                mode="Synthetic generalized momentum",
                extra={
                    "source": "Same generated path; generalized momentum exit",
                    "selected_params": {
                        "k": p["k"],
                        "delta": int(p["delta"]),
                        "L": int(p["lookback_L"]),
                        "mom_c": p["mom_c"],
                    },
                },
            )
        elif extension == "regime_hmm":
            sigma_seed = estimate_sigma_seed(self.synthetic_lp)
            data = run_strategy_on_log_prices(
                self.synthetic_lp,
                k=p["k"],
                delta=int(p["delta"]),
                sigma_seed=sigma_seed,
                c_buy=p["cost"],
                c_sell=p["cost"],
                drift_process_var=p["drift_q"] * 1e-8,
                max_funnel_lookback=int(p["lookback_L"]),
                trailing_stop=p["trail_a"],
                trend_entry_z=p["z_trend"],
                use_hmm_regime=True,
                hmm_states=3,
                mode="Synthetic regime HMM",
                extra={
                    "source": "Same generated path; 3-state HMM funnel",
                    "selected_params": {
                        "k": p["k"],
                        "delta": int(p["delta"]),
                        "L": int(p["lookback_L"]),
                        "states": 3,
                    },
                },
            )
        elif extension == "randomized_stopping":
            sigma_seed = estimate_sigma_seed(self.synthetic_lp)
            data = run_strategy_on_log_prices(
                self.synthetic_lp,
                k=p["k"],
                delta=int(p["delta"]),
                sigma_seed=sigma_seed,
                c_buy=p["cost"],
                c_sell=p["cost"],
                drift_process_var=p["drift_q"] * 1e-8,
                max_funnel_lookback=int(p["lookback_L"]),
                trailing_stop=p["trail_a"],
                trend_entry_z=p["z_trend"],
                randomized_stopping=True,
                mode="Synthetic randomized stopping",
                extra={
                    "source": "Same generated path; stochastic eligible exits",
                    "selected_params": {
                        "k": p["k"],
                        "delta": int(p["delta"]),
                        "L": int(p["lookback_L"]),
                        "random_slope": DEFAULT_RANDOM_STOP_SLOPE,
                        "random_seed": DEFAULT_RANDOM_STOP_SEED,
                    },
                },
            )
        else:
            QMessageBox.information(
                self,
                "Extension unavailable",
                "This extension is not implemented yet.",
            )
            return
        self.sim_data = data
        self.scrubber.setMaximum(data["N"])
        self.scrubber.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._set_frame(0)

    def _choose_extension(self):
        items = ["Generalized Momentum", "Regime HMM", "Randomized Stopping"]
        choice, ok = QInputDialog.getItem(
            self,
            "Choose extension",
            "Run an extension on the current simulated path:",
            items,
            0,
            False,
        )
        if not ok:
            return None
        if choice == "Generalized Momentum":
            return "generalized_momentum"
        if choice == "Regime HMM":
            return "regime_hmm"
        if choice == "Randomized Stopping":
            return "randomized_stopping"
        return None

    def _play_learn(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶  Play")
        if self.synthetic_lp is None:
            QMessageBox.information(
                self,
                "No simulated path",
                "Click Simulate first so Play-Learn can reuse the same path.",
            )
            return
        optimize_params = self._choose_play_learn_params(
            title="Choose Play-Learn parameters",
            prompt="Choose up to 4 parameters to optimize every 5 bars.",
        )
        if optimize_params is None:
            return
        p = {k: sl.value() / scale for k, (sl, scale) in self.sliders.items()}
        try:
            data = run_online_adaptive_strategy(
                self.synthetic_lp,
                c_buy=p["cost"],
                c_sell=p["cost"],
                drift_process_var=p["drift_q"] * 1e-8,
                learning_rate=p["rho"],
                k_init=p["k"],
                delta_init=int(p["delta"]),
                max_funnel_lookback=int(p["lookback_L"]),
                trailing_stop=p["trail_a"],
                trend_entry_z=p["z_trend"],
                optimize_params=optimize_params,
                source="Same synthetic path as Simulate",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Play-Learn failed", str(exc))
            return
        data["mode"] = "Synthetic play-learn online adaptive"
        data["source"] = (
            "Same generated path; optimizing "
            + ", ".join(optimize_params)
            + " every 5 bars"
        )
        self.sim_data = data
        self.scrubber.setMaximum(data["N"])
        self.scrubber.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._set_frame(0)

    def _choose_play_learn_params(
        self,
        title="Choose online-learning parameters",
        prompt="Choose up to 4 parameters to optimize every 5 bars.",
    ):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        info = QLabel(prompt)
        info.setStyleSheet(f"color:{TEXT_C};font-size:11px;")
        layout.addWidget(info)

        choices = [
            ("k", "k  (funnel threshold)"),
            ("delta", "delta  (momentum window)"),
            ("q", "q  (drift adaptation)"),
            ("L", "L  (bounded funnel lookback)"),
            ("a", "a  (trailing stop)"),
            ("z_trend", "z_trend  (trend re-entry)"),
            ("rho", "rho  (learning rate)"),
        ]
        widget = QListWidget()
        widget.setStyleSheet(
            f"QListWidget {{ background:{PANEL_BG}; color:{TEXT_C}; }}"
            "QListWidget::item { padding: 3px; }"
        )
        for key, text in choices:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key in ("k", "delta") else Qt.Unchecked)
            widget.addItem(item)
        layout.addWidget(widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return None
        selected = [
            widget.item(i).data(Qt.UserRole)
            for i in range(widget.count())
            if widget.item(i).checkState() == Qt.Checked
        ]
        if not selected:
            QMessageBox.warning(self, "No parameters selected", "Choose at least one parameter.")
            return None
        if len(selected) > 4:
            QMessageBox.warning(
                self,
                "Too many parameters",
                "Please choose at most 4 parameters for online learning.",
            )
            return None
        return selected

    def _walk_forward(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶  Play")
        p = {k: sl.value() / scale for k, (sl, scale) in self.sliders.items()}
        data = run_walk_forward_demo(
            c_buy=p["cost"],
            c_sell=p["cost"],
            drift_process_var=p["drift_q"] * 1e-8,
        )
        self.sim_data = data
        self.scrubber.setMaximum(data["N"])
        self.scrubber.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._set_frame(0)

    def _real_data(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶  Play")
        symbol = self._choose_real_symbol()
        if symbol is None:
            return
        period = self._choose_real_period()
        if period is None:
            return
        lookback_days, interval = period
        mode = self._choose_real_mode()
        if mode is None:
            return
        optimize_params = None
        if mode == "learn":
            optimize_params = self._choose_play_learn_params(
                title="Choose real-data online parameters",
                prompt="Choose up to 4 parameters to optimize every 5 bars on the fetched Yahoo path.",
            )
            if optimize_params is None:
                return
        p = {k: sl.value() / scale for k, (sl, scale) in self.sliders.items()}
        try:
            full_lp = fetch_yahoo_log_prices(
                symbol=symbol,
                days=lookback_days,
                interval=interval,
            )
            source = f"Yahoo adjusted {symbol.upper()}, {real_period_label(lookback_days, interval)}"
            if mode in ("fixed", "generalized_momentum", "regime_hmm", "randomized_stopping"):
                sigma_seed = estimate_sigma_seed(full_lp)
                extra_params = {
                    "k": p["k"],
                    "delta": int(p["delta"]),
                    "q": p["drift_q"] * 1e-8,
                    "L": int(p["lookback_L"]),
                    "a": p["trail_a"],
                    "z_trend": p["z_trend"],
                }
                generalized_c = None
                use_hmm = False
                randomized = False
                mode_label = f"Real {symbol.upper()} fixed sliders"
                if mode == "generalized_momentum":
                    generalized_c = p["mom_c"]
                    extra_params["mom_c"] = p["mom_c"]
                    mode_label = f"Real {symbol.upper()} generalized momentum"
                elif mode == "regime_hmm":
                    use_hmm = True
                    extra_params["states"] = 3
                    mode_label = f"Real {symbol.upper()} regime HMM"
                elif mode == "randomized_stopping":
                    randomized = True
                    extra_params["random_slope"] = DEFAULT_RANDOM_STOP_SLOPE
                    extra_params["random_seed"] = DEFAULT_RANDOM_STOP_SEED
                    mode_label = f"Real {symbol.upper()} randomized stopping"
                data = run_strategy_on_log_prices(
                    full_lp,
                    k=p["k"],
                    delta=int(p["delta"]),
                    sigma_seed=sigma_seed,
                    c_buy=p["cost"],
                    c_sell=p["cost"],
                    drift_process_var=p["drift_q"] * 1e-8,
                    max_funnel_lookback=int(p["lookback_L"]),
                    trailing_stop=p["trail_a"],
                    trend_entry_z=p["z_trend"],
                    generalized_momentum_c=generalized_c,
                    use_hmm_regime=use_hmm,
                    hmm_states=3,
                    randomized_stopping=randomized,
                    mode=mode_label,
                    extra={
                        "source": source,
                        "selected_params": extra_params,
                    },
                )
            else:
                data = run_online_adaptive_strategy(
                    full_lp,
                    c_buy=p["cost"],
                    c_sell=p["cost"],
                    drift_process_var=p["drift_q"] * 1e-8,
                    learning_rate=p["rho"],
                    k_init=p["k"],
                    delta_init=int(p["delta"]),
                    max_funnel_lookback=int(p["lookback_L"]),
                    trailing_stop=p["trail_a"],
                    trend_entry_z=p["z_trend"],
                    optimize_params=optimize_params,
                    source=source,
                )
                data["mode"] = f"Real {symbol.upper()} online adaptive"
                data["source"] = (
                    source
                    + "; optimizing "
                    + ", ".join(optimize_params)
                    + " every 5 bars"
                )
            data["symbol"] = symbol.upper()
            data["lookback_days"] = lookback_days
            data["interval"] = interval
            data["periods_per_year"] = periods_per_year_for_interval(interval)
        except Exception as exc:
            QMessageBox.warning(self, "Real data fetch failed", str(exc))
            return
        self.sim_data = data
        self.scrubber.setMaximum(data["N"])
        self.scrubber.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._set_frame(0)

    def _choose_real_mode(self):
        items = [
            "Fixed slider values",
            "Online optimization every 5 bars",
            "Generalized Momentum extension",
            "Regime HMM extension",
            "Randomized Stopping extension",
        ]
        choice, ok = QInputDialog.getItem(
            self,
            "Choose real-data mode",
            "How should the algorithm use the fetched Yahoo path?",
            items,
            0,
            False,
        )
        if not ok:
            return None
        if choice.startswith("Online"):
            return "learn"
        if choice.startswith("Generalized"):
            return "generalized_momentum"
        if choice.startswith("Regime"):
            return "regime_hmm"
        if choice.startswith("Randomized"):
            return "randomized_stopping"
        return "fixed"

    def _choose_real_symbol(self):
        items = [f"{symbol} - {name}" for symbol, name in REAL_DATA_CHOICES]
        default_idx = next(
            (idx for idx, (symbol, _) in enumerate(REAL_DATA_CHOICES)
             if symbol == DEFAULT_REAL_SYMBOL),
            0,
        )
        choice, ok = QInputDialog.getItem(
            self,
            "Choose real market data",
            "Select a ticker to fetch from Yahoo Finance:",
            items,
            default_idx,
            False,
        )
        if not ok:
            return None

        symbol = choice.split(" - ", 1)[0].strip().upper()
        if symbol != "CUSTOM TICKER...":
            return symbol

        custom, ok = QInputDialog.getText(
            self,
            "Custom ticker",
            "Enter a Yahoo Finance ticker, e.g. AMD, JPM, BTC-USD:",
        )
        if not ok:
            return None
        custom = custom.strip().upper()
        if not custom:
            QMessageBox.warning(self, "Missing ticker", "Please enter a ticker symbol.")
            return None
        return custom

    def _choose_real_period(self):
        items = [f"{label} - {interval} bars" for label, _, interval in REAL_PERIOD_CHOICES]
        choice, ok = QInputDialog.getItem(
            self,
            "Choose history length",
            "How much history should be fetched, and at what bar interval?",
            items,
            0,
            False,
        )
        if not ok:
            return None
        selected_label = choice.split(" - ", 1)[0]
        for label, days, interval in REAL_PERIOD_CHOICES:
            if label == selected_label:
                return days, interval
        _, days, interval = REAL_PERIOD_CHOICES[0]
        return days, interval

    def _toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("▶  Play")
        else:
            if self.current_t >= (self.sim_data["N"] if self.sim_data else 0):
                self._set_frame(0)
            self.playing = True
            self.btn_play.setText("⏸  Pause")
            self.timer.start()

    def _reset(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶  Play")
        self._set_frame(0)

    def _set_frame(self, t):
        self.current_t = t
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(t)
        self.scrubber.blockSignals(False)
        N = self.sim_data["N"] if self.sim_data else 0
        self.scrub_lbl.setText(f"{t} / {N}")
        self.canvas.render_frame(
            self.sim_data,
            t,
            autoscale=self.btn_autoscale.isChecked(),
        )

        if self.sim_data is not None:
            d   = self.sim_data
            rw  = d["realized_w"][t]
            pw  = d["portfolio_w"][t]
            bh  = d["buy_hold_w"][t]
            metrics = summarize_simulation(d, t)
            ret = metrics["realized_return"] * 100
            zn  = d["Zsig"][t]
            zs  = f"{zn:+.2f}" if not np.isnan(zn) else "n/a"
            pos = "Holding" if d["holding"][t] else "Cash"
            wf_summary = d.get("walk_forward_summary")
            periods_per_year = d.get("periods_per_year", 252)
            sharpe = annualized_sharpe(
                d["portfolio_w"][:t + 1],
                periods_per_year=periods_per_year,
            )
            calmar = calmar_ratio(
                d["portfolio_w"][:t + 1],
                periods_per_year=periods_per_year,
            )

            self.stat_labels["mode"].setText(d.get("mode", "Simulation"))
            self.stat_labels["source"].setText(d.get("source", "Generated path"))
            self.stat_labels["t"].setText(str(t))
            self.stat_labels["price"].setText(f"${d['prices'][t]:.2f}")
            self.stat_labels["wealth"].setText(f"${rw:.2f}")
            self.stat_labels["mtm"].setText(f"${pw:.2f}")
            self.stat_labels["bh"].setText(f"${bh:.2f}")
            self.stat_labels["ret"].setText(f"{ret:+.2f}%")
            self.stat_labels["mtm_ret"].setText(
                f"{metrics['mtm_return'] * 100:+.2f}%")
            self.stat_labels["bh_ret"].setText(
                f"{metrics['buy_hold_return'] * 100:+.2f}%")
            self.stat_labels["gap"].setText(
                f"{metrics['benchmark_gap'] * 100:+.2f}%")
            self.stat_labels["dd"].setText(
                f"{metrics['max_mtm_drawdown'] * 100:.2f}%")
            self.stat_labels["trades"].setText(str(metrics["completed_trades"]))
            self.stat_labels["exposure"].setText(
                f"{metrics['exposure'] * 100:.0f}%")
            self.stat_labels["sharpe"].setText(
                f"{sharpe:.2f}" if np.isfinite(sharpe) else "n/a")
            self.stat_labels["calmar"].setText(
                f"{calmar:.2f}" if np.isfinite(calmar) else "n/a")
            if wf_summary:
                self.stat_labels["wf_ret"].setText(
                    f"{wf_summary['oos_return'] * 100:+.2f}%")
            elif d.get("online_summary"):
                self.stat_labels["wf_ret"].setText(
                    f"{d['online_summary']['updates']} upd")
            else:
                self.stat_labels["wf_ret"].setText("n/a")
            self.stat_labels["z"].setText(zs)
            self.stat_labels["mu_hat"].setText(f"{d['mu_hat'][t]:+.5f}")
            self.stat_labels["sigma_hat"].setText(f"{d['sigma_hat'][t]:.4f}")
            self.stat_labels["h"].setText(f"{d['h']:.5f}")
            self.stat_labels["pos"].setText(pos)
            if "regime_state" in d:
                labels = d.get("regime_labels", [])
                state = int(d["regime_state"][t])
                name = labels[state] if state < len(labels) else f"S{state + 1}"
                probs = d.get("regime_probs")
                conf = probs[t, state] if probs is not None else np.nan
                self.stat_labels["regime"].setText(
                    f"{state + 1}:{name} {conf:.0%}" if np.isfinite(conf)
                    else f"{state + 1}:{name}"
                )
            else:
                self.stat_labels["regime"].setText("n/a")

            self.stat_labels["wealth"].setStyleSheet(
                f"color:{'white' if rw >= 1000 else RED};"
                "font-size:12px;font-weight:bold;")
            self.stat_labels["ret"].setStyleSheet(
                f"color:{GREEN if ret >= 0 else RED};"
                "font-size:12px;font-weight:bold;")
            self.stat_labels["mtm_ret"].setStyleSheet(
                f"color:{GREEN if metrics['mtm_return'] >= 0 else RED};"
                "font-size:12px;font-weight:bold;")
            self.stat_labels["gap"].setStyleSheet(
                f"color:{GREEN if metrics['benchmark_gap'] >= 0 else RED};"
                "font-size:12px;font-weight:bold;")
            self.stat_labels["dd"].setStyleSheet(
                f"color:{RED if metrics['max_mtm_drawdown'] < 0 else TEXT_C};"
                "font-size:12px;font-weight:bold;")
            self.stat_labels["pos"].setStyleSheet(
                f"color:{GREEN if d['holding'][t] else TEXT_C};"
                "font-size:12px;font-weight:bold;")
            regime_state = d.get("regime_state")
            if regime_state is not None:
                state = int(regime_state[t])
                regime_color = [GREEN, AMBER, RED][min(state, 2)]
            else:
                regime_color = TEXT_C
            self.stat_labels["regime"].setStyleSheet(
                f"color:{regime_color};font-size:12px;font-weight:bold;")

    def _on_scrub_press(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶  Play")

    def _on_scrub(self, v):
        if self.sim_data is None:
            return
        self._set_frame(v)

    def _on_speed_change(self, idx):
        self.speed_idx = idx
        self.speed_val_lbl.setText(f"{self.SPEEDS[idx]}x")

    # ── styles ────────────────────────────────────────────────────────────────
    def _apply_dark_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window,          QColor("#1a1d27"))
        pal.setColor(QPalette.WindowText,      QColor(TEXT_C))
        pal.setColor(QPalette.Base,            QColor("#0f1117"))
        pal.setColor(QPalette.AlternateBase,   QColor("#1a1d27"))
        pal.setColor(QPalette.Text,            QColor(TEXT_C))
        pal.setColor(QPalette.Button,          QColor("#262736"))
        pal.setColor(QPalette.ButtonText,      QColor(TEXT_C))
        pal.setColor(QPalette.Highlight,       QColor("#378ADD"))
        pal.setColor(QPalette.HighlightedText, QColor("white"))
        QApplication.setPalette(pal)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

    def _group_style(self):
        return f"""
            QGroupBox {{
                color: {TEXT_C}; font-size: 12px; font-weight: bold;
                border: 0.5px solid #333344; border-radius: 6px;
                margin-top: 8px; padding: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
                color: {TEXT_C};
            }}
        """

    def _slider_style(self):
        return """
            QSlider::groove:horizontal {
                height: 4px; background: #2a2d3e; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #378ADD; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #185FA5; border-radius: 2px;
            }
        """

    def _btn_style(self):
        return f"""
            QPushButton {{
                background: #262736; color: {TEXT_C};
                border: 0.5px solid #333344; border-radius: 5px;
                padding: 6px 10px; font-size: 12px;
            }}
            QPushButton:hover {{ background: #2e3148; }}
            QPushButton:pressed {{ background: #185FA5; color: white; }}
            QPushButton:disabled {{ color: #555566; border-color: #222233; }}
        """


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica Neue", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    
