"""
Interactive Trading Algorithm — Phase 3 Funnel Visualization Version
-----------------------------------------------------
A PyQt5 app with:
  - Sliders for sigma, mu, k, delta, transaction cost, speed
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

Phase 3 includes:
  1. Explicit profit threshold h = log((1+c_buy)/(1-c_sell)).
  2. Sell condition requires log_return > h.
  3. Transaction-cost-adjusted realized wealth update.
  4. Buy-and-hold benchmark added.
  5. Holding state is recorded after same-step decisions.
  6. Gaussian shocks are no longer clipped by default.
  7. Completed trade records, drawdown, exposure, and benchmark-gap metrics.
  8. Visible center, upper, and lower funnel paths on the price chart.

Dependencies:
    pip install numpy matplotlib PyQt5
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QGridLayout, QGroupBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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


# ── Simulation ────────────────────────────────────────────────────────────────
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
):
    """
    Simulate the Phase 2 rule.

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
    prices = np.exp(lp)

    N = n_steps + 1
    cumulative_drift = np.zeros(N)
    cumulative_var = np.zeros(N)
    cumulative_drift[1:] = np.cumsum(np.full(n_steps, mu))
    cumulative_var[1:] = np.cumsum(np.full(n_steps, sigma * sigma))

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
    buy_times.append(1)

    # Buy-and-hold benchmark starts from the same initial buy time.
    bh_buy_t = 1
    bh_buy_price = prices[bh_buy_t]

    for t in range(1, N):
        M = lp[t] - lp[t - delta] if t >= delta else np.nan
        el = t - buy_t if buy_t is not None else 0
        if el >= 0:
            drift_since_ref = cumulative_drift[t] - cumulative_drift[buy_t]
            var_since_ref = cumulative_var[t] - cumulative_var[buy_t]
            center = buy_lp + drift_since_ref
            width = k * np.sqrt(var_since_ref)
            funnel_mid[t] = np.exp(center)
            funnel_up[t] = np.exp(center + width)
            funnel_low[t] = np.exp(center - width)

        if in_pos:
            if el >= 1:
                denom = np.sqrt(var_since_ref)
                Z = (lp[t] - buy_lp - drift_since_ref) / denom if denom > 0 else np.nan
                Zsig[t] = Z
                log_ret = lp[t] - buy_lp

                # Phase 1 corrected sell rule:
                # Z_t > k, momentum weakened, and realized log-return exceeds h.
                if (
                    el >= delta + 1
                    and not np.isnan(Z)
                    and Z > k
                    and not np.isnan(M)
                    and M <= 0
                    and log_ret > h
                ):
                    net_mult = net_liquidation_multiplier(
                        prices[t], buy_price, c_buy=c_buy, c_sell=c_sell
                    )
                    trade_log.append({
                        "buy_t": buy_t,
                        "sell_t": t,
                        "gross_log_return": log_ret,
                        "net_multiplier": net_mult,
                    })
                    wealth *= net_mult
                    in_pos = False
                    sell_times.append(t)
                    # Last sell time becomes reference point for the symmetric re-entry rule.
                    buy_lp = lp[t]
                    buy_t  = t
                    buy_price = prices[t]
        else:
            # Symmetric re-entry rule: buy after a lower-funnel deviation and momentum recovery.
            if buy_t is not None and t >= delta + 1 and not np.isnan(M):
                if el >= 1:
                    denom = np.sqrt(var_since_ref)
                    Zd = (lp[t] - buy_lp - drift_since_ref) / denom if denom > 0 else np.nan
                    if not np.isnan(Zd) and Zd < -k and M >= 0:
                        in_pos = True
                        buy_t  = t
                        buy_lp = lp[t]
                        buy_price = prices[t]
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

    return dict(
        prices=prices, lp=lp,
        realized_w=realized_w, portfolio_w=portfolio_w, buy_hold_w=buy_hold_w,
        funnel_mid=funnel_mid, funnel_up=funnel_up, funnel_low=funnel_low,
        Zsig=Zsig, holding=holding,
        buy_times=set(buy_times), sell_times=set(sell_times),
        trade_log=trade_log,
        N=n_steps, k=k, h=h, c_buy=c_buy, c_sell=c_sell,
        clip_noise=clip_noise,
    )


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

    def render_frame(self, data, t):
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
        k      = data["k"]

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
        self.ax_p.set_xlim(0, max(end, 10))
        self.ax_p.set_ylim(py.min() * 0.995, py.max() * 1.005)

        # ── Z panel ──
        zv = Zsig[:end].copy()
        self.ln_z.set_data(xs, zv)
        self.ln_kp.set_data(xs, np.full(end,  k))
        self.ln_kn.set_data(xs, np.full(end, -k))
        self.vl_z.set_xdata([t, t])
        valid = zv[~np.isnan(zv)]
        zlo = min(valid.min() if len(valid) else -k, -k) - 0.3
        zhi = max(valid.max() if len(valid) else  k,  k) + 0.3
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
        self.ax_w.set_xlim(0, max(end, 10))
        self.ax_w.set_ylim(wlo - wsp * 0.05, whi + wsp * 0.08)

        self.draw_idle()


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    SPEEDS = [1, 2, 3, 5, 10, 20]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Algorithm — Interactive Phase 3")
        self.resize(1280, 760)
        self._apply_dark_palette()

        self.sim_data  = None
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
        self.canvas = TradingCanvas()
        root.addWidget(self.canvas, stretch=1)

        # ── right: controls ───────────────────────────────────────────────────
        ctrl_panel = QWidget()
        ctrl_panel.setMinimumWidth(360)
        ctrl_panel.setMaximumWidth(440)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(6)
        root.addWidget(ctrl_panel)

        # Parameters group
        param_box = QGroupBox("Parameters")
        param_box.setStyleSheet(self._group_style())
        param_grid = QGridLayout(param_box)
        param_grid.setHorizontalSpacing(6)
        param_grid.setVerticalSpacing(3)

        self.sliders = {}
        defs = [
            ("sigma  (volatility)", "sigma", 5,  40,  15, 1000),
            ("mu  (drift/step)",    "mu",  -20,  30,   5, 10000),
            ("k  (threshold)",      "k",     5,  25,  12, 10),
            ("delta  (window)",     "delta", 1,  20,   8, 1),
            ("cost/side",           "cost",  0,  50,   0, 10000),
        ]
        for row, (lbl, key, mn, mx, val, scale) in enumerate(defs):
            label = QLabel(lbl)
            label.setStyleSheet(f"color:{TEXT_C};font-size:11px;")
            val_lbl = QLabel(self._fmt(key, val / scale))
            val_lbl.setStyleSheet(
                "color:white;font-size:11px;font-weight:bold;min-width:56px;")
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            sl = QSlider(Qt.Horizontal)
            sl.setMinimum(mn); sl.setMaximum(mx); sl.setValue(val)
            sl.setStyleSheet(self._slider_style())
            sl.valueChanged.connect(
                lambda v, k=key, s=scale, vl=val_lbl:
                vl.setText(self._fmt(k, v / s)))
            self.sliders[key] = (sl, scale)
            param_grid.addWidget(label,   row * 2,     0, 1, 2)
            param_grid.addWidget(sl,      row * 2 + 1, 0)
            param_grid.addWidget(val_lbl, row * 2 + 1, 1)

        ctrl_layout.addWidget(param_box)

        # Playback group
        play_box = QGroupBox("Playback")
        play_box.setStyleSheet(self._group_style())
        play_layout = QVBoxLayout(play_box)
        play_layout.setSpacing(5)

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
        self.btn_gen   = self._btn("Generate",  self._generate)
        self.btn_play  = self._btn("▶  Play",   self._toggle_play, enabled=False)
        self.btn_reset = self._btn("↺  Reset",  self._reset,       enabled=False)
        btn_grid.addWidget(self.btn_gen,   0, 0, 1, 2)
        btn_grid.addWidget(self.btn_play,  1, 0)
        btn_grid.addWidget(self.btn_reset, 1, 1)
        play_layout.addLayout(btn_grid)
        ctrl_layout.addWidget(play_box)

        # Legend
        legend_box = QGroupBox("Legend")
        legend_box.setStyleSheet(self._group_style())
        leg_layout = QVBoxLayout(legend_box)
        leg_layout.setSpacing(2)
        for color, text in [
            (BLUE,   "Price path"),
            (F_UPPER, "Upper funnel"),
            (F_LOWER, "Lower funnel"),
            (GREEN,  "Buy signal"),
            (RED,    "Sell signal"),
            (TEAL,   "Z statistic / buy-hold"),
            (AMBER,  "Realized wealth"),
            (PURPLE, "Mark-to-market value"),
        ]:
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color};font-size:14px;")
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{TEXT_C};font-size:10px;")
            row.addWidget(dot); row.addWidget(lbl); row.addStretch()
            leg_layout.addLayout(row)
        ctrl_layout.addWidget(legend_box)

        # Live stats group
        stats_box = QGroupBox("Live stats")
        stats_box.setStyleSheet(self._group_style())
        stats_layout = QGridLayout(stats_box)
        stats_layout.setHorizontalSpacing(10)
        stats_layout.setVerticalSpacing(4)
        stats_layout.setContentsMargins(6, 10, 6, 6)

        self.stat_labels = {}
        stat_defs = [
            ("t",        "t"),
            ("price",    "Price"),
            ("pos",      "Position"),
            ("z",        "Z"),
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
        if key == "k":     return f"{v:.1f}"
        if key == "delta": return f"{int(v)}"
        if key == "cost":  return f"{v * 10000:.0f} bp"
        return str(v)

    def _btn(self, text, slot, enabled=True):
        b = QPushButton(text)
        b.setEnabled(enabled)
        b.setStyleSheet(self._btn_style())
        b.clicked.connect(slot)
        return b

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
        )
        self.sim_data = data
        self.scrubber.setMaximum(data["N"])
        self.scrubber.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._set_frame(0)

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
        self.canvas.render_frame(self.sim_data, t)

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
            self.stat_labels["z"].setText(zs)
            self.stat_labels["h"].setText(f"{d['h']:.5f}")
            self.stat_labels["pos"].setText(pos)

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
    
