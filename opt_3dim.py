"""
Layer 2.5 three-dimensional optimization study.

This sidecar script does not modify main.py or Paper.tex. It loads the
strategy backend from main.py, runs seven experiments that add one extra
parameter to the baseline (k, delta) walk-forward search, and writes a
standalone LaTeX report that can be rendered to PDF.

Examples:
    python3 opt_3dim.py --symbol SPY --period "5 years"
    python3 opt_3dim.py --source synthetic --fast
    latexmk -pdf Reports/opt_3dim_report.tex
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
MAIN_PATH = ROOT / "main.py"
REPORT_DIR = ROOT / "Reports"
DEFAULT_REPORT = REPORT_DIR / "opt_3dim_report.tex"

LAMBDA_GAP = 0.15

PERIODS = {
    "5 years": (5 * 365.25, "1d"),
    "2 years": (2 * 365.25, "1d"),
    "1 year": (365.25, "1d"),
    "1 month": (31, "1h"),
    "1 week": (7, "15m"),
}

DEFAULTS = {
    "q": 1e-7,
    "alpha": 0.06,
    "beta": 0.90,
    "L": 60,
    "a": 0.04,
    "z_trend": 0.35,
    "rho": 0.25,
}

GRIDS = {
    "q": [0.0, 5e-8, 1e-7, 2e-7, 5e-7],
    "alpha": [0.03, 0.06, 0.10, 0.15],
    "beta": [0.80, 0.88, 0.90, 0.94],
    "L": [30, 60, 90, 120],
    "a": [0.02, 0.04, 0.06, 0.08],
    "z_trend": [0.00, 0.25, 0.35, 0.50, 0.75],
    "rho": [0.10, 0.25, 0.40, 0.60],
}

EXPERIMENT_ORDER = ["baseline", "q", "alpha", "beta", "L", "a", "z_trend", "rho"]


def load_backend() -> dict[str, Any]:
    """
    Load the non-GUI strategy functions from main.py.

    main.py imports PyQt5 at module import time. This loader executes only the
    backend section before the Matplotlib/PyQt canvas classes so the study can
    run in command-line Python environments without PyQt installed.
    """
    src = MAIN_PATH.read_text()
    backend_src = src.split("# ── Matplotlib canvas", 1)[0]
    lines = backend_src.splitlines()
    filtered: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("from PyQt5"):
            if line.rstrip().endswith("("):
                i += 1
                while i < len(lines) and lines[i].strip() != ")":
                    i += 1
                i += 1
            else:
                i += 1
            continue
        if (
            line.startswith("import matplotlib")
            or line.startswith("matplotlib.")
            or line.startswith("from matplotlib")
        ):
            i += 1
            continue
        filtered.append(line)
        i += 1

    namespace: dict[str, Any] = {}
    exec("\n".join(filtered), namespace)
    return namespace


def experiment_defaults(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(DEFAULTS)
    if overrides:
        params.update(overrides)
    return params


def strategy_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "drift_process_var": params["q"],
        "garch_alpha": params["alpha"],
        "garch_beta": params["beta"],
        "max_funnel_lookback": params["L"],
        "trailing_stop": params["a"],
        "trend_entry_z": params["z_trend"],
    }


def score_metrics(metrics: dict[str, float], lambda_gap: float = LAMBDA_GAP) -> float:
    return (
        metrics["mtm_return"]
        + metrics["max_mtm_drawdown"]
        + lambda_gap * metrics["benchmark_gap"]
    )


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def average_holding_bars(trade_log: list[dict[str, Any]]) -> float | None:
    holds = [trade["sell_t"] - trade["buy_t"] for trade in trade_log]
    if not holds:
        return None
    return float(np.mean(holds))


def summarize_data(
    backend: dict[str, Any],
    data: dict[str, Any],
    periods_per_year: float,
) -> dict[str, Any]:
    metrics = backend["summarize_simulation"](data)
    return {
        "realized_return": metrics["realized_return"],
        "mtm_return": metrics["mtm_return"],
        "buy_hold_return": metrics["buy_hold_return"],
        "benchmark_gap": metrics["benchmark_gap"],
        "max_drawdown": metrics["max_mtm_drawdown"],
        "buy_hold_drawdown": metrics["max_buy_hold_drawdown"],
        "exposure": metrics["exposure"],
        "completed_trades": metrics["completed_trades"],
        "avg_holding_bars": average_holding_bars(data["trade_log"]),
        "sharpe": finite_float(
            backend["annualized_sharpe"](
                data["portfolio_w"],
                periods_per_year=periods_per_year,
            )
        ),
        "calmar": finite_float(
            backend["calmar_ratio"](
                data["portfolio_w"],
                periods_per_year=periods_per_year,
            )
        ),
    }


def select_fold_params(
    backend: dict[str, Any],
    train_lp: np.ndarray,
    k_grid: list[float],
    delta_grid: list[int],
    third_name: str | None,
    third_grid: list[Any],
    defaults: dict[str, Any],
    c_buy: float,
    c_sell: float,
) -> dict[str, Any] | None:
    sigma_seed = backend["estimate_sigma_seed"](train_lp)
    best: dict[str, Any] | None = None
    third_values = [None] if third_name is None else third_grid

    for k_value, delta_value, third_value in itertools.product(
        k_grid,
        delta_grid,
        third_values,
    ):
        if delta_value >= len(train_lp) - 2:
            continue
        params = experiment_defaults(defaults)
        if third_name is not None:
            params[third_name] = third_value

        try:
            data = backend["run_strategy_on_log_prices"](
                train_lp,
                k=k_value,
                delta=delta_value,
                sigma_seed=sigma_seed,
                c_buy=c_buy,
                c_sell=c_sell,
                mode="Training fold",
                **strategy_kwargs(params),
            )
        except Exception:
            continue

        metrics = backend["summarize_simulation"](data)
        score = score_metrics(metrics)
        if best is None or score > best["score"]:
            best = {
                "k": k_value,
                "delta": delta_value,
                "third": third_value,
                "third_name": third_name,
                "score": score,
                "sigma_seed": sigma_seed,
                "params": params,
            }
    return best


def run_walk_forward_experiment(
    backend: dict[str, Any],
    full_lp: np.ndarray,
    experiment: str,
    k_grid: list[float],
    delta_grid: list[int],
    train_size: int,
    test_size: int,
    c_buy: float,
    c_sell: float,
    periods_per_year: float,
) -> dict[str, Any]:
    third_name = None if experiment == "baseline" else experiment
    third_grid = [] if third_name is None else GRIDS[third_name]
    fold_results = []
    folds = []

    if len(full_lp) < train_size + test_size + 2:
        raise ValueError(
            "Dataset is too short for the requested train/test sizes. "
            "Use smaller --train-size/--test-size or a longer period."
        )

    for train_start in range(0, len(full_lp) - train_size - test_size, test_size):
        train_end = train_start + train_size
        test_end = train_end + test_size
        train_lp = full_lp[train_start : train_end + 1]
        test_lp = full_lp[train_end : test_end + 1]

        chosen = select_fold_params(
            backend,
            train_lp,
            k_grid,
            delta_grid,
            third_name,
            third_grid,
            DEFAULTS,
            c_buy,
            c_sell,
        )
        if chosen is None:
            continue

        data = backend["run_strategy_on_log_prices"](
            test_lp,
            k=chosen["k"],
            delta=chosen["delta"],
            sigma_seed=chosen["sigma_seed"],
            c_buy=c_buy,
            c_sell=c_sell,
            mode=f"{experiment} OOS fold",
            **strategy_kwargs(chosen["params"]),
        )
        metrics = backend["summarize_simulation"](data)
        folds.append(
            {
                "train": (train_start, train_end),
                "test": (train_end, test_end),
                "k": chosen["k"],
                "delta": chosen["delta"],
                "third": chosen["third"],
                "third_name": third_name,
                "mtm_return": metrics["mtm_return"],
                "benchmark_gap": metrics["benchmark_gap"],
                "max_drawdown": metrics["max_mtm_drawdown"],
                "sharpe": backend["annualized_sharpe"](
                    data["portfolio_w"],
                    periods_per_year=periods_per_year,
                ),
                "calmar": backend["calmar_ratio"](
                    data["portfolio_w"],
                    periods_per_year=periods_per_year,
                ),
            }
        )
        fold_results.append(data)

    if not fold_results:
        raise RuntimeError(f"{experiment} produced no valid folds.")

    stitched = backend["stitch_walk_forward_tests"](
        fold_results,
        folds,
        source=f"{experiment} 3D study",
        mode=f"{experiment} stitched OOS",
    )
    summary = summarize_data(backend, stitched, periods_per_year)
    summary["folds"] = folds
    summary["experiment"] = experiment
    summary["optimized_parameters"] = ["k", "delta"] + (
        [] if third_name is None else [third_name]
    )
    summary["final_selected"] = {
        "k": folds[-1]["k"],
        "delta": folds[-1]["delta"],
        **({third_name: folds[-1]["third"]} if third_name else {}),
    }
    summary["fold_count"] = len(folds)
    return summary


def run_rho_online_experiment(
    backend: dict[str, Any],
    full_lp: np.ndarray,
    k_grid: list[float],
    delta_grid: list[int],
    train_size: int,
    test_size: int,
    c_buy: float,
    c_sell: float,
    periods_per_year: float,
) -> dict[str, Any]:
    """
    Special rho experiment.

    rho affects smoothing in online adaptation, not a static single-fold
    strategy. For this experiment the three dimensions are (k_init,
    delta_init, rho), with q/alpha/beta/L/a/z_trend fixed at defaults.
    """
    folds = []
    fold_results = []
    third_grid = GRIDS["rho"]

    for train_start in range(0, len(full_lp) - train_size - test_size, test_size):
        train_end = train_start + train_size
        test_end = train_end + test_size
        train_lp = full_lp[train_start : train_end + 1]
        test_lp = full_lp[train_end : test_end + 1]
        best = None

        for k_init, delta_init, rho in itertools.product(k_grid, delta_grid, third_grid):
            if len(train_lp) < 12:
                continue
            try:
                train_data = backend["run_online_adaptive_strategy"](
                    train_lp,
                    c_buy=c_buy,
                    c_sell=c_sell,
                    learning_rate=rho,
                    k_init=k_init,
                    delta_init=delta_init,
                    source="rho training fold",
                    **strategy_kwargs(DEFAULTS),
                )
            except Exception:
                continue
            metrics = backend["summarize_simulation"](train_data)
            score = score_metrics(metrics)
            if best is None or score > best["score"]:
                best = {
                    "k": k_init,
                    "delta": delta_init,
                    "rho": rho,
                    "score": score,
                }

        if best is None:
            continue

        test_data = backend["run_online_adaptive_strategy"](
            test_lp,
            c_buy=c_buy,
            c_sell=c_sell,
            learning_rate=best["rho"],
            k_init=best["k"],
            delta_init=best["delta"],
            source="rho OOS fold",
            **strategy_kwargs(DEFAULTS),
        )
        metrics = backend["summarize_simulation"](test_data)
        folds.append(
            {
                "train": (train_start, train_end),
                "test": (train_end, test_end),
                "k": best["k"],
                "delta": best["delta"],
                "third": best["rho"],
                "third_name": "rho",
                "mtm_return": metrics["mtm_return"],
                "benchmark_gap": metrics["benchmark_gap"],
                "max_drawdown": metrics["max_mtm_drawdown"],
                "sharpe": backend["annualized_sharpe"](
                    test_data["portfolio_w"],
                    periods_per_year=periods_per_year,
                ),
                "calmar": backend["calmar_ratio"](
                    test_data["portfolio_w"],
                    periods_per_year=periods_per_year,
                ),
            }
        )
        fold_results.append(test_data)

    if not fold_results:
        raise RuntimeError("rho experiment produced no valid folds.")

    stitched = backend["stitch_walk_forward_tests"](
        fold_results,
        folds,
        source="rho 3D online study",
        mode="rho stitched OOS",
    )
    summary = summarize_data(backend, stitched, periods_per_year)
    summary["folds"] = folds
    summary["experiment"] = "rho"
    summary["optimized_parameters"] = ["k_init", "delta_init", "rho"]
    summary["final_selected"] = {
        "k_init": folds[-1]["k"],
        "delta_init": folds[-1]["delta"],
        "rho": folds[-1]["third"],
    }
    summary["fold_count"] = len(folds)
    return summary


def run_gui_continuous_experiment(
    backend: dict[str, Any],
    full_lp: np.ndarray,
    experiment: str,
    c_buy: float,
    c_sell: float,
    periods_per_year: float,
) -> dict[str, Any]:
    """
    Run the GUI-style continuous online account.

    This follows the Real Data button mechanics: one continuous account,
    k0=1.2, delta0=3, 5-bar updates, and no fold resets. For added-parameter
    rows, the added parameter is swept over its grid and the best full-sample
    setting is reported as exploratory sensitivity.
    """
    grid = [None] if experiment == "baseline" else GRIDS[experiment]
    best = None

    for value in grid:
        params = experiment_defaults()
        learning_rate = DEFAULTS["rho"]
        selected = {"k0": 1.2, "delta0": 3}

        if experiment == "rho":
            learning_rate = value
            selected["rho"] = value
        elif experiment != "baseline":
            params[experiment] = value
            selected[experiment] = value

        try:
            data = backend["run_online_adaptive_strategy"](
                full_lp,
                c_buy=c_buy,
                c_sell=c_sell,
                learning_rate=learning_rate,
                k_init=1.2,
                delta_init=3,
                source=f"GUI-style {experiment}",
                **strategy_kwargs(params),
            )
        except Exception:
            continue

        metrics = backend["summarize_simulation"](data)
        score = score_metrics(metrics)
        if best is None or score > best["score"]:
            summary = summarize_data(backend, data, periods_per_year)
            summary["experiment"] = experiment
            summary["optimized_parameters"] = (
                ["fixed GUI defaults"]
                if experiment == "baseline"
                else [experiment]
            )
            summary["final_selected"] = selected
            summary["score"] = score
            summary["updates"] = data.get("online_summary", {}).get("updates")
            summary["final_live"] = data.get("selected_params", {})
            best = summary

    if best is None:
        raise RuntimeError(f"GUI-style {experiment} produced no valid run.")
    return best


def load_dataset(
    backend: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.source == "synthetic":
        lp = backend["generate_regime_price_path"](n_steps=args.synthetic_steps, seed=args.seed)
        return lp, {
            "source": f"Synthetic regime path, seed {args.seed}",
            "symbol": "Synthetic",
            "lookback": f"{args.synthetic_steps} simulated bars",
            "interval": "1d",
            "periods_per_year": 252,
        }

    days, interval = PERIODS[args.period]
    if args.interval is not None:
        interval = args.interval
    lp = backend["fetch_yahoo_log_prices"](
        symbol=args.symbol,
        days=days,
        interval=interval,
    )
    return lp, {
        "source": f"Yahoo adjusted {args.symbol.upper()}",
        "symbol": args.symbol.upper(),
        "lookback": args.period,
        "interval": interval,
        "periods_per_year": backend["periods_per_year_for_interval"](interval),
    }


def format_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.2f}\\%"


def format_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "--"
    return f"{value:.{digits}f}"


def format_param_value(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if abs(value) < 1e-4 and value != 0:
            return f"{value:.1e}"
        return f"{value:.3g}"
    return str(value)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def selected_param_string(summary: dict[str, Any]) -> str:
    items = [
        f"{name}={format_param_value(value)}"
        for name, value in summary["final_selected"].items()
    ]
    return latex_escape(", ".join(items))


def result_rows(
    results: list[dict[str, Any]],
    include_realized: bool = False,
) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = next(r for r in results if r["experiment"] == "baseline")
    rows = []
    for result in results:
        delta_mtm = result["mtm_return"] - baseline["mtm_return"]
        cells = [
            latex_escape(result["experiment"]),
            latex_escape(", ".join(result["optimized_parameters"])),
        ]
        if include_realized:
            cells.append(format_pct(result["realized_return"]))
        cells.extend(
            [
                format_pct(result["mtm_return"]),
                format_pct(result["buy_hold_return"]),
                format_pct(result["benchmark_gap"]),
                format_pct(result["max_drawdown"]),
                format_num(result["sharpe"]),
                format_num(result["calmar"]),
                str(result["completed_trades"]),
                format_num(result["avg_holding_bars"]),
                format_pct(delta_mtm),
                selected_param_string(result),
            ]
        )
        rows.append(" & ".join(cells) + r" \\")

    best = max(results, key=lambda item: item["mtm_return"])
    best_gap = max(results, key=lambda item: item["benchmark_gap"])
    best_sharpe = max(
        results,
        key=lambda item: item["sharpe"] if item["sharpe"] is not None else -np.inf,
    )
    return rows, best, best_gap, best_sharpe


def make_report(
    path: Path,
    metadata: dict[str, Any],
    stitched_results: list[dict[str, Any]],
    gui_results: list[dict[str, Any]],
    args: argparse.Namespace,
    comparison_sections: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows, best, best_gap, best_sharpe = result_rows(stitched_results)
    gui_rows, gui_best, gui_best_gap, gui_best_sharpe = result_rows(
        gui_results,
        include_realized=True,
    )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.65in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{hyperref}}

\title{{Three-Dimensional Parameter Expansion Study}}
\author{{Generated by opt\_3dim.py}}
\date{{{generated_at}}}

\begin{{document}}
\maketitle

\section*{{Purpose}}
This side experiment asks which single additional parameter is most useful when
added to the current two-dimensional walk-forward search over $(k,\delta)$.
The full Layer 2 design vector is
\[
\theta=(k,\delta,q,\alpha,\beta,L,a,z_{{\mathrm{{trend}}}},\rho).
\]
The baseline optimizes only $(k,\delta)$. Each expansion optimizes $(k,\delta,x)$
for one additional parameter $x$, except for the $\rho$ experiment where
$(k_0,\delta_0,\rho)$ is optimized because $\rho$ affects online smoothing
rather than a static single-fold rule. The report contains two protocols:
stitched walk-forward out-of-sample folds and GUI-style continuous online runs.

\section*{{Dataset, Protocol, and Search Grids}}
\noindent
\begin{{minipage}}[t]{{0.47\textwidth}}
\subsection*{{Dataset and Protocol}}
\small
\begin{{tabular}}{{ll}}
\toprule
Data source & {latex_escape(metadata["source"])} \\
Symbol & {latex_escape(metadata["symbol"])} \\
Lookback & {latex_escape(metadata["lookback"])} \\
Bar interval & {latex_escape(metadata["interval"])} \\
Bars & {len(args.lp) - 1 if hasattr(args, "lp") else "n/a"} \\
Train size & {args.train_size} bars \\
Test size & {args.test_size} bars \\
Cost per side & {args.cost_bps:.1f} bps \\
Protocol & Stitched OOS folds \\
Objective & $R^m + D^m + {LAMBDA_GAP:.2f}G$ \\
\bottomrule
\end{{tabular}}
\end{{minipage}}
\hfill
\begin{{minipage}}[t]{{0.49\textwidth}}
\subsection*{{Search Grids}}
\small
\begin{{tabular}}{{ll}}
\toprule
Parameter & Grid \\
\midrule
$k$ & {latex_escape(str(args.k_grid))} \\
$\delta$ & {latex_escape(str(args.delta_grid))} \\
$q$ & {latex_escape(str(GRIDS["q"]))} \\
$\alpha$ & {latex_escape(str(GRIDS["alpha"]))} \\
$\beta$ & {latex_escape(str(GRIDS["beta"]))} \\
$L$ & {latex_escape(str(GRIDS["L"]))} \\
$a$ & {latex_escape(str(GRIDS["a"]))} \\
$z_{{\mathrm{{trend}}}}$ & {latex_escape(str(GRIDS["z_trend"]))} \\
$\rho$ & {latex_escape(str(GRIDS["rho"]))} \\
\bottomrule
\end{{tabular}}
\end{{minipage}}
\normalsize

\subsection*{{Fixed Initial / Design Parameters}}
\small
\begin{{tabular}}{{ll}}
\toprule
Quantity & Value used unless explicitly optimized \\
\midrule
Initial $k_0$ & 1.2 in GUI online mode; grid-selected in this stitched OOS report \\
Initial $\delta_0$ & 3 in GUI online mode; grid-selected in this stitched OOS report \\
$q$ & {DEFAULTS["q"]:.1e} \\
$\alpha$ & {DEFAULTS["alpha"]:.2f} \\
$\beta$ & {DEFAULTS["beta"]:.2f} \\
$L$ & {DEFAULTS["L"]} bars \\
$a$ & {DEFAULTS["a"]:.2f} log-price units \\
$z_{{\mathrm{{trend}}}}$ & {DEFAULTS["z_trend"]:.2f} \\
$\rho$ & {DEFAULTS["rho"]:.2f}; only active in the online $\rho$ experiment \\
\bottomrule
\end{{tabular}}
\normalsize

\paragraph{{Protocol note.}}
The stitched rows are formal out-of-sample folds. A training window is used to
choose parameters, the following test window is evaluated, and only test-window
wealth curves are stitched together. The GUI-style rows instead run one
continuous online-adaptive account over the full dataset, with
$(k_0,\delta_0)=(1.2,3)$ and 5-bar parameter updates. Therefore a GUI Real Data
run should be compared with the GUI-style table, not with the stitched baseline.

\section*{{Stitched Walk-Forward Results}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{p{{0.9in}}p{{1.1in}}rrrrrrrrrp{{1.25in}}}}
\toprule
Experiment & Optimized & MtM & BH & Gap & Max DD & Sharpe & Calmar & Trades & Avg hold & $\Delta$ MtM & Final selected \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\normalsize

\section*{{GUI-Style Continuous Online Results}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{p{{0.85in}}p{{0.95in}}rrrrrrrrrrp{{1.15in}}}}
\toprule
Experiment & Swept & Real. & MtM & BH & Gap & Max DD & Sharpe & Calmar & Trades & Avg hold & $\Delta$ MtM & Selected / fixed \\
\midrule
{chr(10).join(gui_rows)}
\bottomrule
\end{{tabular}}
\normalsize

\section*{{Reading the Tables}}
For the stitched table, MtM is the stitched out-of-sample mark-to-market return. BH is the buy-and-hold
return on the same stitched test windows. Gap is $W_T^m/W_T^{{bh}}-1$.
Max DD is the maximum mark-to-market drawdown. Avg hold is measured in bars.
For the GUI-style table, MtM is the full continuous online mark-to-market return
over the selected dataset. $\Delta$ MtM is measured relative to the baseline row
within the same table.

\section*{{High-Level Takeaways}}
\begin{{itemize}}
    \item Stitched best MtM return: {latex_escape(best["experiment"])} with {format_pct(best["mtm_return"])}.
    \item Stitched best benchmark gap: {latex_escape(best_gap["experiment"])} with {format_pct(best_gap["benchmark_gap"])}.
    \item Stitched best Sharpe ratio: {latex_escape(best_sharpe["experiment"])} with {format_num(best_sharpe["sharpe"])}.
    \item GUI-style best MtM return: {latex_escape(gui_best["experiment"])} with {format_pct(gui_best["mtm_return"])}.
    \item GUI-style best benchmark gap: {latex_escape(gui_best_gap["experiment"])} with {format_pct(gui_best_gap["benchmark_gap"])}.
    \item GUI-style best Sharpe ratio: {latex_escape(gui_best_sharpe["experiment"])} with {format_num(gui_best_sharpe["sharpe"])}.
\end{{itemize}}

{comparison_sections}

\section*{{Caveats}}
This report is exploratory. It changes one extra parameter at a time and does
not prove that a parameter is universally useful. Larger grids and more assets
increase the risk of overfitting, so results should be checked across multiple
symbols, periods, and bar intervals before being promoted into the main paper.
The GUI-style parameter sweeps are full-sample sensitivity checks, not strict
out-of-sample evidence.

\end{{document}}
"""
    path.write_text(tex)


def make_comparison_section(
    title: str,
    metadata: dict[str, Any],
    stitched_results: list[dict[str, Any]],
    gui_results: list[dict[str, Any]],
    train_size: int,
    test_size: int,
    bar_count: int,
) -> str:
    rows, best, best_gap, best_sharpe = result_rows(stitched_results)
    gui_rows, gui_best, gui_best_gap, gui_best_sharpe = result_rows(
        gui_results,
        include_realized=True,
    )
    return rf"""
\section*{{Comparison Dataset: {latex_escape(title)}}}
This section repeats the same experiment structure on {latex_escape(metadata["symbol"])}
using {latex_escape(metadata["lookback"])} of {latex_escape(metadata["interval"])} bars.
Because this sample is shorter than the SPY 5-year dataset, the stitched
walk-forward folds use a shorter training window of {train_size} bars and a
test window of {test_size} bars.

\begin{{tabular}}{{ll}}
\toprule
Data source & {latex_escape(metadata["source"])} \\
Symbol & {latex_escape(metadata["symbol"])} \\
Lookback & {latex_escape(metadata["lookback"])} \\
Bar interval & {latex_escape(metadata["interval"])} \\
Bars & {bar_count} \\
Train size & {train_size} bars \\
Test size & {test_size} bars \\
\bottomrule
\end{{tabular}}

\subsection*{{{latex_escape(title)}: Stitched Walk-Forward}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{p{{0.85in}}p{{0.95in}}rrrrrrrrrp{{1.15in}}}}
\toprule
Experiment & Optimized & MtM & BH & Gap & Max DD & Sharpe & Calmar & Trades & Avg hold & $\Delta$ MtM & Final selected \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\normalsize

\subsection*{{{latex_escape(title)}: GUI-Style Continuous Online}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{p{{0.85in}}p{{0.95in}}rrrrrrrrrrp{{1.15in}}}}
\toprule
Experiment & Swept & Real. & MtM & BH & Gap & Max DD & Sharpe & Calmar & Trades & Avg hold & $\Delta$ MtM & Selected / fixed \\
\midrule
{chr(10).join(gui_rows)}
\bottomrule
\end{{tabular}}
\normalsize

\paragraph{{{latex_escape(title)} takeaways.}}
Stitched best MtM is {latex_escape(best["experiment"])} with
{format_pct(best["mtm_return"])}; stitched best Sharpe is
{latex_escape(best_sharpe["experiment"])} with {format_num(best_sharpe["sharpe"])}.
GUI-style best MtM is {latex_escape(gui_best["experiment"])} with
{format_pct(gui_best["mtm_return"])}; GUI-style best Sharpe is
{latex_escape(gui_best_sharpe["experiment"])} with {format_num(gui_best_sharpe["sharpe"])}.
"""


def parse_grid(values: str, cast):
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seven 3D parameter expansion experiments.",
    )
    parser.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--period", choices=list(PERIODS), default="5 years")
    parser.add_argument("--interval", default=None)
    parser.add_argument("--train-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--synthetic-steps", type=int, default=756)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--k-grid", default="0.8,1.1,1.4,1.8,2.2")
    parser.add_argument("--delta-grid", default="1,2,3,5,8")
    parser.add_argument(
        "--only",
        default=",".join(EXPERIMENT_ORDER),
        help="Comma-separated experiments to run.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller grids for quick smoke tests.",
    )
    parser.add_argument(
        "--include-nvda-1y",
        action="store_true",
        help="Append an NVIDIA 1-year/1-day comparison section to the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = load_backend()
    lp, metadata = load_dataset(backend, args)
    args.lp = lp

    k_grid = parse_grid(args.k_grid, float)
    delta_grid = parse_grid(args.delta_grid, int)
    if args.fast:
        k_grid = [0.8, 1.2, 1.8]
        delta_grid = [1, 3]
        for key in GRIDS:
            GRIDS[key] = GRIDS[key][:2]

    experiments = [item.strip() for item in args.only.split(",") if item.strip()]
    invalid = [item for item in experiments if item not in EXPERIMENT_ORDER]
    if invalid:
        raise ValueError(f"Unknown experiments: {invalid}")

    c = args.cost_bps / 10000.0
    periods_per_year = metadata["periods_per_year"]
    results = []
    gui_results = []

    print(f"Dataset: {metadata['source']} / {metadata['lookback']} / {metadata['interval']}")
    print(f"Bars: {len(lp) - 1}, train={args.train_size}, test={args.test_size}")

    for experiment in experiments:
        print(f"Running {experiment}...")
        if experiment == "rho":
            result = run_rho_online_experiment(
                backend,
                lp,
                k_grid,
                delta_grid,
                args.train_size,
                args.test_size,
                c,
                c,
                periods_per_year,
            )
        else:
            result = run_walk_forward_experiment(
                backend,
                lp,
                experiment,
                k_grid,
                delta_grid,
                args.train_size,
                args.test_size,
                c,
                c,
                periods_per_year,
            )
        results.append(result)
        print(
            f"  MtM={100 * result['mtm_return']:.2f}% "
            f"Gap={100 * result['benchmark_gap']:.2f}% "
            f"Sharpe={format_num(result['sharpe'])}"
        )

    for experiment in experiments:
        print(f"Running GUI-style {experiment}...")
        result = run_gui_continuous_experiment(
            backend,
            lp,
            experiment,
            c,
            c,
            periods_per_year,
        )
        gui_results.append(result)
        print(
            f"  MtM={100 * result['mtm_return']:.2f}% "
            f"Gap={100 * result['benchmark_gap']:.2f}% "
            f"Sharpe={format_num(result['sharpe'])}"
        )

    comparison_sections = ""
    if args.include_nvda_1y:
        print("Running NVIDIA 1-year/1-day comparison section...")
        nvda_args = argparse.Namespace(**vars(args))
        nvda_args.source = "yahoo"
        nvda_args.symbol = "NVDA"
        nvda_args.period = "1 year"
        nvda_args.interval = "1d"
        nvda_args.train_size = min(args.train_size, 126)
        nvda_args.test_size = min(args.test_size, 21)
        nvda_lp, nvda_metadata = load_dataset(backend, nvda_args)
        nvda_args.lp = nvda_lp
        nvda_periods_per_year = nvda_metadata["periods_per_year"]
        nvda_stitched = []
        nvda_gui = []

        print(
            f"NVDA bars: {len(nvda_lp) - 1}, "
            f"train={nvda_args.train_size}, test={nvda_args.test_size}"
        )
        for experiment in experiments:
            print(f"Running NVDA stitched {experiment}...")
            if experiment == "rho":
                result = run_rho_online_experiment(
                    backend,
                    nvda_lp,
                    k_grid,
                    delta_grid,
                    nvda_args.train_size,
                    nvda_args.test_size,
                    c,
                    c,
                    nvda_periods_per_year,
                )
            else:
                result = run_walk_forward_experiment(
                    backend,
                    nvda_lp,
                    experiment,
                    k_grid,
                    delta_grid,
                    nvda_args.train_size,
                    nvda_args.test_size,
                    c,
                    c,
                    nvda_periods_per_year,
                )
            nvda_stitched.append(result)
            print(
                f"  MtM={100 * result['mtm_return']:.2f}% "
                f"Sharpe={format_num(result['sharpe'])}"
            )

        for experiment in experiments:
            print(f"Running NVDA GUI-style {experiment}...")
            result = run_gui_continuous_experiment(
                backend,
                nvda_lp,
                experiment,
                c,
                c,
                nvda_periods_per_year,
            )
            nvda_gui.append(result)
            print(
                f"  MtM={100 * result['mtm_return']:.2f}% "
                f"Sharpe={format_num(result['sharpe'])}"
            )

        comparison_sections = make_comparison_section(
            "NVDA 1-Year Daily",
            nvda_metadata,
            nvda_stitched,
            nvda_gui,
            nvda_args.train_size,
            nvda_args.test_size,
            len(nvda_lp) - 1,
        )

    make_report(args.output, metadata, results, gui_results, args, comparison_sections)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
