"""
Layer 2.5 four-dimensional optimization study.

This sidecar mirrors opt_3dim.py, but expands the search from
(k, delta, x) to (k, delta, x, y) for parameter pairs drawn from
q, alpha, beta, L, a, z_trend, and rho.

The full all-pairs study is more expensive than the 3D study. Use --fast
for smoke tests or --pairs to focus on selected pairs.

Examples:
    python3 opt_4dim.py --symbol SPY --period "5 years" --fast
    python3 opt_4dim.py --pairs q+alpha,L+a,z_trend+rho
    latexmk -pdf Reports/opt_4dim_report.tex
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
from pathlib import Path
from typing import Any

import numpy as np

import opt_3dim as opt3


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "Reports"
DEFAULT_REPORT = REPORT_DIR / "opt_4dim_report.tex"
EXTRA_PARAMS = ["q", "alpha", "beta", "L", "a", "z_trend", "rho"]
BASELINE = "baseline"


def pair_name(pair: tuple[str, str]) -> str:
    return "+".join(pair)


def parse_pair_name(name: str) -> tuple[str, str]:
    parts = [part.strip() for part in name.split("+") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Pair must look like x+y, got {name!r}.")
    for part in parts:
        if part not in EXTRA_PARAMS:
            raise ValueError(f"Unknown parameter {part!r}.")
    if parts[0] == parts[1]:
        raise ValueError(f"Pair cannot repeat the same parameter: {name!r}.")
    return tuple(parts)  # type: ignore[return-value]


def all_pairs() -> list[tuple[str, str]]:
    return list(itertools.combinations(EXTRA_PARAMS, 2))


def pair_grid(pair: tuple[str, str]) -> list[tuple[Any, Any]]:
    return list(itertools.product(opt3.GRIDS[pair[0]], opt3.GRIDS[pair[1]]))


def apply_pair(params: dict[str, Any], pair: tuple[str, str], values: tuple[Any, Any]) -> None:
    params[pair[0]] = values[0]
    params[pair[1]] = values[1]


def selected_pair_dict(pair: tuple[str, str], values: tuple[Any, Any]) -> dict[str, Any]:
    return {pair[0]: values[0], pair[1]: values[1]}


def select_fold_params_4d_static(
    backend: dict[str, Any],
    train_lp: np.ndarray,
    k_grid: list[float],
    delta_grid: list[int],
    pair: tuple[str, str] | None,
    c_buy: float,
    c_sell: float,
) -> dict[str, Any] | None:
    sigma_seed = backend["estimate_sigma_seed"](train_lp)
    best = None
    values_grid = [(None, None)] if pair is None else pair_grid(pair)

    for k_value, delta_value, values in itertools.product(k_grid, delta_grid, values_grid):
        if delta_value >= len(train_lp) - 2:
            continue
        params = opt3.experiment_defaults()
        if pair is not None:
            apply_pair(params, pair, values)
        try:
            data = backend["run_strategy_on_log_prices"](
                train_lp,
                k=k_value,
                delta=delta_value,
                sigma_seed=sigma_seed,
                c_buy=c_buy,
                c_sell=c_sell,
                mode="4D training fold",
                **opt3.strategy_kwargs(params),
            )
        except Exception:
            continue
        metrics = backend["summarize_simulation"](data)
        score = opt3.score_metrics(metrics)
        if best is None or score > best["score"]:
            best = {
                "k": k_value,
                "delta": delta_value,
                "values": values,
                "score": score,
                "sigma_seed": sigma_seed,
                "params": params,
            }
    return best


def run_stitched_pair_experiment(
    backend: dict[str, Any],
    full_lp: np.ndarray,
    experiment: str,
    pair: tuple[str, str] | None,
    k_grid: list[float],
    delta_grid: list[int],
    train_size: int,
    test_size: int,
    c_buy: float,
    c_sell: float,
    periods_per_year: float,
) -> dict[str, Any]:
    if len(full_lp) < train_size + test_size + 2:
        raise ValueError("Dataset is too short for the requested train/test sizes.")

    folds = []
    fold_results = []
    for train_start in range(0, len(full_lp) - train_size - test_size, test_size):
        train_end = train_start + train_size
        test_end = train_end + test_size
        train_lp = full_lp[train_start : train_end + 1]
        test_lp = full_lp[train_end : test_end + 1]
        chosen = select_fold_params_4d_static(
            backend,
            train_lp,
            k_grid,
            delta_grid,
            pair,
            c_buy,
            c_sell,
        )
        if chosen is None:
            continue

        test_data = backend["run_strategy_on_log_prices"](
            test_lp,
            k=chosen["k"],
            delta=chosen["delta"],
            sigma_seed=chosen["sigma_seed"],
            c_buy=c_buy,
            c_sell=c_sell,
            mode=f"{experiment} OOS fold",
            **opt3.strategy_kwargs(chosen["params"]),
        )
        metrics = backend["summarize_simulation"](test_data)
        fold = {
            "train": (train_start, train_end),
            "test": (train_end, test_end),
            "k": chosen["k"],
            "delta": chosen["delta"],
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
        if pair is not None:
            fold.update(selected_pair_dict(pair, chosen["values"]))
        folds.append(fold)
        fold_results.append(test_data)

    if not fold_results:
        raise RuntimeError(f"{experiment} produced no valid folds.")

    stitched = backend["stitch_walk_forward_tests"](
        fold_results,
        folds,
        source=f"{experiment} 4D study",
        mode=f"{experiment} stitched OOS",
    )
    summary = opt3.summarize_data(backend, stitched, periods_per_year)
    summary["experiment"] = experiment
    summary["optimized_parameters"] = ["k", "delta"] + ([] if pair is None else list(pair))
    final_selected = {"k": folds[-1]["k"], "delta": folds[-1]["delta"]}
    if pair is not None:
        final_selected.update({name: folds[-1][name] for name in pair})
    summary["final_selected"] = final_selected
    summary["fold_count"] = len(folds)
    return summary


def select_fold_params_4d_online(
    backend: dict[str, Any],
    train_lp: np.ndarray,
    k_grid: list[float],
    delta_grid: list[int],
    pair: tuple[str, str],
    c_buy: float,
    c_sell: float,
) -> dict[str, Any] | None:
    best = None
    for k_init, delta_init, values in itertools.product(k_grid, delta_grid, pair_grid(pair)):
        params = opt3.experiment_defaults()
        apply_pair(params, pair, values)
        rho = params["rho"]
        try:
            train_data = backend["run_online_adaptive_strategy"](
                train_lp,
                c_buy=c_buy,
                c_sell=c_sell,
                learning_rate=rho,
                k_init=k_init,
                delta_init=delta_init,
                source="4D rho training fold",
                **opt3.strategy_kwargs(params),
            )
        except Exception:
            continue
        metrics = backend["summarize_simulation"](train_data)
        score = opt3.score_metrics(metrics)
        if best is None or score > best["score"]:
            best = {
                "k": k_init,
                "delta": delta_init,
                "values": values,
                "params": params,
                "score": score,
            }
    return best


def run_stitched_rho_pair_experiment(
    backend: dict[str, Any],
    full_lp: np.ndarray,
    experiment: str,
    pair: tuple[str, str],
    k_grid: list[float],
    delta_grid: list[int],
    train_size: int,
    test_size: int,
    c_buy: float,
    c_sell: float,
    periods_per_year: float,
) -> dict[str, Any]:
    folds = []
    fold_results = []
    for train_start in range(0, len(full_lp) - train_size - test_size, test_size):
        train_end = train_start + train_size
        test_end = train_end + test_size
        train_lp = full_lp[train_start : train_end + 1]
        test_lp = full_lp[train_end : test_end + 1]
        chosen = select_fold_params_4d_online(
            backend,
            train_lp,
            k_grid,
            delta_grid,
            pair,
            c_buy,
            c_sell,
        )
        if chosen is None:
            continue
        params = chosen["params"]
        test_data = backend["run_online_adaptive_strategy"](
            test_lp,
            c_buy=c_buy,
            c_sell=c_sell,
            learning_rate=params["rho"],
            k_init=chosen["k"],
            delta_init=chosen["delta"],
            source=f"{experiment} OOS fold",
            **opt3.strategy_kwargs(params),
        )
        metrics = backend["summarize_simulation"](test_data)
        fold = {
            "train": (train_start, train_end),
            "test": (train_end, test_end),
            "k": chosen["k"],
            "delta": chosen["delta"],
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
        fold.update(selected_pair_dict(pair, chosen["values"]))
        folds.append(fold)
        fold_results.append(test_data)

    if not fold_results:
        raise RuntimeError(f"{experiment} produced no valid folds.")

    stitched = backend["stitch_walk_forward_tests"](
        fold_results,
        folds,
        source=f"{experiment} 4D online study",
        mode=f"{experiment} stitched OOS",
    )
    summary = opt3.summarize_data(backend, stitched, periods_per_year)
    summary["experiment"] = experiment
    summary["optimized_parameters"] = ["k_init", "delta_init"] + list(pair)
    final_selected = {"k_init": folds[-1]["k"], "delta_init": folds[-1]["delta"]}
    final_selected.update({name: folds[-1][name] for name in pair})
    summary["final_selected"] = final_selected
    summary["fold_count"] = len(folds)
    return summary


def run_gui_pair_experiment(
    backend: dict[str, Any],
    full_lp: np.ndarray,
    experiment: str,
    pair: tuple[str, str] | None,
    c_buy: float,
    c_sell: float,
    periods_per_year: float,
) -> dict[str, Any]:
    best = None
    values_grid = [(None, None)] if pair is None else pair_grid(pair)
    for values in values_grid:
        params = opt3.experiment_defaults()
        selected = {"fixed GUI defaults": ""} if pair is None else {}
        if pair is not None:
            apply_pair(params, pair, values)
            selected = selected_pair_dict(pair, values)
        try:
            data = backend["run_online_adaptive_strategy"](
                full_lp,
                c_buy=c_buy,
                c_sell=c_sell,
                learning_rate=params["rho"],
                k_init=1.2,
                delta_init=3,
                source=f"GUI-style {experiment}",
                **opt3.strategy_kwargs(params),
            )
        except Exception:
            continue
        metrics = backend["summarize_simulation"](data)
        score = opt3.score_metrics(metrics)
        if best is None or score > best["score"]:
            summary = opt3.summarize_data(backend, data, periods_per_year)
            summary["experiment"] = experiment
            summary["optimized_parameters"] = (
                ["fixed GUI defaults"] if pair is None else list(pair)
            )
            summary["final_selected"] = selected
            summary["final_live"] = data.get("selected_params", {})
            summary["score"] = score
            best = summary
    if best is None:
        raise RuntimeError(f"GUI-style {experiment} produced no valid run.")
    return best


def table_rows(
    results: list[dict[str, Any]],
    include_realized: bool = False,
) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return opt3.result_rows(results, include_realized=include_realized)


def make_report(
    path: Path,
    metadata: dict[str, Any],
    stitched_results: list[dict[str, Any]],
    gui_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows, best, best_gap, best_sharpe = table_rows(stitched_results)
    gui_rows, gui_best, gui_best_gap, gui_best_sharpe = table_rows(
        gui_results,
        include_realized=True,
    )
    fast_note = (
        "Yes: reduced screening grid for tractability"
        if args.fast
        else "No: full configured grid"
    )
    grid_rows = [
        ("k", args.k_grid if not args.fast else "0.8, 1.2"),
        ("delta", args.delta_grid if not args.fast else "1, 3"),
    ]
    grid_rows.extend(
        (name, ", ".join(str(value) for value in opt3.GRIDS[name]))
        for name in EXTRA_PARAMS
    )
    grid_tex = "\n".join(
        rf"{opt3.latex_escape(name)} & {opt3.latex_escape(values)} \\"
        for name, values in grid_rows
    )
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.55in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{hyperref}}

\title{{Four-Dimensional Parameter Expansion Study}}
\author{{Generated by opt\_4dim.py}}
\date{{{generated_at}}}

\begin{{document}}
\maketitle

\section*{{Purpose}}
This report extends the 3D side study from $(k,\delta,x)$ to
$(k,\delta,x,y)$, testing all selected pairs $(x,y)$ from the remaining
Layer 2 parameter set
\[
\{{q,\alpha,\beta,L,a,z_{{\mathrm{{trend}}}},\rho\}}.
\]
Pairs involving $\rho$ use online smoothing inside each fold, because $\rho$
does not affect the static single-fold strategy directly.

\section*{{Dataset and Protocol}}
\begin{{tabular}}{{ll}}
\toprule
Data source & {opt3.latex_escape(metadata["source"])} \\
Symbol & {opt3.latex_escape(metadata["symbol"])} \\
Lookback & {opt3.latex_escape(metadata["lookback"])} \\
Bar interval & {opt3.latex_escape(metadata["interval"])} \\
Bars & {len(args.lp) - 1 if hasattr(args, "lp") else "n/a"} \\
Train size & {args.train_size} bars \\
Test size & {args.test_size} bars \\
Cost per side & {args.cost_bps:.1f} bps \\
Objective & $R^m + D^m + {opt3.LAMBDA_GAP:.2f}G$ \\
\bottomrule
\end{{tabular}}

\section*{{Search Grid}}
This run used \texttt{{--fast}}: {opt3.latex_escape(fast_note)}. The 4D grid
can grow quickly because each experiment searches $(k,\delta,x,y)$, so the
fast run should be read as a sensitivity screen rather than a final exhaustive
optimization.

\begin{{tabular}}{{ll}}
\toprule
Parameter & Values searched \\
\midrule
{grid_tex}
\bottomrule
\end{{tabular}}

\section*{{Stitched Walk-Forward Results}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{p{{0.9in}}p{{1.25in}}rrrrrrrrrp{{1.45in}}}}
\toprule
Experiment & Optimized & MtM & BH & Gap & Max DD & Sharpe & Calmar & Trades & Avg hold & $\Delta$ MtM & Final selected \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{GUI-Style Continuous Online Results}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{p{{0.9in}}p{{1.15in}}rrrrrrrrrrp{{1.1in}}}}
\toprule
Experiment & Swept & Real. & MtM & BH & Gap & Max DD & Sharpe & Calmar & Trades & Avg hold & $\Delta$ MtM & End live \\
\midrule
{chr(10).join(gui_rows)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{High-Level Takeaways}}
\begin{{itemize}}
    \item Stitched best MtM return: {opt3.latex_escape(best["experiment"])} with {opt3.format_pct(best["mtm_return"])}.
    \item Stitched best benchmark gap: {opt3.latex_escape(best_gap["experiment"])} with {opt3.format_pct(best_gap["benchmark_gap"])}.
    \item Stitched best Sharpe ratio: {opt3.latex_escape(best_sharpe["experiment"])} with {opt3.format_num(best_sharpe["sharpe"])}.
    \item GUI-style best MtM return: {opt3.latex_escape(gui_best["experiment"])} with {opt3.format_pct(gui_best["mtm_return"])}.
    \item GUI-style best benchmark gap: {opt3.latex_escape(gui_best_gap["experiment"])} with {opt3.format_pct(gui_best_gap["benchmark_gap"])}.
    \item GUI-style best Sharpe ratio: {opt3.latex_escape(gui_best_sharpe["experiment"])} with {opt3.format_num(gui_best_sharpe["sharpe"])}.
\end{{itemize}}

\section*{{Caveats}}
The 4D grid is substantially larger than the 3D grid, so overfitting risk is
higher. Treat this as a sensitivity and screening report, not final evidence.
Pairs involving $\rho$ are not perfectly comparable to static pairs because
they use the online adaptation engine inside folds.

\end{{document}}
"""
    path.write_text(tex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4D parameter pair experiments.")
    parser.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--period", choices=list(opt3.PERIODS), default="5 years")
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
        "--pairs",
        default="all",
        help="Comma-separated pairs like q+alpha,L+a, or 'all'.",
    )
    parser.add_argument("--fast", action="store_true", help="Use smaller grids.")
    return parser.parse_args()


def selected_pairs(pair_arg: str) -> list[tuple[str, str]]:
    if pair_arg.strip().lower() == "all":
        return all_pairs()
    return [parse_pair_name(item.strip()) for item in pair_arg.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    backend = opt3.load_backend()
    lp, metadata = opt3.load_dataset(backend, args)
    args.lp = lp
    k_grid = opt3.parse_grid(args.k_grid, float)
    delta_grid = opt3.parse_grid(args.delta_grid, int)

    if args.fast:
        k_grid = [0.8, 1.2]
        delta_grid = [1, 3]
        for key in opt3.GRIDS:
            opt3.GRIDS[key] = opt3.GRIDS[key][:2]

    pairs = selected_pairs(args.pairs)
    c = args.cost_bps / 10000.0
    ppy = metadata["periods_per_year"]
    stitched_results = []
    gui_results = []

    print(f"Dataset: {metadata['source']} / {metadata['lookback']} / {metadata['interval']}")
    print(f"Bars: {len(lp) - 1}, train={args.train_size}, test={args.test_size}")
    print("Running baseline...")
    stitched_results.append(
        run_stitched_pair_experiment(
            backend,
            lp,
            BASELINE,
            None,
            k_grid,
            delta_grid,
            args.train_size,
            args.test_size,
            c,
            c,
            ppy,
        )
    )
    gui_results.append(
        run_gui_pair_experiment(backend, lp, BASELINE, None, c, c, ppy)
    )

    for pair in pairs:
        name = pair_name(pair)
        print(f"Running stitched {name}...")
        if "rho" in pair:
            stitched = run_stitched_rho_pair_experiment(
                backend,
                lp,
                name,
                pair,
                k_grid,
                delta_grid,
                args.train_size,
                args.test_size,
                c,
                c,
                ppy,
            )
        else:
            stitched = run_stitched_pair_experiment(
                backend,
                lp,
                name,
                pair,
                k_grid,
                delta_grid,
                args.train_size,
                args.test_size,
                c,
                c,
                ppy,
            )
        stitched_results.append(stitched)
        print(
            f"  MtM={100 * stitched['mtm_return']:.2f}% "
            f"Sharpe={opt3.format_num(stitched['sharpe'])}"
        )

        print(f"Running GUI-style {name}...")
        gui = run_gui_pair_experiment(backend, lp, name, pair, c, c, ppy)
        gui_results.append(gui)
        print(
            f"  MtM={100 * gui['mtm_return']:.2f}% "
            f"Sharpe={opt3.format_num(gui['sharpe'])}"
        )

    make_report(args.output, metadata, stitched_results, gui_results, args)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
