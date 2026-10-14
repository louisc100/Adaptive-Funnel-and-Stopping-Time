# Adaptive Funnel and Stopping-Time Trading

Research code for adaptive price funnels and stopping-time trading: a desktop
simulator, historical-data experiments, and an IBKR execution application.

## Project Layout

| Location | Contents |
| --- | --- |
| `trading/gui.py` | Desktop simulation, Yahoo historical-data mode, and shared research calculations |
| `trading/ibkr_bot.py` | TWS connection, quote streaming, orders, state recovery, and live plots |
| `trading/live_strategy.py` | Live signals, anchors, fill accounting, and strategy state |
| `trading/yahoo_paper_trading.py` | Separate local paper account using Yahoo data |
| `experiments/optimize_three_parameters.py` | Walk-forward search over k, delta, and one additional parameter |
| `experiments/optimize_four_parameters.py` | Walk-forward search over k, delta, and two additional parameters |
| `scripts/generate_signal_history_plot.py` | Interactive HTML chart from a signal CSV |
| `tests/` | Offline regression tests |
| `docs/paper/` | Main mathematical paper, LaTeX source and PDF |
| `docs/roadmaps/` | Research and extension roadmaps; earlier extension diagram in `archive/` |
| `docs/research/` | Volatility-harvesting research report |
| `reports/` | Three- and four-parameter optimization reports |
| `data/` | Private runtime logs, charts, caches, and saved states (ignored by Git) |

Run commands from this project's root directory. The existing `venv/` is kept
in place. The applications use NumPy, Matplotlib, PyQt5, and ib_insync.

## Running the Applications

Desktop GUI:

```bash
./venv/bin/python -m trading.gui
```

IBKR command options (does not connect or place an order):

```bash
./venv/bin/python -m trading.ibkr_bot --help
```

The root `main.py` and `ibkr_bridge.py` are compatibility launchers, so existing
GUI and trading commands still work with all their previous flags. The
`--dry-run-strategy` flag enables signal computation; combining it with
`--auto-orders` enables execution on the connected TWS account. Renaming the
application does not change that behavior.

Yahoo-based local paper account:

```bash
./venv/bin/python -m trading.yahoo_paper_trading --help
```

Optimization studies:

```bash
./venv/bin/python -m experiments.optimize_three_parameters --help
./venv/bin/python -m experiments.optimize_four_parameters --help
```

Offline checks:

```bash
./venv/bin/python -m unittest discover -s tests
```

Build the paper with a LaTeX installation:

```bash
latexmk -cd -pdf docs/paper/adaptive_funnel_trading.tex
```

## Naming and Local Files

`Paper/Paper.*` is now `docs/paper/adaptive_funnel_trading.*`.
`Reports/opt_3dim_report.*` and `Reports/opt_4dim_report.*` are now
`reports/three_parameter_optimization.*` and
`reports/four_parameter_optimization.*`.
The original report PDFs retain their historical content and generation labels.

Existing runtime filenames under `data/` are unchanged so saved-state recovery
and terminal commands continue to reference the same files. Generated LaTeX
files, old smoke-test reports, and old caches were moved to local `archive/`.
Both `archive/` and `.privacy-backup/` are ignored by Git; the latter contains
the private repository recovery backup and must not be published.
