# AGENTS.md

## Cursor Cloud specific instructions

This is a Python 3.12 project (CI uses 3.11) for an ML-powered BTC price predictor with a
Streamlit dashboard. Standard commands live in `README.md`; below are the non-obvious
caveats for working in this environment.

### Environment

- Dependencies are installed into a virtualenv at `.venv` (the startup update script creates it
  and runs `pip install -r requirements.txt`). Always invoke tools through `.venv/bin/...`
  (e.g. `.venv/bin/python`, `.venv/bin/streamlit`) rather than the system Python.

### Data / network caveats

- There is no committed price data — `data/price/btc_hourly.parquet` is gitignored. Run
  `python main.py --download` once before `--predict`/`--status`, or `--status` reports
  "No price data available".
- Binance returns HTTP 451 from these runners, so set `BTC_PRICE_PRIMARY_EXCHANGE=bitstamp`
  for any command that fetches price data (`main.py --download`, `main.py --predict`,
  `trade.py --live-tick`). The collector chain falls back automatically, but setting it
  avoids slow failed Binance attempts.

### Run / test / build

- Tests (stdlib `unittest`, no extra deps): `.venv/bin/python -m unittest discover -s tests`.
- No configured linter; use `.venv/bin/python -m py_compile <files>` as a syntax check.
- Prediction engine: `BTC_PRICE_PRIMARY_EXCHANGE=bitstamp .venv/bin/python main.py --predict`.
- Trading tick: `BTC_PRICE_PRIMARY_EXCHANGE=bitstamp .venv/bin/python trade.py --live-tick`.
- Dashboard: `.venv/bin/python -m streamlit run dashboard/app.py --server.port 8501 --server.headless true`
  (serves on http://localhost:8501). It renders with the committed demo data even before any
  prediction runs.

### Notes

- Without trained ML models in `data/validation/models/`, `--predict` falls back to TA
  heuristics (logs a warning); this is expected and still produces output. The trading
  pre-trade gate requires real ML, so live ticks may take no action under heuristics-only.
