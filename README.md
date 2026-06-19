# tradingdashboard

MTF Dashboard Agent — multi-timeframe scanner using Yahoo Finance (yfinance).

Run locally:

```bash
# Install deps
pip install -r requirements.txt

# Run server
USE_MOCK=1 python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

 - Setting `USE_MOCK=1` forces demo/mock OHLCV data (useful offline or behind rate limits).
 - By default the app will attempt live downloads and fallback to mock data when downloads fail.

Optional: Alpha Vantage live data
---------------------------------
- Set `ALPHAVANTAGE_API_KEY` to enable Alpha Vantage as the preferred live provider. Example:

```bash
ALPHAVANTAGE_API_KEY=your_key_here python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

If the key is present the backend will try Alpha Vantage first and fall back to yfinance or mock data when necessary.

- To explicitly prefer Alpha Vantage over yfinance set `PREFER_ALPHAVANTAGE=1` (default). To prefer `yfinance` even when you have an Alpha Vantage key, set `PREFER_ALPHAVANTAGE=0`.

Example preferring yfinance despite having an Alpha Vantage key:

```bash
ALPHAVANTAGE_API_KEY=your_key_here PREFER_ALPHAVANTAGE=0 python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://127.0.0.1:8000/ in your browser.
# MTF Dashboard Agent

A local FastAPI + HTML/CSS/JavaScript dashboard that scans Yahoo Finance data with `yfinance` and summarizes bull, bear, and chop conditions across multiple timeframes.

## Features

- Editable ticker list with manual refresh and 60-second auto-refresh.
- Multi-timeframe columns: Weekly, Daily, 4H, 65m/1H, 30m, 15m, 10m, and 5m.
- Indicator checks for EMA 9/21/50 stacks, SMA 200 side, VWAP side, 15-minute opening range breakout, and volume expansion.
- Green bullish, red bearish, and yellow/gray wait/chop states.

## MacBook run instructions

From the project folder, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000
```

## Notes

Yahoo Finance intraday availability can vary by symbol and interval. The app uses a 1-hour approximation for the requested 65-minute timeframe and resamples 1-hour data into 4-hour bars and 5-minute data into 10-minute bars.
