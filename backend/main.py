from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from json.decoder import JSONDecodeError
from datetime import timedelta
import os
import io
import contextlib
from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MTF Dashboard Agent", version="2.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Environment toggle: set USE_MOCK=1|true to force demo/mock OHLCV data
USE_MOCK = os.environ.get("USE_MOCK", "").lower() in ("1", "true", "yes")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
PREFER_ALPHAVANTAGE = os.environ.get("PREFER_ALPHAVANTAGE", "true").lower() in ("1", "true", "yes")

# Cache for Alpha Vantage responses: {(ticker, interval): (timestamp, df)}
# TTL is 60 seconds to respect rate limits and keep data fresh
AV_CACHE: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
AV_CACHE_TTL = 60

TIMEFRAMES: dict[str, dict[str, Any]] = {
    "W": {"period": "2y", "interval": "1wk"},
    "D": {"period": "1y", "interval": "1d"},
    "4H": {"period": "60d", "interval": "1h", "resample": "4h"},
    "65m/1H": {"period": "60d", "interval": "1h"},
    "30m": {"period": "30d", "interval": "30m"},
    "15m": {"period": "10d", "interval": "15m"},
    "10m": {"period": "5d", "interval": "5m", "resample": "10min"},
    "5m": {"period": "5d", "interval": "5m"},
}


@dataclass
class FrameSignal:
    label: str
    status: str
    close: float | None
    ema9: float | None
    ema21: float | None
    ema50: float | None
    sma200: float | None
    vwap: float | None
    volume_expanding: bool


def clean_ticker_list(raw_tickers: str) -> list[str]:
    tickers = [item.strip().upper() for item in raw_tickers.replace("\n", ",").split(",")]
    return list(dict.fromkeys(ticker for ticker in tickers if ticker))


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df).copy()
    df = df.rename(columns={column: str(column).title() for column in df.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    df = df[required].dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    resampled = df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return resampled.dropna(subset=["Open", "High", "Low", "Close"])


def _try_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    attempts = 3
    delay = 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            logger.debug(f"yfinance attempt {attempt + 1}/{attempts} for {ticker} interval={interval}")
            # suppress yfinance noisy prints by redirecting stdout/stderr
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                data = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    prepost=False,
                )
            if data.empty:
                logger.debug(f"yfinance returned empty data for {ticker} interval={interval}")
                last_exc = None
                continue
            logger.info(f"yfinance succeeded for {ticker} interval={interval}")
            return normalize_frame(data)
        except (JSONDecodeError, ValueError) as exc:
            last_exc = exc
            logger.warning(f"yfinance attempt {attempt + 1} failed for {ticker}: {exc}")
            if attempt < attempts - 1:
                time.sleep(delay)
                delay *= 2
                continue
            # final attempt failed; do not raise here — return empty to allow caller fallback
            break
        except Exception as exc:  # fallback retry for intermittent network/errors
            last_exc = exc
            logger.warning(f"yfinance attempt {attempt + 1} error for {ticker}: {exc}")
            if attempt < attempts - 1:
                time.sleep(delay)
                delay *= 2
                continue
            # final attempt failed; don't raise, let caller fallback
            break
    logger.info(f"yfinance exhausted retries for {ticker} interval={interval}; falling back")
    # return empty to indicate no data from yfinance (caller may fallback)
    return pd.DataFrame()


@lru_cache(maxsize=512)
def download_history(ticker: str, period: str, interval: str, cache_bucket: int) -> pd.DataFrame:
    del cache_bucket
    if USE_MOCK:
        return normalize_frame(generate_mock_history(ticker, period, interval))

    last_exc: Exception | None = None

    def try_av() -> pd.DataFrame:
        try:
            logger.debug(f"Attempting Alpha Vantage for {ticker} interval={interval}")
            av_df = fetch_av_history(ticker, period, interval, ALPHAVANTAGE_API_KEY)
            if not av_df.empty:
                logger.info(f"Alpha Vantage succeeded for {ticker} interval={interval}")
                return normalize_frame(av_df)
            logger.debug(f"Alpha Vantage returned empty for {ticker} interval={interval}")
        except Exception as exc:
            nonlocal last_exc
            logger.warning(f"Alpha Vantage error for {ticker} interval={interval}: {exc}")
            last_exc = exc
        return pd.DataFrame()

    # Order attempts based on preference flag
    if ALPHAVANTAGE_API_KEY and PREFER_ALPHAVANTAGE:
        df = try_av()
        if not df.empty:
            return df
        df = _try_yfinance(ticker, period, interval)
        if not df.empty:
            return df
    elif ALPHAVANTAGE_API_KEY and not PREFER_ALPHAVANTAGE:
        df = _try_yfinance(ticker, period, interval)
        if not df.empty:
            return df
        df = try_av()
        if not df.empty:
            return df
    else:
        df = _try_yfinance(ticker, period, interval)
        if not df.empty:
            return df

    # If we get here no live provider returned data. Provide a mock OHLCV dataset
    # so the dashboard can function for demos and offline use.
    try:
        mock = generate_mock_history(ticker, period, interval)
        return normalize_frame(mock)
    except Exception:
        if last_exc:
            raise last_exc
        raise


def generate_mock_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    # Map interval strings to pandas frequency
    freq_map = {
        "1wk": "W",
        "1d": "D",
        "1h": "h",
        "30m": "30min",
        "15m": "15min",
        "5m": "5min",
        "10m": "10min",
    }
    # Determine frequency from interval
    freq = freq_map.get(interval, "D")

    # Choose a span for mock data based on period
    if "y" in period:
        days = 365
    elif "60d" in period:
        days = 60
    elif "30d" in period:
        days = 30
    elif "10d" in period:
        days = 10
    elif "5d" in period:
        days = 5
    else:
        days = 30

    end = pd.Timestamp.utcnow().floor("min")
    start = end - pd.Timedelta(days=days)
    rng = pd.date_range(start=start, end=end, freq=freq)
    if rng.empty:
        rng = pd.date_range(end=end, periods=50, freq=freq)

    # generate a simple price series
    seed = abs(hash(ticker)) % (2**32)
    rng_state = np.random.RandomState(seed)
    base = 100.0 + (seed % 100) * 0.1
    prices = base + pd.Series(range(len(rng))).astype(float).cumsum() * 0.01
    noise = pd.Series(rng_state.normal(scale=0.5, size=len(rng)))
    close = prices + noise
    openp = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([openp, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([openp, close], axis=1).min(axis=1) - 0.5
    volume = (pd.Series(1000000, index=rng) * (1 + 0.1 * np.sin(np.linspace(0, 3.14, len(rng))))).astype(int)

    df = pd.DataFrame({"Open": openp.values, "High": high.values, "Low": low.values, "Close": close.values, "Volume": volume.values}, index=rng)
    return df


def fetch_av_history(ticker: str, period: str, interval: str, api_key: str) -> pd.DataFrame:
    """Fetch OHLCV from Alpha Vantage with caching. Returns DataFrame indexed by UTC timestamps."""
    # Check cache first
    cache_key = (ticker, interval)
    now = time.time()
    if cache_key in AV_CACHE:
        ts, cached_df = AV_CACHE[cache_key]
        if now - ts < AV_CACHE_TTL:
            logger.info(f"Alpha Vantage cache hit for {ticker} interval={interval}")
            return cached_df
        else:
            logger.debug(f"Alpha Vantage cache expired for {ticker} interval={interval}")
            del AV_CACHE[cache_key]

    logger.debug(f"Fetching Alpha Vantage data for {ticker} interval={interval}")
    base = "https://www.alphavantage.co/query"
    # map our interval to AV function/interval
    if interval == "1wk":
        params = {"function": "TIME_SERIES_WEEKLY", "symbol": ticker, "apikey": api_key, "datatype": "json"}
        key = "Weekly Time Series"
    elif interval == "1d":
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker, "apikey": api_key, "datatype": "json"}
        key = "Time Series (Daily)"
    else:
        # intraday mapping: use closest supported interval
        av_interval = {
            "1h": "60min",
            "65m/1H": "60min",
            "30m": "30min",
            "15m": "15min",
            "10m": "5min",
            "5m": "5min",
        }.get(interval, "60min")
        params = {"function": "TIME_SERIES_INTRADAY", "symbol": ticker, "interval": av_interval, "apikey": api_key, "datatype": "json", "outputsize": "compact"}
        key = f"Time Series ({av_interval})"

    try:
        resp = requests.get(base, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if key not in data:
            logger.warning(f"Alpha Vantage missing key '{key}' for {ticker}; may be rate limited")
            return pd.DataFrame()

        ts_data = data[key]
        # convert to DataFrame
        df = pd.DataFrame.from_dict(ts_data, orient="index")
        # columns from AV are like '1. open', '2. high', etc.
        df = df.rename(columns=lambda c: c.split('. ', 1)[-1].title())
        # ensure numeric types
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # index to datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        # Alpha Vantage times are in local exchange time; assume UTC for now
        df.index = df.index.tz_localize(None)
        result = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
        # Cache the result
        AV_CACHE[cache_key] = (time.time(), result)
        logger.info(f"Alpha Vantage succeeded for {ticker} interval={interval}; cached")
        return result
    except Exception as exc:
        logger.error(f"Alpha Vantage fetch failed for {ticker} interval={interval}: {exc}")
        return pd.DataFrame()

def latest_float(series: pd.Series) -> float | None:
    if series.empty or pd.isna(series.iloc[-1]):
        return None
    return float(series.iloc[-1])


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume = df["Volume"].replace(0, np.nan)
    session_key = df.index.date if hasattr(df.index, "date") else pd.Series(0, index=df.index)
    pv = (typical_price * df["Volume"]).groupby(session_key).cumsum()
    cv = volume.groupby(session_key).cumsum()
    return pv / cv


def compute_signal(label: str, df: pd.DataFrame) -> FrameSignal:
    if df.empty or len(df) < 50:
        return FrameSignal(label, "WAIT", None, None, None, None, None, None, False)

    close = df["Close"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    sma200 = close.rolling(200, min_periods=50).mean()
    vwap = compute_vwap(df)
    avg_volume = df["Volume"].rolling(20, min_periods=5).mean()

    last_close = latest_float(close)
    last_ema9 = latest_float(ema9)
    last_ema21 = latest_float(ema21)
    last_ema50 = latest_float(ema50)
    last_sma200 = latest_float(sma200)
    last_vwap = latest_float(vwap)
    last_volume = latest_float(df["Volume"])
    last_avg_volume = latest_float(avg_volume)

    bull_stack = bool(last_ema9 and last_ema21 and last_ema50 and last_ema9 > last_ema21 > last_ema50)
    bear_stack = bool(last_ema9 and last_ema21 and last_ema50 and last_ema9 < last_ema21 < last_ema50)
    volume_expanding = bool(last_volume is not None and last_avg_volume is not None and last_volume > last_avg_volume * 1.2)

    status = "CHOP"
    if bull_stack:
        status = "BULL"
    elif bear_stack:
        status = "BEAR"

    return FrameSignal(label, status, last_close, last_ema9, last_ema21, last_ema50, last_sma200, last_vwap, volume_expanding)


def compute_orb(df_15m: pd.DataFrame) -> dict[str, Any]:
    if df_15m.empty:
        return {"high": None, "low": None, "range": None, "status": "WAIT"}

    latest_day = df_15m.index[-1].date()
    day_frame = df_15m[df_15m.index.date == latest_day]
    if day_frame.empty:
        return {"high": None, "low": None, "range": None, "status": "WAIT"}

    opening_bar = day_frame.iloc[0]
    orb_high = float(opening_bar["High"])
    orb_low = float(opening_bar["Low"])
    orb_range = orb_high - orb_low
    last_close = float(day_frame.iloc[-1]["Close"])

    if last_close > orb_high:
        status = "BULL BREAK"
    elif last_close < orb_low:
        status = "BEAR BREAK"
    else:
        status = "INSIDE ORB"

    return {"high": orb_high, "low": orb_low, "range": orb_range, "status": status}


def frame_for_ticker(ticker: str, label: str, config: dict[str, Any], cache_bucket: int) -> pd.DataFrame:
    df = download_history(ticker, config["period"], config["interval"], cache_bucket).copy()
    if "resample" in config:
        df = resample_ohlcv(df, config["resample"])
    return df


def scan_ticker(ticker: str, cache_bucket: int) -> dict[str, Any]:
    signals: dict[str, FrameSignal] = {}
    frames: dict[str, pd.DataFrame] = {}

    for label, config in TIMEFRAMES.items():
        try:
            frames[label] = frame_for_ticker(ticker, label, config, cache_bucket)
        except Exception as exc:
            # Fallback to per-timeframe mock if live fetching/resampling fails for this timeframe
            logger.warning(f"Failed to fetch {label} for {ticker}: {exc}; using mock fallback")
            try:
                frames[label] = normalize_frame(generate_mock_history(ticker, config["period"], config["interval"]))
                logger.info(f"Using mock fallback for {label} on {ticker}")
            except Exception as mock_exc:
                logger.error(f"Mock fallback also failed for {label} on {ticker}: {mock_exc}")
                frames[label] = pd.DataFrame()
        signals[label] = compute_signal(label, frames[label])

    intraday = signals["15m"]
    orb = compute_orb(frames["15m"])
    close = intraday.close
    vwap = intraday.vwap
    sma200 = intraday.sma200

    vwap_side = "WAIT" if close is None or vwap is None else ("ABOVE" if close > vwap else "BELOW")
    sma_side = "WAIT" if close is None or sma200 is None else ("ABOVE" if close > sma200 else "BELOW")

    bull_votes = sum(1 for signal in signals.values() if signal.status == "BULL")
    bear_votes = sum(1 for signal in signals.values() if signal.status == "BEAR")
    bull_bonus = sum([vwap_side == "ABOVE", sma_side == "ABOVE", orb["status"] == "BULL BREAK", intraday.volume_expanding])
    bear_bonus = sum([vwap_side == "BELOW", sma_side == "BELOW", orb["status"] == "BEAR BREAK", intraday.volume_expanding])
    score = bull_votes + bull_bonus - bear_votes - bear_bonus

    if score >= 6:
        bias = "BULLISH"
    elif score <= -6:
        bias = "BEARISH"
    else:
        bias = "CHOP"

    if bias == "BULLISH" and orb["status"] == "BULL BREAK" and vwap_side == "ABOVE" and sma_side == "ABOVE":
        entry_status = "LONG WATCH"
    elif bias == "BEARISH" and orb["status"] == "BEAR BREAK" and vwap_side == "BELOW" and sma_side == "BELOW":
        entry_status = "SHORT WATCH"
    else:
        entry_status = "WAIT"

    return {
        "ticker": ticker,
        "timeframes": {label: signal.status for label, signal in signals.items()},
        "orb": orb,
        "vwapSide": vwap_side,
        "sma200Side": sma_side,
        "score": score,
        "bias": bias,
        "entryStatus": entry_status,
        "volumeExpanding": intraday.volume_expanding,
        "lastPrice": close,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/scan")
def scan(tickers: str = Query("AAPL,MSFT,NVDA,SPY,QQQ")) -> dict[str, Any]:
    ticker_list = clean_ticker_list(tickers)
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Add at least one ticker.")

    cache_bucket = int(pd.Timestamp.utcnow().timestamp() // 55)
    rows = []
    errors = []
    for ticker in ticker_list[:30]:
        try:
            rows.append(scan_ticker(ticker, cache_bucket))
        except Exception as exc:  # keeps one bad symbol from breaking the dashboard
            errors.append({"ticker": ticker, "message": str(exc)})

    return {"rows": rows, "errors": errors, "updatedAt": pd.Timestamp.utcnow().isoformat()}
