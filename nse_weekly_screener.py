#!/usr/bin/env python3
"""
nse_weekly_screener.py
======================

Screens the full universe of NSE cash-segment equities using weekly
technical and volume conditions.

A stock passes a given week only if ALL of the following are true:

    1. Weekly RSI(14) >= 60
    2. Weekly Volume > 1 week ago Close
    3. Weekly ADX(14) > 15
    4. 1 week ago Volume >= 2 weeks ago Volume * 0.5
    5. Weekly Volume > SMA(Weekly Volume, 20)
    6. Weekly Close > 1 week ago Close
    7. Weekly Close > EMA(Weekly Close, 20) * 0.7
    8. Weekly Volume > 1,000,000
    9. EMA(Weekly Close, 20) > EMA(Weekly Close, 100)

Output columns: date, stock_name, ticker  (saved to CSV as well).

Usage:
    python nse_weekly_screener.py --start 2018-01-01 --end 2026-12-31
    python nse_weekly_screener.py --debug RELIANCE
    python nse_weekly_screener.py --limit 25 --workers 8
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas.tseries.offsets import DateOffset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

MIN_WEEKLY_CANDLES = 100          # need >= 100 weekly bars for EMA(100)
VOLUME_THRESHOLD = 1_000_000      # condition 8
RSI_PERIOD = 14
ADX_PERIOD = 14
VOL_SMA_PERIOD = 20
EMA_FAST = 20
EMA_SLOW = 100

RESULT_COLUMNS = ["date", "stock_name", "ticker"]

# Small fallback universe used only if the live NSE list cannot be downloaded.
FALLBACK_UNIVERSE: list[tuple[str, str]] = [
    ("RELIANCE", "Reliance Industries Limited"),
    ("TCS", "Tata Consultancy Services Limited"),
    ("HDFCBANK", "HDFC Bank Limited"),
    ("INFY", "Infosys Limited"),
    ("ICICIBANK", "ICICI Bank Limited"),
    ("SBIN", "State Bank of India"),
    ("TATAMOTORS", "Tata Motors Limited"),
    ("ITC", "ITC Limited"),
    ("LT", "Larsen & Toubro Limited"),
    ("AXISBANK", "Axis Bank Limited"),
]

log = logging.getLogger("nse_screener")

# yfinance logs every failed download at ERROR level; that is expected for a
# full-universe NSE run (delisted / unmapped symbols) and clutters the console.
# Keep only CRITICAL from yfinance; real per-symbol failures are summarised by us.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #

def get_nse_universe() -> pd.DataFrame:
    """Download the official NSE equity list and return the EQ-series universe.

    Returns
    -------
    pd.DataFrame
        Columns: ``symbol`` (NSE symbol) and ``stock_name`` (company name).
        Falls back to a small built-in list if the download fails.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
        raw.columns = [c.strip() for c in raw.columns]
        eq = raw[raw["SERIES"].astype(str).str.strip() == "EQ"].copy()
        universe = eq[["SYMBOL", "NAME OF COMPANY"]].rename(
            columns={"SYMBOL": "symbol", "NAME OF COMPANY": "stock_name"}
        )
        universe["symbol"] = universe["symbol"].astype(str).str.strip()
        universe["stock_name"] = universe["stock_name"].astype(str).str.strip()
        universe = universe.dropna().drop_duplicates("symbol").reset_index(drop=True)
        log.info("Downloaded NSE universe: %d EQ symbols", len(universe))
        return universe
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not download NSE universe (%s). Using fallback list.", exc)
        return pd.DataFrame(FALLBACK_UNIVERSE, columns=["symbol", "stock_name"])


# --------------------------------------------------------------------------- #
# Data fetching
# --------------------------------------------------------------------------- #

def fetch_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame | None:
    """Fetch daily OHLCV for one NSE symbol from Yahoo Finance.

    Parameters
    ----------
    symbol : str
        NSE symbol (``.NS`` suffix is added automatically).
    start_date, end_date : str
        Inclusive date range in ``YYYY-MM-DD`` format.
    retries : int
        Number of attempts before giving up.
    backoff : float
        Base seconds for exponential backoff between retries.

    Returns
    -------
    pd.DataFrame | None
        Daily OHLCV indexed by date, or ``None`` if unavailable.
    """
    ticker = f"{symbol}.NS"
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,      # adjusted OHLCV
                progress=False,
                threads=False,
                timeout=30,            # seconds; default 10s is too tight under load
            )
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            # Newer yfinance versions return MultiIndex columns even for 1 ticker.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as exc:  # noqa: BLE001
            log.debug("fetch %s attempt %d failed: %s", ticker, attempt, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    log.warning("Giving up on %s after %d attempts", ticker, retries)
    return None


# --------------------------------------------------------------------------- #
# Weekly resampling
# --------------------------------------------------------------------------- #

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly bars (week ending Friday).

    If Friday is an exchange holiday the bar naturally ends on the last
    trading day of that week because resampling bins actual observations.
    """
    weekly = df.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    # Drop weeks with no trading at all.
    weekly = weekly.dropna(subset=["Close"])
    weekly["Volume"] = weekly["Volume"].fillna(0)
    return weekly


# --------------------------------------------------------------------------- #
# Indicators (manual Wilder-style formulas; no external TA dependency)
# --------------------------------------------------------------------------- #

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing == EMA with alpha = 1/period (adjust=False)."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # If average loss is exactly 0, RSI is 100 by convention.
    return rsi.fillna(100.0).where(avg_loss.notna(), np.nan)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = _wilder_smooth(tr, period).replace(0.0, np.nan)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return _wilder_smooth(dx, period)


def compute_indicators(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator and lagged columns needed by the filters.

    Everything is computed causally from data up to and including each week,
    so there is no look-ahead bias.
    """
    df = df_weekly.copy()
    close, volume = df["Close"], df["Volume"]

    df["rsi_14"] = _rsi(close, RSI_PERIOD)
    df["adx_14"] = _adx(df["High"], df["Low"], close, ADX_PERIOD)
    df["vol_sma_20"] = volume.rolling(VOL_SMA_PERIOD).mean()
    df["ema_20"] = close.ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean()
    df["ema_100"] = close.ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean()

    df["close_lag1"] = close.shift(1)      # 1 week ago Close
    df["vol_lag1"] = volume.shift(1)       # 1 week ago Volume
    df["vol_lag2"] = volume.shift(2)       # 2 weeks ago Volume
    return df


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

def apply_filters(df_weekly: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: True where ALL 9 weekly conditions hold.

    Condition 2 is implemented exactly as specified even though the units
    differ (volume vs. price).
    """
    df = df_weekly
    conditions = [
        df["rsi_14"] >= 60,                              # 1
        df["Volume"] > df["close_lag1"],                 # 2 (exactly as written)
        df["adx_14"] > 15,                               # 3
        df["vol_lag1"] >= df["vol_lag2"] * 0.5,          # 4
        df["Volume"] > df["vol_sma_20"],                 # 5
        df["Close"] > df["close_lag1"],                  # 6
        df["Close"] > df["ema_20"] * 0.7,                # 7
        df["Volume"] > VOLUME_THRESHOLD,                 # 8
        df["ema_20"] > df["ema_100"],                    # 9
    ]
    mask = conditions[0]
    for cond in conditions[1:]:
        mask &= cond
    return mask.fillna(False)


# --------------------------------------------------------------------------- #
# Screening engine
# --------------------------------------------------------------------------- #

def _process_symbol(
    symbol: str,
    stock_name: str,
    start_date: str,
    end_date: str,
    as_of_date: pd.Timestamp | None,
    latest_only: bool,
) -> tuple[list[dict], str]:
    """Run the full pipeline for one symbol.

    Returns (list of passing rows, status) where status is one of
    'ok', 'no_data', 'insufficient_history'.
    """
    daily = fetch_ohlcv(symbol, start_date, end_date)
    if daily is None or daily.empty:
        return [], "no_data"

    weekly = compute_indicators(to_weekly(daily))
    if len(weekly) < MIN_WEEKLY_CANDLES:
        return [], "insufficient_history"

    passes = apply_filters(weekly)
    if as_of_date is not None:
        # Evaluate only the latest week on/before the requested date.
        eligible = weekly.index[weekly.index <= as_of_date]
        if len(eligible) == 0:
            return [], "no_data"
        latest = eligible.max()
        passes = passes & (weekly.index == latest)
    elif latest_only:
        # Default: evaluate only the most recent week (the run date).
        passes = passes & (weekly.index == weekly.index.max())

    rows = [
        {"date": ts.strftime("%Y-%m-%d"), "stock_name": stock_name, "ticker": symbol}
        for ts in weekly.index[passes]
    ]
    return rows, "ok"


def run_screen(
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    as_of_date: str | None = None,
    workers: int = 8,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Screen every symbol in ``universe`` and return passing rows.

    Parameters
    ----------
    universe : pd.DataFrame
        Columns ``symbol`` and ``stock_name``.
    start_date, end_date : str
        Daily data range used to build weekly bars.
    as_of_date : str | None
        If given, evaluate only the latest week on/before this date.
    workers : int
        Thread-pool size for parallel downloads.
    latest_only : bool
        When True (default) and ``as_of_date`` is not given, only the most
        recent week per symbol is evaluated. Set False for full history.
    """
    as_of_ts = pd.Timestamp(as_of_date) if as_of_date else None
    results: list[dict] = []
    fetched = enough_history = 0
    total = len(universe)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_symbol,
                row.symbol,
                row.stock_name,
                start_date,
                end_date,
                as_of_ts,
                latest_only,
            ): row.symbol
            for row in universe.itertuples(index=False)
        }
        for done_count, fut in enumerate(as_completed(futures), start=1):
            rows, status = fut.result()
            if status != "no_data":
                fetched += 1
            if status == "ok":
                enough_history += 1
                results.extend(rows)
            if done_count % 50 == 0 or done_count == total:
                log.info("Progress: %d/%d symbols processed", done_count, total)

    # ------------------------- validation summary ------------------------- #
    log.info("Validation counts:")
    log.info("  universe size ................ %d", total)
    log.info("  symbols with data fetched .... %d", fetched)
    log.info("  symbols with >= %d wk bars ... %d", MIN_WEEKLY_CANDLES, enough_history)
    log.info("  total passing rows ........... %d", len(results))

    out = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if not out.empty:
        out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Debug helper
# --------------------------------------------------------------------------- #

def debug_symbol(symbol: str, start_date: str, end_date: str) -> None:
    """Print an end-to-end inspection of one symbol (last 15 weekly bars)."""
    print(f"\n=== DEBUG: {symbol} ===")
    daily = fetch_ohlcv(symbol, start_date, end_date)
    if daily is None:
        print("No data returned.")
        return
    print(f"Daily bars fetched   : {len(daily)} "
          f"({daily.index[0].date()} -> {daily.index[-1].date()})")

    weekly = compute_indicators(to_weekly(daily))
    weekly["pass"] = apply_filters(weekly)
    print(f"Weekly bars          : {len(weekly)} "
          f"(min required: {MIN_WEEKLY_CANDLES})")

    cols = ["Close", "Volume", "rsi_14", "adx_14", "vol_sma_20",
            "ema_20", "ema_100", "pass"]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(weekly[cols].tail(15).round(2))
    n_pass = int(weekly["pass"].sum())
    print(f"Passing weeks        : {n_pass}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weekly NSE stock screener (RSI/ADX/volume conditions)."
    )
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD. Default: 3 years before the "
                             "end date (warm-up buffer so EMA(100) and the Wilder "
                             "indicators are fully matured).")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (default: today).")
    parser.add_argument("--as-of", default=None, dest="as_of",
                        help="Evaluate only the latest week on/before this date.")
    parser.add_argument("--all-history", action="store_true", dest="all_history",
                        help="Report every passing week in the range (backtest). "
                             "Default: only the latest week (the run date).")
    parser.add_argument("--output", default=None,
                        help="Output CSV path. Default: timestamped name like "
                             "results/nse_screen_results_16_October2026_144500.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only screen the first N symbols (for testing).")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel download threads (default: 5). Lower this "
                             "if Yahoo rate-limits you; raise it to go faster.")
    parser.add_argument("--debug", default=None, metavar="SYMBOL",
                        help="Debug one symbol end-to-end and exit.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose (DEBUG) logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Default start: 3-year rolling buffer before the end date.
    if args.start:
        start_date = args.start
    else:
        start_date = (
            (pd.Timestamp(args.end) - DateOffset(years=3)).strftime("%Y-%m-%d")
        )
    log.info("Data window: %s -> %s", start_date, args.end)

    if args.debug:
        debug_symbol(args.debug.strip().upper(), start_date, args.end)
        return 0

    universe = get_nse_universe()
    if args.limit:
        universe = universe.head(args.limit)
        log.info("--limit %d: screening a subset of %d symbols", args.limit, len(universe))

    results = run_screen(
        universe,
        start_date=start_date,
        end_date=args.end,
        as_of_date=args.as_of,
        workers=args.workers,
        latest_only=not args.all_history,
    )

    print("\n===== SCREEN RESULTS =====")
    if results.empty:
        print("(no stocks passed all conditions)")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(results.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%d_%B%Y_%H%M%S")
        out_path = Path("results") / f"nse_screen_results_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    log.info("Saved %d row(s) to %s", len(results), out_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
