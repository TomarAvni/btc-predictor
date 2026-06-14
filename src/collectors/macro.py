"""Macro economic data collector: DXY, S&P 500, Gold, M2 money supply, Treasury yields."""

import pandas as pd
import yfinance as yf

from src.collectors import BaseCollector
from src.utils.cache import DiskCache
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Ticker symbols for macro assets
TICKERS = {
    "dxy": "DX-Y.NYB",       # US Dollar Index
    "sp500": "^GSPC",         # S&P 500
    "gold": "GC=F",           # Gold futures
    "us10y": "^TNX",          # 10-year Treasury yield
    "us02y": "^IRX",          # 2-year Treasury (approx via 13-week)
    "vix": "^VIX",            # Volatility index
}

# Global M2 proxy: US M2 isn't available on yfinance, but we can track
# related ETFs or use Federal Reserve data
M2_PROXY_TICKERS = {
    "tlt": "TLT",             # 20+ year Treasury ETF (inverse of yields)
    "tip": "TIP",             # Treasury Inflation-Protected Securities
}


class MacroCollector(BaseCollector):
    """Collects macro economic indicators that correlate with BTC."""

    name = "macro"
    tier = 4
    update_interval_seconds = 86400

    def __init__(self):
        self.cache = DiskCache()

    async def get_macro_data(self, period: str = "1y") -> pd.DataFrame:
        """Fetch all macro indicators for the specified period.

        Args:
            period: yfinance period string ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        """
        cache_key = f"macro_data_{period}"
        cached = self.cache.get(cache_key, max_age_seconds=43200)  # 12h cache
        if cached:
            df = pd.DataFrame(cached)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], utc=True)
                df.set_index("Date", inplace=True)
            return df

        try:
            frames = {}
            all_tickers = {**TICKERS, **M2_PROXY_TICKERS}

            for name, ticker in all_tickers.items():
                try:
                    data = yf.download(ticker, period=period, interval="1d", progress=False)
                    if not data.empty:
                        frames[name] = data["Close"].rename(name)
                except Exception as e:
                    logger.warning(f"Failed to fetch {name} ({ticker}): {e}")

            if not frames:
                return pd.DataFrame()

            combined = pd.concat(frames.values(), axis=1)
            combined.index = combined.index.tz_localize("UTC") if combined.index.tz is None else combined.index
            combined.sort_index(inplace=True)

            self.cache.set(cache_key, combined.reset_index().to_dict(orient="records"))
            return combined

        except Exception as e:
            logger.warning(f"Macro data fetch failed: {e}")
            return pd.DataFrame()

    def compute_correlations(self, btc_prices: pd.Series, macro_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Compute rolling correlation between BTC and macro indicators.

        Helps identify which macro factors are currently driving BTC.
        """
        if btc_prices.empty or macro_df.empty:
            return pd.DataFrame()

        btc_returns = btc_prices.pct_change()
        correlations = {}

        for col in macro_df.columns:
            macro_returns = macro_df[col].pct_change()
            aligned = pd.concat([btc_returns, macro_returns], axis=1).dropna()
            if len(aligned) > window:
                corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
                correlations[f"corr_{col}"] = corr

        return pd.DataFrame(correlations)

    def interpret_dxy(self, dxy_change_pct: float) -> str:
        """Interpret DXY movement for BTC (typically inverse correlation)."""
        if dxy_change_pct < -0.5:
            return "dxy_weakening_bullish_btc"
        elif dxy_change_pct < -0.1:
            return "dxy_slightly_weak_neutral_positive"
        elif dxy_change_pct < 0.1:
            return "dxy_flat_neutral"
        elif dxy_change_pct < 0.5:
            return "dxy_slightly_strong_neutral_negative"
        else:
            return "dxy_strengthening_bearish_btc"

    async def _collect(self) -> pd.DataFrame:
        """Collect latest macro snapshot."""
        df = await self.get_macro_data(period="1mo")
        if df.empty:
            return df
        # Return just the latest row
        return df.tail(1)

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        """Collect full macro history."""
        df = await self.get_macro_data(period="max")
        if df.empty:
            return df
        mask = df.index >= pd.Timestamp(start, tz="UTC")
        if end:
            mask &= df.index <= pd.Timestamp(end, tz="UTC")
        return df[mask]
