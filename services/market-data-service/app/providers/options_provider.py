import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)


class OptionsDataError(Exception):
    """Raised when the options data provider cannot retrieve data."""


@dataclass
class OptionQuote:
    symbol: str
    strike: float
    expiry: datetime
    option_type: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float


@dataclass
class OptionsChainPayload:
    symbol: str
    underlying_price: float
    expiries: List[datetime]
    calls: List[OptionQuote]
    puts: List[OptionQuote]


class OptionsDataProvider:
    """Fetch options chains using external data sources (currently yfinance)."""

    def __init__(self, max_expiries: int = 5) -> None:
        self.max_expiries = max_expiries

    async def get_chain(self, symbol: str, expiry_filter: Optional[str] = None) -> OptionsChainPayload:
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
        except Exception as exc:  # pragma: no cover - yfinance network failure path
            logger.error("Failed to initialise yfinance ticker for %s: %s", symbol, exc)
            raise OptionsDataError(str(exc)) from exc

        try:
            options_dates = await asyncio.to_thread(lambda: list(ticker.options))
        except Exception as exc:  # pragma: no cover - yfinance network failure path
            logger.error("Failed to fetch options expiries for %s: %s", symbol, exc)
            raise OptionsDataError(str(exc)) from exc

        if not options_dates:
            raise OptionsDataError(f"No options expiries available for {symbol}")

        if expiry_filter and expiry_filter in options_dates:
            target_dates = [expiry_filter]
        else:
            target_dates = options_dates[: self.max_expiries]

        underlying_price = await self._resolve_underlying_price(ticker)
        calls: List[OptionQuote] = []
        puts: List[OptionQuote] = []
        expiries: List[datetime] = []

        for expiry in target_dates:
            try:
                chain = await asyncio.to_thread(ticker.option_chain, expiry)
            except Exception as exc:  # pragma: no cover - yfinance failure path
                logger.warning("Skipping expiry %s for %s due to error: %s", expiry, symbol, exc)
                continue

            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            expiries.append(expiry_dt)

            calls.extend(self._rows_to_quotes(symbol, chain.calls, "call", expiry_dt))
            puts.extend(self._rows_to_quotes(symbol, chain.puts, "put", expiry_dt))

        if not calls and not puts:
            raise OptionsDataError(f"No options quotes retrieved for {symbol}")

        return OptionsChainPayload(
            symbol=symbol.upper(),
            underlying_price=underlying_price,
            expiries=expiries,
            calls=calls,
            puts=puts,
        )

    async def _resolve_underlying_price(self, ticker: "yf.Ticker") -> float:
        def _get_price() -> Optional[float]:
            info = getattr(ticker, "info", {}) or {}
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            if price:
                return float(price)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
            return None

        price = await asyncio.to_thread(_get_price)
        if price is None:
            raise OptionsDataError("Could not determine underlying price")
        return price

    def _rows_to_quotes(self, symbol: str, frame, option_type: str, expiry: datetime) -> List[OptionQuote]:
        quotes: List[OptionQuote] = []
        if frame is None or frame.empty:
            return quotes

        for _, row in frame.iterrows():
            bid = float(row.get("bid", 0.0) or 0.0)
            ask = float(row.get("ask", 0.0) or 0.0)
            last = float(row.get("lastPrice", 0.0) or row.get("last", 0.0) or 0.0)
            volume = int(row.get("volume") or 0)
            open_interest = int(row.get("openInterest") or 0)
            imp_vol = float(row.get("impliedVolatility", 0.0) or 0.0)
            strike = float(row.get("strike", 0.0) or 0.0)

            option_symbol = (
                f"{symbol.upper()}_{expiry.strftime('%Y%m%d')}"
                f"{'C' if option_type == 'call' else 'P'}{strike:.2f}"
            )

            quotes.append(
                OptionQuote(
                    symbol=option_symbol,
                    strike=strike,
                    expiry=expiry,
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    last=last,
                    volume=volume,
                    open_interest=open_interest,
                    implied_volatility=imp_vol,
                )
            )

        return quotes


options_data_provider = OptionsDataProvider()
