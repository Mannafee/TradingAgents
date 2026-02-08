import yfinance as yf
import numpy as np
import pandas as pd
from typing import List

from .models import PortfolioRequest, CandidateStock, RiskTolerance
from .universes import UNIVERSES


class StockScreener:
    """Lightweight stock screening using yfinance data only (no LLM calls)."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def screen(self, request: PortfolioRequest) -> List[CandidateStock]:
        """Screen stocks and return top candidates sorted by composite score."""
        universe_name = request.stock_universe
        tickers = self._get_universe(universe_name)

        if not tickers:
            raise ValueError(f"Unknown stock universe: {universe_name}")

        period = self.config.get("portfolio_screening_period", "1mo")
        data = self._fetch_bulk_data(tickers, period)

        if data.empty:
            raise RuntimeError("Failed to fetch stock data for screening")

        candidates = []
        for ticker in tickers:
            try:
                ticker_data = self._get_ticker_data(data, ticker)
                if ticker_data is None or len(ticker_data) < 5:
                    continue

                price = float(ticker_data["Close"].iloc[-1])

                # Skip stocks that cost more than the budget
                if price > request.budget_usd:
                    continue

                momentum = self._calculate_momentum(ticker_data, request.time_horizon_days)
                volatility = self._calculate_volatility(ticker_data)
                volume = self._calculate_volume_score(ticker_data)

                composite = self._composite_score(
                    momentum, volatility, volume, request.risk_tolerance
                )

                info = self._get_ticker_info(ticker)

                candidates.append(CandidateStock(
                    ticker=ticker,
                    name=info.get("shortName", ticker),
                    sector=info.get("sector", "Unknown"),
                    price=price,
                    market_cap=info.get("marketCap", 0),
                    momentum_score=momentum,
                    volatility_score=volatility,
                    volume_score=volume,
                    composite_score=composite,
                ))
            except Exception:
                continue

        # Sort by composite score descending
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        # Return top N
        return candidates[:request.max_candidates]

    def _get_universe(self, universe_name: str) -> List[str]:
        if universe_name in UNIVERSES:
            return UNIVERSES[universe_name]
        return []

    def _fetch_bulk_data(self, tickers: List[str], period: str = "1mo") -> pd.DataFrame:
        """Fetch price data for all tickers in one yfinance call."""
        try:
            data = yf.download(
                tickers=" ".join(tickers),
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            return data
        except Exception:
            return pd.DataFrame()

    def _get_ticker_data(self, bulk_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Extract a single ticker's data from bulk download."""
        try:
            if isinstance(bulk_data.columns, pd.MultiIndex):
                # Multi-ticker download: columns are (metric, ticker)
                ticker_close = bulk_data[("Close", ticker)].dropna()
                ticker_volume = bulk_data[("Volume", ticker)].dropna()
                ticker_high = bulk_data[("High", ticker)].dropna()
                ticker_low = bulk_data[("Low", ticker)].dropna()
                df = pd.DataFrame({
                    "Close": ticker_close,
                    "Volume": ticker_volume,
                    "High": ticker_high,
                    "Low": ticker_low,
                })
                return df.dropna()
            else:
                # Single ticker download
                return bulk_data[["Close", "Volume", "High", "Low"]].dropna()
        except (KeyError, TypeError):
            return None

    def _calculate_momentum(self, data: pd.DataFrame, time_horizon_days: int) -> float:
        """Momentum = % return over lookback period."""
        lookback = min(time_horizon_days, len(data) - 1, 20)
        if lookback < 1:
            return 0.0
        current = float(data["Close"].iloc[-1])
        past = float(data["Close"].iloc[-1 - lookback])
        if past == 0:
            return 0.0
        return (current - past) / past

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Annualized volatility from daily returns."""
        returns = data["Close"].pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(252))

    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Recent volume vs average volume ratio."""
        if len(data) < 10:
            return 1.0
        recent_vol = float(data["Volume"].iloc[-5:].mean())
        avg_vol = float(data["Volume"].mean())
        if avg_vol == 0:
            return 1.0
        return recent_vol / avg_vol

    def _composite_score(
        self,
        momentum: float,
        volatility: float,
        volume: float,
        risk_tolerance: RiskTolerance,
    ) -> float:
        """Weighted composite score. Weights vary by risk tolerance."""
        if risk_tolerance == RiskTolerance.AGGRESSIVE:
            w_momentum = 0.50
            w_volatility = 0.20  # Higher volatility = more opportunity
            w_volume = 0.30
            vol_factor = volatility  # Reward volatility
        elif risk_tolerance == RiskTolerance.CONSERVATIVE:
            w_momentum = 0.30
            w_volatility = 0.40
            w_volume = 0.30
            vol_factor = -volatility  # Penalize volatility
        else:  # MODERATE
            w_momentum = 0.40
            w_volatility = 0.30
            w_volume = 0.30
            vol_factor = 0.0  # Neutral on volatility

        return (w_momentum * momentum) + (w_volatility * vol_factor) + (w_volume * (volume - 1.0))

    def _get_ticker_info(self, ticker: str) -> dict:
        """Get basic ticker info (name, sector). Cached by yfinance."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return {
                "shortName": info.get("shortName", ticker),
                "sector": info.get("sector", "Unknown"),
                "marketCap": info.get("marketCap", 0),
            }
        except Exception:
            return {"shortName": ticker, "sector": "Unknown", "marketCap": 0}
