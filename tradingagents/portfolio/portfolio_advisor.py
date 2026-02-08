import asyncio
import datetime
from typing import List, Optional, Callable

import yfinance as yf

from tradingagents.client.codex import CodexClient
from tradingagents.config import DEFAULT_CONFIG
from tradingagents.pipeline.runner import TradingPipeline

from .models import PortfolioRequest, PortfolioResult, CandidateStock
from .screener import StockScreener
from .allocator import PortfolioAllocator


class PortfolioAdvisor:
    """Top-level orchestrator for the portfolio advisory pipeline."""

    def __init__(self, config: dict = None, client: CodexClient = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.client = client or CodexClient()
        self.screener = StockScreener(config=self.config)
        self.allocator = PortfolioAllocator(
            self.client, self.config["deep_think_llm"],
        )

    async def advise(
        self,
        request: PortfolioRequest,
        selected_analysts: list = None,
        progress_callback: Optional[Callable] = None,
        activity_callback: Optional[Callable] = None,
    ) -> PortfolioResult:
        """Run the full portfolio advisory pipeline.

        Args:
            request: The portfolio request with budget, time horizon, etc.
            selected_analysts: List of analyst types to use.
            progress_callback: Optional callback(phase, message) for progress updates.
            activity_callback: Optional callback(event_dict) for detailed ticker/agent events.
        """
        if selected_analysts is None:
            selected_analysts = ["market", "news", "fundamentals"]

        if not request.analysis_date:
            request.analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Step 0: Convert currency to USD
        if progress_callback:
            progress_callback("currency", f"Converting {request.currency} to USD...")
        request.budget_usd = self._convert_currency(request.budget, request.currency)

        # Step 1: Screen stocks
        if progress_callback:
            progress_callback("screening", f"Screening {request.stock_universe} stocks...")
        candidates = self.screener.screen(request)

        if not candidates:
            return PortfolioResult(
                request=request,
                portfolio_summary="No suitable stocks found within budget constraints.",
            )

        if progress_callback:
            tickers_str = ", ".join(c.ticker for c in candidates)
            progress_callback("screening_done", f"Selected {len(candidates)} candidates: {tickers_str}")
        if activity_callback:
            activity_callback({
                "kind": "candidates_selected",
                "tickers": [c.ticker for c in candidates],
                "candidates": [
                    {
                        "ticker": c.ticker,
                        "name": c.name,
                        "sector": c.sector,
                        "price": c.price,
                        "composite_score": c.composite_score,
                    }
                    for c in candidates
                ],
            })

        # Step 2: Deep analysis per candidate (parallel)
        if progress_callback:
            progress_callback("analyzing", f"Deep analyzing {len(candidates)} stocks in parallel...")

        async def _analyze_one(candidate: CandidateStock) -> CandidateStock:
            def _on_pipeline_event(event: dict):
                if activity_callback:
                    activity_callback({"ticker": candidate.ticker, **event})

            try:
                pipeline = TradingPipeline(
                    selected_analysts=selected_analysts,
                    config=self.config,
                    client=self.client,
                    on_event=_on_pipeline_event,
                )
                state, signal = await pipeline.run(
                    candidate.ticker, request.analysis_date,
                )
                candidate.signal = signal
                candidate.final_state = state.to_dict()
                candidate.analysis_summary = self._extract_summary(state.to_dict())
                if activity_callback:
                    activity_callback({
                        "kind": "ticker_completed",
                        "ticker": candidate.ticker,
                        "signal": signal,
                        "summary": candidate.analysis_summary[:600],
                    })
            except Exception as e:
                import traceback
                candidate.signal = "ERROR"
                candidate.analysis_summary = f"Analysis failed: {str(e)}"
                print(f"[ERROR] Analysis of {candidate.ticker} failed: {e}")
                traceback.print_exc()
                if activity_callback:
                    activity_callback({
                        "kind": "ticker_error",
                        "ticker": candidate.ticker,
                        "error": str(e),
                    })
            return candidate

        analyzed = await asyncio.gather(
            *[_analyze_one(c) for c in candidates]
        )

        # Step 3: Allocate portfolio
        if progress_callback:
            progress_callback("allocating", "Generating portfolio allocation plan...")

        valid_candidates = [c for c in analyzed if c.signal != "ERROR"]
        if not valid_candidates:
            return PortfolioResult(
                request=request,
                candidates_analyzed=list(analyzed),
                portfolio_summary="All stock analyses failed. Please try again.",
            )

        result = await self.allocator.allocate(request, valid_candidates)

        if progress_callback:
            progress_callback("done", "Portfolio plan complete!")

        return result

    def _convert_currency(self, amount: float, currency: str) -> float:
        """Convert currency to USD using yfinance FX rates."""
        currency = currency.upper()
        if currency == "USD":
            return amount

        try:
            pair = f"{currency}USD=X"
            ticker = yf.Ticker(pair)
            hist = ticker.history(period="1d")
            if not hist.empty:
                rate = float(hist["Close"].iloc[-1])
                return amount * rate
        except Exception:
            pass

        fallback_rates = {
            "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067,
            "CHF": 1.13, "CAD": 0.74, "AUD": 0.65,
        }
        rate = fallback_rates.get(currency, 1.0)
        return amount * rate

    def _extract_summary(self, final_state: dict) -> str:
        """Extract a concise summary from a full analysis state."""
        parts = []
        if final_state.get("final_trade_decision"):
            decision = final_state["final_trade_decision"]
            parts.append(decision[:1000] if len(decision) > 1000 else decision)
        elif final_state.get("trader_investment_plan"):
            plan = final_state["trader_investment_plan"]
            parts.append(plan[:1000] if len(plan) > 1000 else plan)
        return "\n".join(parts) if parts else "No summary available."

    async def close(self):
        """Close the HTTP client."""
        await self.client.close()
