import datetime
from typing import List, Optional, Callable

import yfinance as yf

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client

from .models import PortfolioRequest, PortfolioResult, CandidateStock
from .screener import StockScreener
from .allocator import PortfolioAllocator


class PortfolioAdvisor:
    """Top-level orchestrator for the portfolio advisory pipeline."""

    def __init__(self, config: dict = None, callbacks: list = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.callbacks = callbacks or []

        # Initialize the deep thinking LLM for the allocator
        llm_kwargs = self._get_provider_kwargs()
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        self.deep_thinking_llm = deep_client.get_llm()

        self.screener = StockScreener(config=self.config)
        self.allocator = PortfolioAllocator(self.deep_thinking_llm)

    def _get_provider_kwargs(self) -> dict:
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()
        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level
        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        return kwargs

    def advise(
        self,
        request: PortfolioRequest,
        selected_analysts: list = None,
        progress_callback: Optional[Callable] = None,
    ) -> PortfolioResult:
        """Run the full portfolio advisory pipeline.

        Args:
            request: The portfolio request with budget, time horizon, etc.
            selected_analysts: List of analyst types to use (default: market, news, fundamentals).
            progress_callback: Optional callback(phase, message) for progress updates.
        """
        if selected_analysts is None:
            selected_analysts = ["market", "news", "fundamentals"]

        # Set analysis date to today if not specified
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

        # Step 2: Deep analysis per candidate
        for i, candidate in enumerate(candidates):
            if progress_callback:
                progress_callback(
                    "analyzing",
                    f"Deep analyzing {candidate.ticker} ({i + 1}/{len(candidates)})...",
                )

            try:
                graph = TradingAgentsGraph(
                    selected_analysts=selected_analysts,
                    config=self.config,
                    callbacks=self.callbacks,
                )
                final_state, signal = graph.propagate(
                    candidate.ticker, request.analysis_date
                )
                candidate.signal = signal
                candidate.final_state = final_state
                candidate.analysis_summary = self._extract_summary(final_state)
            except Exception as e:
                candidate.signal = "ERROR"
                candidate.analysis_summary = f"Analysis failed: {str(e)}"

        # Step 3: Allocate portfolio
        if progress_callback:
            progress_callback("allocating", "Generating portfolio allocation plan...")

        # Filter out failed analyses
        valid_candidates = [c for c in candidates if c.signal != "ERROR"]
        if not valid_candidates:
            return PortfolioResult(
                request=request,
                candidates_analyzed=candidates,
                portfolio_summary="All stock analyses failed. Please try again.",
            )

        result = self.allocator.allocate(request, valid_candidates)

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

        # Fallback: common rates (approximate)
        fallback_rates = {
            "EUR": 1.08,
            "GBP": 1.27,
            "JPY": 0.0067,
            "CHF": 1.13,
            "CAD": 0.74,
            "AUD": 0.65,
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
