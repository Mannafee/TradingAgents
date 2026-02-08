import json
import math
from typing import List

from tradingagents.client.codex import CodexClient
from tradingagents.client.types import Message

from .models import (
    PortfolioRequest,
    PortfolioResult,
    CandidateStock,
    StockAllocation,
    RiskTolerance,
)


class PortfolioAllocator:
    """Synthesizes deep analysis results into a portfolio allocation plan."""

    def __init__(self, client: CodexClient, model: str):
        self.client = client
        self.model = model

    async def allocate(
        self,
        request: PortfolioRequest,
        analyzed_candidates: List[CandidateStock],
    ) -> PortfolioResult:
        """Generate portfolio allocation from analyzed candidates."""
        prompt = self._build_allocation_prompt(request, analyzed_candidates)
        messages = [Message(role="user", content=prompt)]
        response = await self.client.complete(messages, model=self.model)
        return self._parse_allocation_response(
            response.text, request, analyzed_candidates
        )

    def _build_allocation_prompt(
        self,
        request: PortfolioRequest,
        candidates: List[CandidateStock],
    ) -> str:
        # Build per-stock analysis summaries
        stock_sections = []
        for c in candidates:
            section = f"""### {c.ticker} ({c.name})
- Sector: {c.sector}
- Current Price: ${c.price:.2f}
- Screening Score: {c.composite_score:.4f} (momentum={c.momentum_score:.4f}, volatility={c.volatility_score:.4f}, volume={c.volume_score:.4f})
- Deep Analysis Signal: **{c.signal}**
"""
            if c.final_state:
                if c.final_state.get("market_report"):
                    market = c.final_state["market_report"]
                    if len(market) > 1500:
                        market = market[:1500] + "..."
                    section += f"\n**Market Analysis:**\n{market}\n"

                if c.final_state.get("final_trade_decision"):
                    decision = c.final_state["final_trade_decision"]
                    if len(decision) > 2000:
                        decision = decision[:2000] + "..."
                    section += f"\n**Final Trade Decision:**\n{decision}\n"

            stock_sections.append(section)

        stocks_text = "\n---\n".join(stock_sections)

        cash_reserve_guidance = {
            RiskTolerance.CONSERVATIVE: "Reserve 15-25% as cash for safety.",
            RiskTolerance.MODERATE: "Reserve 5-15% as cash.",
            RiskTolerance.AGGRESSIVE: "Reserve 0-5% as cash, maximize investment.",
        }

        from datetime import datetime, timedelta
        try:
            start_date = datetime.strptime(request.analysis_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            start_date = datetime.now()
        end_date = start_date + timedelta(days=request.time_horizon_days)

        prompt = f"""You are a portfolio allocation expert. Create an optimal investment plan.

## Investment Parameters
- **Budget**: ${request.budget_usd:.2f} USD (originally {request.budget} {request.currency})
- **Time Horizon**: {request.time_horizon_days} days ({request.analysis_date} to {end_date.strftime('%Y-%m-%d')})
- **Risk Tolerance**: {request.risk_tolerance.value}
- **Goal**: {request.goal}
- **Cash Reserve**: {cash_reserve_guidance[request.risk_tolerance]}

## Analyzed Stocks

{stocks_text}

## Instructions

Based on the deep analysis above, create a portfolio allocation. You MUST respond with valid JSON matching this exact structure:

```json
{{
  "allocations": [
    {{
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "action": "BUY",
      "allocation_pct": 40.0,
      "allocation_amount": 43.40,
      "shares": 1,
      "entry_price_target": 235.00,
      "exit_price_target": 245.00,
      "stop_loss_price": 228.00,
      "entry_timing": "Monday Feb 10, at market open",
      "exit_timing": "Friday Feb 14, before market close",
      "rationale": "Strong bullish signal with positive momentum..."
    }}
  ],
  "cash_reserved": 10.85,
  "portfolio_summary": "Brief summary of the portfolio strategy...",
  "risk_assessment": "Key risks and mitigation...",
  "execution_plan": "Step-by-step execution instructions:\\n1. Monday: ...\\n2. During the week: ...\\n3. Friday: ..."
}}
```

Rules:
1. Only allocate to stocks with BUY signals. Skip SELL signals entirely.
2. HOLD signals may receive small allocations only if the analysis is borderline bullish.
3. Number of shares must be whole numbers. Calculate: shares = floor(allocation_amount / current_price).
4. allocation_amount = budget * allocation_pct / 100. Ensure total allocations + cash = budget.
5. Entry price targets should be near current price (use support levels from market analysis if available).
6. Exit price targets should be realistic for the {request.time_horizon_days}-day horizon (use resistance levels if available).
7. Stop-loss should be set based on support levels or ATR from the market analysis.
8. Be specific about timing: use actual dates and "market open" / "market close" / "limit order at $X".

Respond ONLY with the JSON object, no other text."""

        return prompt

    def _parse_allocation_response(
        self,
        response_text: str,
        request: PortfolioRequest,
        candidates: List[CandidateStock],
    ) -> PortfolioResult:
        """Parse the LLM's allocation response into structured data."""
        json_text = response_text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return self._fallback_allocation(request, candidates)

        allocations = []
        for alloc_data in data.get("allocations", []):
            allocations.append(StockAllocation(
                ticker=alloc_data.get("ticker", ""),
                name=alloc_data.get("name", ""),
                action=alloc_data.get("action", "BUY"),
                allocation_pct=float(alloc_data.get("allocation_pct", 0)),
                allocation_amount=float(alloc_data.get("allocation_amount", 0)),
                shares=int(alloc_data.get("shares", 0)),
                entry_price_target=float(alloc_data.get("entry_price_target", 0)),
                exit_price_target=float(alloc_data.get("exit_price_target", 0)),
                stop_loss_price=float(alloc_data.get("stop_loss_price", 0)),
                entry_timing=alloc_data.get("entry_timing", ""),
                exit_timing=alloc_data.get("exit_timing", ""),
                rationale=alloc_data.get("rationale", ""),
            ))

        total_invested = sum(a.allocation_amount for a in allocations)

        return PortfolioResult(
            request=request,
            candidates_screened=len(candidates),
            candidates_analyzed=candidates,
            allocations=allocations,
            total_invested=total_invested,
            cash_reserved=float(data.get("cash_reserved", request.budget_usd - total_invested)),
            portfolio_summary=data.get("portfolio_summary", ""),
            risk_assessment=data.get("risk_assessment", ""),
            execution_plan=data.get("execution_plan", ""),
        )

    def _fallback_allocation(
        self,
        request: PortfolioRequest,
        candidates: List[CandidateStock],
    ) -> PortfolioResult:
        """Simple equal-weight allocation for BUY signals when LLM parsing fails."""
        buy_candidates = [c for c in candidates if c.signal == "BUY"]
        if not buy_candidates:
            buy_candidates = candidates[:1]

        cash_pct = 0.10 if request.risk_tolerance == RiskTolerance.MODERATE else (
            0.20 if request.risk_tolerance == RiskTolerance.CONSERVATIVE else 0.05
        )
        investable = request.budget_usd * (1 - cash_pct)
        per_stock = investable / len(buy_candidates) if buy_candidates else 0

        allocations = []
        for c in buy_candidates:
            shares = math.floor(per_stock / c.price) if c.price > 0 else 0
            if shares < 1:
                continue
            amount = shares * c.price
            allocations.append(StockAllocation(
                ticker=c.ticker,
                name=c.name,
                action="BUY",
                allocation_pct=round((amount / request.budget_usd) * 100, 1),
                allocation_amount=round(amount, 2),
                shares=shares,
                entry_price_target=round(c.price, 2),
                exit_price_target=round(c.price * 1.03, 2),
                stop_loss_price=round(c.price * 0.97, 2),
                entry_timing="At market open",
                exit_timing=f"By end of {request.time_horizon_days}-day horizon",
                rationale=f"Deep analysis signal: {c.signal}. Composite score: {c.composite_score:.4f}",
            ))

        total_invested = sum(a.allocation_amount for a in allocations)
        return PortfolioResult(
            request=request,
            candidates_screened=len(candidates),
            candidates_analyzed=candidates,
            allocations=allocations,
            total_invested=total_invested,
            cash_reserved=round(request.budget_usd - total_invested, 2),
            portfolio_summary="Equal-weight allocation across BUY-rated stocks (fallback mode).",
            risk_assessment="Standard risk assessment not available (LLM parsing failed).",
            execution_plan="Buy all positions at market open, sell at end of time horizon.",
        )
