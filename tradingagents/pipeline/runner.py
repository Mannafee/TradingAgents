"""Async pipeline orchestrator - replaces LangGraph StateGraph."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tradingagents.client.codex import CodexClient
from tradingagents.config import DEFAULT_CONFIG, set_config
from tradingagents.memory.bm25 import FinancialSituationMemory
from tradingagents.pipeline.state import PipelineState
from tradingagents.pipeline.signal import extract_signal
from tradingagents.pipeline.reflection import Reflector

from tradingagents.agents import (
    MarketAnalyst,
    SocialAnalyst,
    NewsAnalyst,
    FundamentalsAnalyst,
    BullResearcher,
    BearResearcher,
    ResearchManager,
    Trader,
    AggressiveAnalyst,
    ConservativeAnalyst,
    NeutralAnalyst,
    RiskManager,
)


# Map analyst type names to classes
ANALYST_MAP = {
    "market": MarketAnalyst,
    "social": SocialAnalyst,
    "news": NewsAnalyst,
    "fundamentals": FundamentalsAnalyst,
}


class TradingPipeline:
    """Async pipeline that replaces TradingAgentsGraph + LangGraph.

    Usage:
        pipeline = TradingPipeline()
        state, signal = await pipeline.run("AAPL", "2026-02-08")
    """

    def __init__(
        self,
        selected_analysts: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        client: Optional[CodexClient] = None,
        on_phase: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize pipeline.

        Args:
            selected_analysts: Which analysts to run. Defaults to all 4.
            config: Config dict. Merged with DEFAULT_CONFIG.
            client: Pre-configured CodexClient. If None, a new one is created.
            on_phase: Optional callback called with phase name as string.
            on_event: Optional callback called with structured event dicts.
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        set_config(self.config)

        self.selected_analysts = selected_analysts or [
            "market", "social", "news", "fundamentals",
        ]
        self.on_phase = on_phase
        self.on_event = on_event

        # Ensure cache dir exists
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)

        # Client
        self.client = client or CodexClient()

        # Models
        self.deep_model = self.config["deep_think_llm"]
        self.quick_model = self.config["quick_think_llm"]

        # Memories (one per component that learns)
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Reflector
        self.reflector = Reflector(self.client, self.quick_model)

        # State tracking (for reflection after run)
        self.curr_state: Optional[PipelineState] = None

    def _emit(self, phase: str):
        if self.on_phase:
            self.on_phase(phase)

    def _emit_event(self, kind: str, **payload):
        if self.on_event:
            self.on_event({"kind": kind, **payload})

    @staticmethod
    def _clip(text: str, max_chars: int = 600) -> str:
        if not text:
            return ""
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3] + "..."

    async def run(self, ticker: str, trade_date: str) -> Tuple[PipelineState, str]:
        """Run the full analysis pipeline.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            trade_date: Date string (e.g. "2026-02-08").

        Returns:
            (final_state, signal) where signal is "BUY"/"SELL"/"HOLD".
        """
        state = PipelineState(
            company_of_interest=ticker,
            trade_date=str(trade_date),
        )
        self._emit_event("pipeline_start", ticker=ticker, trade_date=str(trade_date))

        # Phase 1: Analysts (parallel)
        self._emit("analysts")
        self._emit_event("phase", phase="analysts")
        state = await self._run_analysts(state)

        # Phase 2: Investment debate (bull/bear, sequential rounds)
        self._emit("investment_debate")
        self._emit_event("phase", phase="investment_debate")
        debate_rounds = self.config.get("max_debate_rounds", 1)
        bull = BullResearcher(self.client, self.deep_model, self.bull_memory)
        bear = BearResearcher(self.client, self.deep_model, self.bear_memory)

        for round_idx in range(debate_rounds):
            state = await bull.run(state)
            self._emit_event(
                "agent_output",
                phase="investment_debate",
                agent="bull",
                round=round_idx + 1,
                content=self._clip(state.investment_debate_state.current_response),
            )
            state = await bear.run(state)
            self._emit_event(
                "agent_output",
                phase="investment_debate",
                agent="bear",
                round=round_idx + 1,
                content=self._clip(state.investment_debate_state.current_response),
            )

        # Phase 3: Research manager judges the debate
        self._emit("research_manager")
        self._emit_event("phase", phase="research_manager")
        research_mgr = ResearchManager(self.client, self.deep_model, self.invest_judge_memory)
        state = await research_mgr.run(state)
        self._emit_event(
            "agent_output",
            phase="research_manager",
            agent="research_manager",
            content=self._clip(state.investment_plan),
        )

        # Phase 4: Trader makes a plan
        self._emit("trader")
        self._emit_event("phase", phase="trader")
        trader = Trader(self.client, self.deep_model, self.trader_memory)
        state = await trader.run(state)
        self._emit_event(
            "agent_output",
            phase="trader",
            agent="trader",
            content=self._clip(state.trader_investment_plan),
        )

        # Phase 5: Risk debate (aggressive/conservative/neutral, sequential rounds)
        self._emit("risk_debate")
        self._emit_event("phase", phase="risk_debate")
        risk_rounds = self.config.get("max_risk_discuss_rounds", 1)
        aggressive = AggressiveAnalyst(self.client, self.deep_model)
        conservative = ConservativeAnalyst(self.client, self.deep_model)
        neutral = NeutralAnalyst(self.client, self.deep_model)

        for round_idx in range(risk_rounds):
            state = await aggressive.run(state)
            self._emit_event(
                "agent_output",
                phase="risk_debate",
                agent="aggressive",
                round=round_idx + 1,
                content=self._clip(state.risk_debate_state.current_aggressive_response),
            )
            state = await conservative.run(state)
            self._emit_event(
                "agent_output",
                phase="risk_debate",
                agent="conservative",
                round=round_idx + 1,
                content=self._clip(state.risk_debate_state.current_conservative_response),
            )
            state = await neutral.run(state)
            self._emit_event(
                "agent_output",
                phase="risk_debate",
                agent="neutral",
                round=round_idx + 1,
                content=self._clip(state.risk_debate_state.current_neutral_response),
            )

        # Phase 6: Risk manager judges
        self._emit("risk_manager")
        self._emit_event("phase", phase="risk_manager")
        risk_mgr = RiskManager(self.client, self.deep_model, self.risk_manager_memory)
        state = await risk_mgr.run(state)
        self._emit_event(
            "agent_output",
            phase="risk_manager",
            agent="risk_manager",
            content=self._clip(state.final_trade_decision),
        )

        # Phase 7: Extract signal
        self._emit("signal")
        self._emit_event("phase", phase="signal")
        signal = await extract_signal(
            self.client, self.quick_model, state.final_trade_decision,
        )
        self._emit_event("signal", signal=signal)

        # Store state for later reflection
        self.curr_state = state

        # Log state
        self._log_state(ticker, trade_date, state)

        return state, signal

    async def _run_analysts(self, state: PipelineState) -> PipelineState:
        """Run selected analysts in parallel using asyncio.gather."""
        analyst_names = []

        for name in self.selected_analysts:
            if name in ANALYST_MAP:
                analyst_names.append(name)

        if not analyst_names:
            return state

        # Each analyst gets its own copy of state to avoid write conflicts.
        copies = [PipelineState(
            company_of_interest=state.company_of_interest,
            trade_date=state.trade_date,
        ) for _ in analyst_names]

        async def _run_one(agent_cls_name, copy):
            cls = ANALYST_MAP[agent_cls_name]
            agent = cls(self.client, self.quick_model)
            self._emit_event(
                "agent_start",
                phase="analysts",
                agent=agent_cls_name,
            )
            result = await agent.run(copy)

            report = ""
            if agent_cls_name == "market":
                report = result.market_report
            elif agent_cls_name == "social":
                report = result.sentiment_report
            elif agent_cls_name == "news":
                report = result.news_report
            elif agent_cls_name == "fundamentals":
                report = result.fundamentals_report

            self._emit_event(
                "agent_output",
                phase="analysts",
                agent=agent_cls_name,
                content=self._clip(report),
            )
            return result

        results = await asyncio.gather(
            *[_run_one(name, copy) for name, copy in zip(analyst_names, copies)]
        )

        # Merge reports from each analyst back into the main state
        for name, result in zip(analyst_names, results):
            if name == "market":
                state.market_report = result.market_report
            elif name == "social":
                state.sentiment_report = result.sentiment_report
            elif name == "news":
                state.news_report = result.news_report
            elif name == "fundamentals":
                state.fundamentals_report = result.fundamentals_report

        return state

    def _log_state(self, ticker: str, trade_date: str, state: PipelineState):
        """Log the final state to a JSON file."""
        results_dir = self.config.get("results_dir", "./results")
        directory = Path(results_dir) / ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w") as f:
            json.dump({str(trade_date): state.to_dict()}, f, indent=4)

    async def reflect_and_remember(self, returns_losses):
        """Reflect on the last run and update all memories."""
        if self.curr_state is None:
            return

        await self.reflector.reflect_and_remember(
            self.curr_state,
            returns_losses,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.close()
