"""Pipeline state dataclasses - replaces LangGraph TypedDict states."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class InvestDebate:
    history: str = ""
    bull_history: str = ""
    bear_history: str = ""
    current_response: str = ""
    judge_decision: str = ""
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "history": self.history,
            "bull_history": self.bull_history,
            "bear_history": self.bear_history,
            "current_response": self.current_response,
            "judge_decision": self.judge_decision,
            "count": self.count,
        }


@dataclass
class RiskDebate:
    history: str = ""
    aggressive_history: str = ""
    conservative_history: str = ""
    neutral_history: str = ""
    latest_speaker: str = ""
    current_aggressive_response: str = ""
    current_conservative_response: str = ""
    current_neutral_response: str = ""
    judge_decision: str = ""
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "history": self.history,
            "aggressive_history": self.aggressive_history,
            "conservative_history": self.conservative_history,
            "neutral_history": self.neutral_history,
            "latest_speaker": self.latest_speaker,
            "current_aggressive_response": self.current_aggressive_response,
            "current_conservative_response": self.current_conservative_response,
            "current_neutral_response": self.current_neutral_response,
            "judge_decision": self.judge_decision,
            "count": self.count,
        }


@dataclass
class PipelineState:
    """Full pipeline state, passed between agents."""
    company_of_interest: str = ""
    trade_date: str = ""

    # Analyst reports
    market_report: str = ""
    sentiment_report: str = ""
    news_report: str = ""
    fundamentals_report: str = ""

    # Research debate
    investment_debate_state: InvestDebate = field(default_factory=InvestDebate)
    investment_plan: str = ""

    # Trader
    trader_investment_plan: str = ""

    # Risk debate
    risk_debate_state: RiskDebate = field(default_factory=RiskDebate)
    final_trade_decision: str = ""

    def to_dict(self) -> dict:
        """Convert to dict (for CLI display, saving, etc.)."""
        return {
            "company_of_interest": self.company_of_interest,
            "trade_date": self.trade_date,
            "market_report": self.market_report,
            "sentiment_report": self.sentiment_report,
            "news_report": self.news_report,
            "fundamentals_report": self.fundamentals_report,
            "investment_debate_state": self.investment_debate_state.to_dict(),
            "investment_plan": self.investment_plan,
            "trader_investment_plan": self.trader_investment_plan,
            "risk_debate_state": self.risk_debate_state.to_dict(),
            "final_trade_decision": self.final_trade_decision,
        }
