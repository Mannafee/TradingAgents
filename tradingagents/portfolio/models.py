from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PortfolioRequest:
    budget: float
    currency: str = "USD"
    time_horizon_days: int = 7
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    goal: str = "maximize profit"
    stock_universe: str = "sp500_top50"
    max_candidates: int = 3
    analysis_date: str = ""
    budget_usd: float = 0.0


@dataclass
class CandidateStock:
    ticker: str
    name: str
    sector: str
    price: float
    market_cap: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    volume_score: float = 0.0
    composite_score: float = 0.0
    signal: str = ""
    final_state: Optional[Dict[str, Any]] = None
    analysis_summary: str = ""


@dataclass
class StockAllocation:
    ticker: str
    name: str
    action: str
    allocation_pct: float
    allocation_amount: float
    shares: int
    entry_price_target: float
    exit_price_target: float
    stop_loss_price: float
    entry_timing: str
    exit_timing: str
    rationale: str


@dataclass
class PortfolioResult:
    request: PortfolioRequest
    candidates_screened: int = 0
    candidates_analyzed: List[CandidateStock] = field(default_factory=list)
    allocations: List[StockAllocation] = field(default_factory=list)
    total_invested: float = 0.0
    cash_reserved: float = 0.0
    portfolio_summary: str = ""
    risk_assessment: str = ""
    execution_plan: str = ""
