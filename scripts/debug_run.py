#!/usr/bin/env python3
"""Non-interactive debug runner for TradingAgents.

Usage examples:
  python scripts/debug_run.py --mode pipeline --tickers NOW,CSCO --analysts market,social
  python scripts/debug_run.py --mode portfolio --max-candidates 2 --analysts market,social
"""

import argparse
import asyncio
import datetime as dt
import os
import traceback
from typing import List, Tuple

from dotenv import load_dotenv

from tradingagents.client.auth import CodexOAuth
from tradingagents.client.codex import CodexClient
from tradingagents.config import DEFAULT_CONFIG
from tradingagents.pipeline.runner import TradingPipeline
from tradingagents.portfolio import PortfolioAdvisor, PortfolioRequest, RiskTolerance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TradingAgents debug runner")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "portfolio"],
        default=os.getenv("DEBUG_MODE", "pipeline"),
        help="Run single-ticker pipeline checks or full portfolio flow",
    )
    parser.add_argument(
        "--tickers",
        default=os.getenv("DEBUG_TICKERS", "NOW,CSCO"),
        help="Comma-separated tickers for pipeline mode",
    )
    parser.add_argument(
        "--analysis-date",
        default=os.getenv("DEBUG_ANALYSIS_DATE", dt.datetime.now().strftime("%Y-%m-%d")),
        help="Analysis date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--analysts",
        default=os.getenv("DEBUG_ANALYSTS", "market,social"),
        help="Comma-separated analysts (market,social,news,fundamentals)",
    )
    parser.add_argument(
        "--quick-model",
        default=os.getenv("DEBUG_QUICK_MODEL", "gpt-5.1-codex-mini"),
        help="Quick-think model",
    )
    parser.add_argument(
        "--deep-model",
        default=os.getenv("DEBUG_DEEP_MODEL", "gpt-5.2-codex"),
        help="Deep-think model",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=float(os.getenv("DEBUG_BUDGET", "10000")),
        help="Budget for portfolio mode",
    )
    parser.add_argument(
        "--currency",
        default=os.getenv("DEBUG_CURRENCY", "USD"),
        help="Currency for portfolio mode",
    )
    parser.add_argument(
        "--time-horizon-days",
        type=int,
        default=int(os.getenv("DEBUG_TIME_HORIZON_DAYS", "14")),
        help="Time horizon for portfolio mode",
    )
    parser.add_argument(
        "--risk-tolerance",
        choices=["conservative", "moderate", "aggressive"],
        default=os.getenv("DEBUG_RISK_TOLERANCE", "moderate"),
        help="Risk tolerance for portfolio mode",
    )
    parser.add_argument(
        "--stock-universe",
        default=os.getenv("DEBUG_STOCK_UNIVERSE", "sp500_top50"),
        help="Stock universe for portfolio mode",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=int(os.getenv("DEBUG_MAX_CANDIDATES", "3")),
        help="Max candidates for portfolio mode",
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_client() -> CodexClient:
    # If API credentials are already set, use them directly.
    if os.getenv("OPENAI_API_KEY"):
        return CodexClient()

    # Otherwise use cached OAuth refresh/login flow.
    auth = CodexOAuth()
    result = auth.load_or_login()
    os.environ["OPENAI_API_KEY"] = result.access_token
    if result.account_id:
        os.environ["CHATGPT_ACCOUNT_ID"] = result.account_id
    os.environ["CODEX_OAUTH_MODE"] = "1"
    return CodexClient(access_token=result.access_token, base_url=result.base_url)


def build_config(args: argparse.Namespace) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    cfg["quick_think_llm"] = args.quick_model
    cfg["deep_think_llm"] = args.deep_model
    cfg["max_debate_rounds"] = 1
    cfg["max_risk_discuss_rounds"] = 1
    return cfg


async def run_pipeline_mode(
    client: CodexClient,
    cfg: dict,
    tickers: List[str],
    analysts: List[str],
    analysis_date: str,
) -> Tuple[int, int]:
    ok = 0
    failed = 0

    async def _run_one(ticker: str):
        nonlocal ok, failed
        try:
            pipeline = TradingPipeline(
                selected_analysts=analysts,
                config=cfg,
                client=client,
            )
            state, signal = await pipeline.run(ticker, analysis_date)
            print(f"[OK] {ticker}: signal={signal}")
            if state.market_report:
                print(f"  market_report chars={len(state.market_report)}")
            if state.sentiment_report:
                print(f"  sentiment_report chars={len(state.sentiment_report)}")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {ticker}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    await asyncio.gather(*[_run_one(t) for t in tickers])
    return ok, failed


async def run_portfolio_mode(
    client: CodexClient,
    cfg: dict,
    args: argparse.Namespace,
    analysts: List[str],
) -> Tuple[int, int]:
    request = PortfolioRequest(
        budget=args.budget,
        currency=args.currency,
        time_horizon_days=args.time_horizon_days,
        risk_tolerance=RiskTolerance(args.risk_tolerance),
        goal="maximize profit",
        stock_universe=args.stock_universe,
        max_candidates=args.max_candidates,
        analysis_date=args.analysis_date,
    )

    advisor = PortfolioAdvisor(config=cfg, client=client)

    def progress(phase: str, message: str):
        now = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {phase}: {message}")

    try:
        result = await advisor.advise(
            request=request,
            selected_analysts=analysts,
            progress_callback=progress,
        )
    except Exception as exc:
        print(f"[ERROR] portfolio: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 0, 1

    print(
        f"[OK] portfolio: analyzed={len(result.candidates_analyzed)} "
        f"allocations={len(result.allocations)}"
    )
    return 1, 0


async def main():
    load_dotenv()
    args = parse_args()
    tickers = _parse_csv(args.tickers)
    analysts = _parse_csv(args.analysts)

    print("=== TradingAgents Debug Run ===")
    print(f"mode={args.mode}")
    print(f"analysis_date={args.analysis_date}")
    print(f"analysts={analysts}")
    if args.mode == "pipeline":
        print(f"tickers={tickers}")

    client = build_client()
    cfg = build_config(args)

    try:
        if args.mode == "pipeline":
            ok, failed = await run_pipeline_mode(
                client=client,
                cfg=cfg,
                tickers=tickers,
                analysts=analysts,
                analysis_date=args.analysis_date,
            )
        else:
            ok, failed = await run_portfolio_mode(
                client=client,
                cfg=cfg,
                args=args,
                analysts=analysts,
            )
    finally:
        await client.close()

    print(f"=== Summary: ok={ok}, failed={failed} ===")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
