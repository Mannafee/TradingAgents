"""TradingAgents Web UI (Streamlit).

Run:
    python -m streamlit run webui/app.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import queue
import threading
import time
import traceback
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from tradingagents.client.auth import CodexOAuth
from tradingagents.client.codex import CodexClient
from tradingagents.config import DEFAULT_CONFIG
from tradingagents.portfolio import PortfolioAdvisor, PortfolioRequest, RiskTolerance


ANALYST_OPTIONS = ["market", "social", "news", "fundamentals"]

ANALYST_LABELS = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

PHASE_LABELS = {
    "queued": "Queued",
    "currency": "Currency Conversion",
    "screening": "Screening Universe",
    "screening_done": "Screening Complete",
    "analyzing": "Deep Analysis",
    "analysts": "Analysts",
    "investment_debate": "Investment Debate",
    "research_manager": "Research Manager",
    "trader": "Trader",
    "risk_debate": "Risk Debate",
    "risk_manager": "Risk Manager",
    "signal": "Signal Extraction",
    "allocating": "Portfolio Allocation",
    "completed": "Completed",
    "error": "Error",
}

AGENT_LABELS = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
    "bull": "Bull Researcher",
    "bear": "Bear Researcher",
    "research_manager": "Research Manager",
    "trader": "Trader",
    "aggressive": "Aggressive Analyst",
    "conservative": "Conservative Analyst",
    "neutral": "Neutral Analyst",
    "risk_manager": "Portfolio Manager",
    "result": "Result",
    "signal": "Signal",
    "error": "Error",
}

AGENT_AVATARS = {
    "market": "📈",
    "social": "🗣️",
    "news": "📰",
    "fundamentals": "📊",
    "bull": "🐂",
    "bear": "🐻",
    "research_manager": "🧠",
    "trader": "💼",
    "aggressive": "⚡",
    "conservative": "🛡️",
    "neutral": "⚖️",
    "risk_manager": "🏛️",
    "result": "✅",
    "signal": "🎯",
    "error": "❌",
}

TIME_HORIZON_OPTIONS = [
    ("1 day (intraday/swing)", 1),
    ("1 week", 7),
    ("2 weeks", 14),
    ("1 month", 30),
    ("3 months", 90),
]

RISK_OPTIONS = [
    ("Conservative - Prioritize capital preservation, lower risk", "conservative"),
    ("Moderate - Balanced risk and reward", "moderate"),
    ("Aggressive - Maximize potential returns, higher risk", "aggressive"),
]

UNIVERSE_OPTIONS = [
    ("S&P 500 Top 50 - Largest, most liquid US stocks (recommended)", "sp500_top50"),
    ("NASDAQ Top 30 - Tech-focused growth stocks", "nasdaq_top30"),
    ("Popular ETFs - Diversified index & sector ETFs", "etf_popular"),
]

CANDIDATE_OPTIONS = [
    ("2 stocks - Faster, lower cost (~4-10 min, ~40 LLM calls)", 2),
    ("3 stocks - Good balance (recommended, ~6-15 min, ~60 LLM calls)", 3),
    ("4 stocks - More diversified (~8-20 min, ~80 LLM calls)", 4),
    ("5 stocks - Maximum coverage (~10-25 min, ~100 LLM calls)", 5),
]

DEPTH_OPTIONS = [
    ("Shallow - Quick research, fewer debate rounds", 1),
    ("Medium - Balanced depth and speed", 3),
    ("Deep - Comprehensive multi-round debate", 5),
]


def _clip(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _phase_label(value: str) -> str:
    return PHASE_LABELS.get(value, value.replace("_", " ").title())


def _now_ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _init_state():
    state = st.session_state
    if "initialized" in state:
        return

    state.initialized = True
    state.running = False
    state.event_queue = queue.Queue()
    state.worker_thread = None
    state.started_at = None
    state.global_phase = "queued"
    state.global_message = ""
    state.candidates: List[Dict[str, Any]] = []
    state.ticker_state: Dict[str, Dict[str, str]] = {}
    state.ticker_chats: Dict[str, List[Dict[str, str]]] = {}
    state.timeline: List[Dict[str, str]] = []
    state.result = None
    state.error = ""
    state.traceback = ""


def _clear_run_state():
    st.session_state.running = False
    st.session_state.event_queue = queue.Queue()
    st.session_state.worker_thread = None
    st.session_state.started_at = None
    st.session_state.global_phase = "queued"
    st.session_state.global_message = ""
    st.session_state.candidates = []
    st.session_state.ticker_state = {}
    st.session_state.ticker_chats = {}
    st.session_state.timeline = []
    st.session_state.result = None
    st.session_state.error = ""
    st.session_state.traceback = ""


def _build_client(auth_mode: str) -> CodexClient:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")

    if auth_mode == "env":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use OAuth mode or export the API key.")
        return CodexClient(access_token=api_key)

    auth = CodexOAuth()
    result = auth.load_or_login()
    os.environ["OPENAI_API_KEY"] = result.access_token
    if result.account_id:
        os.environ["CHATGPT_ACCOUNT_ID"] = result.account_id
    os.environ["CODEX_OAUTH_MODE"] = "1"
    return CodexClient(access_token=result.access_token, base_url=result.base_url)


def _worker_run(params: Dict[str, Any], event_queue: queue.Queue):
    client = None
    try:
        def push_event(payload: Dict[str, Any]):
            event_queue.put(payload)

        client = _build_client(params["auth_mode"])

        cfg = DEFAULT_CONFIG.copy()
        cfg["quick_think_llm"] = params["quick_model"]
        cfg["deep_think_llm"] = params["deep_model"]
        cfg["max_debate_rounds"] = params["research_depth"]
        cfg["max_risk_discuss_rounds"] = params["research_depth"]

        advisor = PortfolioAdvisor(config=cfg, client=client)
        request = PortfolioRequest(
            budget=params["budget"],
            currency=params["currency"],
            time_horizon_days=params["time_horizon_days"],
            risk_tolerance=RiskTolerance(params["risk_tolerance"]),
            goal="maximize profit",
            stock_universe=params["stock_universe"],
            max_candidates=params["max_candidates"],
            analysis_date=params["analysis_date"],
        )

        def progress_callback(phase: str, message: str):
            push_event({
                "type": "progress",
                "ts": _now_ts(),
                "phase": phase,
                "message": message,
            })

        def activity_callback(event: Dict[str, Any]):
            push_event({
                "type": "activity",
                "ts": _now_ts(),
                **event,
            })

        result = asyncio.run(
            advisor.advise(
                request=request,
                selected_analysts=params["analysts"],
                progress_callback=progress_callback,
                activity_callback=activity_callback,
            )
        )
        push_event({
            "type": "done",
            "ts": _now_ts(),
            "result": result,
        })

    except Exception as exc:
        push_event({
            "type": "error",
            "ts": _now_ts(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    finally:
        if client is not None:
            try:
                asyncio.run(client.close())
            except Exception:
                pass
        push_event({"type": "finished", "ts": _now_ts()})


def _ensure_ticker_entry(ticker: str):
    if ticker not in st.session_state.ticker_state:
        st.session_state.ticker_state[ticker] = {
            "phase": "queued",
            "agent": "Waiting",
            "signal": "",
            "last_update": _now_ts(),
        }
    if ticker not in st.session_state.ticker_chats:
        st.session_state.ticker_chats[ticker] = []


def _append_chat(ticker: str, agent_key: str, content: str, ts: str):
    _ensure_ticker_entry(ticker)
    label = AGENT_LABELS.get(agent_key, agent_key.replace("_", " ").title())
    st.session_state.ticker_chats[ticker].append(
        {
            "ts": ts,
            "agent_key": agent_key,
            "agent": label,
            "content": _clip(content),
        }
    )


def _drain_event_queue():
    q = st.session_state.event_queue
    while True:
        try:
            ev = q.get_nowait()
        except queue.Empty:
            break

        ev_type = ev.get("type", "")
        ts = ev.get("ts", _now_ts())

        if ev_type == "progress":
            phase = ev.get("phase", "queued")
            msg = ev.get("message", "")
            st.session_state.global_phase = "completed" if phase == "done" else phase
            st.session_state.global_message = msg
            st.session_state.timeline.append(
                {"ts": ts, "actor": "System", "message": f"{_phase_label(phase)}: {msg}"}
            )
            continue

        if ev_type == "activity":
            kind = ev.get("kind", "")
            ticker = ev.get("ticker", "SYS")

            if kind == "candidates_selected":
                st.session_state.candidates = ev.get("candidates", [])
                tickers = ev.get("tickers", [])
                for t in tickers:
                    _ensure_ticker_entry(t)
                st.session_state.timeline.append(
                    {
                        "ts": ts,
                        "actor": "Screener",
                        "message": f"Selected candidates: {', '.join(tickers)}",
                    }
                )
                continue

            if ticker != "SYS":
                _ensure_ticker_entry(ticker)

            if kind == "phase":
                phase = ev.get("phase", "queued")
                st.session_state.ticker_state[ticker]["phase"] = phase
                st.session_state.ticker_state[ticker]["last_update"] = ts
                st.session_state.timeline.append(
                    {"ts": ts, "actor": ticker, "message": f"Phase -> {_phase_label(phase)}"}
                )
                continue

            if kind == "agent_start":
                agent_key = ev.get("agent", "")
                st.session_state.ticker_state[ticker]["agent"] = AGENT_LABELS.get(
                    agent_key, agent_key
                )
                st.session_state.ticker_state[ticker]["last_update"] = ts
                continue

            if kind == "agent_output":
                agent_key = ev.get("agent", "")
                content = ev.get("content", "")
                st.session_state.ticker_state[ticker]["agent"] = AGENT_LABELS.get(
                    agent_key, agent_key
                )
                st.session_state.ticker_state[ticker]["last_update"] = ts
                _append_chat(ticker, agent_key, content, ts)
                continue

            if kind == "signal":
                signal = ev.get("signal", "")
                st.session_state.ticker_state[ticker]["signal"] = signal
                st.session_state.ticker_state[ticker]["phase"] = "signal"
                st.session_state.ticker_state[ticker]["last_update"] = ts
                _append_chat(ticker, "signal", f"Signal extracted: {signal}", ts)
                continue

            if kind == "ticker_completed":
                signal = ev.get("signal", "")
                summary = ev.get("summary", "")
                st.session_state.ticker_state[ticker]["phase"] = "completed"
                st.session_state.ticker_state[ticker]["agent"] = "Done"
                st.session_state.ticker_state[ticker]["signal"] = signal
                st.session_state.ticker_state[ticker]["last_update"] = ts
                _append_chat(ticker, "result", f"Final signal: {signal}\n\n{summary}", ts)
                continue

            if kind == "ticker_error":
                err = ev.get("error", "Unknown error")
                st.session_state.ticker_state[ticker]["phase"] = "error"
                st.session_state.ticker_state[ticker]["agent"] = "Error"
                st.session_state.ticker_state[ticker]["signal"] = "ERROR"
                st.session_state.ticker_state[ticker]["last_update"] = ts
                _append_chat(ticker, "error", err, ts)
                continue

            continue

        if ev_type == "done":
            st.session_state.result = ev.get("result")
            st.session_state.global_phase = "completed"
            st.session_state.timeline.append(
                {"ts": ts, "actor": "System", "message": "Portfolio analysis completed"}
            )
            continue

        if ev_type == "error":
            st.session_state.error = ev.get("error", "Unknown error")
            st.session_state.traceback = ev.get("traceback", "")
            st.session_state.global_phase = "error"
            st.session_state.timeline.append(
                {"ts": ts, "actor": "System", "message": st.session_state.error}
            )
            continue

        if ev_type == "finished":
            st.session_state.running = False
            continue


def _render_css():
    st.markdown(
        """
        <style>
          .main .block-container {padding-top: 1.2rem; padding-bottom: 1.4rem;}
          .stMetric {background: #f7f9fc; border: 1px solid #e8edf5; border-radius: 14px; padding: 8px 10px;}
          .portfolio-header {font-size: 1.7rem; font-weight: 700; letter-spacing: 0.2px;}
          .portfolio-sub {color: #526173; margin-top: -2px; margin-bottom: 8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_top():
    st.markdown('<div class="portfolio-header">TradingAgents Portfolio Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="portfolio-sub">Live multi-agent analysis with per-stock debate chats and portfolio allocation.</div>',
        unsafe_allow_html=True,
    )


def _render_sidebar():
    with st.sidebar:
        st.header("Run Configuration")

        auth_mode = st.radio(
            "Authentication",
            options=["env", "oauth"],
            format_func=lambda x: "Use OPENAI_API_KEY (Recommended)" if x == "env" else "Login via Codex OAuth",
        )
        budget = st.number_input("Budget", min_value=100.0, value=10000.0, step=500.0)
        currency = st.selectbox("Currency", options=["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"], index=0)
        time_horizon_label = st.selectbox(
            "Time Horizon",
            options=[label for label, _ in TIME_HORIZON_OPTIONS],
            index=2,
        )
        time_horizon_days = dict(TIME_HORIZON_OPTIONS)[time_horizon_label]

        risk_label = st.selectbox(
            "Risk Tolerance",
            options=[label for label, _ in RISK_OPTIONS],
            index=1,
        )
        risk_tolerance = dict(RISK_OPTIONS)[risk_label]

        universe_label = st.selectbox(
            "Stock Universe",
            options=[label for label, _ in UNIVERSE_OPTIONS],
            index=0,
        )
        stock_universe = dict(UNIVERSE_OPTIONS)[universe_label]

        candidate_label = st.selectbox(
            "Analysis Depth (Number of Stocks)",
            options=[label for label, _ in CANDIDATE_OPTIONS],
            index=1,
        )
        max_candidates = dict(CANDIDATE_OPTIONS)[candidate_label]

        analysts = st.multiselect(
            "Analysts",
            options=ANALYST_OPTIONS,
            default=["market", "social", "news", "fundamentals"],
            format_func=lambda x: ANALYST_LABELS.get(x, x),
        )
        analysis_date = st.date_input("Analysis Date", value=dt.date.today()).strftime("%Y-%m-%d")

        st.markdown("---")
        quick_model = st.selectbox(
            "Quick Model",
            options=[
                "gpt-5.1-codex-mini",
                "gpt-5.2",
                "gpt-5.2-codex",
                "gpt-5.3-codex",
            ],
            index=0,
        )
        deep_model = st.selectbox(
            "Deep Model",
            options=[
                "gpt-5.2-codex",
                "gpt-5.3-codex",
                "gpt-5.1-codex-max",
                "gpt-5.2",
                "gpt-5.1-codex-mini",
            ],
            index=0,
        )
        depth_label = st.selectbox(
            "Research Depth",
            options=[label for label, _ in DEPTH_OPTIONS],
            index=0,
        )
        research_depth = dict(DEPTH_OPTIONS)[depth_label]

        disabled = st.session_state.running
        start_clicked = st.button(
            "Start Portfolio Analysis",
            type="primary",
            width="stretch",
            disabled=disabled,
        )

    if not start_clicked:
        return

    if not analysts:
        st.error("Select at least one analyst.")
        return

    _clear_run_state()
    st.session_state.running = True
    st.session_state.started_at = time.time()
    st.session_state.global_phase = "queued"
    st.session_state.global_message = "Initializing portfolio analysis..."

    params = {
        "auth_mode": auth_mode,
        "budget": float(budget),
        "currency": currency,
        "time_horizon_days": int(time_horizon_days),
        "risk_tolerance": risk_tolerance,
        "stock_universe": stock_universe,
        "max_candidates": int(max_candidates),
        "analysis_date": analysis_date,
        "analysts": analysts,
        "quick_model": quick_model,
        "deep_model": deep_model,
        "research_depth": int(research_depth),
    }

    event_q = queue.Queue()
    st.session_state.event_queue = event_q
    worker = threading.Thread(target=_worker_run, args=(params, event_q), daemon=True)
    st.session_state.worker_thread = worker
    worker.start()


def _render_status_metrics():
    running = st.session_state.running
    phase = _phase_label(st.session_state.global_phase)
    msg = st.session_state.global_message or "Waiting..."
    elapsed = 0
    if st.session_state.started_at:
        elapsed = int(time.time() - st.session_state.started_at)

    total = len(st.session_state.ticker_state)
    completed = 0
    failed = 0
    for _, item in st.session_state.ticker_state.items():
        if item["phase"] == "completed":
            completed += 1
        if item["phase"] == "error":
            failed += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "Running" if running else "Idle")
    c2.metric("Global Phase", phase)
    c3.metric("Tickers Done", f"{completed}/{total or '-'}")
    c4.metric("Elapsed", f"{elapsed//60:02d}:{elapsed%60:02d}")
    st.caption(msg)

    if failed > 0:
        st.warning(f"{failed} ticker(s) failed during analysis.")


def _render_progress_and_timeline():
    left, right = st.columns([1.4, 1.0], gap="large")

    with left:
        st.subheader("Per-Stock Progress")
        rows = []
        if st.session_state.candidates:
            for c in st.session_state.candidates:
                ticker = c["ticker"]
                t = st.session_state.ticker_state.get(
                    ticker, {"phase": "queued", "agent": "Waiting", "signal": "", "last_update": "-"}
                )
                rows.append(
                    {
                        "Ticker": ticker,
                        "Name": c.get("name", ""),
                        "Price": f"${c.get('price', 0):.2f}",
                        "Score": f"{c.get('composite_score', 0):.4f}",
                        "Phase": _phase_label(t["phase"]),
                        "Current Agent": t["agent"],
                        "Signal": t["signal"] or "-",
                        "Last Update": t["last_update"],
                    }
                )
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("Start a run to populate stock progress.")

    with right:
        st.subheader("Run Timeline")
        if st.session_state.timeline:
            for ev in reversed(st.session_state.timeline[-14:]):
                st.markdown(f"`{ev['ts']}` **{ev['actor']}**  \n{ev['message']}")
                st.divider()
        else:
            st.info("No events yet.")


def _render_chat_tabs():
    st.subheader("Agent Chat By Stock")
    tickers = []
    if st.session_state.candidates:
        tickers = [c["ticker"] for c in st.session_state.candidates]
    else:
        tickers = list(st.session_state.ticker_chats.keys())

    if not tickers:
        st.info("Chats will appear once ticker analysis begins.")
        return

    tabs = st.tabs(tickers)
    for ticker, tab in zip(tickers, tabs):
        with tab:
            messages = st.session_state.ticker_chats.get(ticker, [])
            if not messages:
                st.caption("No messages yet for this ticker.")
                continue

            for m in messages[-24:]:
                avatar = AGENT_AVATARS.get(m["agent_key"], "💬")
                with st.chat_message("assistant", avatar=avatar):
                    st.markdown(f"**{m['agent']}** · `{m['ts']}`")
                    st.markdown(m["content"])


def _render_final_result():
    result = st.session_state.result
    if not result:
        return

    st.subheader("Portfolio Output")
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates Analyzed", len(result.candidates_analyzed))
    c2.metric("Allocations", len(result.allocations))
    c3.metric("Cash Reserved", f"${result.cash_reserved:.2f}")

    if result.allocations:
        rows = []
        for a in result.allocations:
            rows.append(
                {
                    "Ticker": a.ticker,
                    "Action": a.action,
                    "Allocation %": f"{a.allocation_pct:.1f}%",
                    "Amount": f"${a.allocation_amount:.2f}",
                    "Shares": a.shares,
                    "Entry Target": f"${a.entry_price_target:.2f}",
                    "Stop Loss": f"${a.stop_loss_price:.2f}",
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

    if result.execution_plan:
        with st.expander("Execution Plan", expanded=False):
            st.markdown(result.execution_plan)
    if result.risk_assessment:
        with st.expander("Risk Assessment", expanded=False):
            st.markdown(result.risk_assessment)
    if result.portfolio_summary:
        with st.expander("Portfolio Summary", expanded=True):
            st.markdown(result.portfolio_summary)


def main():
    st.set_page_config(
        page_title="TradingAgents Portfolio Studio",
        page_icon="📊",
        layout="wide",
    )
    _render_css()
    _init_state()
    _drain_event_queue()

    _render_top()
    _render_sidebar()
    _render_status_metrics()

    if st.session_state.error:
        st.error(st.session_state.error)
        with st.expander("Traceback", expanded=False):
            st.code(st.session_state.traceback)

    _render_progress_and_timeline()
    _render_chat_tabs()
    _render_final_result()

    if st.session_state.running:
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
