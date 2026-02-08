"""TradingAgents CLI - main entry point."""

from typing import Any, Dict, Optional
import asyncio
import datetime
import typer
from pathlib import Path
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich.rule import Rule
from rich import box
import time

from tradingagents.config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.prompts import (
    get_ticker,
    get_analysis_date,
    select_analysts,
    select_research_depth,
    select_quick_model,
    select_deep_model,
    run_codex_login,
    get_budget,
    get_time_horizon,
    select_risk_tolerance,
    select_stock_universe,
    select_max_candidates,
)
from cli.stats import StatsTracker
from cli.announcements import fetch_announcements, display_announcements

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,
)

# Ordered analyst keys
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

# Phase -> agents affected
PHASE_AGENTS = {
    "analysts": list(ANALYST_AGENT_NAMES.values()),
    "investment_debate": ["Bull Researcher", "Bear Researcher"],
    "research_manager": ["Research Manager"],
    "trader": ["Trader"],
    "risk_debate": ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"],
    "risk_manager": ["Portfolio Manager"],
    "signal": [],
}

ALL_AGENTS = [
    *ANALYST_AGENT_NAMES.values(),
    "Bull Researcher", "Bear Researcher", "Research Manager",
    "Trader",
    "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst",
    "Portfolio Manager",
]


def format_tokens(n):
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


# ─── Display helpers ───

def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3),
        Layout(name="analysis", ratio=5),
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2),
        Layout(name="messages", ratio=3),
    )
    return layout


def update_display(layout, agent_status, messages, current_report,
                   report_sections, stats_tracker, start_time, selected_analysts, config):
    # Header
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]Built by Tauric Research[/dim]",
            title="TradingAgents",
            border_style="green",
            padding=(0, 2),
            expand=True,
        )
    )

    # Progress table
    progress_table = Table(
        show_header=True, header_style="bold magenta", show_footer=False,
        box=box.SIMPLE_HEAD, padding=(0, 2), expand=True,
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    all_teams = {
        "Analyst Team": [n for k, n in ANALYST_AGENT_NAMES.items() if k in selected_analysts],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    for team, agents in all_teams.items():
        if not agents:
            continue
        first = True
        for agent in agents:
            status = agent_status.get(agent, "pending")
            if status == "in_progress":
                status_cell = Spinner("dots", text="[blue]in_progress[/blue]", style="bold cyan")
            else:
                color = {"pending": "yellow", "completed": "green", "error": "red"}.get(status, "white")
                status_cell = f"[{color}]{status}[/{color}]"
            progress_table.add_row(team if first else "", agent, status_cell)
            first = False
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages
    messages_table = Table(
        show_header=True, header_style="bold magenta", expand=True,
        box=box.MINIMAL, show_lines=True, padding=(0, 1),
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column("Content", style="white", no_wrap=False, ratio=1)

    for ts, mtype, content in reversed(list(messages)[-12:]):
        messages_table.add_row(ts, mtype, Text(content[:200], overflow="fold"))

    layout["messages"].update(
        Panel(messages_table, title="Messages", border_style="blue", padding=(1, 2))
    )

    # Analysis
    if current_report:
        layout["analysis"].update(
            Panel(Markdown(current_report), title="Current Report", border_style="green", padding=(1, 2))
        )
    else:
        layout["analysis"].update(
            Panel("[italic]Waiting for analysis report...[/italic]",
                  title="Current Report", border_style="green", padding=(1, 2))
        )

    # Footer
    agents_completed = sum(1 for s in agent_status.values() if s == "completed")
    agents_total = len(agent_status)
    reports_done = sum(1 for v in report_sections.values() if v is not None)
    reports_total = len(report_sections)

    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]
    if stats_tracker:
        stats = stats_tracker.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            stats_parts.append(f"Tokens: {format_tokens(stats['tokens_in'])}↑ {format_tokens(stats['tokens_out'])}↓")
            cost = StatsTracker.estimate_cost(stats["tokens_in"], stats["tokens_out"], config.get("deep_think_llm", ""))
            stats_parts.append(f"Cost: ${cost:.3f}")

    stats_parts.append(f"Reports: {reports_done}/{reports_total}")

    if start_time:
        elapsed = time.time() - start_time
        stats_parts.append(f"⏱ {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))
    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def save_report_to_disk(state_dict, ticker, save_path):
    """Save complete analysis report to disk."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    for key, name in [("market_report", "Market Analyst"), ("sentiment_report", "Social Analyst"),
                      ("news_report", "News Analyst"), ("fundamentals_report", "Fundamentals Analyst")]:
        if state_dict.get(key):
            analysts_dir.mkdir(exist_ok=True)
            (analysts_dir / f"{key}.md").write_text(state_dict[key])
            analyst_parts.append((name, state_dict[key]))
    if analyst_parts:
        content = "\n\n".join(f"### {n}\n{t}" for n, t in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # Research
    debate = state_dict.get("investment_debate_state", {})
    if debate:
        research_dir = save_path / "2_research"
        research_parts = []
        for key, name in [("bull_history", "Bull Researcher"), ("bear_history", "Bear Researcher"),
                          ("judge_decision", "Research Manager")]:
            if debate.get(key):
                research_dir.mkdir(exist_ok=True)
                (research_dir / f"{key}.md").write_text(debate[key])
                research_parts.append((name, debate[key]))
        if research_parts:
            content = "\n\n".join(f"### {n}\n{t}" for n, t in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # Trader
    if state_dict.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(state_dict["trader_investment_plan"])
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{state_dict['trader_investment_plan']}")

    # Risk
    risk = state_dict.get("risk_debate_state", {})
    if risk:
        risk_dir = save_path / "4_risk"
        risk_parts = []
        for key, name in [("aggressive_history", "Aggressive Analyst"), ("conservative_history", "Conservative Analyst"),
                          ("neutral_history", "Neutral Analyst")]:
            if risk.get(key):
                risk_dir.mkdir(exist_ok=True)
                (risk_dir / f"{key}.md").write_text(risk[key])
                risk_parts.append((name, risk[key]))
        if risk_parts:
            content = "\n\n".join(f"### {n}\n{t}" for n, t in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"])
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_path = save_path / "complete_report.md"
    report_path.write_text(header + "\n\n".join(sections))
    return report_path


def display_complete_report(state_dict):
    """Display the complete analysis report."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    for key, title in [("market_report", "Market Analyst"), ("sentiment_report", "Social Analyst"),
                       ("news_report", "News Analyst"), ("fundamentals_report", "Fundamentals Analyst")]:
        if state_dict.get(key):
            console.print(Panel(Markdown(state_dict[key]), title=title, border_style="blue", padding=(1, 2)))

    debate = state_dict.get("investment_debate_state", {})
    if debate:
        for key, title in [("bull_history", "Bull Researcher"), ("bear_history", "Bear Researcher"),
                           ("judge_decision", "Research Manager")]:
            if debate.get(key):
                console.print(Panel(Markdown(debate[key]), title=title, border_style="magenta", padding=(1, 2)))

    if state_dict.get("trader_investment_plan"):
        console.print(Panel(Markdown(state_dict["trader_investment_plan"]), title="Trader", border_style="yellow", padding=(1, 2)))

    risk = state_dict.get("risk_debate_state", {})
    if risk:
        for key, title in [("aggressive_history", "Aggressive Analyst"), ("conservative_history", "Conservative Analyst"),
                           ("neutral_history", "Neutral Analyst"), ("judge_decision", "Portfolio Manager")]:
            if risk.get(key):
                console.print(Panel(Markdown(risk[key]), title=title, border_style="red", padding=(1, 2)))


# ─── Single-stock analysis ───

def get_user_selections():
    """Get all user selections before starting analysis."""
    # Welcome
    try:
        with open("./cli/static/welcome.txt", "r") as f:
            welcome_ascii = f.read()
    except FileNotFoundError:
        welcome_ascii = ""

    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow:[/bold] I. Analysts → II. Research Debate → III. Trader → IV. Risk Debate → V. Portfolio Manager\n\n"
    welcome_content += "[dim]Built by Tauric Research[/dim]"
    console.print(Align.center(Panel(welcome_content, border_style="green", padding=(1, 2), title="TradingAgents")))
    console.print()

    announcements = fetch_announcements()
    display_announcements(console, announcements)

    def qbox(title, prompt, default=None):
        text = f"[bold]{title}[/bold]\n[dim]{prompt}[/dim]"
        if default:
            text += f"\n[dim]Default: {default}[/dim]"
        return Panel(text, border_style="blue", padding=(1, 2))

    # Step 1: Ticker
    console.print(qbox("Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"))
    ticker = get_ticker()

    # Step 2: Date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(qbox("Step 2: Analysis Date", "Enter the analysis date (YYYY-MM-DD)", default_date))
    analysis_date = get_analysis_date()

    # Step 3: Analysts
    console.print(qbox("Step 3: Analysts Team", "Select your LLM analyst agents"))
    analysts = select_analysts()
    console.print(f"[green]Selected analysts:[/green] {', '.join(a.value for a in analysts)}")

    # Step 4: Research depth
    console.print(qbox("Step 4: Research Depth", "Select your research depth level"))
    research_depth = select_research_depth()

    # Step 5: Login
    console.print(qbox("Step 5: Login", "Authenticate with ChatGPT"))
    client = run_codex_login()

    # Step 6: Models
    console.print(qbox("Step 6: Models", "Select your LLM models"))
    quick_model = select_quick_model()
    deep_model = select_deep_model()

    return {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "analysts": analysts,
        "research_depth": research_depth,
        "client": client,
        "quick_model": quick_model,
        "deep_model": deep_model,
    }


def run_analysis():
    """Run single-stock analysis with live display."""
    from collections import deque

    selections = get_user_selections()

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["quick_model"]
    config["deep_think_llm"] = selections["deep_model"]

    stats_tracker = StatsTracker()
    client = selections["client"]
    client.stats_tracker = stats_tracker

    selected_set = {a.value for a in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # Build agent status
    agent_status = {}
    for key in selected_analyst_keys:
        agent_status[ANALYST_AGENT_NAMES[key]] = "pending"
    for agent in ["Bull Researcher", "Bear Researcher", "Research Manager",
                  "Trader", "Aggressive Analyst", "Conservative Analyst",
                  "Neutral Analyst", "Portfolio Manager"]:
        agent_status[agent] = "pending"

    # Report sections to track
    report_keys = []
    for key in selected_analyst_keys:
        report_keys.append({"market": "market_report", "social": "sentiment_report",
                           "news": "news_report", "fundamentals": "fundamentals_report"}[key])
    report_keys.extend(["investment_plan", "trader_investment_plan", "final_trade_decision"])
    report_sections = {k: None for k in report_keys}

    messages = deque(maxlen=100)
    current_report = None
    start_time = time.time()

    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)

    layout = create_layout()

    # Phase callback - update agent statuses in real time
    def on_phase(phase):
        nonlocal current_report
        ts = datetime.datetime.now().strftime("%H:%M:%S")

        if phase == "analysts":
            for key in selected_analyst_keys:
                agent_status[ANALYST_AGENT_NAMES[key]] = "in_progress"
            messages.append((ts, "System", f"Running {len(selected_analyst_keys)} analysts in parallel..."))

        elif phase == "investment_debate":
            # Mark analysts complete
            for key in selected_analyst_keys:
                agent_status[ANALYST_AGENT_NAMES[key]] = "completed"
            agent_status["Bull Researcher"] = "in_progress"
            agent_status["Bear Researcher"] = "in_progress"
            messages.append((ts, "System", "Starting investment debate..."))

        elif phase == "research_manager":
            agent_status["Bull Researcher"] = "completed"
            agent_status["Bear Researcher"] = "completed"
            agent_status["Research Manager"] = "in_progress"
            messages.append((ts, "System", "Research manager evaluating debate..."))

        elif phase == "trader":
            agent_status["Research Manager"] = "completed"
            agent_status["Trader"] = "in_progress"
            messages.append((ts, "System", "Trader making decision..."))

        elif phase == "risk_debate":
            agent_status["Trader"] = "completed"
            agent_status["Aggressive Analyst"] = "in_progress"
            agent_status["Conservative Analyst"] = "in_progress"
            agent_status["Neutral Analyst"] = "in_progress"
            messages.append((ts, "System", "Starting risk debate..."))

        elif phase == "risk_manager":
            agent_status["Aggressive Analyst"] = "completed"
            agent_status["Conservative Analyst"] = "completed"
            agent_status["Neutral Analyst"] = "completed"
            agent_status["Portfolio Manager"] = "in_progress"
            messages.append((ts, "System", "Risk manager making final decision..."))

        elif phase == "signal":
            agent_status["Portfolio Manager"] = "completed"
            messages.append((ts, "System", "Extracting signal..."))

    # Run async pipeline
    analysis_error = None
    final_state = None
    signal = None

    async def _run():
        nonlocal final_state, signal
        from tradingagents.pipeline.runner import TradingPipeline

        pipeline = TradingPipeline(
            selected_analysts=selected_analyst_keys,
            config=config,
            client=client,
            on_phase=on_phase,
        )
        state, sig = await pipeline.run(selections["ticker"], selections["analysis_date"])
        final_state = state.to_dict()
        signal = sig

    # Use a background thread for async + Live display refresh
    import threading
    done_event = threading.Event()

    def run_in_thread():
        nonlocal analysis_error
        try:
            asyncio.run(_run())
        except Exception as e:
            import traceback
            analysis_error = f"{type(e).__name__}: {e}"
            try:
                (results_dir / "error.log").write_text(f"{analysis_error}\n\n{traceback.format_exc()}")
            except Exception:
                pass
        finally:
            done_event.set()

    thread = threading.Thread(target=run_in_thread, daemon=True)

    with Live(layout, refresh_per_second=4, console=console) as live:
        thread.start()
        while not done_event.is_set():
            update_display(layout, agent_status, messages, current_report,
                          report_sections, stats_tracker, start_time, selected_analyst_keys, config)
            done_event.wait(0.25)
        # Final display update
        for agent in agent_status:
            agent_status[agent] = "completed"
        update_display(layout, agent_status, messages, current_report,
                      report_sections, stats_tracker, start_time, selected_analyst_keys, config)

    thread.join()

    if analysis_error:
        console.print(f"\n[bold red]Analysis failed![/bold red]")
        console.print(f"[red]{analysis_error}[/red]")
        console.print(f"[dim]Full traceback saved to: {results_dir / 'error.log'}[/dim]")
        return

    # Post-analysis display
    console.print(f"\n[bold cyan]Analysis Complete![/bold cyan]  Signal: [bold]{signal}[/bold]")
    stats = stats_tracker.get_stats()
    cost = StatsTracker.estimate_cost(stats["tokens_in"], stats["tokens_out"], config.get("deep_think_llm", ""))
    console.print(
        f"[dim]LLM calls: {stats['llm_calls']} | "
        f"Tokens: {format_tokens(stats['tokens_in'])} in / {format_tokens(stats['tokens_out'])} out | "
        f"Est. cost: ${cost:.4f}[/dim]"
    )
    console.print()

    # Save
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path = Path(typer.prompt("Save path", default=str(default_path)).strip())
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]Report saved to:[/green] {save_path.resolve()}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Display
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


# ─── Portfolio ───

PORTFOLIO_PHASE_LABELS = {
    "queued": "Queued",
    "currency": "Currency",
    "screening": "Screening",
    "screening_done": "Screening Done",
    "analyzing": "Analyzing",
    "analysts": "Analysts",
    "investment_debate": "Invest Debate",
    "research_manager": "Research Judge",
    "trader": "Trader",
    "risk_debate": "Risk Debate",
    "risk_manager": "Risk Judge",
    "signal": "Signal",
    "allocating": "Allocating",
    "done": "Done",
    "completed": "Completed",
    "error": "Error",
}

PORTFOLIO_AGENT_LABELS = {
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
}


def _clip_text(text: str, max_chars: int = 220) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _phase_styled_label(phase: str) -> str:
    label = PORTFOLIO_PHASE_LABELS.get(phase, phase.title())
    color_map = {
        "queued": "yellow",
        "currency": "cyan",
        "screening": "cyan",
        "screening_done": "cyan",
        "analyzing": "cyan",
        "analysts": "cyan",
        "investment_debate": "magenta",
        "research_manager": "bright_magenta",
        "trader": "green",
        "risk_debate": "bright_red",
        "risk_manager": "red",
        "signal": "blue",
        "allocating": "bright_cyan",
        "done": "green",
        "completed": "green",
        "error": "red",
    }
    color = color_map.get(phase, "white")
    return f"[{color}]{label}[/{color}]"


def create_portfolio_live_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    layout["left"].split_column(
        Layout(name="stocks", ratio=3),
        Layout(name="pipeline", ratio=2),
    )
    layout["right"].split_column(
        Layout(name="feed", ratio=2),
        Layout(name="discussion", ratio=3),
    )
    return layout


def update_portfolio_live_display(
    layout: Layout,
    candidates: list,
    stock_phase: Dict[str, str],
    stock_signal: Dict[str, str],
    stock_agent: Dict[str, str],
    activity,
    focus_ticker: str,
    focus_title: str,
    focus_content: str,
    global_phase: str,
    request,
    stats_tracker,
    config: dict,
    start_time: float,
):
    elapsed = max(0, int(time.time() - start_time))
    mm = elapsed // 60
    ss = elapsed % 60

    header = (
        "[bold bright_green]TradingAgents Portfolio Live Desk[/bold bright_green]\n"
        f"[cyan]Phase:[/cyan] {_phase_styled_label(global_phase)}    "
        f"[cyan]Time:[/cyan] {mm:02d}:{ss:02d}    "
        f"[cyan]Budget:[/cyan] {request.budget} {request.currency} (${request.budget_usd:.2f})    "
        f"[cyan]Horizon:[/cyan] {request.time_horizon_days}d    "
        f"[cyan]Risk:[/cyan] {request.risk_tolerance.value}"
    )
    layout["header"].update(
        Panel(header, border_style="bright_blue", title="Live Portfolio Builder", padding=(0, 2))
    )

    stocks_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    stocks_table.add_column("Ticker", style="cyan", justify="center", width=8)
    stocks_table.add_column("Price", style="green", justify="right", width=10)
    stocks_table.add_column("Score", style="yellow", justify="right", width=9)
    stocks_table.add_column("Current Agent", style="magenta", width=20)
    stocks_table.add_column("Phase", justify="center", width=15)
    stocks_table.add_column("Signal", justify="center", width=9)

    for c in candidates:
        ticker = c.ticker
        sig = stock_signal.get(ticker, "")
        sig_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow", "ERROR": "red"}.get(sig, "white")
        stocks_table.add_row(
            ticker,
            f"${c.price:.2f}",
            f"{c.composite_score:.4f}",
            stock_agent.get(ticker, "-"),
            _phase_styled_label(stock_phase.get(ticker, "queued")),
            f"[{sig_color}]{sig or '-'}[/{sig_color}]",
        )

    layout["stocks"].update(
        Panel(stocks_table, title="Per-Stock Progress", border_style="cyan", padding=(1, 1))
    )

    pipeline_table = Table(show_header=False, box=box.MINIMAL, expand=True)
    pipeline_table.add_column("Metric", style="cyan", width=22)
    pipeline_table.add_column("Value", style="white")
    completed = sum(1 for t in stock_phase.values() if t == "completed")
    failed = sum(1 for t in stock_phase.values() if t == "error")
    in_progress = sum(1 for t in stock_phase.values() if t not in {"queued", "completed", "error"})
    pipeline_table.add_row("Tickers completed", f"[green]{completed}[/green]")
    pipeline_table.add_row("Tickers in progress", f"[cyan]{in_progress}[/cyan]")
    pipeline_table.add_row("Tickers failed", f"[red]{failed}[/red]")
    pipeline_table.add_row("Global phase", _phase_styled_label(global_phase))

    stats = stats_tracker.get_stats() if stats_tracker else {"llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0}
    est_cost = StatsTracker.estimate_cost(
        stats["tokens_in"], stats["tokens_out"], config.get("deep_think_llm", "")
    )
    pipeline_table.add_row("LLM calls", str(stats["llm_calls"]))
    pipeline_table.add_row("Tool calls", str(stats["tool_calls"]))
    pipeline_table.add_row("Tokens", f"{format_tokens(stats['tokens_in'])}↑ {format_tokens(stats['tokens_out'])}↓")
    pipeline_table.add_row("Estimated cost", f"${est_cost:.4f}")

    layout["pipeline"].update(
        Panel(pipeline_table, title="Runtime Metrics", border_style="blue", padding=(1, 2))
    )

    feed_table = Table(show_header=True, box=box.SIMPLE, expand=True, padding=(0, 1))
    feed_table.add_column("Time", style="cyan", width=8, justify="center")
    feed_table.add_column("Ticker", style="green", width=8, justify="center")
    feed_table.add_column("Actor", style="magenta", width=20)
    feed_table.add_column("Update", style="white", ratio=1)

    for ts, ticker, actor, content in reversed(list(activity)[-14:]):
        feed_table.add_row(ts, ticker, actor, _clip_text(content, 170))

    layout["feed"].update(
        Panel(feed_table, title="Live Agent Feed", border_style="bright_magenta", padding=(1, 1))
    )

    if not focus_content:
        focus_content = "Waiting for first agent output..."
    discussion_md = f"### {focus_title}\n\n{focus_content}"
    layout["discussion"].update(
        Panel(Markdown(discussion_md), title=f"Discussion Focus: {focus_ticker}", border_style="green", padding=(1, 2))
    )

    footer_table = Table(show_header=False, box=None, expand=True, padding=(0, 2))
    footer_table.add_column("line", justify="center")
    footer_table.add_row(
        "Phases: Analysts -> Invest Debate -> Research Judge -> Trader -> Risk Debate -> Risk Judge -> Signal"
    )
    layout["footer"].update(Panel(footer_table, border_style="grey62"))


def display_portfolio_result(result):
    """Display portfolio allocation result."""
    console.print()
    console.print(Rule("Portfolio Investment Plan", style="bold green"))
    console.print()

    req = result.request
    header = Table(show_header=False, box=None, padding=(0, 2))
    header.add_column("Key", style="cyan")
    header.add_column("Value", style="white")
    header.add_row("Budget", f"{req.budget} {req.currency} (${req.budget_usd:.2f} USD)")
    header.add_row("Time Horizon", f"{req.time_horizon_days} days")
    header.add_row("Risk Tolerance", req.risk_tolerance.value.capitalize())
    header.add_row("Stocks Screened", str(result.candidates_screened))
    header.add_row("Stocks Analyzed", str(len(result.candidates_analyzed)))
    console.print(Panel(header, title="Investment Parameters", border_style="cyan"))
    console.print()

    if result.candidates_analyzed:
        cand_table = Table(title="Candidates Analyzed", box=box.ROUNDED)
        cand_table.add_column("Ticker", style="cyan", justify="center")
        cand_table.add_column("Name", style="white")
        cand_table.add_column("Price", style="green", justify="right")
        cand_table.add_column("Score", style="yellow", justify="right")
        cand_table.add_column("Signal", justify="center")
        for c in result.candidates_analyzed:
            color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(c.signal, "white")
            cand_table.add_row(c.ticker, c.name, f"${c.price:.2f}", f"{c.composite_score:.4f}",
                             f"[{color}]{c.signal}[/{color}]")
        console.print(cand_table)
        console.print()

    if result.allocations:
        alloc_table = Table(title="Portfolio Allocation", box=box.ROUNDED)
        alloc_table.add_column("Stock", style="cyan", justify="center")
        alloc_table.add_column("Action", justify="center")
        alloc_table.add_column("Alloc %", style="yellow", justify="right")
        alloc_table.add_column("Amount", style="green", justify="right")
        alloc_table.add_column("Shares", justify="right")
        alloc_table.add_column("Entry", style="white")
        alloc_table.add_column("Stop-Loss", style="red", justify="right")
        for a in result.allocations:
            color = {"BUY": "green", "HOLD": "yellow"}.get(a.action, "white")
            alloc_table.add_row(
                a.ticker, f"[{color}]{a.action}[/{color}]",
                f"{a.allocation_pct:.1f}%", f"${a.allocation_amount:.2f}", str(a.shares),
                f"${a.entry_price_target:.2f}", f"${a.stop_loss_price:.2f}",
            )
        if result.cash_reserved > 0:
            alloc_table.add_row("[dim]Cash[/dim]", "[dim]HOLD[/dim]",
                              f"{(result.cash_reserved / req.budget_usd * 100):.1f}%",
                              f"${result.cash_reserved:.2f}", "-", "-", "-")
        console.print(alloc_table)
        console.print()

    if result.execution_plan:
        console.print(Panel(Markdown(result.execution_plan), title="Execution Plan", border_style="green", padding=(1, 2)))
    if result.risk_assessment:
        console.print(Panel(Markdown(result.risk_assessment), title="Risk Assessment", border_style="red", padding=(1, 2)))
    if result.portfolio_summary:
        console.print(Panel(Markdown(result.portfolio_summary), title="Portfolio Summary", border_style="blue", padding=(1, 2)))


def save_portfolio_report(result, save_path):
    """Save portfolio result to disk."""
    save_path.mkdir(parents=True, exist_ok=True)
    req = result.request
    lines = [
        "# Portfolio Investment Plan", "",
        f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "## Investment Parameters",
        f"- Budget: {req.budget} {req.currency} (${req.budget_usd:.2f} USD)",
        f"- Time Horizon: {req.time_horizon_days} days",
        f"- Risk Tolerance: {req.risk_tolerance.value}",
        f"- Stocks Screened: {result.candidates_screened}",
        f"- Stocks Analyzed: {len(result.candidates_analyzed)}", "",
    ]

    if result.candidates_analyzed:
        lines.append("## Candidates Analyzed")
        lines.append("| Ticker | Name | Price | Score | Signal |")
        lines.append("|--------|------|-------|-------|--------|")
        for c in result.candidates_analyzed:
            lines.append(f"| {c.ticker} | {c.name} | ${c.price:.2f} | {c.composite_score:.4f} | {c.signal} |")
        lines.append("")

    if result.allocations:
        lines.append("## Portfolio Allocation")
        lines.append("| Stock | Action | Alloc % | Amount | Shares |")
        lines.append("|-------|--------|---------|--------|--------|")
        for a in result.allocations:
            lines.append(f"| {a.ticker} | {a.action} | {a.allocation_pct:.1f}% | ${a.allocation_amount:.2f} | {a.shares} |")
        lines.append("")

    if result.execution_plan:
        lines.extend(["## Execution Plan", "", result.execution_plan, ""])
    if result.risk_assessment:
        lines.extend(["## Risk Assessment", "", result.risk_assessment, ""])
    if result.portfolio_summary:
        lines.extend(["## Portfolio Summary", "", result.portfolio_summary, ""])

    report_path = save_path / "portfolio_plan.md"
    report_path.write_text("\n".join(lines))
    return report_path


def run_portfolio_analysis():
    """Portfolio advisory mode."""
    # Welcome
    welcome_content = "[bold green]TradingAgents Portfolio Advisor[/bold green]\n\n"
    welcome_content += "Tell us your budget, time horizon, and risk tolerance.\n"
    welcome_content += "We'll screen stocks, analyze candidates, and build your investment plan.\n\n"
    welcome_content += "[bold]Pipeline:[/bold] Stock Screening → Deep Analysis → Portfolio Allocation\n\n"
    welcome_content += "[dim]Built by Tauric Research[/dim]"
    console.print(Align.center(Panel(welcome_content, border_style="green", padding=(1, 2), title="Portfolio Advisor")))
    console.print()

    def qbox(title, prompt):
        return Panel(f"[bold]{title}[/bold]\n[dim]{prompt}[/dim]", border_style="blue", padding=(1, 2))

    # Prompts
    console.print(qbox("Step 1: Budget", "How much do you want to invest?"))
    budget, currency = get_budget()

    console.print(qbox("Step 2: Time Horizon", "How long to hold?"))
    time_horizon = get_time_horizon()

    console.print(qbox("Step 3: Risk Tolerance", "What is your risk appetite?"))
    risk_tolerance = select_risk_tolerance()

    console.print(qbox("Step 4: Stock Universe", "Which stocks to screen?"))
    stock_universe = select_stock_universe()

    console.print(qbox("Step 5: Analysis Depth", "How many top candidates?"))
    max_candidates = select_max_candidates()

    console.print(qbox("Step 6: Analysts", "Select analysts per stock"))
    analysts = select_analysts()

    console.print(qbox("Step 7: Login", "Authenticate with ChatGPT"))
    client = run_codex_login()

    console.print(qbox("Step 8: Models", "Select your LLM models"))
    quick_model = select_quick_model()
    deep_model = select_deep_model()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["quick_think_llm"] = quick_model
    config["deep_think_llm"] = deep_model

    stats_tracker = StatsTracker()
    client.stats_tracker = stats_tracker

    selected_set = {a.value for a in analysts}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    from tradingagents.portfolio import PortfolioRequest, RiskTolerance

    request = PortfolioRequest(
        budget=budget,
        currency=currency,
        time_horizon_days=time_horizon,
        risk_tolerance=RiskTolerance(risk_tolerance),
        goal="maximize profit",
        stock_universe=stock_universe,
        max_candidates=max_candidates,
        analysis_date=datetime.datetime.now().strftime("%Y-%m-%d"),
    )

    start_time = time.time()

    # Screen stocks
    console.print()
    with console.status("[bold cyan]Screening stocks...[/bold cyan]", spinner="dots"):
        from tradingagents.portfolio.screener import StockScreener
        from tradingagents.portfolio.portfolio_advisor import PortfolioAdvisor

        # Currency conversion
        advisor_helper = PortfolioAdvisor.__new__(PortfolioAdvisor)
        advisor_helper.config = config
        request.budget_usd = advisor_helper._convert_currency(request.budget, request.currency)

        screener = StockScreener(config=config)
        candidates = screener.screen(request)

    if not candidates:
        console.print("[red]No suitable stocks found within budget. Try a larger budget.[/red]")
        return

    tickers_str = ", ".join(f"[bold]{c.ticker}[/bold] (${c.price:.2f})" for c in candidates)
    console.print(f"[green]Selected {len(candidates)} candidates:[/green] {tickers_str}")
    console.print()

    from collections import deque
    import threading

    layout = create_portfolio_live_layout()
    stock_phase = {c.ticker: "queued" for c in candidates}
    stock_signal = {c.ticker: "" for c in candidates}
    stock_agent = {c.ticker: "Waiting" for c in candidates}
    activity = deque(maxlen=240)
    focus_ticker = candidates[0].ticker
    focus_title = "System"
    focus_content = "Waiting for live agent output..."
    global_phase = "analyzing"

    portfolio_error = None
    result = None
    done_event = threading.Event()
    state_lock = threading.Lock()

    def _add_activity(ticker: str, actor: str, content: str, focus: bool = False):
        nonlocal focus_ticker, focus_title, focus_content
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        activity.append((ts, ticker, actor, content))
        if focus:
            focus_ticker = ticker
            focus_title = actor
            focus_content = content

    def progress(phase, msg):
        nonlocal global_phase
        with state_lock:
            if phase == "done":
                global_phase = "completed"
            else:
                global_phase = phase
            _add_activity("SYS", "System", f"{phase}: {msg}")

    def on_activity(event: Dict[str, Any]):
        ticker = event.get("ticker", "SYS")
        kind = event.get("kind", "")
        with state_lock:
            if kind == "phase":
                phase = event.get("phase", "queued")
                stock_phase[ticker] = phase
                _add_activity(
                    ticker,
                    "Phase",
                    f"Entered {PORTFOLIO_PHASE_LABELS.get(phase, phase)}",
                )
                return

            if kind == "agent_start":
                agent_key = event.get("agent", "")
                label = PORTFOLIO_AGENT_LABELS.get(agent_key, agent_key.replace("_", " ").title())
                stock_agent[ticker] = label
                _add_activity(ticker, label, "Started analysis")
                return

            if kind == "agent_output":
                agent_key = event.get("agent", "")
                label = PORTFOLIO_AGENT_LABELS.get(agent_key, agent_key.replace("_", " ").title())
                stock_agent[ticker] = label
                message = event.get("content", "") or "Completed step."
                _add_activity(ticker, label, message, focus=True)
                return

            if kind == "signal":
                signal = event.get("signal", "")
                stock_signal[ticker] = signal
                stock_phase[ticker] = "signal"
                _add_activity(ticker, "Signal", f"Extracted signal: {signal}")
                return

            if kind == "ticker_completed":
                signal = event.get("signal", "")
                summary = event.get("summary", "") or "Analysis completed."
                stock_signal[ticker] = signal
                stock_phase[ticker] = "completed"
                stock_agent[ticker] = "Done"
                _add_activity(ticker, "Result", f"Final signal: {signal}\n\n{summary}", focus=True)
                return

            if kind == "ticker_error":
                err = event.get("error", "unknown error")
                stock_signal[ticker] = "ERROR"
                stock_phase[ticker] = "error"
                stock_agent[ticker] = "Error"
                _add_activity(ticker, "Error", err, focus=True)

    # Run async portfolio analysis
    async def _run_portfolio():
        from tradingagents.portfolio.portfolio_advisor import PortfolioAdvisor

        advisor = PortfolioAdvisor(config=config, client=client)
        return await advisor.advise(
            request,
            selected_analysts=selected_analyst_keys,
            progress_callback=progress,
            activity_callback=on_activity,
        )

    def _run_in_thread():
        nonlocal portfolio_error, result, global_phase
        try:
            result = asyncio.run(_run_portfolio())
            with state_lock:
                global_phase = "completed"
        except Exception as e:
            import traceback
            portfolio_error = f"{type(e).__name__}: {e}"
            with state_lock:
                global_phase = "error"
                _add_activity("SYS", "Fatal", portfolio_error, focus=True)
            traceback.print_exc()
        finally:
            done_event.set()

    thread = threading.Thread(target=_run_in_thread, daemon=True)

    with Live(layout, refresh_per_second=4, console=console):
        thread.start()
        while not done_event.is_set():
            with state_lock:
                update_portfolio_live_display(
                    layout=layout,
                    candidates=candidates,
                    stock_phase=stock_phase,
                    stock_signal=stock_signal,
                    stock_agent=stock_agent,
                    activity=activity,
                    focus_ticker=focus_ticker,
                    focus_title=focus_title,
                    focus_content=focus_content,
                    global_phase=global_phase,
                    request=request,
                    stats_tracker=stats_tracker,
                    config=config,
                    start_time=start_time,
                )
            done_event.wait(0.25)

        with state_lock:
            update_portfolio_live_display(
                layout=layout,
                candidates=candidates,
                stock_phase=stock_phase,
                stock_signal=stock_signal,
                stock_agent=stock_agent,
                activity=activity,
                focus_ticker=focus_ticker,
                focus_title=focus_title,
                focus_content=focus_content,
                global_phase=global_phase,
                request=request,
                stats_tracker=stats_tracker,
                config=config,
                start_time=start_time,
            )

    thread.join()

    if portfolio_error:
        console.print(f"\n[bold red]Portfolio analysis failed:[/bold red] {portfolio_error}")
        return

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]Portfolio analysis complete![/bold green] ({int(elapsed)}s)")
    stats = stats_tracker.get_stats()
    cost = StatsTracker.estimate_cost(stats["tokens_in"], stats["tokens_out"], config.get("deep_think_llm", ""))
    console.print(
        f"[dim]LLM: {stats['llm_calls']} | Tools: {stats['tool_calls']} | "
        f"Tokens: {format_tokens(stats['tokens_in'])} in / {format_tokens(stats['tokens_out'])} out | "
        f"Cost: ${cost:.4f}[/dim]"
    )

    display_portfolio_result(result)

    save_choice = typer.prompt("\nSave portfolio plan?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "results" / "portfolio" / timestamp
        save_path = Path(typer.prompt("Save path", default=str(default_path)).strip())
        try:
            report_file = save_portfolio_report(result, save_path)
            console.print(f"\n[green]Portfolio plan saved to:[/green] {save_path.resolve()}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")


@app.command()
def analyze():
    """Analyze a single stock with multi-agent debate."""
    run_analysis()


@app.command()
def portfolio():
    """Portfolio advisory mode - screen, analyze, and allocate."""
    run_portfolio_analysis()


if __name__ == "__main__":
    app()
