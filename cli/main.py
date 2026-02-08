from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Social Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Social Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._last_message_id = None

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._last_message_id = None

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None, config=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

        # Cost estimate
        if config and (stats["tokens_in"] > 0 or stats["tokens_out"] > 0):
            # Use the more expensive model for a conservative estimate
            model = config.get("deep_think_llm", "")
            provider = config.get("llm_provider", "")
            cost = StatsCallbackHandler.estimate_cost(
                stats["tokens_in"], stats["tokens_out"], model, provider
            )
            stats_parts.append(f"Cost: ${cost:.3f}")

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open("./cli/static/welcome.txt", "r") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Select analysts
    console.print(
        create_question_box(
            "Step 3: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 4: Research depth
    console.print(
        create_question_box(
            "Step 4: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 5: OpenAI backend
    console.print(
        create_question_box(
            "Step 5: OpenAI backend", "Select which service to talk to"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    
    # Step 6: Thinking agents
    console.print(
        create_question_box(
            "Step 6: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 7: Provider-specific thinking configuration
    thinking_level = None
    reasoning_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(
            create_question_box(
                "Step 7: Thinking Mode",
                "Configure Gemini thinking mode"
            )
        )
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box(
                "Step 7: Reasoning Effort",
                "Configure OpenAI reasoning effort level"
            )
        )
        reasoning_effort = ask_openai_reasoning_effort()

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
    }


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save complete analysis report to disk with organized subfolders."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"])
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"])
        analyst_parts.append(("Social Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"])
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"])
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"])
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"])
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"])
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"])
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"])
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"])
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"])
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"])
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections))
    return save_path / "complete_report.md"


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Social Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk):
    """Update all analyst statuses based on current report state.

    Logic:
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]
        has_report = bool(chunk.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
            message_buffer.update_report_section(report_key, chunk[report_key])
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def run_analysis():
    # First get all user selections
    selections = get_user_selections()

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w") as f:
                        f.write(content)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time, config=config)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time, config=config)

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout, stats_handler=stats_handler, start_time=start_time, config=config)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time, config=config)

        # Initialize state and get graph args with callbacks
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process messages if present (skip duplicates via message ID)
            if len(chunk["messages"]) > 0:
                last_message = chunk["messages"][-1]
                msg_id = getattr(last_message, "id", None)

                if msg_id != message_buffer._last_message_id:
                    message_buffer._last_message_id = msg_id

                    # Add message to buffer
                    msg_type, content = classify_message_type(last_message)
                    if content and content.strip():
                        message_buffer.add_message(msg_type, content)

                    # Handle tool calls
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if isinstance(tool_call, dict):
                                message_buffer.add_tool_call(
                                    tool_call["name"], tool_call["args"]
                                )
                            else:
                                message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(message_buffer, chunk)

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge:
                    if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time, config=config)

            trace.append(chunk)

        # Get final state and decision
        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time, config=config)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]")
    if stats_handler:
        stats = stats_handler.get_stats()
        cost = StatsCallbackHandler.estimate_cost(
            stats["tokens_in"], stats["tokens_out"],
            config.get("deep_think_llm", ""), config.get("llm_provider", ""),
        )
        console.print(
            f"[dim]LLM calls: {stats['llm_calls']} | "
            f"Tokens: {format_tokens(stats['tokens_in'])} in / {format_tokens(stats['tokens_out'])} out | "
            f"Est. cost: ${cost:.4f}[/dim]"
        )
    console.print()

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


@app.command()
def analyze():
    run_analysis()


def get_portfolio_selections():
    """Get all user selections for portfolio advisory mode."""
    from cli.utils import (
        get_budget,
        get_time_horizon,
        select_risk_tolerance,
        select_stock_universe,
        select_max_candidates,
        select_llm_provider,
        select_shallow_thinking_agent,
        select_deep_thinking_agent,
        select_analysts,
        ask_openai_reasoning_effort,
        ask_gemini_thinking_config,
    )

    # Display welcome
    welcome_content = "[bold green]TradingAgents Portfolio Advisor[/bold green]\n\n"
    welcome_content += "Tell us your budget, time horizon, and risk tolerance.\n"
    welcome_content += "We'll screen stocks, analyze the best candidates, and build your investment plan.\n\n"
    welcome_content += "[bold]Pipeline:[/bold] Stock Screening → Deep Analysis → Portfolio Allocation\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Portfolio Advisor",
        subtitle="Multi-Agent Investment Planning",
    )
    from rich.align import Align
    console.print(Align.center(welcome_box))
    console.print()

    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Budget and currency
    console.print(
        create_question_box(
            "Step 1: Investment Budget",
            "How much do you want to invest?",
        )
    )
    budget, currency = get_budget()
    console.print(f"[green]Budget:[/green] {budget} {currency}")

    # Step 2: Time horizon
    console.print(
        create_question_box(
            "Step 2: Time Horizon",
            "How long do you want to hold your investments?",
        )
    )
    time_horizon_days = get_time_horizon()
    console.print(f"[green]Time horizon:[/green] {time_horizon_days} days")

    # Step 3: Risk tolerance
    console.print(
        create_question_box(
            "Step 3: Risk Tolerance",
            "What is your risk appetite?",
        )
    )
    risk_tolerance = select_risk_tolerance()
    console.print(f"[green]Risk tolerance:[/green] {risk_tolerance}")

    # Step 4: Stock universe
    console.print(
        create_question_box(
            "Step 4: Stock Universe",
            "Which stocks to screen from?",
        )
    )
    stock_universe = select_stock_universe()

    # Step 5: Number of stocks to analyze
    console.print(
        create_question_box(
            "Step 5: Analysis Depth",
            "How many top candidates to deeply analyze?",
        )
    )
    max_candidates = select_max_candidates()

    # Step 6: Analysts
    console.print(
        create_question_box(
            "Step 6: Analyst Team",
            "Select which analysts to use per stock",
        )
    )
    selected_analysts = select_analysts()

    # Step 7: LLM Provider
    console.print(
        create_question_box(
            "Step 7: LLM Provider",
            "Select which LLM service to use",
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()

    # Step 8: Thinking agents
    console.print(
        create_question_box(
            "Step 8: Thinking Agents",
            "Select your thinking agents",
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 9: Provider-specific config
    thinking_level = None
    reasoning_effort = None
    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(
            create_question_box("Step 9: Thinking Mode", "Configure Gemini thinking mode")
        )
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box("Step 9: Reasoning Effort", "Configure OpenAI reasoning effort")
        )
        reasoning_effort = ask_openai_reasoning_effort()

    return {
        "budget": budget,
        "currency": currency,
        "time_horizon_days": time_horizon_days,
        "risk_tolerance": risk_tolerance,
        "stock_universe": stock_universe,
        "max_candidates": max_candidates,
        "analysts": selected_analysts,
        "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
    }


def display_portfolio_result(result):
    """Display the portfolio allocation result in a rich formatted output."""
    from tradingagents.portfolio.models import PortfolioResult

    console.print()
    console.print(Rule("Portfolio Investment Plan", style="bold green"))
    console.print()

    req = result.request
    # Header
    header = Table(show_header=False, box=None, padding=(0, 2))
    header.add_column("Key", style="cyan")
    header.add_column("Value", style="white")
    header.add_row("Budget", f"{req.budget} {req.currency} (${req.budget_usd:.2f} USD)")
    header.add_row("Time Horizon", f"{req.time_horizon_days} days")
    header.add_row("Risk Tolerance", req.risk_tolerance.value.capitalize())
    header.add_row("Goal", req.goal)
    header.add_row("Stocks Screened", str(result.candidates_screened))
    header.add_row("Stocks Analyzed", str(len(result.candidates_analyzed)))
    console.print(Panel(header, title="Investment Parameters", border_style="cyan"))
    console.print()

    # Candidates analyzed
    if result.candidates_analyzed:
        cand_table = Table(title="Candidates Analyzed", box=box.ROUNDED)
        cand_table.add_column("Ticker", style="cyan", justify="center")
        cand_table.add_column("Name", style="white")
        cand_table.add_column("Sector", style="dim")
        cand_table.add_column("Price", style="green", justify="right")
        cand_table.add_column("Score", style="yellow", justify="right")
        cand_table.add_column("Signal", justify="center")
        for c in result.candidates_analyzed:
            signal_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(c.signal, "white")
            cand_table.add_row(
                c.ticker,
                c.name,
                c.sector,
                f"${c.price:.2f}",
                f"{c.composite_score:.4f}",
                f"[{signal_color}]{c.signal}[/{signal_color}]",
            )
        console.print(cand_table)
        console.print()

    # Allocation table
    if result.allocations:
        alloc_table = Table(title="Portfolio Allocation", box=box.ROUNDED)
        alloc_table.add_column("Stock", style="cyan", justify="center")
        alloc_table.add_column("Action", justify="center")
        alloc_table.add_column("Alloc %", style="yellow", justify="right")
        alloc_table.add_column("Amount", style="green", justify="right")
        alloc_table.add_column("Shares", justify="right")
        alloc_table.add_column("Entry", style="white")
        alloc_table.add_column("Exit", style="white")
        alloc_table.add_column("Stop-Loss", style="red", justify="right")

        for a in result.allocations:
            action_color = {"BUY": "green", "HOLD": "yellow", "SKIP": "dim"}.get(a.action, "white")
            alloc_table.add_row(
                a.ticker,
                f"[{action_color}]{a.action}[/{action_color}]",
                f"{a.allocation_pct:.1f}%",
                f"${a.allocation_amount:.2f}",
                str(a.shares),
                f"${a.entry_price_target:.2f} - {a.entry_timing}",
                f"${a.exit_price_target:.2f} - {a.exit_timing}",
                f"${a.stop_loss_price:.2f}",
            )

        # Cash row
        if result.cash_reserved > 0:
            alloc_table.add_row(
                "[dim]Cash[/dim]",
                "[dim]HOLD[/dim]",
                f"{(result.cash_reserved / req.budget_usd * 100):.1f}%",
                f"${result.cash_reserved:.2f}",
                "-", "-", "-", "-",
            )

        console.print(alloc_table)
        console.print()

    # Execution plan
    if result.execution_plan:
        console.print(Panel(
            Markdown(result.execution_plan),
            title="Execution Plan",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()

    # Risk assessment
    if result.risk_assessment:
        console.print(Panel(
            Markdown(result.risk_assessment),
            title="Risk Assessment",
            border_style="red",
            padding=(1, 2),
        ))
        console.print()

    # Portfolio summary
    if result.portfolio_summary:
        console.print(Panel(
            Markdown(result.portfolio_summary),
            title="Portfolio Summary",
            border_style="blue",
            padding=(1, 2),
        ))


def save_portfolio_report(result, save_path: Path):
    """Save portfolio result to disk."""
    save_path.mkdir(parents=True, exist_ok=True)

    req = result.request
    lines = [
        f"# Portfolio Investment Plan",
        f"",
        f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Investment Parameters",
        f"- Budget: {req.budget} {req.currency} (${req.budget_usd:.2f} USD)",
        f"- Time Horizon: {req.time_horizon_days} days",
        f"- Risk Tolerance: {req.risk_tolerance.value}",
        f"- Goal: {req.goal}",
        f"- Stocks Screened: {result.candidates_screened}",
        f"- Stocks Analyzed: {len(result.candidates_analyzed)}",
        f"",
    ]

    # Candidates
    if result.candidates_analyzed:
        lines.append("## Candidates Analyzed")
        lines.append("")
        lines.append("| Ticker | Name | Sector | Price | Score | Signal |")
        lines.append("|--------|------|--------|-------|-------|--------|")
        for c in result.candidates_analyzed:
            lines.append(f"| {c.ticker} | {c.name} | {c.sector} | ${c.price:.2f} | {c.composite_score:.4f} | {c.signal} |")
        lines.append("")

    # Allocations
    if result.allocations:
        lines.append("## Portfolio Allocation")
        lines.append("")
        lines.append("| Stock | Action | Alloc % | Amount | Shares | Entry Price | Exit Price | Stop-Loss | Entry Timing | Exit Timing |")
        lines.append("|-------|--------|---------|--------|--------|-------------|------------|-----------|--------------|-------------|")
        for a in result.allocations:
            lines.append(
                f"| {a.ticker} | {a.action} | {a.allocation_pct:.1f}% | ${a.allocation_amount:.2f} | {a.shares} | ${a.entry_price_target:.2f} | ${a.exit_price_target:.2f} | ${a.stop_loss_price:.2f} | {a.entry_timing} | {a.exit_timing} |"
            )
        if result.cash_reserved > 0:
            lines.append(f"| Cash | HOLD | {(result.cash_reserved / req.budget_usd * 100):.1f}% | ${result.cash_reserved:.2f} | - | - | - | - | - | - |")
        lines.append("")

    if result.execution_plan:
        lines.append("## Execution Plan")
        lines.append("")
        lines.append(result.execution_plan)
        lines.append("")

    if result.risk_assessment:
        lines.append("## Risk Assessment")
        lines.append("")
        lines.append(result.risk_assessment)
        lines.append("")

    if result.portfolio_summary:
        lines.append("## Portfolio Summary")
        lines.append("")
        lines.append(result.portfolio_summary)
        lines.append("")

    # Per-stock rationales
    if result.allocations:
        lines.append("## Stock Rationales")
        lines.append("")
        for a in result.allocations:
            if a.rationale:
                lines.append(f"### {a.ticker} - {a.name}")
                lines.append(a.rationale)
                lines.append("")

    report_path = save_path / "portfolio_plan.md"
    report_path.write_text("\n".join(lines))
    return report_path


def create_portfolio_layout():
    """Create layout for portfolio mode - parallel analysis view."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="portfolio_progress"),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="messages", ratio=1),
        Layout(name="analysis", ratio=1),
    )
    return layout


class StockTracker:
    """Thread-safe per-stock analysis tracker."""

    TEAM_PHASES = {
        "Analyst Team": ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"],
        "Research": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading": ["Trader"],
        "Risk Mgmt": ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"],
    }

    def __init__(self, ticker, selected_analysts):
        import threading
        self.ticker = ticker
        self.lock = threading.Lock()
        self.agent_status = {}
        self.current_team = ""
        self.current_agent = ""
        self.messages = deque(maxlen=50)
        self.latest_report = ""
        self.final_state = None
        self.signal = ""
        self.done = False
        self.error = None
        self._last_message_id = None

        # Init agent status
        analyst_map = {"market": "Market Analyst", "social": "Social Analyst",
                       "news": "News Analyst", "fundamentals": "Fundamentals Analyst"}
        for key in selected_analysts:
            if key in analyst_map:
                self.agent_status[analyst_map[key]] = "pending"
        for agents in list(self.TEAM_PHASES.values())[1:]:
            for agent in agents:
                self.agent_status[agent] = "pending"

    def get_current_phase_display(self):
        """Return a short string showing current phase, e.g. 'Market Analyst'."""
        with self.lock:
            if self.done:
                if self.error:
                    return "[red]ERROR[/red]"
                signal_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(self.signal, "white")
                return f"[{signal_color}]{self.signal}[/{signal_color}]"
            if self.current_agent:
                return f"[blue]{self.current_agent}[/blue]"
            return "[dim]starting...[/dim]"


def _analyze_stock_worker(candidate, config, selected_analyst_keys, stats_handler, tracker, results_dir, analysis_date):
    """Worker function: run full analysis for one stock in a thread."""
    try:
        graph = TradingAgentsGraph(
            selected_analyst_keys,
            config=config,
            debug=True,
            callbacks=[stats_handler],
        )

        # Create per-stock dirs
        stock_dir = results_dir / candidate.ticker
        stock_dir.mkdir(parents=True, exist_ok=True)
        report_dir = stock_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        log_file = stock_dir / "message_tool.log"
        log_file.touch(exist_ok=True)

        init_state = graph.propagator.create_initial_state(candidate.ticker, analysis_date)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        trace = []
        for chunk in graph.graph.stream(init_state, **args):
            with tracker.lock:
                # Process messages
                if len(chunk.get("messages", [])) > 0:
                    last_message = chunk["messages"][-1]
                    msg_id = getattr(last_message, "id", None)
                    if msg_id != tracker._last_message_id:
                        tracker._last_message_id = msg_id
                        content = extract_content_string(getattr(last_message, "content", None))
                        if content and content.strip():
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            tracker.messages.append((timestamp, candidate.ticker, content[:150]))

                # Track analyst status from reports
                for analyst_key, report_key in [
                    ("market", "market_report"), ("social", "sentiment_report"),
                    ("news", "news_report"), ("fundamentals", "fundamentals_report"),
                ]:
                    agent_name = {"market": "Market Analyst", "social": "Social Analyst",
                                  "news": "News Analyst", "fundamentals": "Fundamentals Analyst"}.get(analyst_key)
                    if agent_name and agent_name in tracker.agent_status:
                        if chunk.get(report_key):
                            tracker.agent_status[agent_name] = "completed"
                            tracker.current_team = "Analyst Team"
                            tracker.current_agent = agent_name
                            tracker.latest_report = chunk[report_key][:500]
                            # Write report file
                            (report_dir / f"{report_key}.md").write_text(chunk[report_key])

                # Research team
                if chunk.get("investment_debate_state"):
                    debate = chunk["investment_debate_state"]
                    if debate.get("bull_history", "").strip():
                        tracker.agent_status["Bull Researcher"] = "in_progress"
                        tracker.current_team = "Research"
                        tracker.current_agent = "Bull Researcher"
                    if debate.get("bear_history", "").strip():
                        tracker.agent_status["Bear Researcher"] = "in_progress"
                        tracker.current_agent = "Bear Researcher"
                    if debate.get("judge_decision", "").strip():
                        tracker.agent_status["Bull Researcher"] = "completed"
                        tracker.agent_status["Bear Researcher"] = "completed"
                        tracker.agent_status["Research Manager"] = "completed"
                        tracker.current_agent = "Research Manager"
                        tracker.latest_report = debate["judge_decision"][:500]
                        (report_dir / "investment_plan.md").write_text(debate["judge_decision"])

                # Trader
                if chunk.get("trader_investment_plan"):
                    tracker.agent_status["Trader"] = "completed"
                    tracker.current_team = "Trading"
                    tracker.current_agent = "Trader"
                    tracker.latest_report = chunk["trader_investment_plan"][:500]
                    (report_dir / "trader_investment_plan.md").write_text(chunk["trader_investment_plan"])

                # Risk management
                if chunk.get("risk_debate_state"):
                    risk = chunk["risk_debate_state"]
                    if risk.get("aggressive_history", "").strip():
                        tracker.agent_status["Aggressive Analyst"] = "in_progress"
                        tracker.current_team = "Risk Mgmt"
                        tracker.current_agent = "Aggressive Analyst"
                    if risk.get("conservative_history", "").strip():
                        tracker.agent_status["Conservative Analyst"] = "in_progress"
                        tracker.current_agent = "Conservative Analyst"
                    if risk.get("neutral_history", "").strip():
                        tracker.agent_status["Neutral Analyst"] = "in_progress"
                        tracker.current_agent = "Neutral Analyst"
                    if risk.get("judge_decision", "").strip():
                        tracker.agent_status["Aggressive Analyst"] = "completed"
                        tracker.agent_status["Conservative Analyst"] = "completed"
                        tracker.agent_status["Neutral Analyst"] = "completed"
                        tracker.agent_status["Portfolio Manager"] = "completed"
                        tracker.current_agent = "Portfolio Manager"
                        tracker.latest_report = risk["judge_decision"][:500]
                        (report_dir / "final_trade_decision.md").write_text(risk["judge_decision"])

                # Set in_progress for next pending analyst
                found_active = False
                for agent_name, status in tracker.agent_status.items():
                    if status == "in_progress":
                        found_active = True
                        tracker.current_agent = agent_name
                        break
                if not found_active:
                    for agent_name, status in tracker.agent_status.items():
                        if status == "pending":
                            tracker.agent_status[agent_name] = "in_progress"
                            tracker.current_agent = agent_name
                            break

            trace.append(chunk)

        # Extract final result
        if trace:
            final_state = trace[-1]
            signal = graph.process_signal(final_state.get("final_trade_decision", ""))
            with tracker.lock:
                tracker.final_state = final_state
                tracker.signal = signal
                tracker.done = True
                for agent in tracker.agent_status:
                    tracker.agent_status[agent] = "completed"
        else:
            with tracker.lock:
                tracker.signal = "ERROR"
                tracker.error = "No output from graph"
                tracker.done = True

    except Exception as e:
        with tracker.lock:
            tracker.error = str(e)
            tracker.signal = "ERROR"
            tracker.done = True


def _get_team_status_icon(tracker, team_agents):
    """Get a status icon for a team based on its agents' statuses."""
    with tracker.lock:
        statuses = {a: tracker.agent_status.get(a, "pending") for a in team_agents if a in tracker.agent_status}
    if not statuses:
        return "[dim]-[/dim]"
    completed = sum(1 for s in statuses.values() if s == "completed")
    total = len(statuses)
    if completed == total:
        return f"[green]done {completed}/{total}[/green]"
    in_progress = [a for a, s in statuses.items() if s == "in_progress"]
    if in_progress:
        # Show the active agent's short name
        short_name = in_progress[0].replace(" Analyst", "").replace(" Researcher", "")
        return Spinner("dots", text=f"[blue]{short_name} {completed}/{total}[/blue]", style="bold cyan")
    return f"[dim]pending {completed}/{total}[/dim]"


def update_parallel_display(layout, trackers, candidates, request, stats_handler, start_time, phase, config=None):
    """Update the full dashboard for parallel portfolio analysis."""
    # Header
    layout["header"].update(
        Panel(
            f"[bold green]TradingAgents Portfolio Advisor[/bold green]\n"
            f"[white]Budget: {request.budget} {request.currency} (${request.budget_usd:.2f} USD) | "
            f"Horizon: {request.time_horizon_days} days | Risk: {request.risk_tolerance.value}[/white]",
            title="Portfolio Advisor",
            border_style="green",
            padding=(0, 2),
            expand=True,
        )
    )

    # Portfolio progress table - shows all stocks with per-team breakdown
    portfolio_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        expand=True,
    )
    portfolio_table.add_column("Stock", style="cyan bold", justify="center", width=8)
    portfolio_table.add_column("Price", style="green", justify="right", width=9)
    portfolio_table.add_column("Analysts", justify="center", width=14)
    portfolio_table.add_column("Research", justify="center", width=14)
    portfolio_table.add_column("Trading", justify="center", width=14)
    portfolio_table.add_column("Risk Mgmt", justify="center", width=14)
    portfolio_table.add_column("Signal", justify="center", width=10)

    for c in candidates:
        tracker = trackers.get(c.ticker)
        if tracker:
            # Get per-team status
            analyst_agents = [a for a in ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"]
                           if a in tracker.agent_status]
            research_agents = ["Bull Researcher", "Bear Researcher", "Research Manager"]
            trading_agents = ["Trader"]
            risk_agents = ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"]

            analysts_cell = _get_team_status_icon(tracker, analyst_agents)
            research_cell = _get_team_status_icon(tracker, research_agents)
            trading_cell = _get_team_status_icon(tracker, trading_agents)
            risk_cell = _get_team_status_icon(tracker, risk_agents)

            # Signal column
            if tracker.done:
                if tracker.error:
                    signal_cell = "[red]ERROR[/red]"
                else:
                    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(tracker.signal, "white")
                    signal_cell = f"[bold {signal_color}]{tracker.signal}[/{signal_color}]"
            else:
                signal_cell = "[dim]...[/dim]"

            portfolio_table.add_row(
                c.ticker,
                f"${c.price:.2f}",
                analysts_cell,
                research_cell,
                trading_cell,
                risk_cell,
                signal_cell,
            )
        else:
            portfolio_table.add_row(c.ticker, f"${c.price:.2f}", *["[dim]pending[/dim]"] * 4, "[dim]...[/dim]")

    # Allocation row
    portfolio_table.add_row("─" * 8, "─" * 9, "─" * 14, "─" * 14, "─" * 14, "─" * 14, "─" * 10, style="dim")
    if phase == "allocating":
        portfolio_table.add_row(
            "[bold]Alloc[/bold]", "", "",
            Spinner("dots", text="[blue]building plan[/blue]", style="bold cyan"),
            "", "", "",
        )
    elif phase == "done":
        portfolio_table.add_row("[bold]Alloc[/bold]", "", "", "", "", "", "[green]done[/green]")
    else:
        portfolio_table.add_row("[bold]Alloc[/bold]", "", "", "", "", "", "[dim]waiting[/dim]")

    layout["portfolio_progress"].update(
        Panel(portfolio_table, title="Portfolio Progress (Parallel)", border_style="magenta", padding=(0, 1))
    )

    # Messages panel - combined from all trackers
    messages_table = Table(
        show_header=True, header_style="bold magenta",
        box=box.MINIMAL, show_lines=True, padding=(0, 1), expand=True,
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Stock", style="green", width=8, justify="center")
    messages_table.add_column("Content", style="white", no_wrap=False, ratio=1)

    all_messages = []
    for ticker, tracker in trackers.items():
        with tracker.lock:
            for ts, tk, content in tracker.messages:
                all_messages.append((ts, tk, content))
    all_messages.sort(key=lambda x: x[0], reverse=True)
    for ts, tk, content in all_messages[:15]:
        wrapped = Text(content, overflow="fold")
        messages_table.add_row(ts, tk, wrapped)

    layout["messages"].update(
        Panel(messages_table, title="Messages (All Stocks)", border_style="blue", padding=(0, 1))
    )

    # Analysis panel - show latest report from any tracker
    latest_report = ""
    latest_ticker = ""
    for ticker, tracker in trackers.items():
        with tracker.lock:
            if tracker.latest_report:
                latest_report = tracker.latest_report
                latest_ticker = ticker
    if latest_report:
        layout["analysis"].update(
            Panel(Markdown(latest_report), title=f"Latest Report ({latest_ticker})", border_style="green", padding=(1, 2))
        )
    else:
        layout["analysis"].update(
            Panel("[italic]Waiting for analysis reports...[/italic]", title="Current Report", border_style="green", padding=(1, 2))
        )

    # Footer with statistics
    stats_parts = []
    done_count = sum(1 for t in trackers.values() if t.done)
    stats_parts.append(f"Stocks: {done_count}/{len(trackers)}")
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            stats_parts.append(f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193")

        # Cost estimate
        if config and (stats["tokens_in"] > 0 or stats["tokens_out"] > 0):
            model = config.get("deep_think_llm", "")
            provider = config.get("llm_provider", "")
            cost = StatsCallbackHandler.estimate_cost(
                stats["tokens_in"], stats["tokens_out"], model, provider
            )
            stats_parts.append(f"Cost: ${cost:.3f}")

    if start_time:
        elapsed = time.time() - start_time
        stats_parts.append(f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))
    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def run_portfolio_analysis():
    """Portfolio advisory mode - parallel analysis with live dashboard."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    selections = get_portfolio_selections()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"]
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")

    stats_handler = StatsCallbackHandler()

    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    from tradingagents.portfolio import PortfolioRequest, RiskTolerance
    from tradingagents.portfolio.screener import StockScreener
    from tradingagents.portfolio.models import PortfolioResult

    request = PortfolioRequest(
        budget=selections["budget"],
        currency=selections["currency"],
        time_horizon_days=selections["time_horizon_days"],
        risk_tolerance=RiskTolerance(selections["risk_tolerance"]),
        goal="maximize profit",
        stock_universe=selections["stock_universe"],
        max_candidates=selections["max_candidates"],
        analysis_date=selections["analysis_date"],
    )

    start_time = time.time()

    # --- Phase 1: Currency conversion ---
    from tradingagents.portfolio.portfolio_advisor import PortfolioAdvisor
    advisor_helper = PortfolioAdvisor.__new__(PortfolioAdvisor)
    request.budget_usd = advisor_helper._convert_currency(request.budget, request.currency)

    # --- Phase 2: Stock Screening ---
    console.print()
    with console.status("[bold cyan]Screening stocks...[/bold cyan]", spinner="dots"):
        screener = StockScreener(config=config)
        candidates = screener.screen(request)

    if not candidates:
        console.print("[red]No suitable stocks found within your budget. Try a larger budget.[/red]")
        return

    tickers_str = ", ".join(f"[bold]{c.ticker}[/bold] (${c.price:.2f})" for c in candidates)
    console.print(f"[green]Screened and selected {len(candidates)} candidates:[/green] {tickers_str}")
    console.print()

    # --- Phase 3: Parallel deep analysis ---
    total_stocks = len(candidates)
    results_dir = Path(config["results_dir"]) / "portfolio" / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create per-stock trackers
    trackers = {}
    for c in candidates:
        trackers[c.ticker] = StockTracker(c.ticker, selected_analyst_keys)

    layout = create_portfolio_layout()

    with Live(layout, refresh_per_second=4, console=console) as live:
        # Launch all stock analyses in parallel
        with ThreadPoolExecutor(max_workers=total_stocks) as executor:
            futures = {}
            for candidate in candidates:
                future = executor.submit(
                    _analyze_stock_worker,
                    candidate, config, selected_analyst_keys,
                    stats_handler, trackers[candidate.ticker],
                    results_dir, selections["analysis_date"],
                )
                futures[future] = candidate

            # Main display loop - refresh while threads are running
            while not all(t.done for t in trackers.values()):
                update_parallel_display(layout, trackers, candidates, request, stats_handler, start_time, "analyzing", config)
                time.sleep(0.25)

            # Final update after all threads done
            update_parallel_display(layout, trackers, candidates, request, stats_handler, start_time, "analyzing", config)

        # Copy results from trackers back to candidates
        for candidate in candidates:
            tracker = trackers[candidate.ticker]
            candidate.signal = tracker.signal
            candidate.final_state = tracker.final_state

        # --- Phase 4: Portfolio Allocation ---
        update_parallel_display(layout, trackers, candidates, request, stats_handler, start_time, "allocating", config)

        from tradingagents.llm_clients import create_llm_client
        llm_kwargs = {}
        provider = config.get("llm_provider", "").lower()
        if provider == "google" and config.get("google_thinking_level"):
            llm_kwargs["thinking_level"] = config["google_thinking_level"]
        elif provider == "openai" and config.get("openai_reasoning_effort"):
            llm_kwargs["reasoning_effort"] = config["openai_reasoning_effort"]
        if stats_handler:
            llm_kwargs["callbacks"] = [stats_handler]

        deep_client = create_llm_client(
            provider=config["llm_provider"],
            model=config["deep_think_llm"],
            base_url=config.get("backend_url"),
            **llm_kwargs,
        )
        allocator_llm = deep_client.get_llm()

        from tradingagents.portfolio.allocator import PortfolioAllocator
        allocator = PortfolioAllocator(allocator_llm)

        valid_candidates = [c for c in candidates if c.signal != "ERROR"]
        if valid_candidates:
            result = allocator.allocate(request, valid_candidates)
        else:
            result = PortfolioResult(
                request=request,
                candidates_analyzed=candidates,
                portfolio_summary="All stock analyses failed. Please try again.",
            )

        update_parallel_display(layout, trackers, candidates, request, stats_handler, start_time, "done", config)

    # --- Post Live: show results ---
    elapsed = time.time() - start_time
    console.print(f"\n[bold green]Portfolio analysis complete![/bold green] ({int(elapsed)}s)")

    if stats_handler:
        stats = stats_handler.get_stats()
        cost = StatsCallbackHandler.estimate_cost(
            stats["tokens_in"], stats["tokens_out"],
            config.get("deep_think_llm", ""), config.get("llm_provider", ""),
        )
        console.print(
            f"[dim]LLM calls: {stats['llm_calls']} | "
            f"Tool calls: {stats['tool_calls']} | "
            f"Tokens: {format_tokens(stats['tokens_in'])} in / {format_tokens(stats['tokens_out'])} out | "
            f"Est. cost: ${cost:.4f}[/dim]"
        )

    display_portfolio_result(result)

    save_choice = typer.prompt("\nSave portfolio plan?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "results" / "portfolio" / timestamp
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path),
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_portfolio_report(result, save_path)
            console.print(f"\n[green]Portfolio plan saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")


@app.command()
def portfolio():
    """Portfolio advisory mode - suggest stocks to buy/sell with budget allocation."""
    run_portfolio_analysis()


if __name__ == "__main__":
    app()
