"""Fundamentals Analyst agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.client.types import Message
from tradingagents.pipeline.state import PipelineState
from tradingagents.tools.registry import get_tools


SYSTEM_PROMPT = """You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK; another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop. You have access to the following tools: {tool_names}.
{analyst_prompt}For your reference, the current date is {current_date}. The company we want to look at is {ticker}"""

ANALYST_PROMPT = """You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions. Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read. Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."""


class FundamentalsAnalyst(BaseAgent):
    async def run(self, state: PipelineState) -> PipelineState:
        schemas, executor = get_tools("fundamentals")
        tool_names = ", ".join(s["name"] for s in schemas)

        system = SYSTEM_PROMPT.format(
            tool_names=tool_names,
            analyst_prompt=ANALYST_PROMPT,
            current_date=state.trade_date,
            ticker=state.company_of_interest,
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=state.company_of_interest),
        ]

        response = await self.client.complete_with_tools(
            messages, tools=schemas, tool_executor=executor, model=self.model,
        )

        state.fundamentals_report = response.text
        return state
