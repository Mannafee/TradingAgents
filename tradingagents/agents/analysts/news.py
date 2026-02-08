"""News Analyst agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.client.types import Message
from tradingagents.pipeline.state import PipelineState
from tradingagents.tools.registry import get_tools


SYSTEM_PROMPT = """You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK; another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop. You have access to the following tools: {tool_names}.
{analyst_prompt}For your reference, the current date is {current_date}. We are looking at the company {ticker}"""

ANALYST_PROMPT = """You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions. Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""


class NewsAnalyst(BaseAgent):
    async def run(self, state: PipelineState) -> PipelineState:
        schemas, executor = get_tools("news")
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

        state.news_report = response.text
        return state
