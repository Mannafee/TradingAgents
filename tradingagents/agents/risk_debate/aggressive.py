"""Aggressive Risk Analyst agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.pipeline.state import PipelineState


class AggressiveAnalyst(BaseAgent):
    async def run(self, state: PipelineState) -> PipelineState:
        risk = state.risk_debate_state

        prompt = f"""As the Aggressive Risk Analyst, your role is to actively champion high-reward, high-risk opportunities, emphasizing bold strategies and competitive advantages. When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential, and innovative benefits—even when these come with elevated risk. Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views. Specifically, respond directly to each point made by the conservative and neutral analysts, countering with data-driven rebuttals and persuasive reasoning. Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative. Here is the trader's decision:

{state.trader_investment_plan}

Your task is to create a compelling case for the trader's decision by questioning and critiquing the conservative and neutral stances to demonstrate why your high-reward perspective offers the best path forward. Incorporate insights from the following sources into your arguments:

Market Research Report: {state.market_report}
Social Media Sentiment Report: {state.sentiment_report}
Latest World Affairs Report: {state.news_report}
Company Fundamentals Report: {state.fundamentals_report}
Here is the current conversation history: {risk.history} Here are the last arguments from the conservative analyst: {risk.current_conservative_response} Here are the last arguments from the neutral analyst: {risk.current_neutral_response}. If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage actively by addressing any specific concerns raised, refuting the weaknesses in their logic, and asserting the benefits of risk-taking to outpace market norms. Maintain a focus on debating and persuading, not just presenting data. Challenge each counterpoint to underscore why a high-risk approach is optimal. Output conversationally as if you are speaking without any special formatting."""

        response = await self._invoke(prompt)
        argument = f"Aggressive Analyst: {response}"

        risk.history = risk.history + "\n" + argument
        risk.aggressive_history = risk.aggressive_history + "\n" + argument
        risk.current_aggressive_response = argument
        risk.count += 1

        return state
