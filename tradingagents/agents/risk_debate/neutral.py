"""Neutral Risk Analyst agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.pipeline.state import PipelineState


class NeutralAnalyst(BaseAgent):
    async def run(self, state: PipelineState) -> PipelineState:
        risk = state.risk_debate_state

        prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies.Here is the trader's decision:

{state.trader_investment_plan}

Your task is to challenge both the Aggressive and Conservative Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:

Market Research Report: {state.market_report}
Social Media Sentiment Report: {state.sentiment_report}
Latest World Affairs Report: {state.news_report}
Company Fundamentals Report: {state.fundamentals_report}
Here is the current conversation history: {risk.history} Here is the last response from the aggressive analyst: {risk.current_aggressive_response} Here is the last response from the conservative analyst: {risk.current_conservative_response}. If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage actively by analyzing both sides critically, addressing weaknesses in the aggressive and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds, providing growth potential while safeguarding against extreme volatility. Focus on debating rather than simply presenting data, aiming to show that a balanced view can lead to the most reliable outcomes. Output conversationally as if you are speaking without any special formatting."""

        response = await self._invoke(prompt)
        argument = f"Neutral Analyst: {response}"

        risk.history = risk.history + "\n" + argument
        risk.neutral_history = risk.neutral_history + "\n" + argument
        risk.current_neutral_response = argument
        risk.count += 1

        return state
