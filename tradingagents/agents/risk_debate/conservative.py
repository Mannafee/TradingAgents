"""Conservative Risk Analyst agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.pipeline.state import PipelineState


class ConservativeAnalyst(BaseAgent):
    async def run(self, state: PipelineState) -> PipelineState:
        risk = state.risk_debate_state

        prompt = f"""As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{state.trader_investment_plan}

Your task is to actively counter the arguments of the Aggressive and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Market Research Report: {state.market_report}
Social Media Sentiment Report: {state.sentiment_report}
Latest World Affairs Report: {state.news_report}
Company Fundamentals Report: {state.fundamentals_report}
Here is the current conversation history: {risk.history} Here is the last response from the aggressive analyst: {risk.current_aggressive_response} Here is the last response from the neutral analyst: {risk.current_neutral_response}. If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting."""

        response = await self._invoke(prompt)
        argument = f"Conservative Analyst: {response}"

        risk.history = risk.history + "\n" + argument
        risk.conservative_history = risk.conservative_history + "\n" + argument
        risk.current_conservative_response = argument
        risk.count += 1

        return state
