"""
AdvisorIQ — LLM Narrator

Translates structured quant outputs into plain-English client communications.
The LLM is purely a translator — every number was computed by the quantitative layers.
"""

import logging
import json
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# System prompt for the narrator
NARRATOR_SYSTEM_PROMPT = """You are a financial advisor's AI assistant that translates quantitative 
portfolio analysis into clear, client-friendly language. 

Rules:
- Every number you mention was computed by the quantitative system — you are a translator, not an analyst.
- Write exactly 3 paragraphs: situation overview, key changes, and what it means for the client.
- Use plain English. No jargon without immediate explanation.
- Be specific with numbers but round appropriately (e.g., "about 15%" not "14.73%").
- When discussing risk changes, frame them in terms of the client's goals.
- Never give direct investment advice — describe what the analysis shows.
- Keep the tone professional but warm — this is for real people's money."""


def build_narrator_prompt(client_result: dict, signal_data: dict) -> str:
    """Build the prompt for the LLM narrator from structured data."""
    return f"""Translate this portfolio analysis into a 3-paragraph client summary.

CLIENT PROFILE:
- Name: {client_result['name']}
- Risk profile: {client_result['risk_profile']}
- Target volatility: {client_result['target_vol']:.0%}
- Goals: {client_result['goals']}

CURRENT MARKET REGIME: {signal_data.get('regime', 'NORMAL')}
Regime probabilities: {json.dumps(signal_data.get('regime_probs', {}), indent=0)}

PORTFOLIO ANALYSIS:
- Current portfolio vol: {client_result['current_stats']['annual_vol']:.1%}
- Target vol: {client_result['target_vol']:.0%}
- Vol misaligned: {client_result['vol_misaligned']}
- Total drift from optimal: {client_result['total_drift']:.1%}

KEY WEIGHT CHANGES (IV-adjusted optimal vs current):
{json.dumps({k: v for k, v in client_result['drift'].items() if abs(v) > 0.02}, indent=2)}

IV-ADJUSTED PORTFOLIO STATS:
- Expected return: {client_result['iv_adjusted_stats']['expected_return']:.1%}
- Volatility: {client_result['iv_adjusted_stats']['annual_vol']:.1%}
- Sharpe ratio: {client_result['iv_adjusted_stats']['sharpe_ratio']:.2f}

ACTIVE ALERTS:
{json.dumps(signal_data.get('alerts', []), indent=2)}

STRESS TEST RESULTS:
{json.dumps(client_result['stress_tests'], indent=2)}

Write exactly 3 paragraphs. Be specific with numbers. Frame everything relative to the client's goals."""


def generate_narrative(
    client_result: dict,
    signal_data: dict,
    api_key: str = None,
) -> str:
    """
    Generate client-facing narrative using the Anthropic API.
    Falls back to a template-based narrative if API unavailable.
    """
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            prompt = build_narrator_prompt(client_result, signal_data)

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=NARRATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning("LLM API call failed: %s — using template fallback", e)

    # Template fallback
    return _template_narrative(client_result, signal_data)


def _template_narrative(client_result: dict, signal_data: dict) -> str:
    """Template-based fallback when API is unavailable."""
    name = client_result["name"].split()[0]
    regime = signal_data.get("regime", "NORMAL")
    curr_vol = client_result["current_stats"]["annual_vol"]
    target_vol = client_result["target_vol"]
    drift = client_result["total_drift"]
    alerts = signal_data.get("alerts", [])

    regime_desc = {
        "LOW_VOL": "calm, low-volatility conditions",
        "NORMAL": "normal market conditions",
        "STRESS": "elevated stress",
        "CRISIS": "crisis-level stress",
    }.get(regime, "current market conditions")

    para1 = (
        f"Markets are currently in {regime_desc}. "
        f"Your portfolio's current volatility is running at {curr_vol:.0%}, "
        f"{'which is aligned with' if abs(curr_vol - target_vol) < 0.03 else 'which differs from'} "
        f"your target of {target_vol:.0%}. "
    )

    if alerts:
        para1 += f"Our options-market analysis has flagged {len(alerts)} asset(s) with elevated risk signals."
    else:
        para1 += "No individual assets are currently flagged for unusual risk."

    # Key changes
    big_drifts = {k: v for k, v in client_result["drift"].items() if abs(v) > 0.03}
    if big_drifts:
        changes = []
        for t, d in sorted(big_drifts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
            direction = "increase" if d < 0 else "reduce"  # drift = current - optimal
            changes.append(f"{direction} {t} by about {abs(d):.0%}")
        para2 = f"Based on the analysis, the key suggested adjustments are to {', '.join(changes)}. "
        para2 += f"Overall, your portfolio has drifted {drift:.0%} from the options-market-informed optimal."
    else:
        para2 = f"Your current allocation is close to the options-market-informed optimal, with only {drift:.0%} total drift. No major rebalancing is suggested at this time."

    # Stress context
    stress = client_result["stress_tests"]
    worst_scenario = min(stress.items(), key=lambda x: x[1]["current"])
    para3 = (
        f"In a {worst_scenario[0].replace('_', ' ')} scenario, your current portfolio would see "
        f"an estimated {abs(worst_scenario[1]['current']):.0%} drawdown, compared to "
        f"{abs(worst_scenario[1]['iv_adjusted']):.0%} with the recommended allocation. "
        f"This analysis incorporates real-time options market pricing to identify risks "
        f"before they appear in historical price data."
    )

    return f"{para1}\n\n{para2}\n\n{para3}"


def generate_chatbot_response(
    question: str,
    client_result: dict,
    signal_data: dict,
    api_key: str = None,
) -> str:
    """Handle a follow-up question from the chatbot."""
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            context = build_narrator_prompt(client_result, signal_data)

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=NARRATOR_SYSTEM_PROMPT + "\n\nAnswer the follow-up question using the context provided. Be concise.",
                messages=[
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning("Chatbot API call failed: %s", e)

    return "I'm unable to connect to the AI service right now. Please try again in a moment."
