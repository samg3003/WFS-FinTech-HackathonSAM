"""
AdvisorIQ — Client Profile Definitions
Each client has a risk tolerance (target annual vol), investment goals,
and a current portfolio allocation. The system optimises per-client.
"""
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class ClientProfile:
    name: str
    client_id: str
    risk_profile: str                      # conservative / moderate / growth / aggressive
    target_annual_vol: float               # target portfolio vol (annualised, decimal)
    target_return: float                   # target annual return (decimal)
    max_single_position: float             # max weight in any single asset
    current_weights: Dict[str, float]      # current allocation (sums to ~1.0)
    goals: str                             # plain-English investment goals
    constraints: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# The Five Clients
# ─────────────────────────────────────────────────────────────────────

CLIENTS: Dict[str, ClientProfile] = {

    "margaret": ClientProfile(
        name="Margaret Chen",
        client_id="margaret",
        risk_profile="conservative",
        target_annual_vol=0.08,
        target_return=0.05,
        max_single_position=0.15,
        current_weights={
            "AAPL": 0.05, "MSFT": 0.05, "GOOGL": 0.03, "AMZN": 0.02, "NVDA": 0.02,
            "JPM": 0.08, "BAC": 0.05, "V": 0.05,
            "TSLA": 0.00, "META": 0.00,
            "GLD": 0.25, "TLT": 0.40,
        },
        goals="Capital preservation with modest income. Approaching retirement in 3 years.",
        constraints=["No TSLA/META exposure", "Min 30% bonds/gold"],
    ),

    "james": ClientProfile(
        name="James Whitfield",
        client_id="james",
        risk_profile="moderate",
        target_annual_vol=0.12,
        target_return=0.08,
        max_single_position=0.15,
        current_weights={
            "AAPL": 0.10, "MSFT": 0.10, "GOOGL": 0.08, "AMZN": 0.05, "NVDA": 0.05,
            "JPM": 0.08, "BAC": 0.05, "V": 0.07,
            "TSLA": 0.02, "META": 0.03,
            "GLD": 0.15, "TLT": 0.22,
        },
        goals="Balanced growth. Mid-career professional building long-term wealth.",
        constraints=["Max 60% equities"],
    ),

    "sarah": ClientProfile(
        name="Sarah Okonkwo",
        client_id="sarah",
        risk_profile="growth",
        target_annual_vol=0.16,
        target_return=0.12,
        max_single_position=0.20,
        current_weights={
            "AAPL": 0.12, "MSFT": 0.12, "GOOGL": 0.10, "AMZN": 0.08, "NVDA": 0.10,
            "JPM": 0.06, "BAC": 0.04, "V": 0.06,
            "TSLA": 0.05, "META": 0.07,
            "GLD": 0.10, "TLT": 0.10,
        },
        goals="Aggressive growth, 15+ year horizon. Willing to tolerate drawdowns.",
        constraints=[],
    ),

    "david": ClientProfile(
        name="David Park",
        client_id="david",
        risk_profile="aggressive",
        target_annual_vol=0.22,
        target_return=0.15,
        max_single_position=0.25,
        current_weights={
            "AAPL": 0.15, "MSFT": 0.12, "GOOGL": 0.10, "AMZN": 0.08, "NVDA": 0.20,
            "JPM": 0.03, "BAC": 0.02, "V": 0.05,
            "TSLA": 0.10, "META": 0.10,
            "GLD": 0.03, "TLT": 0.02,
        },
        goals="Maximum growth. Tech-concentrated, high risk tolerance. 20+ year horizon.",
        constraints=[],
    ),

    "elena": ClientProfile(
        name="Elena Vasquez",
        client_id="elena",
        risk_profile="moderate-growth",
        target_annual_vol=0.14,
        target_return=0.10,
        max_single_position=0.18,
        current_weights={
            "AAPL": 0.12, "MSFT": 0.10, "GOOGL": 0.08, "AMZN": 0.06, "NVDA": 0.08,
            "JPM": 0.06, "BAC": 0.04, "V": 0.06,
            "TSLA": 0.04, "META": 0.06,
            "GLD": 0.12, "TLT": 0.18,
        },
        goals="Growth with downside protection. Business owner, needs liquidity optionality.",
        constraints=["Max 70% equities"],
    ),
}
