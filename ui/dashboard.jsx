import { useState, useEffect, useCallback, useRef } from "react";
import { AlertTriangle, Shield, TrendingUp, TrendingDown, ChevronRight, MessageCircle, Send, X, Activity, BarChart3, Users, Zap, ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";

// ─── MOCK DATA (replace with /api/* calls when backend is running) ───
const MOCK_DATA = {
  regime: "NORMAL",
  regime_probs: { LOW_VOL: 0.15, NORMAL: 0.72, STRESS: 0.13 },
  alerts: [
    "NVDA: ELEVATED_FEAR (IVR=1.63, regime=NORMAL)",
    "TSLA: ELEVATED_FEAR (IVR=1.58, regime=NORMAL)",
  ],
  tickers: {
    AAPL: { ticker: "AAPL", atm_iv: 0.221, predicted_hv: 0.189, ivr: 1.169, fear_level: "NONE", iv_percentile: 0.45, recommended_action: "hold" },
    MSFT: { ticker: "MSFT", atm_iv: 0.198, predicted_hv: 0.175, ivr: 1.131, fear_level: "NONE", iv_percentile: 0.38, recommended_action: "hold" },
    GOOGL: { ticker: "GOOGL", atm_iv: 0.245, predicted_hv: 0.208, ivr: 1.178, fear_level: "NONE", iv_percentile: 0.52, recommended_action: "hold" },
    AMZN: { ticker: "AMZN", atm_iv: 0.268, predicted_hv: 0.231, ivr: 1.16, fear_level: "NONE", iv_percentile: 0.48, recommended_action: "hold" },
    NVDA: { ticker: "NVDA", atm_iv: 0.482, predicted_hv: 0.296, ivr: 1.628, fear_level: "ELEVATED_FEAR", iv_percentile: 0.88, recommended_action: "reduce_moderate" },
    JPM: { ticker: "JPM", atm_iv: 0.185, predicted_hv: 0.162, ivr: 1.142, fear_level: "NONE", iv_percentile: 0.35, recommended_action: "hold" },
    BAC: { ticker: "BAC", atm_iv: 0.221, predicted_hv: 0.198, ivr: 1.116, fear_level: "NONE", iv_percentile: 0.41, recommended_action: "hold" },
    V: { ticker: "V", atm_iv: 0.178, predicted_hv: 0.155, ivr: 1.148, fear_level: "NONE", iv_percentile: 0.33, recommended_action: "hold" },
    TSLA: { ticker: "TSLA", atm_iv: 0.589, predicted_hv: 0.373, ivr: 1.579, fear_level: "ELEVATED_FEAR", iv_percentile: 0.82, recommended_action: "reduce_moderate" },
    META: { ticker: "META", atm_iv: 0.312, predicted_hv: 0.265, ivr: 1.177, fear_level: "NONE", iv_percentile: 0.55, recommended_action: "hold" },
    GLD: { ticker: "GLD", atm_iv: 0.142, predicted_hv: 0.128, ivr: 1.109, fear_level: "NONE", iv_percentile: 0.28, recommended_action: "hold" },
    TLT: { ticker: "TLT", atm_iv: 0.168, predicted_hv: 0.151, ivr: 1.113, fear_level: "NONE", iv_percentile: 0.31, recommended_action: "hold" },
  },
  clients: {
    margaret: {
      client_id: "margaret", name: "Margaret Chen", risk_profile: "conservative",
      target_vol: 0.08, current_vol: 0.072, vol_misaligned: false, total_drift: 0.06,
      goals: "Capital preservation with modest income. Approaching retirement in 3 years.",
      iv_adjusted_sharpe: 0.82, current_sharpe: 0.71,
      expected_return: 0.052, n_alerts: 0,
      current_weights: { AAPL: 0.05, MSFT: 0.05, GOOGL: 0.03, AMZN: 0.02, NVDA: 0.02, JPM: 0.08, BAC: 0.05, V: 0.05, TSLA: 0.00, META: 0.00, GLD: 0.25, TLT: 0.40 },
      iv_adjusted_optimal: { AAPL: 0.06, MSFT: 0.06, GOOGL: 0.03, AMZN: 0.02, NVDA: 0.01, JPM: 0.07, BAC: 0.04, V: 0.06, TSLA: 0.00, META: 0.00, GLD: 0.27, TLT: 0.38 },
      stress_tests: { "2008_GFC": { current: -0.08, iv_adjusted: -0.06 }, "2020_COVID": { current: -0.05, iv_adjusted: -0.04 }, "2022_RATE_SHOCK": { current: -0.12, iv_adjusted: -0.09 } },
      narrative: "Markets are currently in normal conditions. Your portfolio\u2019s volatility at 7.2% is well-aligned with your 8% target, which is exactly where you want to be as you approach retirement.\n\nThe analysis suggests minor adjustments: slightly increasing gold allocation and reducing some financial sector exposure. Overall drift is minimal at 6%, indicating your portfolio is well-positioned.\n\nIn a 2022-style rate shock, your current allocation would see about 12% drawdown versus 9% with the recommended allocation. The options market is not signaling unusual risk for your conservative holdings."
    },
    james: {
      client_id: "james", name: "James Whitfield", risk_profile: "moderate",
      target_vol: 0.12, current_vol: 0.118, vol_misaligned: false, total_drift: 0.09,
      goals: "Balanced growth. Mid-career professional building long-term wealth.",
      iv_adjusted_sharpe: 0.91, current_sharpe: 0.78,
      expected_return: 0.081, n_alerts: 1,
      current_weights: { AAPL: 0.10, MSFT: 0.10, GOOGL: 0.08, AMZN: 0.05, NVDA: 0.05, JPM: 0.08, BAC: 0.05, V: 0.07, TSLA: 0.02, META: 0.03, GLD: 0.15, TLT: 0.22 },
      iv_adjusted_optimal: { AAPL: 0.11, MSFT: 0.11, GOOGL: 0.08, AMZN: 0.06, NVDA: 0.03, JPM: 0.08, BAC: 0.04, V: 0.08, TSLA: 0.01, META: 0.03, GLD: 0.16, TLT: 0.21 },
      stress_tests: { "2008_GFC": { current: -0.18, iv_adjusted: -0.14 }, "2020_COVID": { current: -0.12, iv_adjusted: -0.10 }, "2022_RATE_SHOCK": { current: -0.20, iv_adjusted: -0.16 } },
      narrative: "Markets are in normal conditions, and your balanced portfolio is tracking close to your 12% volatility target at 11.8%.\n\nThe key suggested change is reducing NVDA exposure from 5% to 3% \u2014 the options market is pricing in elevated risk for NVDA that our model confirms hasn\u2019t yet appeared in price history. This frees capital to add slightly to blue-chip positions.\n\nStress testing shows your current allocation would lose about 20% in a rate shock scenario versus 16% with the options-informed adjustment. The overall drift of 9% suggests a modest rebalance would be beneficial."
    },
    sarah: {
      client_id: "sarah", name: "Sarah Okonkwo", risk_profile: "growth",
      target_vol: 0.16, current_vol: 0.171, vol_misaligned: true, total_drift: 0.12,
      goals: "Aggressive growth, 15+ year horizon. Willing to tolerate drawdowns.",
      iv_adjusted_sharpe: 0.95, current_sharpe: 0.82,
      expected_return: 0.121, n_alerts: 2,
      current_weights: { AAPL: 0.12, MSFT: 0.12, GOOGL: 0.10, AMZN: 0.08, NVDA: 0.10, JPM: 0.06, BAC: 0.04, V: 0.06, TSLA: 0.05, META: 0.07, GLD: 0.10, TLT: 0.10 },
      iv_adjusted_optimal: { AAPL: 0.14, MSFT: 0.13, GOOGL: 0.10, AMZN: 0.09, NVDA: 0.05, JPM: 0.06, BAC: 0.04, V: 0.07, TSLA: 0.02, META: 0.07, GLD: 0.11, TLT: 0.12 },
      stress_tests: { "2008_GFC": { current: -0.29, iv_adjusted: -0.23 }, "2020_COVID": { current: -0.21, iv_adjusted: -0.17 }, "2022_RATE_SHOCK": { current: -0.31, iv_adjusted: -0.25 } },
      narrative: "Markets are normal but your portfolio is running slightly hot at 17.1% vol against your 16% target. This is driven by concentrated positions in NVDA and TSLA, both flagged by our options-market analysis.\n\nThe recommended changes: reduce NVDA from 10% to 5% and TSLA from 5% to 2%. Both show elevated implied-to-predicted volatility ratios, meaning the options market sees risk that historical data hasn\u2019t captured yet. Redirect to large-cap and bond exposure.\n\nIn a 2008-style crisis, the adjustment reduces your estimated drawdown from 29% to 23% \u2014 a meaningful difference over your 15-year horizon."
    },
    david: {
      client_id: "david", name: "David Park", risk_profile: "aggressive",
      target_vol: 0.22, current_vol: 0.248, vol_misaligned: true, total_drift: 0.18,
      goals: "Maximum growth. Tech-concentrated, high risk tolerance. 20+ year horizon.",
      iv_adjusted_sharpe: 0.88, current_sharpe: 0.69,
      expected_return: 0.152, n_alerts: 2,
      current_weights: { AAPL: 0.15, MSFT: 0.12, GOOGL: 0.10, AMZN: 0.08, NVDA: 0.20, JPM: 0.03, BAC: 0.02, V: 0.05, TSLA: 0.10, META: 0.10, GLD: 0.03, TLT: 0.02 },
      iv_adjusted_optimal: { AAPL: 0.16, MSFT: 0.14, GOOGL: 0.12, AMZN: 0.10, NVDA: 0.09, JPM: 0.04, BAC: 0.03, V: 0.06, TSLA: 0.04, META: 0.08, GLD: 0.06, TLT: 0.08 },
      stress_tests: { "2008_GFC": { current: -0.41, iv_adjusted: -0.29 }, "2020_COVID": { current: -0.31, iv_adjusted: -0.23 }, "2022_RATE_SHOCK": { current: -0.43, iv_adjusted: -0.32 } },
      narrative: "Your portfolio volatility at 24.8% significantly exceeds your 22% target, primarily due to concentrated NVDA (20%) and TSLA (10%) positions \u2014 both flagged with elevated fear signals.\n\nThe options market is pricing in substantially more risk for NVDA than our model predicts will materialise historically. The recommended rebalance trims NVDA from 20% to 9% and TSLA from 10% to 4%, redirecting to diversified tech and adding some defensive exposure. This brings your Sharpe ratio from 0.69 to 0.88.\n\nCritically, in a 2008-style scenario, your current portfolio faces a 41% drawdown. The IV-adjusted allocation reduces this to 29% \u2014 your 20-year horizon survives both, but the adjusted portfolio recovers faster."
    },
    elena: {
      client_id: "elena", name: "Elena Vasquez", risk_profile: "moderate-growth",
      target_vol: 0.14, current_vol: 0.145, vol_misaligned: false, total_drift: 0.10,
      goals: "Growth with downside protection. Business owner, needs liquidity optionality.",
      iv_adjusted_sharpe: 0.93, current_sharpe: 0.80,
      expected_return: 0.101, n_alerts: 1,
      current_weights: { AAPL: 0.12, MSFT: 0.10, GOOGL: 0.08, AMZN: 0.06, NVDA: 0.08, JPM: 0.06, BAC: 0.04, V: 0.06, TSLA: 0.04, META: 0.06, GLD: 0.12, TLT: 0.18 },
      iv_adjusted_optimal: { AAPL: 0.13, MSFT: 0.12, GOOGL: 0.08, AMZN: 0.07, NVDA: 0.04, JPM: 0.06, BAC: 0.04, V: 0.07, TSLA: 0.02, META: 0.06, GLD: 0.13, TLT: 0.18 },
      stress_tests: { "2008_GFC": { current: -0.23, iv_adjusted: -0.18 }, "2020_COVID": { current: -0.16, iv_adjusted: -0.13 }, "2022_RATE_SHOCK": { current: -0.25, iv_adjusted: -0.20 } },
      narrative: "Your portfolio is well-positioned in current normal market conditions, with volatility at 14.5% tracking close to your 14% target.\n\nThe main adjustment: reduce NVDA from 8% to 4% and redirect to MSFT and gold. As a business owner needing liquidity optionality, the defensive shift protects against correlated drawdowns between your business income and tech-heavy portfolio.\n\nThe options market shows one active flag (NVDA). In a severe downturn, the recommended allocation reduces your maximum drawdown from 25% to 20%, preserving capital you might need to access."
    },
  },
};

// ─── DESIGN TOKENS ───
const COLORS = {
  bg: "#0a0e17",
  surface: "#111827",
  surfaceHover: "#1a2332",
  border: "#1e293b",
  borderAccent: "#334155",
  text: "#e2e8f0",
  textMuted: "#94a3b8",
  textDim: "#64748b",
  accent: "#22d3ee",
  accentGlow: "rgba(34, 211, 238, 0.15)",
  green: "#34d399",
  greenDim: "rgba(52, 211, 153, 0.15)",
  red: "#f87171",
  redDim: "rgba(248, 113, 113, 0.15)",
  orange: "#fbbf24",
  orangeDim: "rgba(251, 191, 36, 0.15)",
  purple: "#a78bfa",
};

const REGIME_COLORS = {
  LOW_VOL: { bg: COLORS.greenDim, text: COLORS.green, label: "Low Vol" },
  NORMAL: { bg: COLORS.accentGlow, text: COLORS.accent, label: "Normal" },
  STRESS: { bg: COLORS.orangeDim, text: COLORS.orange, label: "Stress" },
  CRISIS: { bg: COLORS.redDim, text: COLORS.red, label: "Crisis" },
};

const FEAR_COLORS = {
  NONE: { bg: "transparent", text: COLORS.textMuted, icon: null },
  ELEVATED_FEAR: { bg: COLORS.orangeDim, text: COLORS.orange, icon: "⚠" },
  HIGH_FEAR: { bg: COLORS.redDim, text: COLORS.red, icon: "🔴" },
};

// ─── COMPONENTS ───

function RegimeBadge({ regime }) {
  const r = REGIME_COLORS[regime] || REGIME_COLORS.NORMAL;
  return (
    <span style={{ background: r.bg, color: r.text, padding: "4px 12px", borderRadius: 20, fontSize: 13, fontWeight: 600, letterSpacing: 0.5 }}>
      {r.label}
    </span>
  );
}

function RegimeBar({ probs }) {
  const order = ["LOW_VOL", "NORMAL", "STRESS", "CRISIS"];
  const active = order.filter(k => probs[k]);
  return (
    <div style={{ display: "flex", gap: 2, height: 6, borderRadius: 3, overflow: "hidden", marginTop: 8 }}>
      {active.map(k => (
        <div key={k} style={{ flex: probs[k], background: REGIME_COLORS[k]?.text || "#666", opacity: 0.8, transition: "flex 0.5s ease" }} title={`${k}: ${(probs[k]*100).toFixed(0)}%`} />
      ))}
    </div>
  );
}

function StatCard({ label, value, subtitle, color, icon: Icon }) {
  return (
    <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: "16px 20px", flex: 1, minWidth: 160 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        {Icon && <Icon size={16} color={color || COLORS.textMuted} />}
        <span style={{ color: COLORS.textMuted, fontSize: 12, textTransform: "uppercase", letterSpacing: 1 }}>{label}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || COLORS.text, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
      {subtitle && <div style={{ color: COLORS.textDim, fontSize: 12, marginTop: 4 }}>{subtitle}</div>}
    </div>
  );
}

function ClientCard({ client, signals, onSelect, isSelected }) {
  const alertCount = Object.values(signals.tickers || {}).filter(t =>
    client.current_weights[t.ticker] > 0 && t.fear_level !== "NONE"
  ).length;
  const volDelta = client.current_vol - client.target_vol;
  const isHot = volDelta > 0.02;
  const isCold = volDelta < -0.02;

  return (
    <div
      onClick={() => onSelect(client.client_id)}
      style={{
        background: isSelected ? COLORS.surfaceHover : COLORS.surface,
        border: `1px solid ${isSelected ? COLORS.accent : COLORS.border}`,
        borderRadius: 14,
        padding: 20,
        cursor: "pointer",
        transition: "all 0.2s ease",
        position: "relative",
        overflow: "hidden",
      }}
      onMouseEnter={e => { if (!isSelected) e.currentTarget.style.borderColor = COLORS.borderAccent; }}
      onMouseLeave={e => { if (!isSelected) e.currentTarget.style.borderColor = COLORS.border; }}
    >
      {/* Alert indicator */}
      {(alertCount > 0 || client.vol_misaligned) && (
        <div style={{ position: "absolute", top: 12, right: 12, display: "flex", gap: 6 }}>
          {alertCount > 0 && (
            <span style={{ background: COLORS.orangeDim, color: COLORS.orange, borderRadius: 12, padding: "2px 8px", fontSize: 11, fontWeight: 600 }}>
              {alertCount} alert{alertCount > 1 ? "s" : ""}
            </span>
          )}
          {client.vol_misaligned && (
            <span style={{ background: COLORS.redDim, color: COLORS.red, borderRadius: 12, padding: "2px 8px", fontSize: 11, fontWeight: 600 }}>
              Vol drift
            </span>
          )}
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: `linear-gradient(135deg, ${COLORS.accent}22, ${COLORS.purple}22)`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 18, fontWeight: 700, color: COLORS.accent
        }}>
          {client.name.split(" ").map(n => n[0]).join("")}
        </div>
        <div>
          <div style={{ fontWeight: 600, color: COLORS.text, fontSize: 15 }}>{client.name}</div>
          <div style={{ color: COLORS.textDim, fontSize: 12, textTransform: "capitalize" }}>{client.risk_profile}</div>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <div>
          <div style={{ color: COLORS.textDim, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>Current Vol</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 16, fontWeight: 600, color: isHot ? COLORS.red : isCold ? COLORS.accent : COLORS.text }}>
            {(client.current_vol * 100).toFixed(1)}%
            {isHot && <ArrowUpRight size={14} style={{ marginLeft: 2 }} />}
            {isCold && <ArrowDownRight size={14} style={{ marginLeft: 2 }} />}
          </div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ color: COLORS.textDim, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>Target</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 16, fontWeight: 600, color: COLORS.textMuted }}>{(client.target_vol * 100).toFixed(0)}%</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ color: COLORS.textDim, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>Sharpe</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 16, fontWeight: 600, color: COLORS.green }}>{client.iv_adjusted_sharpe}</div>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12, paddingTop: 12, borderTop: `1px solid ${COLORS.border}` }}>
        <div style={{ flex: 1, fontSize: 12, color: COLORS.textDim }}>
          Drift: <span style={{ color: client.total_drift > 0.1 ? COLORS.orange : COLORS.textMuted, fontWeight: 600 }}>{(client.total_drift * 100).toFixed(0)}%</span>
        </div>
        <ChevronRight size={16} color={COLORS.textDim} />
      </div>
    </div>
  );
}

function WeightBar({ ticker, current, optimal, maxW = 0.25 }) {
  const delta = optimal - current;
  const scale = 100 / maxW;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 0" }}>
      <div style={{ width: 48, fontSize: 12, fontWeight: 600, color: COLORS.text, fontFamily: "'JetBrains Mono', monospace" }}>{ticker}</div>
      <div style={{ flex: 1, position: "relative", height: 18, background: COLORS.bg, borderRadius: 4, overflow: "hidden" }}>
        <div style={{ position: "absolute", height: "100%", width: `${current * scale}%`, background: `${COLORS.accent}44`, borderRadius: 4, transition: "width 0.3s" }} />
        <div style={{ position: "absolute", height: "100%", width: 2, left: `${optimal * scale}%`, background: COLORS.green, transition: "left 0.3s" }} />
      </div>
      <div style={{ width: 50, textAlign: "right", fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: Math.abs(delta) > 0.02 ? (delta > 0 ? COLORS.green : COLORS.red) : COLORS.textDim }}>
        {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
      </div>
    </div>
  );
}

function SignalRow({ signal }) {
  const fear = FEAR_COLORS[signal.fear_level];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 12px", borderRadius: 8, background: signal.fear_level !== "NONE" ? fear.bg : "transparent" }}>
      <div style={{ width: 48, fontWeight: 600, fontSize: 13, color: COLORS.text, fontFamily: "'JetBrains Mono', monospace" }}>{signal.ticker}</div>
      <div style={{ flex: 1, display: "flex", gap: 16 }}>
        <div style={{ fontSize: 12 }}>
          <span style={{ color: COLORS.textDim }}>IV </span>
          <span style={{ color: COLORS.text, fontFamily: "'JetBrains Mono', monospace" }}>{(signal.atm_iv * 100).toFixed(1)}%</span>
        </div>
        <div style={{ fontSize: 12 }}>
          <span style={{ color: COLORS.textDim }}>Pred </span>
          <span style={{ color: COLORS.text, fontFamily: "'JetBrains Mono', monospace" }}>{(signal.predicted_hv * 100).toFixed(1)}%</span>
        </div>
        <div style={{ fontSize: 12 }}>
          <span style={{ color: COLORS.textDim }}>IVR </span>
          <span style={{ color: signal.ivr > 1.3 ? COLORS.orange : COLORS.text, fontWeight: signal.ivr > 1.3 ? 600 : 400, fontFamily: "'JetBrains Mono', monospace" }}>{signal.ivr.toFixed(2)}</span>
        </div>
      </div>
      <div style={{ fontSize: 12, fontWeight: 600, color: fear.text }}>
        {signal.fear_level === "NONE" ? "" : signal.fear_level.replace("_", " ")}
      </div>
    </div>
  );
}

function ChatPanel({ clientId, clientData, signals, onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    setMessages([{
      role: "assistant",
      content: clientData?.narrative || "Hello! I can help explain the portfolio analysis. What would you like to know?"
    }]);
  }, [clientId, clientData?.narrative]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return;
    const q = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: q }]);
    setLoading(true);

    try {
      const resp = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: `You are a financial advisor's AI assistant. Answer questions about the portfolio analysis below. Be specific with numbers. Use plain English.\n\nClient: ${clientData?.name} (${clientData?.risk_profile})\nTarget vol: ${(clientData?.target_vol*100)}%\nCurrent vol: ${(clientData?.current_vol*100).toFixed(1)}%\nRegime: ${signals?.regime}\nAlerts: ${JSON.stringify(signals?.alerts)}\nStress tests: ${JSON.stringify(clientData?.stress_tests)}`,
          messages: [{ role: "user", content: q }]
        })
      });
      const data = await resp.json();
      const text = data.content?.map(c => c.text || "").join("") || "I couldn't process that request.";
      setMessages(prev => [...prev, { role: "assistant", content: text }]);
    } catch {
      setMessages(prev => [...prev, { role: "assistant", content: "I'm having trouble connecting right now. The key information is in the analysis above." }]);
    }
    setLoading(false);
  }, [input, loading, clientData, signals]);

  return (
    <div style={{ position: "fixed", bottom: 0, right: 0, width: 420, height: "70vh", background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: "16px 0 0 0", display: "flex", flexDirection: "column", zIndex: 100, boxShadow: "0 -4px 30px rgba(0,0,0,0.4)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 18px", borderBottom: `1px solid ${COLORS.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <MessageCircle size={16} color={COLORS.accent} />
          <span style={{ fontWeight: 600, color: COLORS.text, fontSize: 14 }}>Chat — {clientData?.name?.split(" ")[0]}</span>
        </div>
        <X size={18} color={COLORS.textDim} style={{ cursor: "pointer" }} onClick={onClose} />
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ alignSelf: m.role === "user" ? "flex-end" : "flex-start", maxWidth: "85%", background: m.role === "user" ? `${COLORS.accent}22` : COLORS.bg, border: `1px solid ${m.role === "user" ? `${COLORS.accent}44` : COLORS.border}`, borderRadius: 12, padding: "10px 14px", fontSize: 13, color: COLORS.text, lineHeight: 1.5, whiteSpace: "pre-wrap" }}>
            {m.content}
          </div>
        ))}
        {loading && <div style={{ color: COLORS.textDim, fontSize: 12, fontStyle: "italic" }}>Thinking...</div>}
        <div ref={scrollRef} />
      </div>
      <div style={{ padding: "12px 16px", borderTop: `1px solid ${COLORS.border}`, display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
          placeholder="Ask about signals, regime, allocations..."
          style={{ flex: 1, background: COLORS.bg, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: "8px 12px", color: COLORS.text, fontSize: 13, outline: "none" }}
        />
        <button onClick={sendMessage} style={{ background: COLORS.accent, border: "none", borderRadius: 8, padding: "8px 12px", cursor: "pointer", display: "flex", alignItems: "center" }}>
          <Send size={16} color={COLORS.bg} />
        </button>
      </div>
    </div>
  );
}

// ─── MAIN DASHBOARD ───

export default function AdvisorIQDashboard() {
  const [data] = useState(MOCK_DATA);
  const [selectedClient, setSelectedClient] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);

  const clientData = selectedClient ? data.clients[selectedClient] : null;
  const alertTickers = Object.values(data.tickers).filter(t => t.fear_level !== "NONE");

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ borderBottom: `1px solid ${COLORS.border}`, padding: "16px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.purple})`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Shield size={18} color="#fff" />
          </div>
          <span style={{ fontSize: 20, fontWeight: 700, letterSpacing: -0.5 }}>AdvisorIQ</span>
          <span style={{ color: COLORS.textDim, fontSize: 13, marginLeft: 8 }}>Options-Informed Portfolio Intelligence</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: COLORS.textDim, fontSize: 12 }}>REGIME</span>
            <RegimeBadge regime={data.regime} />
          </div>
          {data.alerts.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, background: COLORS.orangeDim, padding: "4px 12px", borderRadius: 20 }}>
              <AlertTriangle size={14} color={COLORS.orange} />
              <span style={{ color: COLORS.orange, fontSize: 13, fontWeight: 600 }}>{data.alerts.length} Alert{data.alerts.length > 1 ? "s" : ""}</span>
            </div>
          )}
        </div>
      </div>

      <div style={{ padding: "24px 32px", maxWidth: 1400, margin: "0 auto" }}>
        {/* Top Stats */}
        <div style={{ display: "flex", gap: 16, marginBottom: 24 }}>
          <StatCard label="Market Regime" value={REGIME_COLORS[data.regime]?.label} color={REGIME_COLORS[data.regime]?.text} icon={Activity} subtitle={
            <RegimeBar probs={data.regime_probs} />
          } />
          <StatCard label="Active Alerts" value={data.alerts.length} color={data.alerts.length > 0 ? COLORS.orange : COLORS.green} icon={AlertTriangle} subtitle={data.alerts.length === 0 ? "All clear" : alertTickers.map(t => t.ticker).join(", ")} />
          <StatCard label="Clients" value={Object.keys(data.clients).length} icon={Users} subtitle={`${Object.values(data.clients).filter(c => c.vol_misaligned).length} vol-misaligned`} />
          <StatCard label="Tickers Monitored" value={Object.keys(data.tickers).length} icon={BarChart3} subtitle="12 liquid options names" />
        </div>

        <div style={{ display: "flex", gap: 24 }}>
          {/* Left: Client Cards */}
          <div style={{ width: 380, flexShrink: 0 }}>
            <div style={{ color: COLORS.textMuted, fontSize: 12, textTransform: "uppercase", letterSpacing: 1, marginBottom: 12, fontWeight: 600 }}>Client Book</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {Object.values(data.clients).map(client => (
                <ClientCard
                  key={client.client_id}
                  client={client}
                  signals={data}
                  onSelect={setSelectedClient}
                  isSelected={selectedClient === client.client_id}
                />
              ))}
            </div>
          </div>

          {/* Right: Detail Panel */}
          <div style={{ flex: 1, minWidth: 0 }}>
            {clientData ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                {/* Client Header */}
                <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 24 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                    <div>
                      <h2 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>{clientData.name}</h2>
                      <div style={{ color: COLORS.textDim, fontSize: 13, marginTop: 4 }}>{clientData.goals}</div>
                    </div>
                    <button
                      onClick={() => setChatOpen(!chatOpen)}
                      style={{ background: `${COLORS.accent}22`, border: `1px solid ${COLORS.accent}44`, borderRadius: 10, padding: "8px 16px", cursor: "pointer", display: "flex", alignItems: "center", gap: 6, color: COLORS.accent, fontSize: 13, fontWeight: 600 }}
                    >
                      <MessageCircle size={14} /> Ask AI
                    </button>
                  </div>

                  {/* Key Metrics Row */}
                  <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
                    {[
                      { label: "Current Vol", value: `${(clientData.current_vol*100).toFixed(1)}%`, color: clientData.vol_misaligned ? COLORS.red : COLORS.text },
                      { label: "Target Vol", value: `${(clientData.target_vol*100).toFixed(0)}%`, color: COLORS.textMuted },
                      { label: "Exp. Return", value: `${(clientData.expected_return*100).toFixed(1)}%`, color: COLORS.green },
                      { label: "Sharpe (IV-adj)", value: clientData.iv_adjusted_sharpe.toFixed(2), color: COLORS.accent },
                      { label: "Drift", value: `${(clientData.total_drift*100).toFixed(0)}%`, color: clientData.total_drift > 0.1 ? COLORS.orange : COLORS.textMuted },
                    ].map(m => (
                      <div key={m.label}>
                        <div style={{ color: COLORS.textDim, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>{m.label}</div>
                        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 20, fontWeight: 700, color: m.color }}>{m.value}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Narrative */}
                <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 24 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <Zap size={16} color={COLORS.accent} />
                    <span style={{ color: COLORS.textMuted, fontSize: 12, textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>AI Analysis</span>
                  </div>
                  <div style={{ color: COLORS.text, fontSize: 14, lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
                    {clientData.narrative}
                  </div>
                </div>

                {/* Weight Comparison */}
                <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 24 }}>
                  <div style={{ color: COLORS.textMuted, fontSize: 12, textTransform: "uppercase", letterSpacing: 1, fontWeight: 600, marginBottom: 4 }}>
                    Portfolio Allocation — Current vs IV-Adjusted Optimal
                  </div>
                  <div style={{ display: "flex", gap: 8, marginBottom: 12, fontSize: 11, color: COLORS.textDim }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 12, height: 8, background: `${COLORS.accent}44`, borderRadius: 2 }} /> Current</span>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}><div style={{ width: 2, height: 8, background: COLORS.green }} /> Optimal</span>
                  </div>
                  {Object.keys(clientData.current_weights).filter(t => clientData.current_weights[t] > 0 || (clientData.iv_adjusted_optimal[t] || 0) > 0).map(ticker => (
                    <WeightBar
                      key={ticker}
                      ticker={ticker}
                      current={clientData.current_weights[ticker] || 0}
                      optimal={clientData.iv_adjusted_optimal[ticker] || 0}
                    />
                  ))}
                </div>

                {/* Stress Tests */}
                <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 24 }}>
                  <div style={{ color: COLORS.textMuted, fontSize: 12, textTransform: "uppercase", letterSpacing: 1, fontWeight: 600, marginBottom: 16 }}>Stress Test Scenarios</div>
                  <div style={{ display: "flex", gap: 16 }}>
                    {Object.entries(clientData.stress_tests).map(([scenario, vals]) => (
                      <div key={scenario} style={{ flex: 1, background: COLORS.bg, borderRadius: 10, padding: 16 }}>
                        <div style={{ fontSize: 12, color: COLORS.textDim, fontWeight: 600, marginBottom: 12 }}>{scenario.replace(/_/g, " ")}</div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ fontSize: 11, color: COLORS.textDim }}>Current</span>
                          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 14, fontWeight: 600, color: COLORS.red }}>{(vals.current * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ fontSize: 11, color: COLORS.textDim }}>IV-Adjusted</span>
                          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 14, fontWeight: 600, color: COLORS.green }}>{(vals.iv_adjusted * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ fontSize: 11, color: COLORS.accent, marginTop: 8, fontWeight: 500 }}>
                          Δ {((vals.iv_adjusted - vals.current) * 100).toFixed(0)}% less drawdown
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Signal Table */}
                <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 24 }}>
                  <div style={{ color: COLORS.textMuted, fontSize: 12, textTransform: "uppercase", letterSpacing: 1, fontWeight: 600, marginBottom: 12 }}>IVR Signal Monitor</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                    {Object.values(data.tickers)
                      .sort((a, b) => b.ivr - a.ivr)
                      .map(signal => <SignalRow key={signal.ticker} signal={signal} />)}
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 400, background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14 }}>
                <div style={{ textAlign: "center", color: COLORS.textDim }}>
                  <Users size={48} style={{ marginBottom: 16, opacity: 0.3 }} />
                  <div style={{ fontSize: 16, fontWeight: 500 }}>Select a client to view analysis</div>
                  <div style={{ fontSize: 13, marginTop: 8 }}>Click a client card to see their portfolio details, signals, and AI narrative.</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Chat Panel */}
      {chatOpen && clientData && (
        <ChatPanel
          clientId={selectedClient}
          clientData={clientData}
          signals={data}
          onClose={() => setChatOpen(false)}
        />
      )}
    </div>
  );
}
