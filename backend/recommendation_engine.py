"""
Recommendation Engine — the "agent" layer.

Combines Day 2 (anomaly detection) + Day 4 (failure risk) outputs into
structured findings, then generates human-readable recommendations.

Two-layer design:
    1. RULE-BASED CORE (always runs, zero dependencies, deterministic)
       -> guarantees every anomaly gets SOME actionable recommendation
    2. OPTIONAL LLM ENRICHMENT (Gemini, same pattern as your
       Cardboard Processing project's RFQ Assistant)
       -> rephrases the structured facts into more natural, varied
          language. If the API call fails or no key is set, the
          rule-based text is used as-is -- the agent never breaks.

This separation matters for the interview answer: "what if the LLM API
is down or rate-limited?" -> "the recommendation engine still works,
it just uses templated text instead of LLM-generated text."
"""

import pandas as pd
import os

# ---------------------------------------------------------------------------
# 1. Rule-based recommendation templates
# ---------------------------------------------------------------------------

MACHINE_TYPE_ACTIONS = {
    "Cutter": "inspect the cutting blade for wear or misalignment",
    "Press": "check press pressure calibration and hydraulic seals",
    "Folder": "inspect folding arm alignment and belt tension",
}


def build_finding(row, machine_type: str) -> dict:
    """Turns one flagged anomaly/risk row into a structured finding."""
    def capped_z(z):
        # extremely low-variance baselines can produce inflated z-scores;
        # cap the *displayed* value for readability without changing the
        # underlying breach detection (which already ran on the raw z)
        return min(abs(z), 6.0)

    issues = []
    if row.get("efficiency_pct_breach"):
        issues.append(f"production efficiency dropped {capped_z(row['efficiency_pct_zscore']):.1f} std-dev below normal")
    if row.get("defect_rate_pct_breach"):
        issues.append(f"defect rate spiked {capped_z(row['defect_rate_pct_zscore']):.1f} std-dev above normal")
    if row.get("downtime_pct_breach"):
        issues.append(f"downtime rose {capped_z(row['downtime_pct_zscore']):.1f} std-dev above normal")

    return {
        "date": row["date"],
        "machine_id": row["machine_id"],
        "severity": row["severity"],
        "issues": issues,
    }


def generate_rule_based_recommendation(finding: dict, machine_type: str, failure_probability: float = None) -> str:
    machine_id = finding["machine_id"]
    severity = finding["severity"]
    action = MACHINE_TYPE_ACTIONS.get(machine_type, "inspect the machine for mechanical faults")

    if severity == "HIGH":
        urgency = "Schedule maintenance within 24 hours."
    elif severity == "MEDIUM":
        urgency = "Schedule maintenance within 3-5 days."
    else:
        urgency = "Monitor over the next few shifts; no immediate action required."

    issue_text = "; ".join(finding["issues"]) if finding["issues"] else "abnormal performance pattern detected"

    risk_note = ""
    if failure_probability is not None and failure_probability >= 0.5:
        risk_note = f" Failure risk model estimates a {failure_probability*100:.0f}% probability of breakdown if unaddressed."

    return (
        f"Machine {machine_id}: {issue_text.capitalize()}. "
        f"Recommend engineers {action}.{risk_note} {urgency}"
    )


# ---------------------------------------------------------------------------
# 2. Optional LLM enrichment (Gemini) -- plug-and-play, safe to skip
# ---------------------------------------------------------------------------

def enrich_with_llm(rule_based_text: str, finding: dict) -> str:
    """
    Rephrases the rule-based recommendation using Gemini, if GEMINI_API_KEY
    is set in the environment. Falls back to the rule-based text on any
    failure (missing key, network error, API error) -- the agent must
    never crash or go silent because of the LLM layer.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return rule_based_text  # no key configured -> use rule-based text as-is

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are a manufacturing operations assistant. Rewrite the following "
            "machine maintenance recommendation in clear, professional, concise "
            "language suitable for a factory supervisor. Keep it to 2-3 sentences. "
            "Do not invent facts not present in the original.\n\n"
            f"Original: {rule_based_text}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # LLM layer is enrichment only -- any failure silently falls back
        print(f"  [LLM enrichment skipped: {e}]")
        return rule_based_text


# ---------------------------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------------------------

def run_recommendation_engine(anomalies: pd.DataFrame, failure_risk: pd.DataFrame, machines: pd.DataFrame, use_llm: bool = False) -> pd.DataFrame:
    machine_type_map = machines.set_index("machine_id")["machine_type"].to_dict()

    flagged = anomalies[anomalies["is_anomaly"]].copy()

    # attach failure probability for the same machine+date, where available
    risk_lookup = failure_risk.set_index(["machine_id", "date"])["failure_probability"].to_dict()

    recommendations = []
    for _, row in flagged.iterrows():
        machine_type = machine_type_map.get(row["machine_id"], "Unknown")
        finding = build_finding(row, machine_type)
        fail_prob = risk_lookup.get((row["machine_id"], row["date"]))

        rule_text = generate_rule_based_recommendation(finding, machine_type, fail_prob)
        final_text = enrich_with_llm(rule_text, finding) if use_llm else rule_text

        recommendations.append({
            "date": row["date"],
            "machine_id": row["machine_id"],
            "severity": finding["severity"],
            "failure_probability": fail_prob,
            "recommendation": final_text,
        })

    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    anomalies = pd.read_csv("../data/anomaly_flags.csv", parse_dates=["date"])
    failure_risk = pd.read_csv("../data/failure_risk.csv", parse_dates=["date"])
    machines = pd.read_csv("../data/machines.csv")

    # USE_LLM=1 in environment turns on Gemini enrichment; default is rule-based only
    use_llm = os.environ.get("USE_LLM", "0") == "1"

    recs = run_recommendation_engine(anomalies, failure_risk, machines, use_llm=use_llm)
    recs = recs.sort_values(["severity", "date"], ascending=[True, True])
    recs.to_csv("../data/recommendations.csv", index=False)

    print(f"Generated {len(recs)} recommendations ({'LLM-enriched' if use_llm else 'rule-based'}).\n")
    print("=== Sample HIGH severity recommendations ===")
    high = recs[recs["severity"] == "HIGH"].head(5)
    for _, r in high.iterrows():
        print(f"\n[{r['date'].date()}] {r['recommendation']}")
