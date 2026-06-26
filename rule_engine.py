"""
xZero Rule Engine v3.3
Implementerar use_case_engine och action_strategy_engine deterministiskt.
LLM:en väljer patterns och context — regelmotorn sköter logiken.
"""

from typing import Optional

# ── Lookup-tabeller (use_case_engine.json) ────────────────────────────────────

PATTERN_TO_DECISIONS: dict[str, list[str]] = {
    "inventory_imbalance":      ["volume_decision", "timing_decision"],
    "flow_inefficiency":        ["allocation_decision", "resource_decision"],
    "capacity_misalignment":    ["resource_decision", "allocation_decision"],
    "pricing_leakage":          ["pricing_decision"],
    "forecast_error":           ["volume_decision", "timing_decision"],
    "fragmentation":            ["process_decision", "enterprise_decision"],
    "knowledge_dependency":     ["process_decision", "resource_decision"],
    "process_variability":      ["process_decision"],
    "utilization_gap":          ["resource_decision"],
    "timing_mismatch":          ["timing_decision"],
    "project_margin_leakage":   ["project_decision", "risk_decision", "pricing_decision"],
    "risk_underestimation":     ["risk_decision"],
    "execution_variability":    ["process_decision", "resource_decision"],
    "customer_concentration":   ["portfolio_decision", "risk_decision"],
    "portfolio_misalignment":   ["portfolio_decision", "pricing_decision"],
    "credit_risk_leakage":      ["risk_decision", "pricing_decision"],
    "interest_margin_leakage":  ["pricing_decision", "portfolio_decision"],
    "regulatory_constraint":    ["process_decision", "risk_decision"],
    "skill_mismatch":           ["resource_decision"],
    "talent_utilization_gap":   ["resource_decision"],
    "bench_leakage":            ["resource_decision"],
    "staffing_friction":        ["resource_decision", "process_decision"],
    "retention_risk":           ["resource_decision", "enterprise_decision"],
    "hybrid_complexity":        ["process_decision", "project_decision", "resource_decision"],
}

DECISION_TO_MECHANISMS: dict[str, list[str]] = {
    "volume_decision":    ["demand_forecast", "scenario_simulation", "reorder_logic"],
    "timing_decision":    ["timing_forecast", "price_signal_monitoring", "event_prediction"],
    "allocation_decision":["allocation_optimization", "network_optimization", "priority_rules"],
    "pricing_decision":   ["price_elasticity_model", "risk_adjusted_pricing", "margin_simulation"],
    "resource_decision":  ["capacity_planning", "matching_logic", "utilization_optimization"],
    "risk_decision":      ["risk_scoring", "loss_prediction", "scenario_analysis"],
    "process_decision":   ["decision_standardization", "rule_engine", "workflow_structuring"],
    "project_decision":   ["project_scoring", "project_margin_model", "bid_risk_model"],
    "portfolio_decision": ["portfolio_optimization", "mix_analysis", "capital_allocation_model"],
    "enterprise_decision":["governance_rules", "operating_model_design", "integration_framework"],
}

MECHANISM_CLASSIFICATION: dict[str, str] = {
    "demand_forecast":          "optimization",
    "scenario_simulation":      "optimization",
    "reorder_logic":            "optimization",
    "timing_forecast":          "optimization",
    "price_signal_monitoring":  "optimization",
    "event_prediction":         "optimization",
    "allocation_optimization":  "optimization",
    "network_optimization":     "optimization",
    "priority_rules":           "stabilization",
    "price_elasticity_model":   "optimization",
    "risk_adjusted_pricing":    "optimization",
    "margin_simulation":        "optimization",
    "capacity_planning":        "human_capital",
    "matching_logic":           "human_capital",
    "utilization_optimization": "human_capital",
    "risk_scoring":             "optimization",
    "loss_prediction":          "optimization",
    "scenario_analysis":        "optimization",
    "decision_standardization": "stabilization",
    "rule_engine":              "stabilization",
    "workflow_structuring":     "stabilization",
    "project_scoring":          "optimization",
    "project_margin_model":     "optimization",
    "bid_risk_model":           "optimization",
    "portfolio_optimization":   "optimization",
    "mix_analysis":             "optimization",
    "capital_allocation_model": "optimization",
    "governance_rules":         "stabilization",
    "operating_model_design":   "stabilization",
    "integration_framework":    "stabilization",
}

MECHANISM_LABELS_SV: dict[str, str] = {
    "demand_forecast":          "efterfrågeprognos",
    "scenario_simulation":      "scenariosimulering",
    "reorder_logic":            "påfyllnadslogik",
    "timing_forecast":          "tidsprognos",
    "price_signal_monitoring":  "prissignalövervakning",
    "event_prediction":         "händelseprognos",
    "allocation_optimization":  "allokeringsoptimering",
    "network_optimization":     "nätverksoptimering",
    "priority_rules":           "prioriteringsregler",
    "price_elasticity_model":   "priselasticitetsmodell",
    "risk_adjusted_pricing":    "riskjusterad prissättning",
    "margin_simulation":        "marginalsimulering",
    "capacity_planning":        "kapacitetsplanering",
    "matching_logic":           "matchningslogik",
    "utilization_optimization": "beläggningsoptimering",
    "risk_scoring":             "riskmodellering",
    "loss_prediction":          "förlustprognos",
    "scenario_analysis":        "scenarioanalys",
    "decision_standardization": "beslutsstandardisering",
    "rule_engine":              "regelstyrning",
    "workflow_structuring":     "strukturerade arbetsflöden",
    "project_scoring":          "projektscoring",
    "project_margin_model":     "projektmarginalmodell",
    "bid_risk_model":           "anbuds- och riskmodell",
    "portfolio_optimization":   "portföljoptimering",
    "mix_analysis":             "mixanalys",
    "capital_allocation_model": "kapitalallokeringsmodell",
    "governance_rules":         "styrningsregler",
    "operating_model_design":   "operativ modellutformning",
    "integration_framework":    "integrationsramverk",
    "simple_capacity_planning": "enkel kapacitetsplanering",
}

USE_CASE_TEMPLATES: dict[str, str] = {
    "volume_decision":    "Optimera volymbeslut baserat på {mechanism_sv}",
    "timing_decision":    "Optimera timingbeslut baserat på {mechanism_sv}",
    "allocation_decision":"Optimera allokering baserat på {mechanism_sv}",
    "pricing_decision":   "Optimera prissättning baserat på {mechanism_sv}",
    "resource_decision":  "Optimera resursallokering baserat på {mechanism_sv}",
    "risk_decision":      "Förbättra riskbeslut baserat på {mechanism_sv}",
    "process_decision":   "Standardisera beslutsprocesser genom {mechanism_sv}",
    "project_decision":   "Förbättra projektbeslut baserat på {mechanism_sv}",
    "portfolio_decision": "Optimera portföljbeslut baserat på {mechanism_sv}",
    "enterprise_decision":"Stärka styrning och beslutskapacitet genom {mechanism_sv}",
}

# context_preferences: promote/demote per dimension
PROMOTE: dict[str, dict[str, list[str]]] = {
    "decision_mode": {
        "continuous": ["demand_forecast", "timing_forecast", "allocation_optimization",
                       "capacity_planning", "utilization_optimization", "risk_scoring"],
        "episodic":   ["scenario_analysis", "project_margin_model", "bid_risk_model",
                       "portfolio_optimization", "governance_rules"],
    },
    "company_scale": {
        "micro":  ["priority_rules", "rule_engine", "simple_capacity_planning"],
        "small":  ["capacity_planning", "margin_simulation", "risk_scoring"],
        "mid":    ["allocation_optimization", "portfolio_optimization", "project_margin_model"],
        "large":  ["network_optimization", "portfolio_optimization",
                   "integration_framework", "operating_model_design"],
    },
    "system_ambition": {
        "lightweight": ["priority_rules", "rule_engine", "margin_simulation"],
        "standard":    ["capacity_planning", "risk_scoring",
                        "allocation_optimization", "project_margin_model"],
        "advanced":    ["network_optimization", "portfolio_optimization",
                        "capital_allocation_model", "integration_framework"],
    },
    "human_capital_intensity": {
        "high":   ["capacity_planning", "matching_logic",
                   "utilization_optimization", "decision_standardization"],
        "medium": ["capacity_planning", "workflow_structuring"],
    },
}

DEMOTE: dict[str, dict[str, list[str]]] = {
    "decision_mode": {
        "continuous": ["project_margin_model", "bid_risk_model"],
        "episodic":   ["reorder_logic", "timing_forecast"],
    },
    "company_scale": {
        "micro": ["network_optimization", "portfolio_optimization", "capital_allocation_model"],
    },
    "system_ambition": {
        "lightweight": ["network_optimization", "capital_allocation_model"],
    },
    "human_capital_intensity": {
        "low": ["matching_logic", "utilization_optimization"],
    },
}

PRIORITY_THRESHOLDS = {"very_high": 13, "high": 10, "medium": 7}


# ── Hjälpfunktioner ────────────────────────────────────────────────────────────

def _priority_level(score: float) -> str:
    if score >= PRIORITY_THRESHOLDS["very_high"]: return "very_high"
    if score >= PRIORITY_THRESHOLDS["high"]:      return "high"
    if score >= PRIORITY_THRESHOLDS["medium"]:    return "medium"
    return "low"


def _promote_score(mech: str, dimension: str, value: str) -> float:
    return 2.0 if mech in PROMOTE.get(dimension, {}).get(value, []) else 0.0


def _demote_score(mech: str, dimension: str, value: str) -> float:
    return -2.0 if mech in DEMOTE.get(dimension, {}).get(value, []) else 0.0


# ── Use Case Engine ────────────────────────────────────────────────────────────

def run_use_case_engine(canonical: dict) -> dict:
    """
    Tar canonical JSON och returnerar prioriterade use cases.
    Regelbaserad — ingen LLM.
    """
    bi      = canonical.get("business_interpretation", {})
    elir    = canonical.get("elir", {})
    dual    = canonical.get("dual_mode", {})

    primary_patterns  = bi.get("primary_patterns", [])
    secondary_patterns= bi.get("secondary_patterns", [])
    hc_patterns       = bi.get("human_capital_patterns", [])

    decision_mode  = bi.get("decision_mode", "unknown")
    company_scale  = bi.get("company_scale", "mid")
    system_ambition= bi.get("system_ambition", "standard")
    hc_intensity   = bi.get("human_capital_intensity", "medium")

    # I-drivare för prioritetsformel
    i_scores   = elir.get("I", {}).get("driver_scores", {})
    freq       = float(i_scores.get("frequency")     or 3)
    data       = float(i_scores.get("data")          or 3)
    regulation = float(i_scores.get("regulation")    or 3)
    lock_in    = float(i_scores.get("physical_lock_in") or 3)
    feasibility= ((6 - regulation) + (6 - lock_in)) / 2

    # Samla alla kandidatmekanismer med källinformation
    mech_scores:  dict[str, float] = {}
    mech_sources: dict[str, dict]  = {}

    all_patterns = list(dict.fromkeys(primary_patterns + secondary_patterns + hc_patterns))
    for pattern in all_patterns:
        for dt in PATTERN_TO_DECISIONS.get(pattern, []):
            for mech in DECISION_TO_MECHANISMS.get(dt, []):
                if mech not in mech_scores:
                    mech_scores[mech]  = 0.0
                    mech_sources[mech] = {"source_pattern": pattern, "decision_type": dt}
                # Mönsterbonus
                if pattern in primary_patterns:   mech_scores[mech] += 3
                elif pattern in secondary_patterns: mech_scores[mech] += 1
                if pattern in hc_patterns:        mech_scores[mech] += 2

    # Lägg till drivare-bonusar och kontext-justeringar
    for mech in list(mech_scores.keys()):
        mech_scores[mech] += freq + data + feasibility
        for dim, val in [
            ("decision_mode", decision_mode),
            ("company_scale", company_scale),
            ("system_ambition", system_ambition),
            ("human_capital_intensity", hc_intensity),
        ]:
            mech_scores[mech] += _promote_score(mech, dim, val)
            mech_scores[mech] += _demote_score(mech, dim, val)

    # Bygg use case-lista sorterad på score
    generated = []
    for mech, score in sorted(mech_scores.items(), key=lambda x: -x[1]):
        src   = mech_sources[mech]
        dt    = src["decision_type"]
        cls   = MECHANISM_CLASSIFICATION.get(mech, "optimization")
        sv    = MECHANISM_LABELS_SV.get(mech, mech)
        tmpl  = USE_CASE_TEMPLATES.get(dt, "Förbättra beslut baserat på {mechanism_sv}")
        generated.append({
            "source_pattern": src["source_pattern"],
            "decision_type":  dt,
            "mechanism":      mech,
            "classification": cls,
            "use_case_text":  tmpl.format(mechanism_sv=sv),
            "priority_score": round(score, 1),
            "priority_level": _priority_level(score),
        })

    opt_uc  = [u["use_case_text"] for u in generated if u["classification"] == "optimization"]
    stab_uc = [u["use_case_text"] for u in generated if u["classification"] == "stabilization"]
    hc_uc   = [u["use_case_text"] for u in generated if u["classification"] == "human_capital"]
    top_uc  = [u["use_case_text"] for u in generated
               if u["priority_level"] in ("very_high", "high")][:5]

    return {
        "optimization_use_cases":   opt_uc,
        "stabilization_use_cases":  stab_uc,
        "human_capital_use_cases":  hc_uc,
        "generated_use_cases":      generated,
        "top_use_cases":            top_uc,
    }


# ── Action Strategy Engine ─────────────────────────────────────────────────────

def determine_strategy_type(canonical: dict) -> str:
    """Väljer strategy_type deterministiskt enligt action_strategy_engine.json."""
    bi          = canonical.get("business_interpretation", {})
    rec         = canonical.get("xzero_recommendation", {})
    dual        = canonical.get("dual_mode", {})
    org_maturity= rec.get("org_maturity", {})
    primary     = bi.get("primary_patterns", [])
    scale       = bi.get("company_scale", "mid")
    ambition    = bi.get("system_ambition", "standard")
    mode        = bi.get("decision_mode", "unknown")

    if dual.get("enabled", False):
        return "split_strategy"
    if scale == "micro" or ambition == "lightweight":
        return "lightweight_first"
    if (org_maturity.get("process") == "low" or
        org_maturity.get("decision") == "low" or
        "fragmentation" in primary or
        "knowledge_dependency" in primary):
        return "stabilize_first"
    if (mode == "continuous" and
        org_maturity.get("process") in ("medium", "high") and
        org_maturity.get("decision") in ("medium", "high")):
        return "optimize_first"
    return "stabilize_first"


def build_action_plan_context(canonical: dict, use_case_output: dict) -> dict:
    """Sammanställer allt kontext som LLM:en behöver för att skriva åtgärdsplanen."""
    strategy_type = determine_strategy_type(canonical)
    bi   = canonical.get("business_interpretation", {})
    rec  = canonical.get("xzero_recommendation", {})
    elir = canonical.get("elir", {})
    meta = canonical.get("scan_meta", {})

    return {
        "company_name":       meta.get("company_name", ""),
        "strategy_type":      strategy_type,
        "top_use_cases":      use_case_output.get("top_use_cases", []),
        "optimization_use_cases":  use_case_output.get("optimization_use_cases", []),
        "stabilization_use_cases": use_case_output.get("stabilization_use_cases", []),
        "human_capital_use_cases": use_case_output.get("human_capital_use_cases", []),
        "decision_mode":      bi.get("decision_mode", "unknown"),
        "company_scale":      bi.get("company_scale", "mid"),
        "system_ambition":    bi.get("system_ambition", "standard"),
        "human_capital_intensity": bi.get("human_capital_intensity", "medium"),
        "structure_type":     bi.get("structure_type", "unknown"),
        "primary_patterns":   bi.get("primary_patterns", []),
        "org_maturity":       rec.get("org_maturity", {}),
        "dual_mode_enabled":  canonical.get("dual_mode", {}).get("enabled", False),
        "total_potential_msek": elir.get("total_potential_msek"),
        "i_msek":             elir.get("I", {}).get("msek"),
    }
