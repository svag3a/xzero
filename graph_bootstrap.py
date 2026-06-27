"""
Graph Bootstrap — creates and updates OpportunityGraphs from scan/workshop data.

Two entry points:
  bootstrap_from_scan()       → called after /save, builds initial graph
  update_from_analysis()      → called after post-workshop analysis, enriches graph

Graph IDs follow the convention  scan-{scan_id}  so the link between
a scan row and its graph is derivable without an extra DB column.
"""
from __future__ import annotations

import json

import anthropic

from opportunity_graph import (
    ActionNode, AssumptionNode, DataSourceNode, DecisionNode, ELIR, ELIRParam,
    EvidenceNode, GapNode, HypothesisNode, KPINode, MechanismNode,
    OpportunityGraph, OpportunityNode, RelationType, StakeholderNode,
    UseCaseNode,
)

MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def graph_id_for_scan(scan_id: int) -> str:
    return f"scan-{scan_id}"


# ── Scan → initial graph ───────────────────────────────────────────────────────

def bootstrap_from_scan(scan_id: int, scan_data: dict,
                        hypotheses_json: str | None) -> OpportunityGraph:
    """
    Build an initial OpportunityGraph from scan row data.
    Creates: Opportunity, Hypotheses, Mechanisms, Decisions, Use Cases,
    and one implicit Assumption per hypothesis.
    No LLM call — fully deterministic.
    """
    gid = graph_id_for_scan(scan_id)
    graph = OpportunityGraph(gid)

    # ── Opportunity node ───────────────────────────────────────────────────────
    revenue = (scan_data.get("revenue_msek") or 0) * 1_000_000
    opp = graph.add_node(OpportunityNode(
        company_name=scan_data.get("company_name") or "Okänt bolag",
        industry=scan_data.get("industry") or "",
        revenue_estimate_sek=revenue,
        status="indikativ",
        confidence=55.0,
        elir=ELIR(
            exposure=ELIRParam(
                min_pct=_safe(scan_data.get("e_pct"), 0) * 0.9,
                max_pct=_safe(scan_data.get("e_pct"), 0) * 1.1,
                confidence=50.0,
            ),
            leakage=ELIRParam(
                min_pct=_safe(scan_data.get("l_pct"), 0) * 0.85,
                max_pct=_safe(scan_data.get("l_pct"), 0) * 1.15,
                confidence=45.0,
            ),
            improvement=ELIRParam(
                min_pct=_safe(scan_data.get("i_pct"), 0) * 0.8,
                max_pct=_safe(scan_data.get("i_pct"), 0) * 1.2,
                confidence=40.0,
            ),
            realization=ELIRParam(
                min_pct=_safe(scan_data.get("r_pct"), 0) * 0.8,
                max_pct=_safe(scan_data.get("r_pct"), 0) * 1.2,
                confidence=35.0,
            ),
            value_min_sek=_safe(scan_data.get("total_potential_msek"), 0) * 0.7 * 1_000_000,
            value_max_sek=_safe(scan_data.get("total_potential_msek"), 0) * 1.3 * 1_000_000,
        ),
    ))

    if not hypotheses_json:
        return graph

    try:
        hypotheses = json.loads(hypotheses_json)
    except Exception:
        return graph

    for rank, hyp in enumerate(hypotheses, start=1):
        _add_hypothesis_subgraph(graph, hyp, rank)

    return graph


def _add_hypothesis_subgraph(graph: OpportunityGraph, hyp: dict,
                              rank: int) -> None:
    """Add one hypothesis + mechanism + assumption + decision + use case."""
    title = hyp.get("title") or f"Hypotes {rank}"
    conf = 45.0 + (4 - min(rank, 4)) * 5  # rank 1 → 60%, rank 4 → 45%

    h_node = graph.add_node(HypothesisNode(
        title=title,
        description=hyp.get("description") or "",
        priority=max(1, min(4, 5 - rank)),
        status="ej_behandlad",
        confidence=conf,
        tags=[hyp.get("hypothesis_id", "")] if hyp.get("hypothesis_id") else [],
    ))

    # Implicit assumption — represents the world-model belief behind the mechanism
    mech_data = hyp.get("mechanism") or {}
    assumption_stmt = (
        f"Antagande bakom '{title[:60]}': "
        f"{mech_data.get('description', 'mekanismen är korrekt som beskriven')[:120]}"
    )
    a_node = graph.add_node(AssumptionNode(
        statement=assumption_stmt,
        risk_level="medium",
        is_implicit=True,
        status="implicit",
        confidence=50.0,
    ))
    graph.add_edge(a_node.id, RelationType.UNDERBYGGER, h_node.id)

    # Mechanism
    mech_desc = mech_data.get("description") or title
    m_node = graph.add_node(MechanismNode(
        description=mech_desc,
        status="aktiv",
        confidence=conf,
    ))
    graph.add_edge(h_node.id, RelationType.FORKLARAS_AV, m_node.id)

    # Decision
    dec_data = hyp.get("decision") or {}
    dec_name = dec_data.get("decision_name") or f"Beslut kopplat till {title[:40]}"
    d_node = graph.add_node(DecisionNode(
        description=dec_name,
        frequency=dec_data.get("decision_frequency") or "",
        status="identifierat",
        confidence=45.0,
    ))
    graph.add_edge(m_node.id, RelationType.PEKAR_PA, d_node.id)

    # Use case (candidate)
    uc_data = hyp.get("candidate_use_case") or {}
    if uc_data.get("name"):
        uc_node = graph.add_node(UseCaseNode(
            title=uc_data["name"],
            module=uc_data.get("xzero_module") or "",
            status="föreslagen",
            confidence=40.0,
            notes=uc_data.get("expected_effect") or "",
        ))
        graph.add_edge(d_node.id, RelationType.FORBATTRAS_AV, uc_node.id)


# ── Workshop analysis → graph update ──────────────────────────────────────────

_EXTRACT_PROMPT = """\
Du läser en post-workshop-analys och ett workshop-session-objekt för ett xZero Opportunity Discovery-engagemang.

Din uppgift: extrahera strukturerade noduppdateringar för Opportunity Graph.

Svara ENDAST med ett JSON-objekt enligt schemat nedan — inga förklaringar utanför JSON.

{
  "hypothesis_updates": [
    {
      "hypothesis_title_substring": "<tillräckligt av titeln för att matcha>",
      "status": "ej_behandlad|diskuteras|delvis_verifierad|verifierad|falsifierad|mekanism_andrad",
      "confidence": <0–100>,
      "notes": "<valfritt>"
    }
  ],
  "mechanism_pivots": [
    {
      "hypothesis_title_substring": "<matcha hypotesen>",
      "new_description": "<ny mekanism som framkom i workshopen>",
      "reason": "<varför mekanismen ändrades>"
    }
  ],
  "evidence_nodes": [
    {
      "statement": "<vad som sades eller observerades>",
      "speaker": "<namn eller tom sträng>",
      "direction": "stodjer|motbevisar|neutral",
      "hypothesis_title_substring": "<vilken hypotes det berör>",
      "confidence_delta": <0–40>
    }
  ],
  "stakeholder_nodes": [
    {
      "name": "<namn>",
      "title": "<roll/titel>",
      "role": "sponsor|skeptiker|dataagare|beslutsfattare|okant_mandat",
      "notes": "<valfritt>"
    }
  ],
  "gap_nodes": [
    {
      "description": "<vad som saknas>",
      "priority": "low|medium|high|critical",
      "elir_dimension": "E|L|I|R|",
      "linked_hypothesis_substring": "<hypotesen luckan tillhör, eller tom>"
    }
  ],
  "action_nodes": [
    {
      "description": "<konkret nästa steg>",
      "assigned_to": "<vem, eller tom>",
      "deadline": "<deadline, eller tom>",
      "gap_description_substring": "<matcha vilket gap åtgärden stänger>"
    }
  ],
  "kpi_nodes": [
    {
      "name": "<KPI-namn>",
      "unit": "<enhet>",
      "elir_dimension": "L|I|R|E",
      "baseline_value": <tal eller null>,
      "baseline_year": <år eller null>,
      "target_value": <tal eller null>,
      "linked_hypothesis_substring": "<hypotes>"
    }
  ]
}"""


def update_from_analysis(graph: OpportunityGraph, analysis_markdown: str,
                         session: dict, client: anthropic.AnthropicBedrock) -> OpportunityGraph:
    """
    Enrich the graph with nodes extracted from the post-workshop analysis.
    Makes one Bedrock call to extract structured data from the analysis text.
    Returns the mutated graph.
    """
    company = session.get("company_name") or graph.get_opportunity().company_name if graph.get_opportunity() else ""

    user_msg = (
        f"BOLAG: {company}\n\n"
        f"POST-WORKSHOP-ANALYS:\n{analysis_markdown[:8000]}\n\n"
        f"SESSION (hypoteser och anteckningar):\n"
        + json.dumps(
            {
                "hypotheses": [
                    {
                        "title": h.get("title"),
                        "validation_status": h.get("validation_status"),
                        "new_findings": h.get("new_findings", []),
                        "evidence_collected": h.get("evidence_collected", []),
                    }
                    for h in session.get("hypotheses", [])
                ]
            },
            ensure_ascii=False,
        )
    )

    msg = client.messages.create(
        model=MODEL_ID,
        max_tokens=4096,
        system=_EXTRACT_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = msg.content[0].text

    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        updates = json.loads(raw[start:end]) if start >= 0 and end > start else {}
    except Exception:
        updates = {}

    _apply_updates(graph, updates)
    return graph


def _apply_updates(graph: OpportunityGraph, updates: dict) -> None:
    hyp_nodes = graph.find_nodes()

    def _find_hyp(substring: str):
        if not substring:
            return None
        sl = substring.lower()
        for n in hyp_nodes:
            title = getattr(n, "title", "") or ""
            if sl in title.lower():
                return n
        return None

    # Hypothesis status/confidence updates
    for upd in updates.get("hypothesis_updates") or []:
        node = _find_hyp(upd.get("hypothesis_title_substring", ""))
        if node:
            kwargs = {}
            if upd.get("status"):
                kwargs["status"] = upd["status"]
            if upd.get("confidence") is not None:
                kwargs["confidence"] = float(upd["confidence"])
            if upd.get("notes"):
                kwargs["notes"] = upd["notes"]
            if kwargs:
                graph.update_node(node.id, **kwargs)

    # Mechanism pivots
    for pivot in updates.get("mechanism_pivots") or []:
        hyp = _find_hyp(pivot.get("hypothesis_title_substring", ""))
        if not hyp:
            continue
        from opportunity_graph import MechanismNode, NodeType, RelationType
        # Mark existing active mechanisms as replaced
        for old_mech in graph.neighbors(hyp.id, RelationType.FORKLARAS_AV):
            if old_mech.status == "aktiv":
                graph.update_node(old_mech.id, status="ersatt")
        # Add new mechanism
        new_mech = graph.add_node(MechanismNode(
            description=pivot.get("new_description", ""),
            status="aktiv",
            confidence=60.0,
            notes=pivot.get("reason", ""),
        ))
        graph.add_edge(hyp.id, RelationType.FORKLARAS_AV, new_mech.id)
        graph.update_node(hyp.id, status="mekanism_andrad")

    # Evidence nodes
    for ev_data in updates.get("evidence_nodes") or []:
        stmt = ev_data.get("statement", "")
        if not stmt:
            continue
        direction = ev_data.get("direction", "stodjer")
        ev = graph.add_node(EvidenceNode(
            statement=stmt,
            speaker=ev_data.get("speaker", ""),
            source="post-workshop-analys",
            direction=direction,
            confidence_delta=float(ev_data.get("confidence_delta", 10)),
            confidence=70.0,
        ))
        hyp = _find_hyp(ev_data.get("hypothesis_title_substring", ""))
        if hyp:
            rel = RelationType.STODJER if direction == "stodjer" else RelationType.MOTBEVISAR
            graph.add_edge(ev.id, rel, hyp.id)

    # Stakeholder nodes
    existing_names = {
        n.name.lower()
        for n in graph.find_nodes()
        if hasattr(n, "name")
    }
    for sk_data in updates.get("stakeholder_nodes") or []:
        name = sk_data.get("name", "")
        if not name or name.lower() in existing_names:
            continue
        role_raw = sk_data.get("role", "okant_mandat")
        role = {
            "dataagare": "dataägare",
            "okant_mandat": "okänt_mandat",
        }.get(role_raw, role_raw)
        graph.add_node(StakeholderNode(
            name=name,
            title=sk_data.get("title", ""),
            role=role,
            notes=sk_data.get("notes", ""),
            confidence=65.0,
        ))
        existing_names.add(name.lower())

    # Gap nodes
    gap_nodes_added: list[tuple[str, object]] = []
    for g_data in updates.get("gap_nodes") or []:
        desc = g_data.get("description", "")
        if not desc:
            continue
        gap = graph.add_node(GapNode(
            description=desc,
            priority=g_data.get("priority", "medium"),
            elir_dimension=g_data.get("elir_dimension", ""),
            status="identifierat",
            confidence=40.0,
        ))
        linked_hyp = _find_hyp(g_data.get("linked_hypothesis_substring", ""))
        if linked_hyp:
            graph.add_edge(gap.id, RelationType.SAKNAS_I, linked_hyp.id)
        gap_nodes_added.append((desc.lower()[:60], gap))

    # Action nodes
    for a_data in updates.get("action_nodes") or []:
        action_desc = a_data.get("description", "")
        if not action_desc:
            continue
        action = graph.add_node(ActionNode(
            description=action_desc,
            assigned_to=a_data.get("assigned_to", ""),
            deadline=a_data.get("deadline", ""),
            status="identifierad",
            confidence=75.0,
        ))
        # Link to gap if matchable
        gap_substr = (a_data.get("gap_description_substring") or "").lower()[:60]
        if gap_substr:
            for gap_desc, gap_node in gap_nodes_added:
                if gap_substr in gap_desc or gap_desc in gap_substr:
                    graph.add_edge(action.id, RelationType.STANGER, gap_node.id)
                    break

    # KPI nodes
    for k_data in updates.get("kpi_nodes") or []:
        kpi_name = k_data.get("name", "")
        if not kpi_name:
            continue
        from opportunity_graph import KPINode
        kpi = graph.add_node(KPINode(
            name=kpi_name,
            unit=k_data.get("unit", ""),
            elir_dimension=k_data.get("elir_dimension", "L"),
            baseline_value=k_data.get("baseline_value"),
            baseline_year=k_data.get("baseline_year"),
            target_value=k_data.get("target_value"),
            status="definierad",
            confidence=55.0,
        ))
        # Link to any matching use case for this hypothesis
        hyp = _find_hyp(k_data.get("linked_hypothesis_substring", ""))
        if hyp:
            from opportunity_graph import NodeType
            for uc in graph.find_nodes(NodeType.USE_CASE):
                decisions = graph.predecessors(uc.id, RelationType.FORBATTRAS_AV)
                for dec in decisions:
                    mechs = graph.predecessors(dec.id, RelationType.PEKAR_PA)
                    for mech in mechs:
                        if hyp.id in [e.to_id for e in graph.edges_from(hyp.id, RelationType.FORKLARAS_AV)
                                      if e.to_id == mech.id]:
                            graph.add_edge(uc.id, RelationType.MATS_AV, kpi.id)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
