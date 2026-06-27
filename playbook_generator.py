"""
Playbook Generator — builds a Workshop Playbook from the current graph state.

Reads the graph to understand which hypotheses need validation, which
assumptions are implicit and dangerous, and which data sources are unconfirmed.
Then calls Bedrock Claude to produce contextually-rich workshop questions.
"""
from __future__ import annotations

import json
import os

import anthropic

from gap_hunter import find_implicit_assumption_gaps, rank_gaps
from opportunity_graph import NodeType, OpportunityGraph, RelationType

MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def generate_playbook(graph: OpportunityGraph) -> dict:
    opp = graph.get_opportunity()
    if not opp:
        raise ValueError("Graph har ingen Opportunity-nod")

    ranked_gaps = rank_gaps(graph)
    implicit_warnings = find_implicit_assumption_gaps(graph)
    graph_summary = _build_graph_summary(graph)

    prompt = _build_prompt(graph_summary, ranked_gaps[:5], implicit_warnings)
    raw = _call_bedrock(prompt)
    playbook = _parse_json_response(raw)

    return {
        "opportunity_id": graph.opportunity_id,
        "company": opp.company_name,
        "graph_summary": graph_summary,
        "playbook": playbook,
    }


def _build_graph_summary(graph: OpportunityGraph) -> dict:
    opp = graph.get_opportunity()
    hypotheses = graph.find_nodes(NodeType.HYPOTHESIS)
    assumptions = graph.find_nodes(NodeType.ASSUMPTION)
    gaps = graph.find_open_gaps()
    stakeholders = graph.find_nodes(NodeType.STAKEHOLDER)
    data_sources = graph.find_nodes(NodeType.DATA_SOURCE)

    return {
        "company": opp.company_name if opp else "",
        "industry": opp.industry if opp else "",
        "elir_value_range_sek": (
            f"{opp.elir.value_min_sek:,.0f}–{opp.elir.value_max_sek:,.0f}"
            if opp else ""
        ),
        "hypotheses": [
            {
                "title": h.title,
                "status": h.status,
                "confidence": h.confidence,
                "priority": h.priority,
                "assumptions": [
                    a.statement
                    for a in graph.predecessors(h.id, RelationType.UNDERBYGGER)
                ],
            }
            for h in sorted(hypotheses, key=lambda h: h.priority, reverse=True)
        ],
        "implicit_assumptions": [
            a.statement for a in assumptions if a.is_implicit
        ],
        "open_gaps": [
            {
                "description": g.description,
                "priority": g.priority,
                "elir": g.elir_dimension,
            }
            for g in gaps
        ],
        "stakeholders": [
            {"name": s.name, "title": s.title, "role": s.role}
            for s in stakeholders
        ],
        "data_sources": [
            {"description": ds.description, "status": ds.status}
            for ds in data_sources
        ],
    }


def _build_prompt(summary: dict, top_gaps: list, implicit_warnings: list) -> str:
    company = summary.get("company", "kunden")
    return f"""Du är xZero:s AI-konsult och ska generera ett Workshop Playbook för {company}.

GRAPH-SAMMANDRAG:
{json.dumps(summary, ensure_ascii=False, indent=2)}

TOP-5 GAP (Gap Hunter-analys):
{json.dumps(
    [{"desc": r.gap.description, "score": r.score,
      "elir": r.elir_label, "reasoning": r.reasoning}
     for r in top_gaps],
    ensure_ascii=False, indent=2
)}

VARNINGAR OM IMPLICITA ANTAGANDEN:
{json.dumps(implicit_warnings, ensure_ascii=False, indent=2)}

Generera ett Workshop Playbook. Svara ENDAST med ett JSON-objekt, inga förklaringar utanför JSON.

{{
  "opening_frame": {{
    "duration_min": <int>,
    "objective": "<vad workshopen ska uppnå>",
    "framing_statement": "<hur ni presenterar syftet för kunden, 2–3 meningar>"
  }},
  "hypothesis_blocks": [
    {{
      "hypothesis_title": "<string>",
      "priority": "high|medium|low",
      "validation_questions": ["<fråga>", ...],
      "assumption_probes": ["<fråga som testar det implicita antagandet>", ...],
      "success_signal": "<hur ni vet om hypotesen är bekräftad under mötet>"
    }}
  ],
  "data_source_confirmations": [
    {{
      "data_source": "<beskrivning>",
      "questions": ["<fråga>", ...]
    }}
  ],
  "stakeholder_mapping": {{
    "objective": "<string>",
    "questions": ["<fråga>", ...]
  }},
  "closing_commitments": {{
    "objective": "<string>",
    "questions": ["<fråga som leder till konkreta åtaganden>", ...]
  }}
}}"""


def _call_bedrock(prompt: str) -> str:
    client = anthropic.AnthropicBedrock(
        aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    msg = client.messages.create(
        model=MODEL_ID,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _parse_json_response(raw: str) -> dict:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {"raw": raw, "parse_error": True}
