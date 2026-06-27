"""
Gap Hunter — ranks open knowledge gaps by urgency.

Score = priority_weight × uncertainty × elir_multiplier
  priority_weight:  critical=4, high=3, medium=2, low=1
  uncertainty:      1 − avg_confidence_of_linked_nodes / 100
  elir_multiplier:  1.0 + ELIR_dimension_confidence / 200
                    (higher stakes → gap matters more)

Surfaces the highest-impact unknown questions first so
workshop prep and post-analysis focus on what actually matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from opportunity_graph import (
    GapNode, HypothesisNode, NodeType, OpportunityGraph, RelationType,
)

_PRIORITY_WEIGHT = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}
_ELIR_LABEL = {
    "E": "Exponering", "L": "Läckage",
    "I": "Förbättringsmöjlighet", "R": "Realisering", "": "Oklar",
}


@dataclass
class RankedGap:
    gap: GapNode
    score: float
    linked_node_labels: list[str]
    elir_label: str
    has_action: bool
    reasoning: str


def rank_gaps(graph: OpportunityGraph) -> list[RankedGap]:
    """Return all open gaps sorted by urgency score, highest first."""
    opp = graph.get_opportunity()
    elir = opp.elir if opp else None
    results: list[RankedGap] = []

    for gap in graph.find_open_gaps():
        linked = graph.neighbors(gap.id, RelationType.SAKNAS_I)
        avg_conf = (sum(n.confidence for n in linked) / len(linked)
                    if linked else 50.0)

        pw = _PRIORITY_WEIGHT.get(gap.priority, 2.0)
        uncertainty = 1.0 - avg_conf / 100.0
        em = _elir_multiplier(gap.elir_dimension, elir)
        score = round(pw * uncertainty * em, 3)

        actions = graph.predecessors(gap.id, RelationType.STANGER)
        results.append(RankedGap(
            gap=gap,
            score=score,
            linked_node_labels=[_label(n) for n in linked],
            elir_label=_ELIR_LABEL.get(gap.elir_dimension, "Oklar"),
            has_action=bool(actions),
            reasoning=_reasoning(gap, avg_conf, bool(actions)),
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def find_implicit_assumption_gaps(graph: OpportunityGraph) -> list[dict]:
    """
    Hypotheses with no explicit assumptions — the silent killers.
    The Sweden Pelagic pivot was caused by exactly this: an implicit
    assumption ('inköpsflexibilitet') that was never surfaced as a node.
    """
    result = []
    for hyp in graph.find_nodes(NodeType.HYPOTHESIS):
        assumptions = graph.predecessors(hyp.id, RelationType.UNDERBYGGER)
        explicit = [a for a in assumptions if not a.is_implicit]
        if not explicit:
            result.append({
                "hypothesis_id": hyp.id,
                "hypothesis_title": hyp.title,
                "implicit_count": len([a for a in assumptions if a.is_implicit]),
                "warning": "Inga explicita antaganden — risk för oupptäckt pivot",
            })
    return result


def summarize(graph: OpportunityGraph) -> dict:
    ranked = rank_gaps(graph)
    implicit_warnings = find_implicit_assumption_gaps(graph)
    violations = graph.validate()

    return {
        "total_open_gaps": len(ranked),
        "gaps_without_actions": len(graph.find_gaps_without_actions()),
        "implicit_assumption_warnings": implicit_warnings,
        "invariant_violations": len(violations),
        "top_gaps": [
            {
                "id": r.gap.id,
                "description": r.gap.description,
                "priority": r.gap.priority,
                "elir_dimension": r.elir_label,
                "score": r.score,
                "has_action": r.has_action,
                "linked_to": r.linked_node_labels,
                "reasoning": r.reasoning,
            }
            for r in ranked[:10]
        ],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _elir_multiplier(dimension: str, elir) -> float:
    if not elir or not dimension:
        return 1.0
    param = elir.param(dimension)
    return 1.0 + (param.confidence / 200.0) if param else 1.0


def _label(node) -> str:
    for attr in ("title", "statement", "description", "name", "raw_text"):
        val = getattr(node, attr, None)
        if val:
            return f"{node.type.value}: {str(val)[:50]}"
    return node.type.value


def _reasoning(gap: GapNode, avg_conf: float, has_action: bool) -> str:
    parts = []
    if gap.priority == "critical":
        parts.append("Kritisk prioritet")
    if avg_conf < 50:
        parts.append(f"Länkade noder har låg konfidens ({avg_conf:.0f}%)")
    if not has_action:
        parts.append("Ingen åtgärd tilldelad")
    if gap.elir_dimension:
        parts.append(f"Blockerar ELIR-{gap.elir_dimension}")
    return " · ".join(parts) if parts else "Standardbedömning"
