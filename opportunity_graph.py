"""
Opportunity Graph — JSON-based domain model.
Implements all 13 node types, relationships, invariants (M1-M9),
and query API from the Opportunity Graph Ontology v1.1.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    OPPORTUNITY  = "opportunity"
    ASSUMPTION   = "assumption"
    HYPOTHESIS   = "hypothesis"
    MECHANISM    = "mechanism"
    EVIDENCE     = "evidence"
    GAP          = "gap"
    DECISION     = "decision"
    USE_CASE     = "use_case"
    STAKEHOLDER  = "stakeholder"
    DATA_SOURCE  = "data_source"
    OBSERVATION  = "observation"
    KPI          = "kpi"
    ACTION       = "action"


class RelationType(str, Enum):
    UNDERBYGGER   = "underbygger"     # Assumption → Hypothesis
    FORKLARAS_AV  = "forklaras_av"    # Hypothesis → Mechanism
    PEKAR_PA      = "pekar_pa"        # Mechanism → Decision
    FORBATTRAS_AV = "forbattras_av"   # Decision → Use Case
    STODJER       = "stodjer"         # Evidence → Hypothesis/Assumption
    MOTBEVISAR    = "motbevisar"      # Evidence → Hypothesis/Assumption
    SAKNAS_I      = "saknas_i"        # Gap → Hypothesis/Assumption/Decision
    STANGER       = "stanger"         # Action → Gap
    AGER          = "ager"            # Stakeholder → Decision
    KONTROLLERAR  = "kontrollerar"    # Stakeholder → Data Source
    TILLDELAS     = "tilldelas"       # Action → Stakeholder
    MATS_AV       = "mats_av"         # Use Case → KPI
    KVANTIFIERAR  = "kvantifierar"    # KPI → Opportunity
    RESOLVAS_TILL = "resolvas_till"   # Observation → Evidence/Gap/Assumption
    KRAVER_DATA   = "kraver_data"     # Use Case → Data Source


# ── ELIR value model ───────────────────────────────────────────────────────────

class ELIRParam(BaseModel):
    min_pct: float = 0.0
    max_pct: float = 0.0
    confidence: float = 20.0  # 0–100

class ELIR(BaseModel):
    exposure:    ELIRParam = Field(default_factory=ELIRParam)
    leakage:     ELIRParam = Field(default_factory=ELIRParam)
    improvement: ELIRParam = Field(default_factory=ELIRParam)
    realization: ELIRParam = Field(default_factory=ELIRParam)
    value_min_sek: float = 0.0
    value_max_sek: float = 0.0

    def param(self, dimension: str) -> Optional[ELIRParam]:
        return {"E": self.exposure, "L": self.leakage,
                "I": self.improvement, "R": self.realization}.get(dimension)


# ── Base node ──────────────────────────────────────────────────────────────────

def _new_id() -> str:
    return str(uuid.uuid4())

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Node(BaseModel):
    id: str = Field(default_factory=_new_id)
    type: NodeType
    confidence: float = Field(50.0, ge=0.0, le=100.0)
    created_at: str = Field(default_factory=_now)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class Edge(BaseModel):
    id: str = Field(default_factory=_new_id)
    from_id: str
    relation: RelationType
    to_id: str
    created_at: str = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Concrete node types ────────────────────────────────────────────────────────

class OpportunityNode(Node):
    type: NodeType = NodeType.OPPORTUNITY
    company_name: str
    industry: str = ""
    revenue_estimate_sek: float = 0.0
    elir: ELIR = Field(default_factory=ELIR)
    status: str = "indikativ"  # indikativ | aktiv | verifierad


class HypothesisNode(Node):
    type: NodeType = NodeType.HYPOTHESIS
    title: str
    description: str = ""
    priority: int = Field(2, ge=1, le=4)  # 4 = highest
    status: str = "ej_behandlad"
    # ej_behandlad | diskuteras | delvis_verifierad | verifierad | falsifierad | mekanism_andrad


class AssumptionNode(Node):
    type: NodeType = NodeType.ASSUMPTION
    statement: str
    risk_level: str = "medium"  # low | medium | high
    is_implicit: bool = True
    status: str = "implicit"    # implicit | explicit | verifierat | falsifierat


class MechanismNode(Node):
    type: NodeType = NodeType.MECHANISM
    description: str
    status: str = "föreslagen"  # föreslagen | aktiv | ersatt | falsifierad


class EvidenceNode(Node):
    type: NodeType = NodeType.EVIDENCE
    statement: str
    source: str = ""
    speaker: str = ""
    direction: str = "stodjer"  # stodjer | motbevisar | neutral
    confidence_delta: float = 0.0


class GapNode(Node):
    type: NodeType = NodeType.GAP
    description: str
    priority: str = "medium"    # low | medium | high | critical
    elir_dimension: str = ""    # E | L | I | R
    status: str = "identifierat"  # identifierat | bearbetas | stängt


class DecisionNode(Node):
    type: NodeType = NodeType.DECISION
    description: str
    frequency: str = ""
    status: str = "identifierat"  # identifierat | ägare_känd | kvantifierat


class UseCaseNode(Node):
    type: NodeType = NodeType.USE_CASE
    title: str
    module: str = ""
    expected_value_min_sek: float = 0.0
    expected_value_max_sek: float = 0.0
    status: str = "föreslagen"  # föreslagen | prioriterad | i_pilot | live


class StakeholderNode(Node):
    type: NodeType = NodeType.STAKEHOLDER
    name: str
    title: str = ""
    role: str = "okänt_mandat"
    # sponsor | skeptiker | dataägare | beslutsfattare | okänt_mandat
    present_at_workshop: bool = False
    notes: str = ""


class DataSourceNode(Node):
    type: NodeType = NodeType.DATA_SOURCE
    description: str
    data_type: str = ""
    system_name: str = ""
    owner: str = ""
    status: str = "identifierad"  # identifierad | bekräftad | tillgänglig | blockerad


class ObservationNode(Node):
    type: NodeType = NodeType.OBSERVATION
    raw_text: str
    source: str = ""   # transcript | document | interview
    speaker: str = ""
    timestamp: str = ""
    status: str = "ohanterad"  # ohanterad | resolvad


class KPINode(Node):
    type: NodeType = NodeType.KPI
    name: str
    description: str = ""
    unit: str = ""
    elir_dimension: str = "L"   # E | L | I | R
    baseline_value: Optional[float] = None
    baseline_year: Optional[int] = None
    target_value: Optional[float] = None
    actual_value: Optional[float] = None
    status: str = "definierad"  # definierad | baslinjerad | mätt | roi_verifierad


class ActionNode(Node):
    type: NodeType = NodeType.ACTION
    description: str
    assigned_to: str = ""
    deadline: str = ""
    status: str = "identifierad"  # identifierad | tilldelad | genomförd


# ── Node registry ──────────────────────────────────────────────────────────────

NODE_CLASSES: dict[NodeType, type[Node]] = {
    NodeType.OPPORTUNITY:  OpportunityNode,
    NodeType.ASSUMPTION:   AssumptionNode,
    NodeType.HYPOTHESIS:   HypothesisNode,
    NodeType.MECHANISM:    MechanismNode,
    NodeType.EVIDENCE:     EvidenceNode,
    NodeType.GAP:          GapNode,
    NodeType.DECISION:     DecisionNode,
    NodeType.USE_CASE:     UseCaseNode,
    NodeType.STAKEHOLDER:  StakeholderNode,
    NodeType.DATA_SOURCE:  DataSourceNode,
    NodeType.OBSERVATION:  ObservationNode,
    NodeType.KPI:          KPINode,
    NodeType.ACTION:       ActionNode,
}


# ── Invariant violation ────────────────────────────────────────────────────────

class InvariantViolation(BaseModel):
    rule: str
    message: str
    node_ids: list[str] = Field(default_factory=list)


# ── Graph ──────────────────────────────────────────────────────────────────────

class OpportunityGraph:
    """
    In-memory directed graph. Persist to/from JSON for storage in SQLite.

    Nodes are keyed by id. Edges are stored as a flat list.
    All query methods return plain lists — no lazy loading, no cursors.
    """

    def __init__(self, opportunity_id: str):
        self.opportunity_id = opportunity_id
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    # ── Node CRUD ──────────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> Node:
        self._nodes[node.id] = node
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs) -> Optional[Node]:
        node = self._nodes.get(node_id)
        if not node:
            return None
        data = node.model_dump()
        data.update(kwargs)
        updated = NODE_CLASSES[NodeType(data["type"])](**data)
        self._nodes[node_id] = updated
        return updated

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        self._edges = [e for e in self._edges
                       if e.from_id != node_id and e.to_id != node_id]
        return True

    def find_nodes(self, type: Optional[NodeType] = None, **filters) -> list[Node]:
        nodes = list(self._nodes.values())
        if type:
            nodes = [n for n in nodes if n.type == type]
        for k, v in filters.items():
            nodes = [n for n in nodes if getattr(n, k, None) == v]
        return nodes

    # ── Edge CRUD ──────────────────────────────────────────────────────────────

    def add_edge(self, from_id: str, relation: RelationType, to_id: str,
                 **metadata) -> Edge:
        edge = Edge(from_id=from_id, relation=relation, to_id=to_id,
                    metadata=metadata)
        self._edges.append(edge)
        return edge

    def remove_edges(self, from_id: str, relation: RelationType,
                     to_id: str) -> int:
        before = len(self._edges)
        self._edges = [e for e in self._edges
                       if not (e.from_id == from_id and
                               e.relation == relation and
                               e.to_id == to_id)]
        return before - len(self._edges)

    def edges_from(self, node_id: str,
                   relation: Optional[RelationType] = None) -> list[Edge]:
        edges = [e for e in self._edges if e.from_id == node_id]
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return edges

    def edges_to(self, node_id: str,
                 relation: Optional[RelationType] = None) -> list[Edge]:
        edges = [e for e in self._edges if e.to_id == node_id]
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return edges

    def neighbors(self, node_id: str,
                  relation: Optional[RelationType] = None) -> list[Node]:
        return [self._nodes[e.to_id] for e in self.edges_from(node_id, relation)
                if e.to_id in self._nodes]

    def predecessors(self, node_id: str,
                     relation: Optional[RelationType] = None) -> list[Node]:
        return [self._nodes[e.from_id] for e in self.edges_to(node_id, relation)
                if e.from_id in self._nodes]

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_opportunity(self) -> Optional[OpportunityNode]:
        nodes = self.find_nodes(NodeType.OPPORTUNITY)
        return nodes[0] if nodes else None  # type: ignore[return-value]

    def find_open_gaps(self) -> list[GapNode]:
        return [n for n in self.find_nodes(NodeType.GAP)
                if n.status != "stängt"]  # type: ignore[return-value]

    def find_implicit_assumptions(self) -> list[AssumptionNode]:
        return [n for n in self.find_nodes(NodeType.ASSUMPTION)
                if n.is_implicit and n.status == "implicit"]  # type: ignore[return-value]

    def find_hypotheses_at_risk(self) -> list[HypothesisNode]:
        """Hypotheses whose supporting assumptions have been falsified."""
        at_risk = []
        for hyp in self.find_nodes(NodeType.HYPOTHESIS):
            assumptions = self.predecessors(hyp.id, RelationType.UNDERBYGGER)
            if any(a.status == "falsifierat" for a in assumptions):
                at_risk.append(hyp)
        return at_risk  # type: ignore[return-value]

    def find_gaps_without_actions(self) -> list[GapNode]:
        """M9: high/critical gaps that have no linked Action."""
        result = []
        for gap in self.find_open_gaps():
            if gap.priority not in ("high", "critical"):
                continue
            if not self.predecessors(gap.id, RelationType.STANGER):
                result.append(gap)
        return result

    def find_unresolved_observations(self) -> list[ObservationNode]:
        return [n for n in self.find_nodes(NodeType.OBSERVATION)
                if n.status == "ohanterad"]  # type: ignore[return-value]

    # ── Confidence propagation ────────────────────────────────────────────────

    def apply_evidence(self, evidence_id: str) -> list[str]:
        """
        Push a EvidenceNode's confidence_delta to all nodes it points to.
        Returns the list of updated node ids.
        """
        ev = self.get_node(evidence_id)
        if not ev or ev.type != NodeType.EVIDENCE:
            return []

        relation = (RelationType.STODJER if ev.direction == "stodjer"
                    else RelationType.MOTBEVISAR)
        delta = ev.confidence_delta if ev.direction == "stodjer" else -abs(ev.confidence_delta)

        updated = []
        for target in self.neighbors(evidence_id, relation):
            new_conf = max(0.0, min(100.0, target.confidence + delta))
            self.update_node(target.id, confidence=new_conf)
            updated.append(target.id)
        return updated

    # ── Invariant validation ──────────────────────────────────────────────────

    def validate(self) -> list[InvariantViolation]:
        v: list[InvariantViolation] = []

        for hyp in self.find_nodes(NodeType.HYPOTHESIS):
            if not self.predecessors(hyp.id, RelationType.UNDERBYGGER):
                v.append(InvariantViolation(
                    rule="M1",
                    message=f"Hypothesis '{hyp.title}' har inga Assumptions.",
                    node_ids=[hyp.id],
                ))

        for mech in self.find_nodes(NodeType.MECHANISM):
            if mech.status == "aktiv" and not self.neighbors(mech.id, RelationType.PEKAR_PA):
                v.append(InvariantViolation(
                    rule="M2",
                    message=f"Aktiv Mechanism saknar Decision: '{mech.description[:60]}'",
                    node_ids=[mech.id],
                ))

        for ev in self.find_nodes(NodeType.EVIDENCE):
            has_target = (self.neighbors(ev.id, RelationType.STODJER) or
                          self.neighbors(ev.id, RelationType.MOTBEVISAR))
            if not has_target:
                v.append(InvariantViolation(
                    rule="M3",
                    message=f"Evidence utan riktning: '{ev.statement[:60]}'",
                    node_ids=[ev.id],
                ))

        for gap in self.find_nodes(NodeType.GAP):
            if not self.neighbors(gap.id, RelationType.SAKNAS_I):
                v.append(InvariantViolation(
                    rule="M4",
                    message=f"Gap utan referens: '{gap.description[:60]}'",
                    node_ids=[gap.id],
                ))

        for uc in self.find_nodes(NodeType.USE_CASE):
            if not self.predecessors(uc.id, RelationType.FORBATTRAS_AV):
                v.append(InvariantViolation(
                    rule="M5",
                    message=f"Use Case saknar Decision: '{uc.title}'",
                    node_ids=[uc.id],
                ))
            if not self.neighbors(uc.id, RelationType.KRAVER_DATA):
                v.append(InvariantViolation(
                    rule="M5",
                    message=f"Use Case saknar Data Source: '{uc.title}'",
                    node_ids=[uc.id],
                ))
            if not self.neighbors(uc.id, RelationType.MATS_AV):
                v.append(InvariantViolation(
                    rule="M8",
                    message=f"Use Case saknar KPI: '{uc.title}'",
                    node_ids=[uc.id],
                ))

        for obs in self.find_nodes(NodeType.OBSERVATION):
            if obs.status == "ohanterad" and not self.neighbors(obs.id, RelationType.RESOLVAS_TILL):
                v.append(InvariantViolation(
                    rule="M7",
                    message=f"Ohanterad Observation: '{obs.raw_text[:60]}'",
                    node_ids=[obs.id],
                ))

        for gap in self.find_gaps_without_actions():
            v.append(InvariantViolation(
                rule="M9",
                message=f"Hög/kritisk Gap utan Action: '{gap.description[:60]}'",
                node_ids=[gap.id],
            ))

        return v

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        opp = self.get_opportunity()
        hypotheses = self.find_nodes(NodeType.HYPOTHESIS)
        avg_conf = (sum(h.confidence for h in hypotheses) / len(hypotheses)
                    if hypotheses else 0.0)
        violations = self.validate()

        return {
            "opportunity_id": self.opportunity_id,
            "company": opp.company_name if opp else "",
            "status": opp.status if opp else "",
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "hypothesis_count": len(hypotheses),
            "avg_hypothesis_confidence": round(avg_conf, 1),
            "open_gaps": len(self.find_open_gaps()),
            "unresolved_observations": len(self.find_unresolved_observations()),
            "invariant_violations": len(violations),
            "violations": [v.model_dump() for v in violations],
            "elir": opp.elir.model_dump() if opp else {},
        }

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "OpportunityGraph":
        graph = cls(data["opportunity_id"])
        for nd in data.get("nodes", []):
            ntype = NodeType(nd["type"])
            graph._nodes[nd["id"]] = NODE_CLASSES[ntype](**nd)
        for ed in data.get("edges", []):
            graph._edges.append(Edge(**ed))
        return graph

    @classmethod
    def from_json(cls, s: str) -> "OpportunityGraph":
        return cls.from_dict(json.loads(s))
