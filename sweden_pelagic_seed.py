"""
Sweden Pelagic seed data — full post-workshop graph state.

Captures the complete Opportunity Journey for Sweden Pelagic AB:
  · Scan hypotheses (three, including one that later pivoted)
  · The pivotal Assumption collapse: 'inköpsflexibilitet'
  · Mechanism pivot from råvaruinköp → personalplanering
  · Evidence from the workshop (Martin Kuhlin's 2.4 MSEK finding)
  · Open gaps and their Actions
  · One prioritized Use Case with KPI and Data Source

Run directly to print the graph JSON:
  python sweden_pelagic_seed.py
"""
from __future__ import annotations

from opportunity_graph import (
    ActionNode, AssumptionNode, DataSourceNode, DecisionNode, ELIR,
    ELIRParam, EvidenceNode, GapNode, HypothesisNode, KPINode, MechanismNode,
    ObservationNode, OpportunityGraph, OpportunityNode, RelationType,
    StakeholderNode, UseCaseNode,
)


def build() -> OpportunityGraph:
    g = OpportunityGraph("sweden-pelagic-001")

    # ── Opportunity ────────────────────────────────────────────────────────────
    opp = g.add_node(OpportunityNode(
        company_name="Sweden Pelagic AB",
        industry="Fiskindustri / Pelagisk beredning",
        revenue_estimate_sek=120_000_000,
        status="aktiv",
        confidence=70.0,
        elir=ELIR(
            exposure=ELIRParam(min_pct=80, max_pct=95, confidence=75),
            leakage=ELIRParam(min_pct=12, max_pct=18, confidence=65),
            improvement=ELIRParam(min_pct=40, max_pct=60, confidence=55),
            realization=ELIRParam(min_pct=50, max_pct=70, confidence=50),
            value_min_sek=1_500_000,
            value_max_sek=3_500_000,
        ),
        notes="Biologiskt kvotstyrd råvara. Säsongsproduktion sill/skarpsill.",
    ))

    # ── Hypotheses ─────────────────────────────────────────────────────────────

    h1 = g.add_node(HypothesisNode(
        title="Suboptimalt råvaruutnyttjande driver kronisk underprestanda",
        description=(
            "Bolaget lämnar 12–18% av möjlig intjäning på bordet p.g.a. suboptimalt "
            "utnyttjande av biologisk råvara. Mekanismen pivoterade under workshopen: "
            "inte inköpsoptimering utan felaktig personaldimensionering relativt sillankomst."
        ),
        priority=4,
        status="delvis_verifierad",
        confidence=72.0,
    ))

    h2 = g.add_node(HypothesisNode(
        title="Lagerhantering och logistik genererar onödiga kostnader",
        description="Lagerstyrning och distributionslogistik är inte optimerade för biologisk säsongsproduktion.",
        priority=2,
        status="ej_behandlad",
        confidence=40.0,
    ))

    h3 = g.add_node(HypothesisNode(
        title="Kundlönsamhetsvariation är hög men osynlig",
        description="Bruttomarginal varierar kraftigt per kund och produkt men spåras inte systematiskt.",
        priority=3,
        status="diskuteras",
        confidence=55.0,
    ))

    # ── Assumptions ───────────────────────────────────────────────────────────

    # The pivotal assumption that collapsed — was NEVER explicit in the scan
    a_inkop = g.add_node(AssumptionNode(
        statement="Bolaget kan välja när och hur mycket råvara de köper",
        risk_level="high",
        is_implicit=True,
        status="falsifierat",
        confidence=10.0,
        notes=(
            "Aldrig formulerat som en hypotes i scanen. Kollapsade under workshopen: "
            "biologisk kvot och sillankomst styr helt — inköpsflexibilitet saknas."
        ),
    ))

    a_personal = g.add_node(AssumptionNode(
        statement="Personalvolymen kan dimensioneras i förväg relativt förväntad sillankomst",
        risk_level="medium",
        is_implicit=False,
        status="verifierat",
        confidence=70.0,
        notes="Bekräftades av Martin Kuhlin. Historisk data finns för minst 3 säsonger.",
    ))

    a_data_finns = g.add_node(AssumptionNode(
        statement="Historisk data om sillankomst och personalvolym finns digitalt",
        risk_level="medium",
        is_implicit=False,
        status="explicit",
        confidence=60.0,
    ))

    a_kund_data = g.add_node(AssumptionNode(
        statement="Data för lönsamhetsanalys per kund finns i befintliga system",
        risk_level="low",
        is_implicit=False,
        status="verifierat",
        confidence=75.0,
        notes="Anna Andreasson bekräftade att datan finns.",
    ))

    # ── Mechanisms ────────────────────────────────────────────────────────────

    mech_original = g.add_node(MechanismNode(
        description="Reaktivt råvaruinköp: köper när priset är högt och kvoten halvförbrukad",
        status="ersatt",
        confidence=15.0,
        notes="Ersatt under workshopen när inköpsflexibilitetsantagandet kollapsade.",
    ))

    mech_aktiv = g.add_node(MechanismNode(
        description=(
            "Felaktig personaldimensionering relativt sillankomst: "
            "för många/få kontraktsanställda kallas in per säsong, "
            "vilket skapar antingen lönekostnadsspill eller produktionsglapp."
        ),
        status="aktiv",
        confidence=70.0,
    ))

    # ── Evidence ──────────────────────────────────────────────────────────────

    ev_glapp = g.add_node(EvidenceNode(
        statement="Martin Kuhlin har identifierat ett glapp på 2,4 MSEK 2025 mellan personalvolym och sillleveranser",
        source="Workshop 2025-06",
        speaker="Martin Kuhlin",
        direction="stodjer",
        confidence_delta=35.0,
        confidence=85.0,
    ))

    ev_inkop_falsifierad = g.add_node(EvidenceNode(
        statement="Biologisk kvot och sillankomst styrs helt av naturliga faktorer — inköpsflexibilitet saknas",
        source="Workshop 2025-06",
        speaker="Martin Kuhlin",
        direction="motbevisar",
        confidence_delta=40.0,
        confidence=90.0,
    ))

    ev_kunddata = g.add_node(EvidenceNode(
        statement="Anna Andreasson bekräftade att data finns för lönsamhetsanalys per kund",
        source="Workshop 2025-06",
        speaker="Anna Andreasson",
        direction="stodjer",
        confidence_delta=20.0,
        confidence=80.0,
    ))

    # ── Observations (from workshop transcript, not yet all resolved) ─────────

    obs_elia = g.add_node(ObservationNode(
        raw_text=(
            "Elia Widmark Jangevik ifrågasatte om personalplaneringssystemet "
            "verkligen kan bli bättre givet sillankomstens oförutsägbarhet"
        ),
        source="Workshop-transkript 2025-06",
        speaker="Elia Widmark Jangevik",
        status="ohanterad",
        confidence=50.0,
    ))

    obs_ekonomisystem = g.add_node(ObservationNode(
        raw_text="Anna Andreasson nämnde Visma för löner — oklart om produktionsdata också ligger där",
        source="Workshop-transkript 2025-06",
        speaker="Anna Andreasson",
        status="ohanterad",
        confidence=50.0,
    ))

    # ── Stakeholders ──────────────────────────────────────────────────────────

    sk_martin = g.add_node(StakeholderNode(
        name="Martin Kuhlin",
        title="Produktionschef",
        role="beslutsfattare",
        present_at_workshop=True,
        confidence=65.0,
        notes="Lämnade mötet tidigt. Beslutsmandat för pilot ej formellt bekräftat.",
    ))

    sk_anna = g.add_node(StakeholderNode(
        name="Anna Andreasson",
        title="Ekonomichef",
        role="dataägare",
        present_at_workshop=True,
        confidence=75.0,
        notes="Verifierade datakällor. Positiv inställning till analyse.",
    ))

    sk_elia = g.add_node(StakeholderNode(
        name="Elia Widmark Jangevik",
        title="Okänd",
        role="skeptiker",
        present_at_workshop=True,
        confidence=50.0,
        notes="Invändning om oförutsägbarhet ej adresserad. Öppen risk.",
    ))

    # ── Decision ──────────────────────────────────────────────────────────────

    decision = g.add_node(DecisionNode(
        description="Hur många kontraktsanställda kallas in, och från vilken vecka under silläsongen?",
        frequency="Säsongsvis (mars–juni)",
        status="ägare_känd",
        confidence=70.0,
    ))

    # ── Data Sources ──────────────────────────────────────────────────────────

    ds_personal = g.add_node(DataSourceNode(
        description="Historiska data om personalvolym och sillleveranser per dag, minst 3 säsonger",
        data_type="Tidsserie / operativ data",
        system_name="Okänt (troligen Visma eller Excel)",
        owner="Martin Kuhlin / Anna Andreasson",
        status="bekräftad",
        confidence=65.0,
        notes="Existensen bekräftad men format och tillgänglighet ej verifierad.",
    ))

    ds_kunder = g.add_node(DataSourceNode(
        description="Försäljnings- och marginaldata per kund och produkt",
        data_type="Ekonomidata",
        system_name="Visma (troligen)",
        owner="Anna Andreasson",
        status="bekräftad",
        confidence=75.0,
    ))

    # ── Use Case ──────────────────────────────────────────────────────────────

    uc = g.add_node(UseCaseNode(
        title="Personalplaneringsprognos silläsong",
        module="FlowZero",
        expected_value_min_sek=700_000,
        expected_value_max_sek=1_200_000,
        status="prioriterad",
        confidence=65.0,
        notes="Förutsätter tillgång till 3+ säsongers historisk data.",
    ))

    # ── KPI ───────────────────────────────────────────────────────────────────

    kpi = g.add_node(KPINode(
        name="Glapp personalvolym/sillvolym (SEK/säsong)",
        description="Skillnad mellan faktiska personalkostnader och optimalt läge givet sillankomst",
        unit="SEK/säsong",
        elir_dimension="L",
        baseline_value=2_400_000,
        baseline_year=2025,
        target_value=800_000,
        status="baslinjerad",
        confidence=70.0,
    ))

    # ── Gaps ──────────────────────────────────────────────────────────────────

    gap_mandat = g.add_node(GapNode(
        description="Martins beslutsmandat för pilot och datadelning är inte formellt bekräftat",
        priority="critical",
        elir_dimension="R",
        status="identifierat",
        confidence=30.0,
    ))

    gap_datasystem = g.add_node(GapNode(
        description="Vilket system innehåller produktionsdata och i vilket format?",
        priority="high",
        elir_dimension="I",
        status="identifierat",
        confidence=40.0,
    ))

    gap_elia = g.add_node(GapNode(
        description="Elias invändning om oförutsägbarhet är ej adresserad — okänt om den blockerar",
        priority="medium",
        elir_dimension="R",
        status="identifierat",
        confidence=50.0,
    ))

    gap_h2_data = g.add_node(GapNode(
        description="H2 (lagerhantering) har inga data och ingen workshoptid — okänd potential",
        priority="low",
        elir_dimension="L",
        status="identifierat",
        confidence=20.0,
    ))

    # ── Actions ───────────────────────────────────────────────────────────────

    action_mandat = g.add_node(ActionNode(
        description=(
            "Boka uppföljningsmöte med Martin Kuhlin för att formellt bekräfta "
            "beslutsmandat och datadelning inför pilot"
        ),
        assigned_to="xZero account manager",
        deadline="Inom 2 veckor",
        status="tilldelad",
        confidence=80.0,
    ))

    action_datasystem = g.add_node(ActionNode(
        description=(
            "Fråga Anna Andreasson: vilket system innehåller produktionsdata, "
            "i vilket format och vem har access?"
        ),
        assigned_to="xZero lösningsarkitekt",
        deadline="Inom 1 vecka",
        status="identifierad",
        confidence=80.0,
    ))

    action_elia = g.add_node(ActionNode(
        description=(
            "Adressera Elias invändning om oförutsägbarhet: visa hur FlowZero "
            "hanterar osäkerhet med konfidensintervall i prognosen"
        ),
        assigned_to="xZero lösningsarkitekt",
        deadline="Inför nästa kundmöte",
        status="identifierad",
        confidence=70.0,
    ))

    # ── Edges — Hypothesis chain ───────────────────────────────────────────────

    g.add_edge(a_inkop.id,    RelationType.UNDERBYGGER,   h1.id)
    g.add_edge(a_personal.id, RelationType.UNDERBYGGER,   h1.id)
    g.add_edge(a_data_finns.id, RelationType.UNDERBYGGER, h1.id)
    g.add_edge(a_kund_data.id,  RelationType.UNDERBYGGER, h3.id)

    g.add_edge(h1.id, RelationType.FORKLARAS_AV, mech_original.id)
    g.add_edge(h1.id, RelationType.FORKLARAS_AV, mech_aktiv.id)

    g.add_edge(mech_aktiv.id, RelationType.PEKAR_PA, decision.id)

    g.add_edge(decision.id, RelationType.FORBATTRAS_AV, uc.id)

    # ── Edges — Evidence ──────────────────────────────────────────────────────

    g.add_edge(ev_glapp.id,               RelationType.STODJER,    h1.id)
    g.add_edge(ev_glapp.id,               RelationType.STODJER,    mech_aktiv.id)
    g.add_edge(ev_inkop_falsifierad.id,   RelationType.MOTBEVISAR, a_inkop.id)
    g.add_edge(ev_kunddata.id,            RelationType.STODJER,    h3.id)
    g.add_edge(ev_kunddata.id,            RelationType.STODJER,    a_kund_data.id)

    # ── Edges — Gaps ──────────────────────────────────────────────────────────

    g.add_edge(gap_mandat.id,    RelationType.SAKNAS_I, decision.id)
    g.add_edge(gap_datasystem.id, RelationType.SAKNAS_I, ds_personal.id)
    g.add_edge(gap_elia.id,      RelationType.SAKNAS_I, h1.id)
    g.add_edge(gap_h2_data.id,   RelationType.SAKNAS_I, h2.id)

    # ── Edges — Actions ───────────────────────────────────────────────────────

    g.add_edge(action_mandat.id,    RelationType.STANGER,   gap_mandat.id)
    g.add_edge(action_mandat.id,    RelationType.TILLDELAS, sk_martin.id)
    g.add_edge(action_datasystem.id, RelationType.STANGER,  gap_datasystem.id)
    g.add_edge(action_datasystem.id, RelationType.TILLDELAS, sk_anna.id)
    g.add_edge(action_elia.id,      RelationType.STANGER,   gap_elia.id)
    g.add_edge(action_elia.id,      RelationType.TILLDELAS, sk_elia.id)

    # ── Edges — Stakeholders ──────────────────────────────────────────────────

    g.add_edge(sk_martin.id, RelationType.AGER,         decision.id)
    g.add_edge(sk_martin.id, RelationType.KONTROLLERAR, ds_personal.id)
    g.add_edge(sk_anna.id,   RelationType.KONTROLLERAR, ds_kunder.id)
    g.add_edge(sk_anna.id,   RelationType.KONTROLLERAR, ds_personal.id)

    # ── Edges — Use Case / KPI / Data Source ──────────────────────────────────

    g.add_edge(uc.id,  RelationType.MATS_AV,    kpi.id)
    g.add_edge(uc.id,  RelationType.KRAVER_DATA, ds_personal.id)
    g.add_edge(kpi.id, RelationType.KVANTIFIERAR, opp.id)

    # ── Edges — Observations (ohanterade) ─────────────────────────────────────
    # These are NOT yet resolved — they show up as M7 violations, as intended.
    # obs_elia and obs_ekonomisystem are left unresolved deliberately.

    return g


if __name__ == "__main__":
    import json
    graph = build()
    print(graph.to_json())

    print("\n── Summary ──────────────────────────────────────────────────────")
    import pprint
    pprint.pprint(graph.summary())

    print("\n── Gap Hunter ───────────────────────────────────────────────────")
    from gap_hunter import summarize
    pprint.pprint(summarize(graph))
