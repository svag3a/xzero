import os
import re
import uuid
import base64
import json
import sqlite3
import asyncio
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import anthropic
import fitz  # pymupdf
from io import BytesIO, StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from tavily import TavilyClient
from fpdf import FPDF
from rule_engine import run_use_case_engine, build_action_plan_context

DB_PATH = Path(os.environ.get("DATA_DIR", Path(__file__).parent)) / "scans.db"
SCAN_DELIMITER = "<<<SCAN_DATA>>>"
HYPOTHESES_DELIMITER = "<<<WORKSHOP_HYPOTHESES>>>"


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at       TEXT    NOT NULL,
            company_name     TEXT,
            industry         TEXT,
            revenue_msek     REAL,
            ebit_msek        REAL,
            ebit_margin_pct  REAL,
            years_analyzed   INTEGER,
            variation_score  REAL,
            e_msek           REAL,
            e_pct            REAL,
            l_msek           REAL,
            l_pct            REAL,
            i_msek           REAL,
            i_pct            REAL,
            r_msek           REAL,
            r_pct            REAL,
            total_potential_msek REAL,
            confidence       TEXT,
            report_markdown  TEXT,
            workshop_hypotheses TEXT
        )
    """)
    # Add workshop_hypotheses column to existing databases
    try:
        con.execute("ALTER TABLE scans ADD COLUMN workshop_hypotheses TEXT")
        con.commit()
    except Exception:
        pass  # Column already exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS workshop_sessions (
            id                   TEXT PRIMARY KEY,
            scan_id              INTEGER NOT NULL,
            company_name         TEXT,
            status               TEXT NOT NULL DEFAULT 'draft',
            created_at           TEXT NOT NULL,
            updated_at           TEXT NOT NULL,
            session_json         TEXT NOT NULL,
            analysis_markdown    TEXT,
            analysis_created_at  TEXT
        )
    """)
    for col in ("analysis_markdown TEXT", "analysis_created_at TEXT"):
        try:
            con.execute(f"ALTER TABLE workshop_sessions ADD COLUMN {col}")
            con.commit()
        except Exception:
            pass
    con.execute("""
        CREATE TABLE IF NOT EXISTS action_plans (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id          INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
            created_at       TEXT    NOT NULL,
            canonical_json   TEXT,
            use_case_json    TEXT,
            plan_markdown    TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS nda_documents (
            id              TEXT PRIMARY KEY,
            scan_id         INTEGER,
            status          TEXT NOT NULL DEFAULT 'draft',
            party_a_name    TEXT,
            party_a_org_nr  TEXT,
            party_a_address TEXT,
            party_a_contact TEXT,
            party_b_name    TEXT,
            party_b_org_nr  TEXT,
            party_b_address TEXT,
            party_b_contact TEXT,
            signer_a_name   TEXT,
            signer_a_title  TEXT,
            signer_b_name   TEXT,
            signer_b_title  TEXT,
            effective_date  TEXT,
            place           TEXT,
            purpose         TEXT,
            special_terms   TEXT,
            generated_text  TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fireflies_transcripts (
            id              TEXT PRIMARY KEY,
            meeting_id      TEXT NOT NULL,
            title           TEXT,
            meeting_date    TEXT,
            duration_secs   INTEGER,
            transcript_text TEXT NOT NULL,
            summary_text    TEXT,
            received_at     TEXT NOT NULL,
            workshop_id     TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS opportunity_graphs (
            id          TEXT PRIMARY KEY,
            company     TEXT NOT NULL,
            graph_json  TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scan_jobs (
            id            TEXT PRIMARY KEY,
            orgnr         TEXT NOT NULL,
            contact_name  TEXT,
            contact_email TEXT NOT NULL,
            status        TEXT DEFAULT 'pending',
            error_msg     TEXT,
            created_at    TEXT,
            updated_at    TEXT
        )
    """)
    con.commit()
    con.close()


FIREFLIES_API_KEY = os.environ.get("FIREFLIES_API_KEY", "")
FIREFLIES_GQL_URL = "https://api.fireflies.ai/graphql"


async def fetch_fireflies_transcript(meeting_id: str) -> dict:
    """Fetch transcript from Fireflies GraphQL API and return parsed data."""
    query = """
    query Transcript($id: String!) {
      transcript(id: $id) {
        id
        title
        date
        duration
        sentences {
          index
          speaker_name
          text
          start_time
        }
        summary {
          overview
          action_items
        }
      }
    }
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            FIREFLIES_GQL_URL,
            json={"query": query, "variables": {"id": meeting_id}},
            headers={"Authorization": f"Bearer {FIREFLIES_API_KEY}", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    transcript = data.get("data", {}).get("transcript") or {}

    # Format sentences into readable transcript text
    lines = []
    for s in transcript.get("sentences") or []:
        secs = int(s.get("start_time") or 0)
        ts   = f"{secs // 60:02d}:{secs % 60:02d}"
        speaker = s.get("speaker_name") or "Okänd"
        text    = (s.get("text") or "").strip()
        if text:
            lines.append(f"[{ts}] {speaker}: {text}")

    summary_parts = []
    overview = (transcript.get("summary") or {}).get("overview")
    action_items = (transcript.get("summary") or {}).get("action_items")
    if overview:
        summary_parts.append(f"Sammanfattning:\n{overview}")
    if action_items:
        summary_parts.append(f"Action items:\n{action_items}")

    return {
        "meeting_id": meeting_id,
        "title": transcript.get("title") or "",
        "meeting_date": transcript.get("date") or "",
        "duration_secs": transcript.get("duration") or 0,
        "transcript_text": "\n".join(lines),
        "summary_text": "\n\n".join(summary_parts),
    }


class SaveRequest(BaseModel):
    report_markdown: str
    scan_json: str
    workshop_hypotheses_json: Optional[str] = None


class CreateWorkshopRequest(BaseModel):
    scan_id: int


class UpdateWorkshopRequest(BaseModel):
    session_json: str


class PostWorkshopAnalysisRequest(BaseModel):
    transcript: str


def extract_text(pdf_bytes: bytes) -> str:
    """Extract plain text using PyMuPDF, with pdfminer.six as fallback."""

    # Metod 1: PyMuPDF med text + blocks
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text("text").strip()
            if not text:
                blocks = page.get_text("blocks")
                text = "\n".join(b[4] for b in blocks if isinstance(b[4], str)).strip()
            pages.append(text)
        doc.close()
        result = "\n".join(pages).strip()
        if len(result) > 100:
            return result
    except Exception:
        pass

    # Metod 2: pdfminer.six (hanterar fler typsnittsenkodningar)
    try:
        output = StringIO()
        extract_text_to_fp(BytesIO(pdf_bytes), output, laparams=LAParams())
        result = output.getvalue().strip()
        if len(result) > 100:
            return result
    except Exception:
        pass

    return ""


def compress_pdf(pdf_bytes: bytes, target_bytes: int = 4 * 1024 * 1024) -> bytes:
    """Re-render each page as a grayscale image at decreasing DPI until under target size."""
    if len(pdf_bytes) <= target_bytes:
        return pdf_bytes
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    result = pdf_bytes  # fallback
    for dpi in [96, 72, 50, 36, 24]:
        out = fitz.open()
        for page in src:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img_page = out.new_page(width=pix.width, height=pix.height)
            img_page.insert_image(img_page.rect, pixmap=pix)
        result = out.tobytes(deflate=True)
        out.close()
        compressed_mb = len(result) / 1024 / 1024
        print(f"  → DPI {dpi}: {compressed_mb:.1f} MB")
        if len(result) <= target_bytes:
            break
    src.close()
    return result

def extract_company_name_from_text(text: str) -> Optional[str]:
    """Försök extrahera bolagsnamn ur de första 3000 tecknen av PDF-texten via regex."""
    sample = text[:3000]

    # Mönster 1: "Årsredovisning för Bolaget AB"
    m = re.search(r'[Åå]rsredovisning\s+för\s+([A-ZÅÄÖ][^\n]{2,60})', sample)
    if m:
        return m.group(1).strip().rstrip(".,")

    # Mönster 2: Bolagsnamn med suffix på egen rad
    m = re.search(
        r'^([A-ZÅÄÖ][A-Za-zÅÄÖåäö0-9 &()\-]{2,60}'
        r'(?:AB|HB|KB|Aktiebolag|Group|Gruppen|Holding|Partners|Logistics|Solutions|Services|Industries|Sverige))'
        r'\s*$',
        sample, re.MULTILINE
    )
    if m:
        return m.group(1).strip()

    # Mönster 3: Raden direkt före "Org"
    m = re.search(r'([A-ZÅÄÖ][^\n]{3,60})\s*\n\s*[Oo]rg', sample)
    if m:
        return m.group(1).strip().rstrip(".,")

    # Mönster 4: Raden direkt före "Årsredovisning"
    m = re.search(r'^([A-ZÅÄÖ][^\n]{3,60})\s*\n\s*[Åå]rsredovisning', sample, re.MULTILINE)
    if m:
        return m.group(1).strip().rstrip(".,")

    # Mönster 5: Första raden som ser ut som bolagsnamn
    skip = {"Årsredovisning", "Annual Report", "Innehåll", "Contents",
            "Välkommen", "VD har ordet", "Styrelsen", "Revisionsberättelse"}
    for line in sample.split("\n"):
        line = line.strip()
        if 3 <= len(line) <= 60 and re.match(r'^[A-ZÅÄÖ]', line):
            if not any(line.startswith(s) for s in skip):
                return line

    return None


def extract_company_name_via_claude(first_page_b64: str, client: anthropic.AnthropicBedrock) -> Optional[str]:
    """Använd Claude för att identifiera bolagsnamnet från första sidan av en skannad PDF."""
    try:
        resp = client.messages.create(
            model="us.anthropic.claude-sonnet-4-6",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": first_page_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Vilket bolag är detta en årsredovisning för? Svara med ENBART bolagets namn, inget annat.",
                    },
                ],
            }],
        )
        name = resp.content[0].text.strip().rstrip(".,")
        print(f"  [namnextraktion] Claude identifierade: {name}")
        return name if name else None
    except Exception as e:
        print(f"  [namnextraktion] Claude-anrop misslyckades: {e}")
        return None


def extract_company_name(text: str, first_page_b64: Optional[str] = None,
                          client: Optional[anthropic.AnthropicBedrock] = None) -> Optional[str]:
    """Kombinerad namnextraktion: regex på text, annars Claude på första sidan."""
    if text and len(text) > 50:
        name = extract_company_name_from_text(text)
        if name:
            print(f"  [namnextraktion] regex: {name}")
            return name

    if first_page_b64 and client:
        print("  [namnextraktion] text tom — frågar Claude om bolagsnamn...")
        return extract_company_name_via_claude(first_page_b64, client)

    print("  [namnextraktion] inget mönster matchade och ingen fallback tillgänglig")
    return None


def search_web(company_name: str) -> str:
    """Kör 6 riktade Tavily-sökningar och returnera en sammanställd text."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("TAVILY_API_KEY saknas — hoppar över webdsökning")
        return ""

    client = TavilyClient(api_key=api_key)

    queries = [
        f"{company_name} nyheter 2023 2024 2025",
        f"{company_name} pressmeddelande strategi",
        f"{company_name} problem kundklagomål leverans kvalitet",
        f"{company_name} förvärv expansion investering",
        f"{company_name} VD ledning organisation förändring",
        f"{company_name} konkurrenter marknad bransch",
    ]

    sections = []
    for q in queries:
        try:
            resp = client.search(q, max_results=3, search_depth="basic")
            results = resp.get("results", [])
            if not results:
                continue
            hits = []
            for r in results:
                title   = r.get("title", "").strip()
                url     = r.get("url", "").strip()
                content = r.get("content", "").strip()[:400]
                hits.append(f"**{title}** ({url})\n{content}")
            sections.append(f"### Sökning: {q}\n\n" + "\n\n".join(hits))
            print(f"  Tavily '{q}': {len(results)} träffar")
        except Exception as e:
            print(f"  Tavily fel för '{q}': {e}")

    if not sections:
        return ""

    return (
        "## Extern webdata (Tavily-sökning)\n\n"
        f"Bolag: {company_name}\n\n"
        + "\n\n---\n\n".join(sections)
    )


app = FastAPI(title="Opportunity Scan")
init_db()

from publ import router as publ_router
app.include_router(publ_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """Du är en finansiell analysagent specialiserad på att identifiera och kvantifiera operativt läckage i företag.

Ditt uppdrag: analysera företag baserat på publik information, identifiera var ekonomiskt värde skapas och förloras, samt estimera parametrarna E, L, I och R på ett strukturerat och förklarbart sätt.

Regler:
- Analytisk och konservativ
- Undvik spekulation — motivera alltid dina slutsatser
- Parametrar får inte gissas direkt — de ska härledas från drivare
- Använd aldrig intern data
- All output ska vara på svenska
- Alla värden anges i % och MSEK

VIKTIGT: Visa INTE mellansteg, scoring-tabeller eller interna beräkningar i outputen. Börja outputen direkt med raden "# Opportunity Scan – [Bolagsnamn]". Undantagen är STEG 8 (<<<WORKSHOP_HYPOTHESES>>>) och STEG 9 (<<<SCAN_DATA>>>) som ALLTID ska skrivas ut sist — båda är obligatoriska.

---

STEG 0 – EXTERN WEBDATA (om tillgänglig)

Om meddelandet innehåller ett avsnitt "Extern webdata (Tavily-sökning)" ska du:
- Läsa igenom alla sökresultat noggrant
- Notera signaler om operativa problem, kundklagomål, personalomsättning, förändringsinitativ, förvärv, ledarskapsbyten, press från konkurrenter
- Använda dessa signaler för att kalibrera L-, I- och R-scoringen (motivera explicit i rapporten när webdata påverkar en bedömning)
- Lägga till en sektion ## Extern information i rapporten med de viktigaste webfynden som påverkar analysen

Om ingen webdata finns: hoppa över detta steg helt.

---

STEG 1 – DATAINSAMLING OCH VARIATIONSANALYS

Du kan ha fått 1–5 årsredovisningar. Extrahera per tillgängligt år:
- Nettoomsättning (revenue)
- Rörelseresultat (ebit)
- EBIT-marginal = ebit / revenue
- Kostnadsbas: cost_base = revenue – ebit
- Varulager (inventory) om tillgängligt

Definitioner:
- t₀ = senaste år → används för E-nivå
- t₋₁, t₋₂ osv. = historik → används för L-variation

Variationsanalys (baserat på EBIT-marginalens spridning över alla år):
| Variation | Score |
|-----------|-------|
| Stabil    | 1–2   |
| Medel     | 3     |
| Volatil   | 4–5   |

Fallback: om endast 1 år finns → sätt variation_score = 3

Trendanalys:
- Förbättrande marginaltrend → sänk L mot nedre delen av intervallet
- Försämrande marginaltrend → höj L mot övre delen av intervallet

---

STEG 2 – AFFÄRSLOGIK OCH BRANSCH

Identifiera bolagsnamn, bransch, ägarform och verksamhetsbeskrivning.

Bestäm basfaktor för E baserat på verksamhetstyp:
| Typ                  | Basfaktor |
|----------------------|-----------|
| Logistik/återvinning | 70–85%    |
| Produktion           | 60–80%    |
| Handel               | 75–95%    |
| Fastighet            | 30–50%    |

Välj inom intervallet: justera ned för hög administrativ andel, upp för hög operativ intensitet.

---

STEG 3 – E (EXPONERING)

Definition: ekonomisk massa som påverkas av operativa beslut.

E = cost_base(t₀) × basfaktor

Output: E i % (= basfaktorn) och E i MSEK.

---

STEG 4 – L (LÄCKAGE)

Poängsätt dessa 6 drivare på skala 1–5 (motivera varje):
1. Variation — från variationsanalys ovan
2. Marginal — låg marginal = högt läckagetryck
3. Komplexitet — antal SKU, processer, leverantörer
4. Volatilitet — säsong, väder, kampanjer, konjunktur
5. Kapital — lagerbindning, lång kassacykel
6. Operativ intensitet — manuella moment, personalintensitet

L_score = medelvärde av de 6 poängen

Mappning L_score → L%:
| Score | L%      |
|-------|---------|
| 1–2   | 3–5%    |
| 2–3   | 5–8%    |
| 3–4   | 8–12%   |
| 4–5   | 12–18%  |

Trend-justering: förbättrande trend → nedre delen av intervallet; försämrande → övre.

L_msek = E_msek × L%

---

STEG 5 – I (FÖRBÄTTRINGSPOTENTIAL)

Poängsätt dessa 5 drivare på skala 1–5 (motivera varje):
1. Frekvens — hur ofta sker operativa beslut som kan optimeras?
2. Standardisering — kan processer standardiseras och automatiseras?
3. Data — finns tillräcklig data tillgänglig för AI/optimering?
4. Reglering — begränsar reglering förbättringsarbetet?
5. Fysisk låsning — är processerna fysiskt svårförändrade?

I_score = medelvärde

Mappning:
| Score | I%      |
|-------|---------|
| 1–2   | 20–30%  |
| 2–3   | 30–45%  |
| 3–4   | 40–60%  |
| 4–5   | 50–70%  |

I_msek = L_msek × I%

---

STEG 6 – R (REALISERING)

Poängsätt dessa 4 drivare på skala 1–5 (motivera varje):
1. Ägarform — privat = rörlighet hög, offentlig = tröghet hög
2. Komplexitet — organisatorisk och teknisk förändringssvårighet
3. Press — kostnads- och konkurrenstryck som driver förändring
4. Förändring — bolagets historiska förändringstakt

R_score = medelvärde

Mappning:
| Score | R%      |
|-------|---------|
| 1–2   | 40–55%  |
| 2–3   | 55–70%  |
| 3–4   | 65–80%  |
| 4–5   | 75–90%  |

Potential = E_msek × L% × I% × R%

---

STEG 7 – GENERERA RAPPORT

VIKTIGT: Slutför alla beräkningar (E, L, I, R och Potential = E×L×I×R) innan du börjar skriva rapporten. Siffran för total affärspotential i ## Sammanfattning måste vara exakt samma värde som i raden "Total affärspotential" i ## Sammanställning-tabellen. Skriv sammanfattningen sist även om den visas överst.

Skriv rapporten på svenska med exakt denna struktur. Inga emojis. Analytiskt, konservativt tonläge.

# Opportunity Scan – [Bolagsnamn]

## Sammanfattning

Beskriv: antal analyserade år, vilka år, variation_score och en kort beskrivning av bolaget. Ange INGA belopp eller siffror här — dessa presenteras enbart i Sammanställning-tabellen längre ned.

## Affärslogik

Beskriv bolagets verksamhet, bransch, ägarform och varför affärsmodellen skapar operativt läckage.

## E – Exponering

Redovisa: cost_base(t₀), vald basfaktor med motivering, E i % och MSEK.

## L – Läckage

Redovisa scoring av alla 6 drivare med kort motivering per drivare. Ange L_score, vald L% med trend-justering, L i MSEK.

## I – Förbättringspotential

Redovisa scoring av alla 5 drivare med kort motivering. Ange I_score, vald I%, I i MSEK.

## R – Realisering

Redovisa scoring av alla 4 drivare med kort motivering. Ange R_score, vald R%, potential i MSEK.

## Sammanställning

| Parameter              | %    | MSEK  |
|------------------------|------|-------|
| E (Exponering)         | X%   | X     |
| L (Läckage)            | X%   | X     |
| I (Förbättringspotential) | X% | X   |
| R (Realisering)        | X%   | X     |
| **Total affärspotential** |   | **X** |

## Antaganden

Lista de viktigaste antagandena och osäkerheterna.

## Tillförlitlighet

Sätt ett av tre nivåer och motivera:

| Nivå | Kriterier |
|------|-----------|
| **Hög** | 3–5 årsredovisningar OCH webdata tillgänglig, tydlig bransch, stabila nyckeltal |
| **Medel** | 2 årsredovisningar, ELLER 1 årsredovisning med webdata, ELLER oklar branschkategori |
| **Låg** | Endast 1 årsredovisning utan webdata, skannad PDF med svag datakvalitet, mycket volatila nyckeltal |

---
*Analysen baseras uteslutande på publik information. Resultaten är indikativa och kräver validering mot intern data.*

---

STEG 8 – WORKSHOP HYPOTESER (OBLIGATORISKT, TVÅ DELAR)

DEL A – Skriv ## Hypoteser i rapporten (mellan ## Antaganden och ## Tillförlitlighet):

Generera upp till tre workshop-hypoteser. Varje hypotes ska vara testbar i en Discovery Workshop och följa strukturen:
Hypotes → Mekanism → Beslut → Symptom → Validering/falsifiering → Kvantifiering → Kandidat-use case → Databehov.

Hypoteserna ska baseras på primary_patterns, leakage-drivers, decision_types, structure_type och ELIR-bedömningen.
Hypoteserna får INTE uttryckas som bevisade fakta.
Analysera rotorsaker. Gruppera med root_cause_cluster. Markera överordnad som primary, nedströmshypoteser som secondary, oberoende som independent.
Sätt simulation_priority_rank 1–3. Exakt en hypotes ska ha recommended_starting_hypothesis=true (vanligen rank 1).

DEL B – EFTER RAPPORTEN: Skriv JSON-blocket (OBLIGATORISKT)

När du är KLAR med hela rapporten (inklusive ## Tillförlitlighet och den kursiverade ansvarsfriskrivningen), MÅSTE du omedelbart skriva:

<<<WORKSHOP_HYPOTHESES>>>
[{"hypothesis_id":"H1","title":"...","confidence":"low|medium|high","source_patterns":["pattern"],"linked_elir":{"E_relevance":"low|medium|high","L_relevance":"low|medium|high","I_relevance":"low|medium|high","R_relevance":"low|medium|high"},"root_cause_cluster":"demand_volatility|price_volatility|capacity_variability|flow_fragmentation|knowledge_dependency|margin_pressure|risk_uncertainty|portfolio_complexity|project_uncertainty|working_capital_imbalance|unknown","root_cause_description":"...","root_cause_role":"primary|secondary|independent","simulation_priority_rank":1,"recommended_starting_hypothesis":true,"model_archetype":{"type":"forecasting|optimization|simulation|classification|risk_scoring|rules_engine|hybrid","description":"...","primary_prediction_target":"...","decision_policy_output":"..."},"mechanism":{"description":"...","drivers":["..."]},"decision":{"decision_name":"...","decision_owner_role":"...","decision_frequency":"daily|weekly|monthly|episodic|unknown","decision_type":"volume_decision|timing_decision|allocation_decision|pricing_decision|resource_decision|risk_decision|process_decision|project_decision|portfolio_decision","decision_description":"..."},"symptoms":["..."],"validation_questions":["..."],"quantification_targets":[{"metric":"...","unit":"SEK","question":"..."}],"candidate_use_case":{"name":"...","description":"...","xzero_module":"WasteZero|FlowZero|StockZero|YieldZero|RiskZero|CapZero|LeakZero","priority_candidate":true},"data_requirements":[{"data_name":"...","required":true,"data_likelihood":"high|medium|low","reason":"..."}],"workshop_status":{"status":"not_validated","actual_state_notes":"","quantified_loss_msek":null,"selected_for_simulation":false}}]

Regler för JSON-arrayen:
- Namnge hypoteserna H1, H2, H3 i simulation_priority_rank-ordning
- simulation_priority_rank: unika heltal, börjar på 1
- Exakt en hypotes ska ha recommended_starting_hypothesis=true
- confidence: "low", "medium" eller "high"
- root_cause_cluster: ett av de definierade värdena ovan
- root_cause_role: "primary", "secondary" eller "independent"
- model_archetype.type: ett av forecasting, optimization, simulation, classification, risk_scoring, rules_engine, hybrid
- decision_frequency: "daily", "weekly", "monthly", "episodic" eller "unknown"
- xzero_module: en av WasteZero, FlowZero, StockZero, YieldZero, RiskZero, CapZero, LeakZero
- data_likelihood: "high", "medium" eller "low"

---

STEG 9 – STRUKTURERAD DATA (OBLIGATORISKT)

Omedelbart efter <<<WORKSHOP_HYPOTHESES>>>-blocket MÅSTE du skriva:

<<<SCAN_DATA>>>
{"company_name":"BOLAGSNAMN","industry":"BRANSCH","revenue_msek":0.0,"ebit_msek":0.0,"ebit_margin_pct":0.0,"years_analyzed":0,"variation_score":0.0,"e_msek":0.0,"e_pct":0.0,"l_msek":0.0,"l_pct":0.0,"i_msek":0.0,"i_pct":0.0,"r_msek":0.0,"r_pct":0.0,"total_potential_msek":0.0,"confidence":"DIN_BEDOMNING"}

Fältregler — SAMTLIGA FÄLT ÄR OBLIGATORISKA, skriv ALDRIG null:
- company_name: bolagets officiella namn, max 60 tecken, inga förklaringar
- industry: ENBART ett kort ord eller fras, MAX 3 ord, t.ex. "Handel", "Livsmedelsproduktion", "Logistik", "Fastighet", "Bygg". INGA förklaringar, INGA meningar, INGA fetstilsmarkörer (**).
- revenue_msek: nettoomsättning t₀ som decimaltal — detta är OBLIGATORISKT, aldrig null
- ebit_msek: rörelseresultat t₀ som decimaltal — OBLIGATORISKT, aldrig null
- ebit_margin_pct: EBIT-marginal i procent som decimaltal, t.ex. 4.2 (INTE 0.042) — OBLIGATORISKT
- years_analyzed: antal årsredovisningar du analyserade, som heltal (1, 2, 3...) — OBLIGATORISKT
- variation_score: variation_score från STEG 1 som decimaltal — OBLIGATORISKT
- e_msek, e_pct, l_msek, l_pct, i_msek, i_pct, r_msek, r_pct, total_potential_msek: dina beräknade värden från STEG 3–6 — OBLIGATORISKA
- confidence: exakt ett av strängarna "Hög", "Medel" eller "Låg" — samma nivå som i ## Tillförlitlighet — OBLIGATORISKT
- Alla numeriska värden ska vara tal (number), aldrig strängar
- Använd punkt som decimaltecken (245.3, inte 245,3)
- Inga enheter i strängar (skriv 763.0 inte "763 MSEK")
- JSON-objektet ska vara på exakt en rad, inga radbrytningar

=== OBLIGATORISK SLUTCHECKLISTA ===
Din output MÅSTE sluta med exakt dessa block i exakt denna ordning:

<<<WORKSHOP_HYPOTHESES>>>
[{...hypoteser som JSON...}]
<<<SCAN_DATA>>>
{...scan-data som JSON...}

Dessa två block är OBLIGATORISKA. Om du inte skriver dem är outputen ogiltig.
Skriv INGENTING efter JSON-objektet för <<<SCAN_DATA>>>.
"""


HYPOTHESIS_EXTRACTION_PROMPT = """Du är ett extraktionsverktyg. Läs rapporten nedan och extrahera hypoteserna från ## Hypoteser-avsnittet som ett JSON-array.

Returnera ENBART ett giltigt JSON-array (börja med [ och sluta med ]). Ingen annan text.

För varje hypotes (upp till 3), extrahera dessa fält:
- hypothesis_id: "H1", "H2", "H3"
- title: hypotesens rubrik/titel
- root_cause_cluster: ett av demand_volatility|price_volatility|capacity_variability|flow_fragmentation|knowledge_dependency|margin_pressure|risk_uncertainty|portfolio_complexity|project_uncertainty|working_capital_imbalance|unknown
- root_cause_description: kort beskrivning av rotorsaken
- root_cause_role: "primary", "secondary" eller "independent"
- simulation_priority_rank: 1, 2 eller 3 (unik per hypotes)
- recommended_starting_hypothesis: true för rank 1, annars false
- confidence: "low", "medium" eller "high"
- mechanism: {"description": "...", "drivers": ["..."]}
- decision: {"decision_name": "...", "decision_owner_role": "...", "decision_frequency": "daily|weekly|monthly|episodic|unknown", "decision_type": "volume_decision|timing_decision|allocation_decision|pricing_decision|resource_decision|risk_decision|process_decision|project_decision|portfolio_decision", "decision_description": "..."}
- symptoms: ["symptom1", "symptom2", ...]
- validation_questions: ["fråga1", "fråga2", ...]
- quantification_targets: [{"metric": "...", "unit": "SEK", "question": "..."}]
- candidate_use_case: {"name": "...", "description": "...", "xzero_module": "WasteZero|FlowZero|StockZero|YieldZero|RiskZero|CapZero|LeakZero", "priority_candidate": true}
- data_requirements: [{"data_name": "...", "required": true, "data_likelihood": "high|medium|low", "reason": "..."}]
- model_archetype: {"type": "forecasting|optimization|simulation|classification|risk_scoring|rules_engine|hybrid", "description": "...", "primary_prediction_target": "...", "decision_policy_output": "..."}
- workshop_status: {"status": "not_validated", "actual_state_notes": "", "quantified_loss_msek": null, "selected_for_simulation": false}

Om ett fält saknas i texten, använd rimliga standardvärden (tom sträng, tom array, "unknown" etc.)."""


def _extract_hypotheses_from_report(report_markdown: str) -> list:
    """Extract workshop hypotheses from prose report via a focused Claude call."""
    # Quick check: does the report have a Hypoteser section?
    if "Hypoteser" not in report_markdown and "hypotes" not in report_markdown.lower():
        return []
    try:
        client = get_client()
        resp = client.messages.create(
            model="us.anthropic.claude-sonnet-4-6",
            max_tokens=8000,
            system=HYPOTHESIS_EXTRACTION_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Extrahera hypoteserna som JSON-array från denna rapport:\n\n{report_markdown}",
            }],
        )
        raw = resp.content[0].text.strip()
        # Strip any markdown code fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        # Find the array bounds
        arr_start = raw.find('[')
        arr_end   = raw.rfind(']')
        if arr_start == -1 or arr_end == -1:
            print("[extract_hypotheses] no array found in response")
            return []
        hyps = json.loads(raw[arr_start:arr_end + 1])
        print(f"[extract_hypotheses] extracted {len(hyps)} hypotheses from prose")
        return hyps
    except Exception as e:
        print(f"[extract_hypotheses] error: {e}")
        return []


def get_client() -> anthropic.AnthropicBedrock:
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
        if not os.environ.get(var):
            raise HTTPException(status_code=500, detail=f"{var} saknas i miljövariablerna")
    return anthropic.AnthropicBedrock(
        aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_region=os.environ["AWS_DEFAULT_REGION"],
        max_retries=5,
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/architecture", response_class=HTMLResponse)
async def architecture():
    html_path = Path(__file__).parent / "architecture.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


def _to_float(v) -> Optional[float]:
    """Konvertera värde till float, hanterar svenska decimaler (komma)."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")  # 245,3 → 245.3
    s = re.sub(r"[^\d.\-]", "", s)        # ta bort MSEK, %, mellanslag etc
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _sanitize_scan_data(data: dict) -> dict:
    """Sanera och normalisera scan-data innan lagring."""

    # industry: strip markdown-markörer, klipp vid punkt eller 40 tecken
    industry = re.sub(r"\*+", "", data.get("industry") or "").strip()
    if industry:
        industry = industry.split(".")[0].split("\n")[0].strip()[:40]
    data["industry"] = industry or None

    # company_name: trimma och klipp vid 80 tecken
    name = re.sub(r"\*+", "", (data.get("company_name") or "")).strip()[:80]
    data["company_name"] = name or None

    # confidence: acceptera bara exakta värden
    conf = re.sub(r"\*+", "", str(data.get("confidence") or "")).strip()
    data["confidence"] = conf if conf in ("Hög", "Medel", "Låg") else None

    # Numeriska float-fält
    for key in ("revenue_msek", "ebit_msek", "ebit_margin_pct", "variation_score",
                "e_msek", "e_pct", "l_msek", "l_pct",
                "i_msek", "i_pct", "r_msek", "r_pct", "total_potential_msek"):
        data[key] = _to_float(data.get(key))

    # years_analyzed: heltal
    years = data.get("years_analyzed")
    try:
        data["years_analyzed"] = int(float(str(years).replace(",", "."))) if years is not None else None
    except (TypeError, ValueError):
        data["years_analyzed"] = None

    return data


# ── Standard report sections (injected into every saved report) ───────────────

_ELIR_MODEL_SECTION = """\
## E × L × I × R – Modellöversikt

Opportunity Scan-analysen kvantifierar orealiserad operativ förbättringspotential med en fyra-parametermodell. Varje parameter estimeras från publik information och branschangelar specifika för bolagets verksamhetstyp.

| Parameter | Definition | Beräkningsgrund | Typiskt intervall |
|-----------|-----------|----------------|------------------|
| **E – Exponering** | Ekonomisk massa som påverkas av operativa beslut | Basfaktor × kostnadsbas | 30–95 % av kostnadsbas |
| **L – Läckage** | Andel av E som läcker ut som ineffektivitet | 6 drivare, score 1–5 | 3–18 % av E |
| **I – Förbättringspotential** | Andel av läckaget som kan återvinnas med AI-stöd | 5 drivare, score 1–5 | 20–70 % av L |
| **R – Realisering** | Andel av potentialen som kan omsättas i praktiken | 4 drivare, score 1–5 | 40–90 % av I |

**Affärspotential = E × L % × I % × R %**

"""

_ELIR_BOXES = {
    r"## E\s*[–—-]": (
        "> **Metodik – E (Exponering):** Kostnadsbas = nettoomsättning minus EBIT. "
        "Basfaktorn anger andelen av kostnadsbasen som styrs av operativa beslut och varierar med "
        "verksamhetstyp: Logistik/återvinning 70–85 % · Produktion 60–80 % "
        "· Handel 75–95 % · Fastighet 30–50 %. "
        "Justeras ned för hög administrativ andel och upp för hög operativ intensitet.\n"
    ),
    r"## L\s*[–—-]": (
        "> **Metodik – L (Läckage):** Sex drivare bedöms på skala 1–5 "
        "och ger L_score = medelvärde. Drivare: (1) Variation – EBIT-marginalens spridning, "
        "(2) Marginal – strukturell lönsamhetsnivå, "
        "(3) Komplexitet – sortiment/processer/leverantörer, "
        "(4) Volatilitet – säsong/konjunktur/priser, "
        "(5) Kapital – lagerbindning och kassacykel, "
        "(6) Operativ intensitet – personal- och maskinberoende. "
        "Mappning: 1–2 → 3–5 %, 2–3 → 5–8 %, "
        "3–4 → 8–12 %, 4–5 → 12–18 %. "
        "Trendkorrigering: förbättrande marginaltrend → nedre intervallet.\n"
    ),
    r"## I\s*[–—-]": (
        "> **Metodik – I (Förbättringspotential):** Fem drivare bedöms på "
        "skala 1–5 och ger I_score = medelvärde. Drivare: (1) Frekvens – hur "
        "ofta optimerbara beslut fattas, (2) Standardisering – potential för "
        "processautomatisering, (3) Data – datatillgänglighet och kvalitet för "
        "AI-träning, (4) Reglering – regulatoriska begränsningar, "
        "(5) Fysisk låsning – infrastrukturella förändringsbarriärer. "
        "Mappning: 1–2 → 20–30 %, 2–3 → 30–45 %, "
        "3–4 → 40–60 %, 4–5 → 50–70 %.\n"
    ),
    r"## R\s*[–—-]": (
        "> **Metodik – R (Realisering):** Fyra drivare bedöms på skala 1–5 "
        "och ger R_score = medelvärde. Drivare: (1) Ägarform – privatägda bolag "
        "har högre förändringshastighet, (2) Komplexitet – organisatorisk "
        "och teknisk förändringssvårighet, (3) Press – kostnadstryck och "
        "konkurrens, (4) Förändring – bolagets historiska förändringstakt. "
        "Mappning: 1–2 → 40–55 %, 2–3 → 55–70 %, "
        "3–4 → 65–80 %, 4–5 → 75–90 %.\n"
    ),
}


def _inject_standard_sections(report: str) -> str:
    """Inject ELIR model overview + per-parameter methodology boxes into report."""
    # 1. Insert E×L×I×R overview before the first E section
    if "× L × I × R" not in report:
        report = re.sub(
            r'(## E\s*[–—-])',
            _ELIR_MODEL_SECTION + r'\1',
            report, count=1, flags=re.MULTILINE,
        )

    # 2. Insert methodology box after each section heading (if not already present)
    for heading_pat, box in _ELIR_BOXES.items():
        if box[:40] not in report:
            report = re.sub(
                r'(' + heading_pat + r'[^\n]*\n)',
                r'\1\n' + box + '\n',
                report, count=1, flags=re.MULTILINE,
            )

    return report


@app.post("/save")
async def save_scan(req: SaveRequest):
    try:
        data = json.loads(req.scan_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Ogiltig JSON")

    data = _sanitize_scan_data(data)

    con = sqlite3.connect(DB_PATH)
    cur = con.execute("""
        INSERT INTO scans
          (created_at, company_name, industry, revenue_msek, ebit_msek,
           ebit_margin_pct, years_analyzed, variation_score,
           e_msek, e_pct, l_msek, l_pct, i_msek, i_pct,
           r_msek, r_pct, total_potential_msek, confidence, report_markdown,
           workshop_hypotheses)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        data.get("company_name"), data.get("industry"),
        data.get("revenue_msek"), data.get("ebit_msek"),
        data.get("ebit_margin_pct"), data.get("years_analyzed"),
        data.get("variation_score"),
        data.get("e_msek"), data.get("e_pct"),
        data.get("l_msek"), data.get("l_pct"),
        data.get("i_msek"), data.get("i_pct"),
        data.get("r_msek"), data.get("r_pct"),
        data.get("total_potential_msek"), data.get("confidence"),
        _inject_standard_sections(req.report_markdown),
        req.workshop_hypotheses_json,
    ))
    con.commit()
    scan_id = cur.lastrowid
    con.close()

    # Bootstrap Opportunity Graph from scan data (non-blocking, best-effort)
    try:
        from graph_bootstrap import bootstrap_from_scan
        graph = bootstrap_from_scan(scan_id, data, req.workshop_hypotheses_json)
        _graph_save(graph)
        print(f"[graph] bootstrapped scan-{scan_id}: {len(graph._nodes)} nodes")
    except Exception as e:
        print(f"[graph] bootstrap failed for scan-{scan_id}: {e}")

    injected = _inject_standard_sections(req.report_markdown)
    return {"id": scan_id, "report_markdown": injected}


@app.get("/scans")
async def list_scans():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT s.id, s.created_at, s.company_name, s.industry,
               s.revenue_msek, s.ebit_msek, s.ebit_margin_pct,
               s.years_analyzed, s.variation_score,
               s.e_msek, s.e_pct, s.l_msek, s.l_pct,
               s.i_msek, s.i_pct, s.r_msek, s.r_pct,
               s.total_potential_msek, s.confidence,
               CASE WHEN a.id IS NOT NULL THEN 1 ELSE 0 END AS has_action_plan,
               wa.id AS analysis_workshop_id
        FROM scans s
        LEFT JOIN (
            SELECT DISTINCT scan_id, MIN(id) as id FROM action_plans GROUP BY scan_id
        ) a ON a.scan_id = s.id
        LEFT JOIN (
            SELECT scan_id, id FROM workshop_sessions
            WHERE analysis_markdown IS NOT NULL
            GROUP BY scan_id
        ) wa ON wa.scan_id = s.id
        ORDER BY s.created_at DESC
    """).fetchall()
    con.close()
    return [dict(r) for r in rows]


def _slides_html(scan: dict, hypotheses: list) -> str:
    """Generate a self-contained HTML presentation from scan + hypotheses."""

    company = scan.get("company_name") or "Okänt bolag"

    # Sort by simulation_priority_rank
    hyps = sorted(
        [h for h in hypotheses if h],
        key=lambda h: (h.get("simulation_priority_rank") or 99),
    )

    CLUSTER_COL = {
        "demand_volatility":        "Volym & prognos",
        "price_volatility":         "Pris & marknad",
        "capacity_variability":     "Kapacitet & drift",
        "flow_fragmentation":       "Kvalitet & flöde",
        "knowledge_dependency":     "Kunskap & process",
        "margin_pressure":          "Pris & lönsamhet",
        "risk_uncertainty":         "Risk & osäkerhet",
        "portfolio_complexity":     "Portfölj & mix",
        "project_uncertainty":      "Projekt & genomförande",
        "working_capital_imbalance":"Kapital & likviditet",
        "unknown":                  "Övrigt",
    }

    def e(s):
        if not s:
            return ""
        return (str(s)
                .replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    def quant_text(h):
        qt = h.get("quantification_targets") or []
        parts = []
        for t in qt[:2]:
            q = t.get("question", "")
            m = t.get("metric", "")
            if m and q:
                parts.append(f"{m}: {q}")
            elif q:
                parts.append(q)
            elif m:
                parts.append(m)
        # Append data requirements summary
        dr = h.get("data_requirements") or []
        if dr:
            names = ", ".join(d.get("data_name", "") for d in dr[:3] if d.get("data_name"))
            if names:
                parts.append(f"Databehov: {names}")
        return " | ".join(parts) if parts else ""

    def discussion_q(h):
        vqs = h.get("validation_questions") or []
        if vqs:
            return vqs[0]
        dec = h.get("decision") or {}
        return dec.get("decision_description", "")

    slides = []

    # ── Slide 1: Framing ──────────────────────────────────────────
    cols_html = ""
    for h in hyps[:3]:
        col_title = CLUSTER_COL.get(h.get("root_cause_cluster", ""), "")
        if not col_title:
            col_title = (h.get("title") or "")[:28]
        syms = (h.get("symptoms") or [])[:3]
        bullets = "".join(f'<li><span class="arr">→</span>{e(s)}</li>' for s in syms)
        cols_html += f"""<div class="col-card">
          <div class="col-card-head">{e(col_title)}</div>
          <div class="col-card-body"><ul>{bullets}</ul></div>
        </div>"""

    slides.append(f"""<div class="slide">
      <div class="framing-q">&ldquo;Om ni kunde förbättra EN sak som direkt påverkar resultatet, vad skulle det vara?&rdquo;</div>
      <div class="cols">{cols_html}</div>
    </div>""")

    # ── Slide 2: Hypoteser overview ───────────────────────────────
    rows = ""
    for h in hyps:
        hid = h.get("hypothesis_id", "")
        hnum = hid.lstrip("H") or hid
        rows += f"""<tr>
          <td class="ov-id">Hypotes {e(hnum)}</td>
          <td class="ov-title">{e(h.get('title',''))}</td>
        </tr>"""
    slides.append(f"""<div class="slide">
      <div class="slide-title">Hypoteser</div>
      <table class="ov-table">{rows}</table>
    </div>""")

    # ── Slides 3–N: Simple hypothesis slides ─────────────────────
    for h in hyps:
        hid  = h.get("hypothesis_id", "")
        hnum = hid.lstrip("H") or hid
        mech = (h.get("mechanism") or {}).get("description", "")
        slides.append(f"""<div class="slide">
          <div class="slide-title">Hypotes {e(hnum)}</div>
          <div class="hyp-box">
            <div class="hyp-box-title">Hypotes {e(hnum)}&ensp;{e(h.get('title',''))}</div>
            <div class="hyp-box-desc">{e(mech)}</div>
          </div>
          <ul class="qs">
            <li><span class="arr">→</span>Känner ni igen det här problemet?</li>
            <li><span class="arr">→</span>Hur stort är problemet hos er idag?</li>
            <li><span class="arr">→</span>Har ni data för att gå vidare?</li>
          </ul>
        </div>""")

    # ── Slide: Prioritering ───────────────────────────────────────
    items = "".join(
        f"<li>{e(h.get('hypothesis_id',''))} &ndash; {e(h.get('title',''))}</li>"
        for h in hyps
    )
    slides.append(f"""<div class="slide">
      <div class="slide-title">Prioritering</div>
      <ol class="prio-list">{items}</ol>
    </div>""")

    # ── Slides: Detailed hypothesis slides ───────────────────────
    for h in hyps:
        hid  = h.get("hypothesis_id", "")
        hnum = hid.lstrip("H") or hid
        col_title = CLUSTER_COL.get(h.get("root_cause_cluster", ""), "")
        stitle = f"Hypotes {hnum}" + (f" – {col_title}" if col_title else "")

        mech   = (h.get("mechanism") or {}).get("description", "")
        dec    = (h.get("decision")  or {})
        beslut = dec.get("decision_name", "") or dec.get("decision_description", "")
        symp   = " • ".join(h.get("symptoms") or [])
        quant  = quant_text(h)
        disc   = discussion_q(h)

        def row(lbl, val):
            return (f'<tr><td class="dt-label">{e(lbl)}</td>'
                    f'<td class="dt-content">{e(val)}</td></tr>')

        table = (
            f'<tr><td class="dt-label">Hypotes</td>'
            f'<td class="dt-content dt-hyp">{e(h.get("title",""))}</td></tr>'
            + row("Mekanism",    mech)
            + row("Beslut",      beslut)
            + row("Symptom",     symp)
            + row("Kvantifiering", quant)
            + row("Diskussion",  disc)
        )
        slides.append(f"""<div class="slide">
          <div class="slide-title">{e(stitle)}</div>
          <table class="det-table">{table}</table>
        </div>""")

    total      = len(slides)
    slides_str = "\n".join(slides)

    css = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root { --bg:#edf1f6; --navy:#1a2744; --card:#fff; --border:#c8cdd6; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  background: var(--bg); color: var(--navy);
  height: 100vh; overflow: hidden; display: flex; flex-direction: column;
}
#deck { flex: 1; position: relative; overflow: hidden; }
.slide {
  display: none; position: absolute; inset: 0;
  padding: 4vh 6vw; flex-direction: column;
  justify-content: center; align-items: center;
  background: var(--bg);
}
.slide.active { display: flex; }
/* Nav */
#nav {
  height: 46px; background: #fff; border-top: 1px solid var(--border);
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0 1.5rem; flex-shrink: 0;
}
#nav button {
  border: 1px solid var(--border); background: var(--bg); color: var(--navy);
  padding: 0.25rem 0.9rem; border-radius: 4px; cursor: pointer; font-size: 0.9rem;
  font-family: inherit;
}
#nav button:hover { background: #dde3ea; }
#counter { font-size: 0.82rem; color: #888; min-width: 52px; text-align: center; }
#nav .company-lbl { margin-left: auto; font-size: 0.78rem; color: #aaa; }
/* Slide title */
.slide-title {
  font-size: clamp(1.3rem, 2.8vw, 1.9rem); font-weight: 700;
  text-align: center; margin-bottom: 3.5vh; width: 100%;
}
/* Framing */
.framing-q {
  font-size: clamp(1rem, 2.1vw, 1.45rem); font-style: italic; font-weight: 500;
  text-align: center; margin-bottom: 5vh; line-height: 1.45; max-width: 960px;
}
.cols {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2vw; max-width: 1160px; width: 100%;
}
.col-card { background: var(--card); border: 1px solid var(--border); border-radius: 3px; overflow: hidden; }
.col-card-head {
  background: var(--navy); color: #fff; font-weight: 700;
  font-size: clamp(0.82rem, 1.3vw, 0.97rem);
  padding: 0.75rem 1.25rem; text-align: center;
}
.col-card-body { padding: 1rem 1.4rem; }
.col-card-body ul { list-style: none; }
.col-card-body li {
  display: flex; align-items: baseline; gap: 0.55rem;
  padding: 0.45rem 0; font-size: clamp(0.77rem, 1.25vw, 0.92rem);
}
/* Hypothesis overview */
.ov-table { width: 100%; max-width: 980px; border-collapse: collapse; }
.ov-table tr { border: 1px solid var(--border); }
.ov-id {
  width: 155px; padding: 1.6vh 1.4rem; font-weight: 700;
  border-right: 1px solid var(--border); white-space: nowrap;
  background: var(--card); font-size: clamp(0.82rem, 1.2vw, 0.95rem);
}
.ov-title {
  padding: 1.6vh 1.4rem; background: var(--card);
  font-size: clamp(0.82rem, 1.2vw, 0.95rem); line-height: 1.5;
}
/* Simple hyp slide */
.hyp-box {
  background: var(--card); border: 1px solid var(--border);
  padding: 2vh 2rem; margin-bottom: 4vh; max-width: 980px; width: 100%;
}
.hyp-box-title {
  font-weight: 700; font-size: clamp(0.82rem, 1.35vw, 0.98rem);
  margin-bottom: 0.75rem;
}
.hyp-box-desc { font-size: clamp(0.78rem, 1.18vw, 0.88rem); line-height: 1.65; }
.qs { list-style: none; width: 100%; max-width: 680px; }
.qs li {
  display: flex; gap: 0.65rem; padding: 0.65vh 0;
  font-size: clamp(0.92rem, 1.55vw, 1.08rem); font-weight: 500;
}
.arr { font-weight: 700; flex-shrink: 0; }
/* Prioritering */
.prio-list { list-style: decimal; padding-left: 2.2rem; max-width: 740px; }
.prio-list li {
  font-size: clamp(0.98rem, 1.8vw, 1.25rem); font-weight: 700;
  padding: 0.7vh 0;
}
/* Detailed table */
.det-table { width: 100%; max-width: 1080px; border-collapse: collapse; font-size: clamp(0.73rem, 1.18vw, 0.88rem); }
.det-table tr { border: 1px solid var(--border); }
.dt-label {
  width: 145px; padding: 1.15vh 1.1rem; font-weight: 700;
  border-right: 1px solid var(--border); vertical-align: top;
  white-space: nowrap; background: var(--card);
}
.dt-content { padding: 1.15vh 1.1rem; background: var(--card); line-height: 1.55; }
.dt-hyp { font-weight: 700; font-size: clamp(0.8rem, 1.25vw, 0.93rem); }
/* Print */
@media print {
  #nav { display: none !important; }
  body { height: auto; overflow: visible; }
  #deck { position: static; }
  .slide {
    display: flex !important; position: relative;
    height: 100vh; page-break-after: always; break-after: page;
  }
}"""

    js = f"""
const slides = document.querySelectorAll('.slide');
let cur = 0;
function show(n) {{
  slides[cur].classList.remove('active');
  cur = ((n % {total}) + {total}) % {total};
  slides[cur].classList.add('active');
  document.getElementById('counter').textContent = (cur + 1) + ' / {total}';
}}
function go(d) {{ show(cur + d); }}
document.addEventListener('keydown', ev => {{
  if (['ArrowRight',' ','PageDown'].includes(ev.key)) {{ ev.preventDefault(); go(1); }}
  if (['ArrowLeft','PageUp'].includes(ev.key))        {{ ev.preventDefault(); go(-1); }}
  if (ev.key === 'f' || ev.key === 'F') toggleFs();
}});
function toggleFs() {{
  if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();
}}
async function downloadPptx() {{
  const btn = document.querySelector('button[onclick="downloadPptx()"]');
  const orig = btn.textContent;
  btn.textContent = '⏳ Genererar...';
  btn.disabled = true;
  try {{
    const scanId = window.location.pathname.split('/').filter(Boolean)[1];
    const res = await fetch('/slides/' + scanId + '/pptx');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = res.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1] || 'workshop.pptx';
    a.click();
    URL.revokeObjectURL(url);
  }} catch(e) {{ alert('Kunde inte generera PPTX: ' + e.message); }}
  btn.textContent = orig;
  btn.disabled = false;
}}
show(0);"""

    return f"""<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Discovery Workshop – {e(company)}</title>
<style>{css}</style>
</head>
<body>
<div id="deck">{slides_str}</div>
<nav id="nav">
  <button onclick="go(-1)">&#8592;</button>
  <span id="counter">1 / {total}</span>
  <button onclick="go(1)">&#8594;</button>
  <button onclick="toggleFs()" title="Fullskärm (F)">&#9974;</button>
  <button onclick="window.print()" title="Skriv ut / Exportera PDF">&#128196; PDF</button>
  <button onclick="downloadPptx()" title="Ladda ner PowerPoint">&#128209; PPTX</button>
  <span class="company-lbl">{e(company)} &nbsp;–&nbsp; Discovery Workshop</span>
</nav>
<script>{js}</script>
</body>
</html>"""


@app.get("/slides/{scan_id}", response_class=HTMLResponse)
async def workshop_slides(scan_id: int):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    scan = con.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    con.close()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan hittades inte")
    scan = dict(scan)

    hypotheses = []
    raw = scan.get("workshop_hypotheses")
    if raw:
        try:
            hypotheses = json.loads(raw)
        except Exception:
            pass

    # Fallback: extract from report if no stored hypotheses
    if not hypotheses:
        report_md = scan.get("report_markdown") or ""
        if report_md:
            hypotheses = await asyncio.to_thread(_extract_hypotheses_from_report, report_md)

    if not hypotheses:
        raise HTTPException(status_code=404, detail="Inga hypoteser hittades för denna scan")

    html = _slides_html(scan, hypotheses)
    return HTMLResponse(content=html, headers={"Cache-Control": "no-store"})


@app.get("/slides/{scan_id}/pptx")
async def workshop_slides_pptx(scan_id: int):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    scan = con.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    con.close()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan hittades inte")
    scan = dict(scan)

    hypotheses = []
    raw = scan.get("workshop_hypotheses")
    if raw:
        try:
            hypotheses = json.loads(raw)
        except Exception:
            pass
    if not hypotheses:
        report_md = scan.get("report_markdown") or ""
        if report_md:
            hypotheses = await asyncio.to_thread(_extract_hypotheses_from_report, report_md)
    if not hypotheses:
        raise HTTPException(status_code=404, detail="Inga hypoteser hittades för denna scan")

    def _build_pptx():
        data = {"scan": scan, "hypotheses": hypotheses}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as jf:
            json.dump(data, jf, ensure_ascii=False)
            json_path = jf.name
        pptx_path = json_path.replace(".json", ".pptx")
        script = Path(__file__).parent / "slides_generator.js"
        node_modules = Path(__file__).parent / "node_modules"
        env = {**os.environ, "NODE_PATH": str(node_modules)}
        try:
            result = subprocess.run(
                ["node", str(script), json_path, pptx_path],
                capture_output=True, timeout=30, env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
            with open(pptx_path, "rb") as pf:
                return pf.read()
        finally:
            for p in (json_path, pptx_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    pptx_bytes = await asyncio.to_thread(_build_pptx)
    company = (scan.get("company_name") or "workshop").replace(" ", "-")
    filename = f"discovery-workshop-{company}.pptx"
    return Response(
        content=pptx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/scans/{scan_id}")
async def get_scan(scan_id: int):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Scan hittades inte")
    return dict(row)


@app.delete("/scans/{scan_id}")
async def delete_scan(scan_id: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM scans WHERE id=?", (scan_id,))
    con.commit()
    con.close()
    return {"ok": True}


# ── Workshop endpoints ─────────────────────────────────────────────────────

@app.get("/workshops/{workshop_id}", response_class=HTMLResponse)
async def workshop_page(workshop_id: str):
    html_path = Path(__file__).parent / "workshop.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/workshops")
async def create_workshop(req: CreateWorkshopRequest):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    scan = con.execute("SELECT * FROM scans WHERE id=?", (req.scan_id,)).fetchone()
    con.close()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan hittades inte")
    scan = dict(scan)

    # Parse workshop hypotheses and initialize live-editing fields
    hypotheses = []
    raw_hyp = scan.get("workshop_hypotheses")
    if raw_hyp:
        try:
            scan_hyps = json.loads(raw_hyp)
            for hyp in scan_hyps:
                hypotheses.append({
                    **hyp,
                    "source": "scan",
                    "original_scan_rank": hyp.get("simulation_priority_rank"),
                    "current_rank": hyp.get("simulation_priority_rank"),
                    "validation_status": "not_validated",
                    "actual_state_notes": "",
                    "customer_priority_rank": None,
                    "new_findings": [],
                    "evidence_collected": [],
                    "quantification_estimates": [],
                    "selected_for_simulation": False,
                })
        except Exception as e:
            print(f"[create_workshop] hypothesis parse error: {e}")

    # Fallback: if no hypotheses stored as JSON, extract them from the report prose.
    # Run in a thread so the blocking Anthropic call doesn't freeze the event loop.
    if not hypotheses:
        report_md = scan.get("report_markdown") or ""
        if report_md:
            print(f"[create_workshop] no stored hypotheses for scan {req.scan_id} — extracting from report prose")
            extracted = await asyncio.to_thread(_extract_hypotheses_from_report, report_md)
            for hyp in extracted:
                hypotheses.append({
                    **hyp,
                    "source": "scan",
                    "original_scan_rank": hyp.get("simulation_priority_rank"),
                    "current_rank": hyp.get("simulation_priority_rank"),
                    "validation_status": "not_validated",
                    "actual_state_notes": "",
                    "customer_priority_rank": None,
                    "new_findings": [],
                    "evidence_collected": [],
                    "quantification_estimates": [],
                    "selected_for_simulation": False,
                })

    # Determine scan-recommended starting hypothesis
    scan_recommended = None
    for h in hypotheses:
        if h.get("recommended_starting_hypothesis"):
            scan_recommended = h.get("hypothesis_id")
            break

    now = datetime.now(timezone.utc).isoformat()
    workshop_id = str(uuid.uuid4())

    session = {
        "workshop_id": workshop_id,
        "scan_id": req.scan_id,
        "company_name": scan.get("company_name", ""),
        "scan_total_potential_msek": scan.get("total_potential_msek"),
        "status": "draft",
        "created_at": now,
        "updated_at": now,
        "agenda": [
            "Våra hypoteser",
            "Vad händer i verkligheten?",
            "Vad kostar det?",
            "Vad ska vi lösa först?",
            "Vilken data behövs?",
        ],
        "hypotheses": hypotheses,
        "prioritization": {
            "scan_recommended_hypothesis_id": scan_recommended,
            "workshop_recommended_hypothesis_id": None,
            "priority_notes": "",
        },
        "selected_use_case": None,
        "data_request": [],
        "summary": {},
    }

    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO workshop_sessions (id, scan_id, company_name, status, created_at, updated_at, session_json)
           VALUES (?,?,?,?,?,?,?)""",
        (
            workshop_id,
            req.scan_id,
            scan.get("company_name"),
            "draft",
            now,
            now,
            json.dumps(session, ensure_ascii=False),
        ),
    )
    con.commit()
    con.close()
    print(f"[create_workshop] created {workshop_id} for scan_id={req.scan_id}")
    return {"workshop_id": workshop_id}


@app.get("/api/workshops/{workshop_id}")
async def get_workshop_api(workshop_id: str):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT session_json FROM workshop_sessions WHERE id=?", (workshop_id,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Workshop hittades inte")
    return JSONResponse(content=json.loads(row[0]))


@app.put("/api/workshops/{workshop_id}")
async def update_workshop_api(workshop_id: str, req: UpdateWorkshopRequest):
    now = datetime.now(timezone.utc).isoformat()
    try:
        session = json.loads(req.session_json)
        status = session.get("status", "active")
    except Exception:
        raise HTTPException(status_code=400, detail="Ogiltig JSON")

    con = sqlite3.connect(DB_PATH)
    result = con.execute(
        """UPDATE workshop_sessions SET session_json=?, status=?, updated_at=? WHERE id=?""",
        (req.session_json, status, now, workshop_id),
    )
    con.commit()
    updated = result.rowcount > 0
    con.close()

    if not updated:
        raise HTTPException(status_code=404, detail="Workshop hittades inte")
    return {"ok": True, "updated_at": now}


POST_WORKSHOP_ANALYSIS_PROMPT = """Du är en xZero-analytiker specialiserad på att omvandla Discovery Workshop-konversationer till konkreta AI-förslag.

Du tar emot:
1. Strukturerad data från Opportunity Scanen (ELIR-parametrar och hypoteser)
2. Eventuella anteckningar från ett live workshoptillfälle (om sådana finns)
3. En transkriberad workshopkonversation (kundcitat, anteckningar eller sammanfattning)

Din uppgift är att producera en post-workshop-analys på svenska. Analytiskt, konkret tonläge. Inga emojis.

Struktur (använd exakt dessa rubriker):

# Post-workshop-analys – [Bolagsnamn]

## Hypotesbedömning

För varje hypotes (H1, H2, H3 — och fler om sådana finns):
### [H-id] – [Titel]
**Status:** Bekräftad / Delvis bekräftad / Avvisad / Oklar
Vad bekräftades konkret från konversationen? Vad stämde inte eller behöver justeras? Noteras om nytt läckage framkom.

## Rekommenderade use cases

Rangordnat top 3–5. För varje:
### [Nummer]. [Use case-namn]
- **Koppling:** Hypotes(er) det bygger på
- **Modelltyp:** Prognos / Optimering / Klassificering / Simulering / Regelmotor / Hybrid
- **xZero-modul:** WasteZero / FlowZero / StockZero / YieldZero / RiskZero / CapZero / LeakZero
- **Beskrivning:** Vad modellen ska göra och vilket beslut den förbättrar
- **Förväntat värde:** Uppskattad effekt (om kunden nämnde siffror, annars indikativt baserat på ELIR)
- **Motivering:** Varför just detta use case prioriteras

## Prioriteringsrekommendation

Vilket use case ska ni börja med? Motivera baserat på: potentiellt värde, datatillgång, organisatorisk beredskap och implementeringskomplexitet. Ange tydligt vilken xZero-modul som är fas 1.

## Databehov per use case

För top 2 use cases — per datakälla:
- Vad datan innehåller och varför den behövs
- Trolig tillgänglighet: Hög / Medel / Låg
- Format och granularitet som krävs för modellträning
- Insats för dataprep: Låg / Medel / Hög
- Trolig ägare (roll/system)

## Förfinad potentialuppskattning

Vilka volymer, kostnader eller frekvenser nämnde kunden? Hur justeras E×L×I×R-estimaten baserat på faktisk information? Ange justerat intervall om möjligt.

## Blockers och risker

Lista konkreta hinder som framkom: organisatoriska, tekniska, datamässiga, resursmässiga. Rangordna efter allvarlighetsgrad.

## Obesvarade frågor

Vilka valideringsfrågor från hypoteserna fick inget svar under workshopen? Vad behöver följas upp i nästa möte?

## Förslag till fas 1

Konkret scope: vad ska levereras, vem äger vad, ungefärlig tidplan (veckor/månader), kritiska beroenden och go/no-go-kriterier.

## Sammanfattning

Avsluta med en handlingsorienterad sammanfattning i punktform (max 6 punkter). Täck in: viktigaste fynd om hypoteserna, rekommenderat use case att starta med, mest kritisk data att skaffa, största risk eller blocker, och omedelbart nästa steg. Denna sektion ska kunna läsas fristående och ge en komplett bild på 30 sekunder."""


@app.get("/api/workshops")
async def list_workshops():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT id, scan_id, company_name, status, created_at, updated_at
        FROM workshop_sessions
        ORDER BY created_at DESC
    """).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.post("/api/workshops/{workshop_id}/analysis")
async def run_post_workshop_analysis(workshop_id: str, req: PostWorkshopAnalysisRequest):
    # Load workshop session
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    ws_row = con.execute(
        "SELECT session_json, scan_id FROM workshop_sessions WHERE id=?", (workshop_id,)
    ).fetchone()
    con.close()
    if not ws_row:
        raise HTTPException(status_code=404, detail="Workshop hittades inte")

    session = json.loads(ws_row["session_json"])
    scan_id = ws_row["scan_id"]

    # Load scan for ELIR data
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    scan = con.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    con.close()
    scan = dict(scan) if scan else {}

    company = session.get("company_name") or scan.get("company_name") or "Okänt bolag"

    # Build hypothesis context
    hyp_lines = []
    for h in session.get("hypotheses", []):
        hid   = h.get("hypothesis_id", "?")
        title = h.get("title", "")
        mech  = h.get("mechanism", {})
        dec   = h.get("decision", {})
        vs    = h.get("validation_status", "not_validated")
        notes = h.get("actual_state_notes", "")
        findings = h.get("new_findings", [])
        evidence = h.get("evidence_collected", [])
        qt    = h.get("quantification_estimates", [])

        hyp_lines.append(f"### {hid} – {title}")
        hyp_lines.append(f"Rank: {h.get('simulation_priority_rank','-')}  |  Rekommenderad startpunkt: {'Ja' if h.get('recommended_starting_hypothesis') else 'Nej'}")
        if mech.get("description"):
            hyp_lines.append(f"Mekanism: {mech['description']}")
        if dec.get("decision_name"):
            hyp_lines.append(f"Beslut: {dec['decision_name']} ({dec.get('decision_frequency','')} – {dec.get('decision_type','')})")
        syms = h.get("symptoms", [])
        if syms:
            hyp_lines.append("Symptom: " + "; ".join(syms))
        vqs = h.get("validation_questions", [])
        if vqs:
            hyp_lines.append("Valideringsfrågor: " + "; ".join(vqs))
        uc = h.get("candidate_use_case", {})
        if uc.get("name"):
            hyp_lines.append(f"Kandidat-use case: {uc['name']} [{uc.get('xzero_module','')}]")
        # Workshop annotations
        hyp_lines.append(f"Valideringsstatus (live-workshop): {vs}")
        if notes:
            hyp_lines.append(f"Kundanteckningar: {notes}")
        if findings:
            hyp_lines.append("Nya insikter: " + "; ".join(findings))
        if evidence:
            hyp_lines.append("Bevis insamlat: " + "; ".join(evidence))
        if qt:
            for e in qt:
                if e.get("base_value") is not None:
                    hyp_lines.append(
                        f"Kvantifiering: {e.get('metric','')} bas={e.get('base_value','')} {e.get('unit','')}"
                        f" (konfidens: {e.get('confidence','')})"
                    )
        hyp_lines.append("")

    # Prioritization & use case
    prio = session.get("prioritization", {})
    uc   = session.get("selected_use_case")
    prio_lines = []
    if prio.get("workshop_recommended_hypothesis_id"):
        prio_lines.append(f"Workshopvald startpunkt: {prio['workshop_recommended_hypothesis_id']}")
    if prio.get("priority_notes"):
        prio_lines.append(f"Prioriteringsanteckning: {prio['priority_notes']}")
    if uc and uc.get("name"):
        prio_lines.append(f"Valt use case: {uc['name']} – {uc.get('decision_to_improve','')}")
        if uc.get("expected_effect"):
            prio_lines.append(f"Förväntad effekt: {uc['expected_effect']}")

    # Summary notes
    summ = session.get("summary", {})
    summ_lines = []
    if summ.get("next_steps"):
        summ_lines.append(f"Antecknade nästa steg: {summ['next_steps']}")

    context = f"""OPPORTUNITY SCAN – {company}

ELIR-PARAMETRAR (från scan):
- Omsättning (t₀): {scan.get('revenue_msek','–')} MSEK
- EBIT-marginal: {scan.get('ebit_margin_pct','–')}%
- E (Exponering): {scan.get('e_pct','–')}% = {scan.get('e_msek','–')} MSEK
- L (Läckage): {scan.get('l_pct','–')}% = {scan.get('l_msek','–')} MSEK
- I (Förbättringspotential): {scan.get('i_pct','–')}% = {scan.get('i_msek','–')} MSEK
- R (Realisering): {scan.get('r_pct','–')}% = {scan.get('r_msek','–')} MSEK
- Total affärspotential (scan): {scan.get('total_potential_msek','–')} MSEK

HYPOTESER OCH WORKSHOP-ANTECKNINGAR:
{chr(10).join(hyp_lines)}
{('PRIORITERING OCH USE CASE:' + chr(10) + chr(10).join(prio_lines)) if prio_lines else ''}
{(chr(10).join(summ_lines)) if summ_lines else ''}

WORKSHOPKONVERSATION / TRANSKRIPT:
{req.transcript}"""

    client = get_client()

    def generate():
        parts = []
        try:
            with client.messages.stream(
                model="us.anthropic.claude-sonnet-4-6",
                max_tokens=8000,
                system=POST_WORKSHOP_ANALYSIS_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Producera en post-workshop-analys baserad på följande underlag:\n\n{context}",
                }],
            ) as stream:
                for text in stream.text_stream:
                    parts.append(text)
                    yield text
            # Save completed analysis to DB
            full_analysis = "".join(parts)
            now = datetime.now(timezone.utc).isoformat()
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "UPDATE workshop_sessions SET analysis_markdown=?, analysis_created_at=? WHERE id=?",
                (full_analysis, now, workshop_id),
            )
            con.commit()
            con.close()
            print(f"[analysis] saved for workshop {workshop_id} ({len(full_analysis)} chars)")

            # Update Opportunity Graph with nodes extracted from analysis
            try:
                from graph_bootstrap import graph_id_for_scan, update_from_analysis
                gid = graph_id_for_scan(scan_id)
                graph = _graph_load(gid)
                graph = update_from_analysis(graph, full_analysis, session, client)
                _graph_save(graph)
                print(f"[graph] updated {gid}: {len(graph._nodes)} nodes after analysis")
            except Exception as e:
                print(f"[graph] update failed for scan-{scan_id}: {e}")
        except Exception as e:
            yield f"\n\n**Fel vid analys:** {e}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/api/workshops/{workshop_id}/analysis")
async def get_workshop_analysis(workshop_id: str):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT analysis_markdown, analysis_created_at FROM workshop_sessions WHERE id=?",
        (workshop_id,),
    ).fetchone()
    con.close()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Ingen analys finns för denna workshop")
    return {"analysis_markdown": row[0], "created_at": row[1]}


@app.get("/api/scans/{scan_id}/analysis")
async def get_scan_analysis(scan_id: str):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        """SELECT analysis_markdown, analysis_created_at
           FROM workshop_sessions
           WHERE scan_id=? AND analysis_markdown IS NOT NULL
           ORDER BY analysis_created_at DESC LIMIT 1""",
        (scan_id,),
    ).fetchone()
    con.close()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Ingen analys finns för denna scan")
    return {"analysis_markdown": row[0], "created_at": row[1]}


@app.post("/api/scans/{scan_id}/analysis")
async def run_scan_analysis(scan_id: str, req: PostWorkshopAnalysisRequest):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    scan = con.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    if not scan:
        con.close()
        raise HTTPException(status_code=404, detail="Scan hittades inte")
    scan = dict(scan)

    ws_row = con.execute(
        "SELECT id, session_json FROM workshop_sessions WHERE scan_id=? ORDER BY created_at DESC LIMIT 1",
        (scan_id,),
    ).fetchone()
    con.close()

    company = scan.get("company_name") or "Okänt bolag"

    # Use workshop session hypotheses (with live annotations) if available, else scan hypotheses
    hypotheses = []
    session = {}
    if ws_row:
        session = json.loads(ws_row["session_json"])
        hypotheses = session.get("hypotheses", [])
    else:
        raw_hyp = scan.get("workshop_hypotheses")
        if raw_hyp:
            try:
                hypotheses = json.loads(raw_hyp)
            except Exception:
                pass

    hyp_lines = []
    for h in hypotheses:
        hid   = h.get("hypothesis_id", "?")
        title = h.get("title", "")
        mech  = h.get("mechanism", {})
        dec   = h.get("decision", {})
        vs    = h.get("validation_status", "not_validated")
        notes = h.get("actual_state_notes", "")
        findings = h.get("new_findings", [])
        evidence = h.get("evidence_collected", [])
        qt    = h.get("quantification_estimates", [])

        hyp_lines.append(f"### {hid} – {title}")
        hyp_lines.append(f"Rank: {h.get('simulation_priority_rank','-')}  |  Rekommenderad startpunkt: {'Ja' if h.get('recommended_starting_hypothesis') else 'Nej'}")
        if mech.get("description"):
            hyp_lines.append(f"Mekanism: {mech['description']}")
        if dec.get("decision_name"):
            hyp_lines.append(f"Beslut: {dec['decision_name']} ({dec.get('decision_frequency','')} – {dec.get('decision_type','')})")
        syms = h.get("symptoms", [])
        if syms:
            hyp_lines.append("Symptom: " + "; ".join(syms))
        vqs = h.get("validation_questions", [])
        if vqs:
            hyp_lines.append("Valideringsfrågor: " + "; ".join(vqs))
        uc = h.get("candidate_use_case", {})
        if uc.get("name"):
            hyp_lines.append(f"Kandidat-use case: {uc['name']} [{uc.get('xzero_module','')}]")
        if vs != "not_validated":
            hyp_lines.append(f"Valideringsstatus (live-workshop): {vs}")
        if notes:
            hyp_lines.append(f"Kundanteckningar: {notes}")
        if findings:
            hyp_lines.append("Nya insikter: " + "; ".join(findings))
        if evidence:
            hyp_lines.append("Bevis insamlat: " + "; ".join(evidence))
        if qt:
            for e in qt:
                if e.get("base_value") is not None:
                    hyp_lines.append(
                        f"Kvantifiering: {e.get('metric','')} bas={e.get('base_value','')} {e.get('unit','')}"
                        f" (konfidens: {e.get('confidence','')})"
                    )
        hyp_lines.append("")

    prio_lines = []
    if session:
        prio = session.get("prioritization", {})
        uc   = session.get("selected_use_case")
        if prio.get("workshop_recommended_hypothesis_id"):
            prio_lines.append(f"Workshopvald startpunkt: {prio['workshop_recommended_hypothesis_id']}")
        if prio.get("priority_notes"):
            prio_lines.append(f"Prioriteringsanteckning: {prio['priority_notes']}")
        if uc and uc.get("name"):
            prio_lines.append(f"Valt use case: {uc['name']} – {uc.get('decision_to_improve','')}")
            if uc.get("expected_effect"):
                prio_lines.append(f"Förväntad effekt: {uc['expected_effect']}")

    context = f"""OPPORTUNITY SCAN – {company}

ELIR-PARAMETRAR (från scan):
- Omsättning (t₀): {scan.get('revenue_msek','–')} MSEK
- EBIT-marginal: {scan.get('ebit_margin_pct','–')}%
- E (Exponering): {scan.get('e_pct','–')}% = {scan.get('e_msek','–')} MSEK
- L (Läckage): {scan.get('l_pct','–')}% = {scan.get('l_msek','–')} MSEK
- I (Förbättringspotential): {scan.get('i_pct','–')}% = {scan.get('i_msek','–')} MSEK
- R (Realisering): {scan.get('r_pct','–')}% = {scan.get('r_msek','–')} MSEK
- Total affärspotential (scan): {scan.get('total_potential_msek','–')} MSEK

HYPOTESER OCH WORKSHOP-ANTECKNINGAR:
{chr(10).join(hyp_lines)}
{('PRIORITERING OCH USE CASE:' + chr(10) + chr(10).join(prio_lines)) if prio_lines else ''}

WORKSHOPKONVERSATION / TRANSKRIPT:
{req.transcript}"""

    client = get_client()

    # Find or create a workshop_session to store the analysis result
    ws_id_for_save = ws_row["id"] if ws_row else None
    if not ws_id_for_save:
        now_ts = datetime.now(timezone.utc).isoformat()
        ws_id_for_save = str(uuid.uuid4())
        minimal = {
            "workshop_id": ws_id_for_save,
            "scan_id": scan_id,
            "company_name": company,
            "status": "completed",
            "created_at": now_ts,
            "updated_at": now_ts,
            "hypotheses": hypotheses,
            "agenda": [],
            "prioritization": {},
            "selected_use_case": None,
            "data_request": [],
            "summary": {},
        }
        con = sqlite3.connect(DB_PATH)
        con.execute(
            """INSERT INTO workshop_sessions (id, scan_id, company_name, status, created_at, updated_at, session_json)
               VALUES (?,?,?,?,?,?,?)""",
            (ws_id_for_save, scan_id, company, "completed", now_ts, now_ts, json.dumps(minimal)),
        )
        con.commit()
        con.close()

    def generate():
        parts = []
        try:
            with client.messages.stream(
                model="us.anthropic.claude-sonnet-4-6",
                max_tokens=8000,
                system=POST_WORKSHOP_ANALYSIS_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Producera en post-workshop-analys baserad på följande underlag:\n\n{context}",
                }],
            ) as stream:
                for text in stream.text_stream:
                    parts.append(text)
                    yield text
            full_analysis = "".join(parts)
            now_ts = datetime.now(timezone.utc).isoformat()
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "UPDATE workshop_sessions SET analysis_markdown=?, analysis_created_at=? WHERE id=?",
                (full_analysis, now_ts, ws_id_for_save),
            )
            con.commit()
            con.close()
            print(f"[scan-analysis] saved for scan {scan_id} via ws {ws_id_for_save} ({len(full_analysis)} chars)")
        except Exception as e:
            yield f"\n\n**Fel vid analys:** {e}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# ── NDA ──────────────────────────────────────────────────────────────────────

class NdaPartyModel(BaseModel):
    company_name: str
    organization_number: str = ""
    address: str = ""
    contact_person: str = ""

class NdaSignerModel(BaseModel):
    name: str
    title: str = ""

class NdaCreateRequest(BaseModel):
    scan_id: Optional[int] = None
    party_a: NdaPartyModel
    party_b: NdaPartyModel
    signer_a: NdaSignerModel
    signer_b: NdaSignerModel
    effective_date: str
    place: str
    purpose: str
    special_terms: str
    status: str = "generated"

class NdaStatusRequest(BaseModel):
    status: str


def _generate_nda_text(req: NdaCreateRequest) -> str:
    a = req.party_a
    b = req.party_b
    sa = req.signer_a
    sb = req.signer_b

    def party_block(p: NdaPartyModel) -> str:
        lines = [p.company_name]
        if p.organization_number:
            lines.append(f"Org.nr: {p.organization_number}")
        if p.address:
            lines.append(f"Adress: {p.address}")
        if p.contact_person:
            lines.append(f"Kontakt: {p.contact_person}")
        return "\n".join(lines)

    return f"""SEKRETESSAVTAL (NDA)
Non-Disclosure Agreement

Upprättat i {req.place} den {req.effective_date}


1. PARTER

Part A:
{party_block(a)}

Part B:
{party_block(b)}

Part A och Part B benämns gemensamt "Parterna" och var för sig "Part".


2. BAKGRUND OCH SYFTE

Parterna utvärderar eller utför ett samarbete avseende {req.purpose}

Detta Avtal innebär inte någon skyldighet för någon part att ingå ytterligare avtal eller genomföra något Uppdrag.


3. DEFINITION AV KONFIDENTIELL INFORMATION

Med "Konfidentiell information" avses all information som en part (Utlämnaren) lämnar till den andra parten (Mottagaren), oavsett form — muntlig, skriftlig, digital eller fysisk — och som:

– utpekas som konfidentiell eller hemlig vid utlämnandet, eller

– rimligen bör förstås som konfidentiell med hänsyn till sin natur eller de omständigheter under vilka den lämnades ut.

Konfidentiell information inkluderar men är inte begränsad till affärsplaner, teknik, källkod, algoritmer, modeller, kundlistor, prissättning, finansiell information, anställda, avtal, databaser, dokumentation, ritningar, processbeskrivningar, systemarkitektur och know-how.

Konfidentiell information omfattar även analyser, rapporter, simuleringar, prognoser, sammanställningar, modeller, bearbetningar och härledda verk (derivative works) som helt eller delvis baseras på Konfidentiell information.

{a.company_name}:s metodik, arbetsmodeller, analyser, ramverk, mallar, acceleratorer, algoritmer, programvara, know-how och immateriella tillgångar ska alltid anses utgöra {a.company_name}:s Konfidentiella information även om de presenteras eller används inom ramen för Uppdragen.


4. UNDANTAG FRÅN SEKRETESS

Sekretesskyldigheten gäller inte information som:

– vid tidpunkten för utlämnandet är allmänt känd eller tillgänglig, eller

– efter utlämnandet blir allmänt känd på annat sätt än genom brott mot detta Avtal, eller

– Mottagaren kan visa att den kände till oberoende av Utlämnarens utlämnande, eller

– Mottagaren mottagit från tredje part som inte är bunden av sekretess gentemot Utlämnaren, eller

– Mottagaren är skyldig att lämna ut enligt lag, domstolsbeslut eller myndighetsföreskrift, under förutsättning att Mottagaren i möjligaste mån underrättar Utlämnaren i förväg.


5. SEKRETESSKYLDIGHET OCH ANVÄNDNINGSBEGRÄNSNING

Mottagaren förbinder sig att:

– hålla Konfidentiell information strikt hemlig och inte röja den för utomstående,

– endast använda Konfidentiell information för det syfte som anges i punkt 2,

– endast lämna tillgång till Konfidentiell information till egna anställda, koncernbolag, samarbetspartners, konsulter eller underentreprenörer som har behov därav för syftet enligt punkt 2 och som omfattas av sekretessåtaganden minst motsvarande detta Avtal,

– omedelbart meddela Utlämnaren om Mottagaren får kännedom om faktisk eller misstänkt obehörig åtkomst, användning eller utlämning.


6. AVTALSTID

Avtalet träder i kraft {req.effective_date} och gäller tills vidare.

Sekretessskyldigheten ska gälla under Avtalets löptid samt därefter under fem (5) år från Avtalets upphörande.

För företagshemligheter, affärskritisk information, källkod, algoritmer, modeller och annan information som till sin natur har ett långsiktigt skyddsvärde ska sekretessskyldigheten kvarstå så länge informationen inte blivit allmänt känd på annat sätt än genom brott mot detta Avtal.


7. IMMATERIELLA RÄTTIGHETER

Detta Avtal överför inga immateriella rättigheter.

Konfidentiell information förblir Utlämnarens egendom.

Inget i detta Avtal ska tolkas som en licens att använda Utlämnarens immateriella rättigheter utöver vad som krävs för det syfte som anges i punkt 2.

Om någon part använder Konfidentiell information för att träna, utvärdera, utveckla eller förbättra algoritmer, AI-modeller, maskininlärningsmodeller eller liknande system, innebär detta inte att äganderätten till den Konfidentiella informationen övergår till den andra parten.

Ingen part förvärvar genom detta Avtal rätt till den andra partens affärsmodeller, metodik, programvara, algoritmer eller know-how.


8. ÅTERLÄMNANDE OCH RADERING

Vid Avtalets upphörande, eller på begäran av Utlämnaren, ska Mottagaren utan dröjsmål återlämna eller förstöra all Konfidentiell information, inklusive kopior och bearbetningar, och på begäran skriftligen bekräfta att så skett.


9. SKADESTÅND OCH PÅFÖLJDER

Parterna erkänner att brott mot detta Avtal kan orsaka skada som inte fullt ut kan kompenseras genom ekonomisk ersättning.

Utlämnaren har därför rätt att begära interimistiskt förbud, vitesföreläggande eller annan rättslig åtgärd för att förhindra eller begränsa skada, utöver eventuell rätt till skadestånd.

Skadeståndsskyldighet gäller för direkt skada som orsakats genom brott mot detta Avtal.


10. TILLÄMPLIG LAG OCH TVISTELÖSNING

Detta Avtal regleras av svensk rätt.

Tvister som uppstår i anledning av detta Avtal ska i första hand lösas genom förhandlingar mellan Parterna.

Om enighet inte nås ska tvisten avgöras av allmän domstol med Göteborgs tingsrätt som första instans.


11. ÖVRIGT

Residual Knowledge

Ingenting i detta Avtal ska hindra någon part från att använda generell kunskap, erfarenhet, kompetens, metoder, idéer eller know-how som förvärvats under samarbetet, under förutsättning att ingen Konfidentiell information röjs eller används i strid med detta Avtal.

Ändring av detta Avtal kräver skriftlig överenskommelse undertecknad av båda Parter.

Om någon bestämmelse i detta Avtal befinns ogiltig, ska övriga bestämmelser förbli i kraft.


12. SÄRSKILDA VILLKOR

{req.special_terms}


13. UNDERSKRIFTER

Detta Avtal har upprättats i {req.place} den {req.effective_date} i två likalydande exemplar, ett till vardera Part.


PART A – {a.company_name}

_________________________________
Namn:  {sa.name}
Titel: {sa.title}
Datum: ___________________________


PART B – {b.company_name}

_________________________________
Namn:  {sb.name}
Titel: {sb.title}
Datum: ___________________________
"""


class NdaPDF(FPDF):
    NAVY   = (26, 39, 68)
    TEXT   = (26, 26, 26)
    MUTED  = (100, 100, 100)
    LIGHT  = (245, 247, 250)

    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.MUTED)
        self.cell(0, 5, f"Sida {self.page_no()}", align="C")

    @staticmethod
    def _ascii(text):
        # Replace characters outside Latin-1 (Helvetica font range)
        replacements = [
            (u'\u2014', '-'),  # em-dash
            (u'\u2013', '-'),  # en-dash
            (u'\u2018', "'"),  # left single quote
            (u'\u2019', "'"),  # right single quote
            (u'\u201c', '"'),  # left double quote
            (u'\u201d', '"'),  # right double quote
            (u'\u2022', '-'),  # bullet
        ]
        for char, repl in replacements:
            text = text.replace(char, repl)
        return text.encode('latin-1', errors='replace').decode('latin-1')
    def render(self, nda_text: str):
        self.add_page()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_left_margin(25)
        self.set_right_margin(25)
        self.set_text_color(*self.TEXT)

        lines = nda_text.split("\n")
        for line in lines:
            stripped = self._ascii(line.strip())

            # Main title (all caps, first line)
            if stripped in ("SEKRETESSAVTAL (NDA)", "Non-Disclosure Agreement"):
                self.set_font("Helvetica", "B", 16 if stripped == "SEKRETESSAVTAL (NDA)" else 11)
                self.set_text_color(*self.NAVY)
                self.cell(0, 8, stripped, ln=True, align="C")
                self.set_text_color(*self.TEXT)
                if stripped == "Non-Disclosure Agreement":
                    self.ln(4)
                continue

            # Section headings (digit + dot + space + caps)
            import re as _re
            if _re.match(r"^\d+\.\s+[A-ZÅÄÖ]", stripped):
                self.ln(3)
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*self.NAVY)
                self.cell(0, 6, stripped, ln=True)
                self.set_text_color(*self.TEXT)
                self.ln(1)
                continue

            # Signature blocks — monospace-like
            if stripped.startswith("PART A") or stripped.startswith("PART B"):
                self.ln(4)
                self.set_font("Helvetica", "B", 10)
                self.cell(0, 6, stripped, ln=True)
                continue

            if stripped.startswith("_"):
                self.set_font("Helvetica", "", 10)
                self.cell(0, 6, stripped, ln=True)
                continue

            if stripped.startswith("Namn:") or stripped.startswith("Titel:") or stripped.startswith("Datum:"):
                self.set_font("Helvetica", "", 9)
                self.cell(0, 5, stripped, ln=True)
                continue

            # Bullet lines
            if stripped.startswith("– ") or stripped.startswith("- "):
                self.set_font("Helvetica", "", 9)
                indent = 8
                self.set_x(self.get_x() + indent)
                self.multi_cell(
                    self.w - self.l_margin - self.r_margin - indent,
                    5, stripped, ln=True
                )
                continue

            # Empty line
            if not stripped:
                self.ln(2)
                continue

            # Normal body text
            self.set_font("Helvetica", "", 9)
            self.multi_cell(0, 5, stripped, ln=True)


@app.get("/nda", response_class=HTMLResponse)
async def serve_nda():
    path = Path(__file__).parent / "nda.html"
    content = path.read_text(encoding="utf-8")
    return HTMLResponse(content=content, headers={"Cache-Control": "no-store"})


@app.post("/api/nda")
async def create_nda(req: NdaCreateRequest):
    generated_text = _generate_nda_text(req)
    now = datetime.now(timezone.utc).isoformat()
    nda_id = str(uuid.uuid4())
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO nda_documents (
            id, scan_id, status,
            party_a_name, party_a_org_nr, party_a_address, party_a_contact,
            party_b_name, party_b_org_nr, party_b_address, party_b_contact,
            signer_a_name, signer_a_title, signer_b_name, signer_b_title,
            effective_date, place, purpose, special_terms,
            generated_text, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            nda_id, req.scan_id, req.status,
            req.party_a.company_name, req.party_a.organization_number,
            req.party_a.address, req.party_a.contact_person,
            req.party_b.company_name, req.party_b.organization_number,
            req.party_b.address, req.party_b.contact_person,
            req.signer_a.name, req.signer_a.title,
            req.signer_b.name, req.signer_b.title,
            req.effective_date, req.place, req.purpose, req.special_terms,
            generated_text, now, now,
        ),
    )
    con.commit()
    con.close()
    return {"id": nda_id, "generated_text": generated_text, "created_at": now}


@app.get("/api/nda")
async def list_ndas():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT id, scan_id, status, party_a_name, party_b_name,
                  effective_date, place, created_at, updated_at
           FROM nda_documents ORDER BY created_at DESC"""
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.get("/api/nda/{nda_id}")
async def get_nda(nda_id: str):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM nda_documents WHERE id=?", (nda_id,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="NDA hittades inte")
    return dict(row)


@app.put("/api/nda/{nda_id}/status")
async def update_nda_status(nda_id: str, req: NdaStatusRequest):
    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "UPDATE nda_documents SET status=?, updated_at=? WHERE id=?",
        (req.status, now, nda_id),
    )
    con.commit()
    con.close()
    return {"id": nda_id, "status": req.status}


@app.get("/api/nda/{nda_id}/pdf")
async def download_nda_pdf(nda_id: str):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM nda_documents WHERE id=?", (nda_id,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="NDA hittades inte")
    row = dict(row)

    pdf = NdaPDF(orientation="P", unit="mm", format="A4")
    pdf.set_title(f"NDA – {row.get('party_b_name', '')}")
    pdf.render(row.get("generated_text", ""))
    pdf_bytes = bytes(pdf.output())

    filename = f"NDA-{(row.get('party_b_name') or 'dokument').replace(' ', '-')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/scans/{scan_id}/ndas")
async def list_scan_ndas(scan_id: int):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT id, status, party_a_name, party_b_name, effective_date, created_at
           FROM nda_documents WHERE scan_id=? ORDER BY created_at DESC""",
        (scan_id,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── PDF (Opportunity Scan) ─────────────────────────────────────────────────

class PdfRequest(BaseModel):
    report_markdown: str
    filename: Optional[str] = "opportunity-scan.pdf"


class OpportunityScanPDF(FPDF):
    """Textbaserad PDF-generator för Opportunity Scan-rapporter."""

    ACCENT   = (26, 58, 92)   # #1a3a5c
    TEXT     = (26, 26, 26)
    MUTED    = (100, 100, 100)
    ROW_ALT  = (245, 247, 250)
    HDR_FG   = (255, 255, 255)

    def header(self):
        pass  # Ingen sidhuvud

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.MUTED)
        self.cell(0, 5, f"Sida {self.page_no()}", align="C")

    def render(self, report_markdown: str):
        self.add_page()
        self.set_auto_page_break(auto=True, margin=18)
        self.set_left_margin(20)
        self.set_right_margin(20)

        # Konvertera markdown till rader och rendera block för block
        lines = report_markdown.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]

            # H1
            if line.startswith("# "):
                self._h1(line[2:].strip())
                i += 1

            # H2
            elif line.startswith("## "):
                self._h2(line[3:].strip())
                i += 1

            # H3
            elif line.startswith("### "):
                self._h3(line[4:].strip())
                i += 1

            # Tabell
            elif line.startswith("|"):
                rows, i = self._collect_table(lines, i)
                self._table(rows)

            # Horisontell linje
            elif re.match(r'^-{3,}$', line.strip()):
                self._hr()
                i += 1

            # Blockquote (> text)
            elif line.startswith("> "):
                bq_lines, i = self._collect_blockquote(lines, i)
                self._blockquote(" ".join(bq_lines))

            # Bullet-lista
            elif line.startswith("- ") or line.startswith("* "):
                items, i = self._collect_list(lines, i)
                self._list(items)

            # Tom rad
            elif line.strip() == "":
                self.ln(2)
                i += 1

            # Vanlig text
            else:
                self._paragraph(line.strip())
                i += 1

    # ── Hjälpmetoder ─────────────────────────────────────────────

    @staticmethod
    def _safe(text: str) -> str:
        """Ersätt unicode-tecken som inte stöds av Latin-1/Helvetica."""
        replacements = {
            "\u2013": "-",   # en dash –
            "\u2014": "-",   # em dash —
            "\u2012": "-",   # figure dash
            "\u2022": "-",   # bullet •
            "\u2019": "'",   # right single quote '
            "\u2018": "'",   # left single quote '
            "\u201c": '"',   # left double quote "
            "\u201d": '"',   # right double quote "
            "\u2026": "...", # ellipsis …
            "\u00d7": "x",   # multiplication sign ×
            "\u00f7": "/",   # division sign ÷
            "\u2192": "->",  # arrow →
            "\u2190": "<-",  # arrow ←
            "\u00b0": " gr", # degree °
            "\u00b1": "+/-", # plus-minus ±
            "\u00b2": "2",   # superscript 2 ²
            "\u00b3": "3",   # superscript 3 ³
            "\u00b9": "1",   # superscript 1 ¹
            "\u2070": "0",   # superscript 0
            "\u20ac": "EUR", # euro €
            "\u00a0": " ",   # non-breaking space
        }
        for uni, asc in replacements.items():
            text = text.replace(uni, asc)
        # Sista utväg: ta bort tecken utanför Latin-1
        return text.encode("latin-1", errors="replace").decode("latin-1")

    @staticmethod
    def _strip_md(text: str) -> str:
        """Ta bort all markdown-formattering (används för tabeller/rubriker)."""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*',     r'\1', text)
        text = re.sub(r'`(.+?)`',       r'\1', text)
        return text

    # Subscript Unicode digits U+2080–U+2089
    _SUB_DIGITS = {chr(0x2080 + i): str(i) for i in range(10)}

    @staticmethod
    def _has_sub(text: str) -> bool:
        return any(('₀' <= c <= '₉') for c in text)

    def _write_with_subs(self, text: str, font_size: float, font_style: str = "") -> None:
        """Write text with visual subscript rendering for ₀–₉ chars."""
        # Split into (segment, is_subscript) runs
        runs = []
        buf = ""
        for ch in text:
            if ch in self._SUB_DIGITS:
                if buf:
                    runs.append((self._safe(buf), False))
                    buf = ""
                runs.append((self._SUB_DIGITS[ch], True))
            else:
                buf += ch
        if buf:
            runs.append((self._safe(buf), False))

        sub_size  = font_size * 0.65
        sub_drop  = font_size * 0.38   # mm to drop baseline for subscript

        y0 = self.get_y()
        for seg, is_sub in runs:
            if is_sub:
                self.set_font("Helvetica", font_style, sub_size)
                self.set_y(y0 + sub_drop)
                self.write(sub_size * 0.45, seg)
                self.set_y(y0)
                self.set_font("Helvetica", font_style, font_size)
            else:
                self.set_font("Helvetica", font_style, font_size)
                self.write(font_size * 0.45, seg)
        self.ln(font_size * 0.45 + 1.5)

    @staticmethod
    def _strip_md_keep_bold(text: str) -> str:
        """Ta bort markdown utom **fetstil**-markörer."""
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Ta bort *kursiv* men inte **fetstil**
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
        return text

    def _write_inline(self, text: str, h: float, size: int = 10,
                      base_style: str = "") -> None:
        """Skriv text med inline **fetstil** och nedsänkta siffror (₀–₉)."""
        sub_size = size * 0.65
        sub_drop = size * 0.38   # mm att sänka baslinjen

        clean  = self._strip_md_keep_bold(text)
        tokens = re.split("([" + "".join(self._SUB_DIGITS.keys()) + "])", clean)

        for token in tokens:
            if not token:
                continue
            if token in self._SUB_DIGITS:
                # cell() i stället för write() – ändrar aldrig Y, eliminerar positionskorruption
                y_cur = self.get_y()
                x_cur = self.get_x()
                digit = self._SUB_DIGITS[token]
                self.set_font("Helvetica", base_style, sub_size)
                w = self.get_string_width(digit)
                self.set_xy(x_cur, y_cur + sub_drop)
                self.cell(w, sub_size * 0.45, digit)
                self.set_xy(x_cur + w, y_cur)
                self.set_font("Helvetica", base_style, size)
            else:
                bold_parts = re.split(r'\*\*(.+?)\*\*', token)
                for i, part in enumerate(bold_parts):
                    if not part:
                        continue
                    self.set_font("Helvetica", "B" if i % 2 == 1 else base_style, size)
                    self.write(h, self._safe(part))

        self.set_font("Helvetica", base_style, size)
        self.ln(h)

    # ── Block-renderare ──────────────────────────────────────────

    def _h1(self, text):
        self.set_font("Helvetica", "B", 17)
        self.set_text_color(*self.ACCENT)
        self.multi_cell(0, 9, self._safe(text), align="L")
        self.set_draw_color(*self.ACCENT)
        self.set_line_width(0.6)
        y = self.get_y() + 1
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(5)

    def _h2(self, text):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.ACCENT)
        self.multi_cell(0, 7, self._safe(text), align="L")
        self.set_draw_color(192, 204, 214)
        self.set_line_width(0.3)
        y = self.get_y() + 1
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)

    def _h3(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.TEXT)
        self.multi_cell(0, 6, self._safe(text), align="L")
        self.ln(1)

    def _paragraph(self, text):
        if not text:
            return
        self.set_text_color(*self.TEXT)
        self._write_inline(text, 5.5, size=10)
        self.ln(1)

    def _list(self, items):
        self.set_text_color(*self.TEXT)
        for item in items:
            self.set_x(self.l_margin + 4)
            self.set_font("Helvetica", "", 10)
            self.cell(5, 5.5, "-")
            self._write_inline(item, 5.5, size=10)
        self.ln(1)

    def _hr(self):
        self.ln(3)
        self.set_draw_color(192, 204, 214)
        self.set_line_width(0.3)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(4)

    def _table(self, rows):
        if not rows:
            return
        self.ln(2)
        usable_w = self.w - self.l_margin - self.r_margin
        n_cols    = max(len(r) for r in rows)
        col_w     = usable_w / n_cols if n_cols else usable_w
        row_h     = 6.5

        for ri, row in enumerate(rows):
            # Hoppa över separator-rader (|---|---|)
            if all(re.match(r'^[-: ]+$', c.strip()) for c in row if c.strip()):
                continue

            is_header = (ri == 0)
            if is_header:
                self.set_fill_color(*self.ACCENT)
                self.set_text_color(*self.HDR_FG)
                self.set_font("Helvetica", "B", 9)
            else:
                fill = self.ROW_ALT if ri % 2 == 0 else (255, 255, 255)
                self.set_fill_color(*fill)
                self.set_text_color(*self.TEXT)
                self.set_font("Helvetica", "", 9)

            for ci, cell in enumerate(row):
                cell_text = self._safe(self._strip_md(cell.strip()))
                self.cell(col_w, row_h, cell_text, border=0, fill=True)
            # Fyll ut tomma kolumner
            for _ in range(n_cols - len(row)):
                self.cell(col_w, row_h, "", border=0, fill=True)
            self.ln(row_h)

        self.ln(3)

    def _collect_table(self, lines, i):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c for c in lines[i].split("|")]
            # Ta bort första och sista tomma elementet
            if cells and cells[0].strip() == "":
                cells = cells[1:]
            if cells and cells[-1].strip() == "":
                cells = cells[:-1]
            rows.append(cells)
            i += 1
        return rows, i

    def _collect_list(self, lines, i):
        items = []
        while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("* ")):
            items.append(lines[i][2:].strip())
            i += 1
        return items, i

    def _collect_blockquote(self, lines, i):
        bq = []
        while i < len(lines) and lines[i].startswith("> "):
            bq.append(lines[i][2:].strip())
            i += 1
        return bq, i

    def _blockquote(self, text: str):
        """Render a methodology info box with left accent and light background."""
        self.ln(2)
        clean = self._safe(self._strip_md(text))
        if not clean:
            return

        x0       = self.l_margin
        y0       = self.get_y()
        usable_w = self.w - self.l_margin - self.r_margin - 8  # 5 accent + 3 gap

        # Pass 1: render text invisibly to measure actual height
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(255, 255, 255)  # white = invisible on white page
        self.set_xy(x0 + 5, y0 + 2)
        self.multi_cell(usable_w, 4.5, clean, align="L")
        box_h = max(self.get_y() - y0 + 2, 8)

        # Draw background and accent bar now that we know the real height
        self.set_fill_color(240, 244, 248)
        self.rect(x0, y0, self.w - self.l_margin - self.r_margin, box_h, "F")
        self.set_fill_color(*self.ACCENT)
        self.rect(x0, y0, 3, box_h, "F")

        # Pass 2: render text on top of background
        self.set_xy(x0 + 5, y0 + 2)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(60, 60, 60)
        self.multi_cell(usable_w, 4.5, clean, align="L")

        self.set_y(y0 + box_h)
        self.set_text_color(*self.TEXT)
        self.ln(3)


@app.post("/pdf")
async def generate_pdf(req: PdfRequest):
    try:
        pdf = OpportunityScanPDF()
        pdf.render(req.report_markdown)
        pdf_bytes = bytes(pdf.output())
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF-generering misslyckades: {e}")

    safe_name = re.sub(r'[^\w\-.]', '-', req.filename or "opportunity-scan.pdf")
    if not safe_name.endswith(".pdf"):
        safe_name += ".pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


EXTRACTION_SYSTEM_PROMPT = """Du är ett extraktionsverktyg. Din uppgift är att läsa en Opportunity Scan-rapport och extrahera strukturerad data till ett exakt JSON-format.

Regler:
- Returnera ENBART ett giltigt JSON-objekt. Ingen annan text.
- Använd ENDAST värdena från enum-listorna nedan.
- Sätt null för fält du inte kan avgöra med rimlig säkerhet.
- primary_patterns: max 3, de dominerande läckagedrivarna.
- secondary_patterns: 1-4 stycken, stödjande mönster.
- human_capital_patterns: endast om personalintensitet är hög.

Tillgängliga enums:
patterns: inventory_imbalance, flow_inefficiency, capacity_misalignment, pricing_leakage, forecast_error, fragmentation, knowledge_dependency, process_variability, utilization_gap, timing_mismatch, project_margin_leakage, risk_underestimation, execution_variability, customer_concentration, portfolio_misalignment, credit_risk_leakage, interest_margin_leakage, regulatory_constraint, skill_mismatch, talent_utilization_gap, bench_leakage, staffing_friction, retention_risk, hybrid_complexity

structure_type: network, pipeline, project, portfolio, hybrid, unknown
decision_mode: continuous, episodic, hybrid, unknown
decision_level: transactional, project, network, portfolio, enterprise, unknown
company_scale: micro (<50 MSEK), small (50-300 MSEK), mid (300-2000 MSEK), large (>2000 MSEK)
system_ambition: lightweight, standard, advanced
human_capital_intensity: low, medium, high
maturity: low, medium, high, unknown

Returnera exakt detta JSON-skelett ifyllt:
{
  "scan_meta": {"company_name": "", "country": "Sverige", "confidence_level": ""},
  "source_data": {"years_analyzed": [], "latest_year": null, "web_signals_used": false, "web_signals_summary": []},
  "financial_profile": {"latest_year_revenue_msek": null, "latest_year_ebit_msek": null, "latest_year_ebit_margin_pct": null, "latest_year_cost_base_msek": null, "financial_trend_summary": ""},
  "elir": {
    "E": {"percent": null, "msek": null, "rationale": ""},
    "L": {"percent": null, "msek": null, "rationale": "", "driver_scores": {"variation": null, "margin": null, "complexity": null, "volatility": null, "capital": null, "operational_intensity": null}},
    "I": {"percent": null, "msek": null, "rationale": "", "driver_scores": {"frequency": null, "standardization": null, "data": null, "regulation": null, "physical_lock_in": null}},
    "R": {"percent": null, "msek": null, "rationale": "", "driver_scores": {"ownership": null, "organizational_complexity": null, "pressure": null, "change_capacity": null}},
    "total_potential_msek": null
  },
  "business_interpretation": {
    "business_logic": "", "industry_label": "",
    "structure_type": "", "decision_mode": "", "decision_level": "",
    "company_scale": "", "system_ambition": "", "human_capital_intensity": "",
    "primary_patterns": [], "secondary_patterns": [], "human_capital_patterns": [],
    "leakage_categories": [], "decision_types": []
  },
  "dual_mode": {"enabled": false},
  "xzero_recommendation": {
    "org_maturity": {"process": "", "data": "", "decision": "", "change": ""}
  },
  "traceability": {"key_evidence": [], "assumptions": []}
}"""

ACTION_PLAN_SYSTEM_PROMPT = """Du är en åtgärdsstrategist för xZero. Du tar emot en strukturerad kontext med:
- company_name och affärsprofil
- strategy_type (bestämd av regelmotor): stabilize_first | optimize_first | split_strategy | lightweight_first
- Prioriterade use cases från regelmotorn
- org_maturity och decision_mode

Din uppgift är att skriva en konkret, genomförbar åtgärdsplan på svenska i markdown-format.

Regler:
- Följ den bestämda strategy_type — motivera varför sekvensen är rätt för detta bolag.
- Börja med det minsta rimliga steget som låser upp värde (fas 1 ska vara smal och explicit).
- Anpassa ambitionsnivån till company_scale och system_ambition.
- Om dual_mode är aktivt, dela strategin i två logiker (kontinuerlig och episodisk).
- Använd de prioriterade use cases som grund — uppfinn inga nya.
- Inga emojis. Analytiskt och konkret tonläge.

Struktur:
# Åtgärdsplan – [Bolagsnamn]

## Strategival
Förklara strategy_type och varför sekvensen passar bolaget.

## Fas 1 – [Mål]
Fokus, leverabler och framgångskriterier.

## Fas 2 – [Mål]
Fokus, leverabler och framgångskriterier.

## Fas 3 – [Mål]
Fokus, leverabler och framgångskriterier.

## Prioriterade use cases
Lista top_use_cases med kort motivering per use case.

## Genomföranderisker
Lista de viktigaste riskerna.

## Varför denna sekvens
Sammanfattande motivering."""


@app.get("/action-plan/{scan_id}")
async def get_action_plan(scan_id: int):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM action_plans WHERE scan_id=? ORDER BY created_at DESC LIMIT 1",
        (scan_id,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Ingen åtgärdsplan finns för denna scan")
    return dict(row)


@app.post("/action-plan/{scan_id}")
async def create_action_plan(scan_id: int):
    # Hämta scan från DB
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    scan = con.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    con.close()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan hittades inte")

    scan = dict(scan)
    report_markdown = scan.get("report_markdown", "")
    if not report_markdown:
        raise HTTPException(status_code=400, detail="Scan saknar rapporttext")

    client = get_client()

    def generate():
        # ── Steg 1: Extrahera canonical JSON (icke-streaming) ──────────────
        yield "__STATUS__Extraherar strukturerad data från rapporten...\n"
        try:
            extraction_resp = client.messages.create(
                model="us.anthropic.claude-sonnet-4-6",
                max_tokens=4000,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Extrahera canonical JSON från denna Opportunity Scan-rapport.\n\n"
                        f"Känd data att inkludera direkt:\n"
                        f"- revenue_msek: {scan.get('revenue_msek')}\n"
                        f"- ebit_msek: {scan.get('ebit_msek')}\n"
                        f"- ebit_margin_pct: {scan.get('ebit_margin_pct')}\n"
                        f"- e_pct: {scan.get('e_pct')}, e_msek: {scan.get('e_msek')}\n"
                        f"- l_pct: {scan.get('l_pct')}, l_msek: {scan.get('l_msek')}\n"
                        f"- i_pct: {scan.get('i_pct')}, i_msek: {scan.get('i_msek')}\n"
                        f"- r_pct: {scan.get('r_pct')}, r_msek: {scan.get('r_msek')}\n"
                        f"- total_potential_msek: {scan.get('total_potential_msek')}\n"
                        f"- years_analyzed: {scan.get('years_analyzed')}\n\n"
                        f"Rapport:\n\n{report_markdown}"
                    ),
                }],
            )
            raw_json = extraction_resp.content[0].text.strip()
            # Rensa eventuella markdown-kodblock
            raw_json = re.sub(r'^```(?:json)?\s*', '', raw_json, flags=re.MULTILINE)
            raw_json = re.sub(r'\s*```$', '', raw_json, flags=re.MULTILINE)
            canonical = json.loads(raw_json)
            print(f"[action-plan] canonical JSON extraherat ({len(raw_json)} tecken)")
        except Exception as e:
            yield f"\n\n**Fel vid datautvinning:** {e}"
            return

        # ── Steg 2: Regelmotor (deterministisk) ───────────────────────────
        yield "__STATUS__Kör regelmotor...\n"
        try:
            use_case_output = run_use_case_engine(canonical)
            context = build_action_plan_context(canonical, use_case_output)
            print(f"[action-plan] strategy_type={context['strategy_type']}, "
                  f"top_use_cases={len(context['top_use_cases'])}")
        except Exception as e:
            yield f"\n\n**Fel i regelmotorn:** {e}"
            return

        # ── Steg 3: Åtgärdsplan (streaming) ──────────────────────────────
        yield "__STATUS__Skriver åtgärdsplan...\n"
        plan_parts = []
        try:
            context_text = json.dumps(context, ensure_ascii=False, indent=2)
            with client.messages.stream(
                model="us.anthropic.claude-sonnet-4-6",
                max_tokens=4000,
                system=ACTION_PLAN_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Skriv åtgärdsplan baserat på denna kontext:\n\n{context_text}"
                    ),
                }],
            ) as stream:
                for chunk in stream.text_stream:
                    plan_parts.append(chunk)
                    yield chunk
        except Exception as e:
            yield f"\n\n**Fel vid generering av åtgärdsplan:** {e}"
            return

        # ── Steg 4: Spara i DB ────────────────────────────────────────────
        plan_markdown = "".join(plan_parts)
        try:
            con = sqlite3.connect(DB_PATH)
            con.execute(
                """INSERT INTO action_plans
                   (scan_id, created_at, canonical_json, use_case_json, plan_markdown)
                   VALUES (?,?,?,?,?)""",
                (
                    scan_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(canonical, ensure_ascii=False),
                    json.dumps(use_case_output, ensure_ascii=False),
                    plan_markdown,
                ),
            )
            con.commit()
            con.close()
            print(f"[action-plan] sparad för scan_id={scan_id}")
        except Exception as e:
            print(f"[action-plan] DB-fel: {e}")

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    if not 1 <= len(files) <= 5:
        raise HTTPException(status_code=400, detail="Ladda upp 1–5 årsredovisningar")

    content = []
    all_texts = []
    first_page_b64 = None  # Första sidan av första skannande PDF — används för namnextraktion
    MAX_TOTAL_PDF_PAGES = 100  # Bedrocks gräns
    pages_per_file = MAX_TOTAL_PDF_PAGES // len(files)  # Fördela jämnt
    print(f"Sidbudget: {pages_per_file} sidor/fil ({len(files)} filer)")

    for i, file in enumerate(files):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Fil {i+1} är inte en PDF")

        pdf_bytes = await file.read()
        if len(pdf_bytes) > 500 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"Fil {i+1} är för stor (max 500 MB)")

        size_mb = len(pdf_bytes) / 1024 / 1024
        print(f"[fil {i+1}] {file.filename} — {size_mb:.1f} MB")

        text = extract_text(pdf_bytes)
        print(f"[fil {i+1}] textextraktion: {len(text)} tecken")
        all_texts.append(text)

        if len(text) > 100:
            # Textbaserad PDF — skicka som text (ingen sidgräns)
            label = f"ÅRSREDOVISNING {i+1} ({file.filename})"
            content.append({
                "type": "text",
                "text": f"<document index=\"{i+1}\" title=\"{label}\">\n{text}\n</document>",
            })
            print(f"[fil {i+1}] skickas som TEXT ({len(text)} tecken)")
        else:
            # Skannad PDF — begränsa inom globalt sidbudget, komprimera, skicka som base64
            src = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(src)
            allowed = min(total_pages, pages_per_file)

            if total_pages > allowed:
                print(f"[fil {i+1}] {total_pages} sidor — trunkerar till {allowed}")
                trimmed = fitz.open()
                trimmed.insert_pdf(src, from_page=0, to_page=allowed - 1)
                pdf_bytes = trimmed.tobytes()
                trimmed.close()
            src.close()
            print(f"[fil {i+1}] skannad PDF — komprimerar...")
            sendable = compress_pdf(pdf_bytes)
            print(f"[fil {i+1}] komprimerad: {len(sendable)/1024/1024:.1f} MB")
            pdf_data = base64.standard_b64encode(sendable).decode("utf-8")
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_data,
                },
            })
            print(f"[fil {i+1}] skickas som BASE64 ({allowed}/{total_pages} sidor)")

            # Spara första sidan av första skannande PDF för namnextraktion
            if first_page_b64 is None:
                try:
                    one_page = fitz.open(stream=pdf_bytes, filetype="pdf")
                    p = fitz.open()
                    p.insert_pdf(one_page, from_page=0, to_page=0)
                    one_page.close()
                    first_page_b64 = base64.standard_b64encode(p.tobytes()).decode("utf-8")
                    p.close()
                except Exception as e:
                    print(f"  [namnextraktion] kunde inte extrahera första sidan: {e}")

    # ── Tavily-sökning ──────────────────────────────────────────────
    client = get_client()
    combined_text = "\n".join(t for t in all_texts if t)
    company_name = extract_company_name(combined_text, first_page_b64, client)
    web_data = ""
    if company_name:
        print(f"Identifierat bolagsnamn: {company_name}")
        web_data = search_web(company_name)
        if web_data:
            print(f"Webdata hämtad: {len(web_data)} tecken")
            content.insert(0, {"type": "text", "text": web_data})
        else:
            print("Ingen webdata hämtad")
    else:
        print("Kunde inte identifiera bolagsnamn — hoppar över webdsökning")

    n = len(files)
    content.append({
        "type": "text",
        "text": (
            f"Du har fått {n} årsredovisning{'ar' if n > 1 else ''} för samma bolag. "
            "Identifiera vilket år respektive dokument avser (t₀ = senaste). "
            "Genomför analysen enligt regelverket och generera en komplett Opportunity Scan."
            + (f" Webdata om {company_name} är inkluderad ovan." if web_data else "")
        ),
    })

    def generate():
        try:
            with client.messages.stream(
                model="us.anthropic.claude-sonnet-4-6",
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                yield "\n\n**Anthropics API är tillfälligt överbelastad.** Vänta någon minut och försök igen."
            elif e.status_code == 413:
                yield "\n\n**Filen är fortfarande för stor efter komprimering.** Prova att dela upp årsredovisningen i mindre delar (t.ex. bara resultat- och balansräkning) och ladda upp den delen."
            else:
                yield f"\n\n**Fel vid API-anrop ({e.status_code}):** {e.message}"
        except anthropic.APIConnectionError:
            yield "\n\n**Kunde inte nå Anthropics API.** Kontrollera din internetanslutning och försök igen."
        except anthropic.APIError as e:
            yield f"\n\n**Fel vid API-anrop:** {e}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# ── FIREFLIES INTEGRATION ─────────────────────────────────────────────────────

@app.post("/api/webhooks/fireflies")
async def fireflies_webhook(payload: dict):
    """Receive Fireflies transcription-complete webhook and store transcript."""
    if payload.get("eventType") != "Transcription_Complete":
        return {"status": "ignored"}

    meeting_id = payload.get("meetingId")
    if not meeting_id:
        raise HTTPException(status_code=400, detail="meetingId saknas")

    if not FIREFLIES_API_KEY:
        raise HTTPException(status_code=500, detail="FIREFLIES_API_KEY ej konfigurerad")

    con = sqlite3.connect(DB_PATH)
    existing = con.execute(
        "SELECT id FROM fireflies_transcripts WHERE meeting_id=?", (meeting_id,)
    ).fetchone()
    con.close()
    if existing:
        return {"status": "already_stored", "id": existing[0]}

    data = await fetch_fireflies_transcript(meeting_id)

    transcript_id = str(uuid.uuid4())
    received_at   = datetime.now(timezone.utc).isoformat()

    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO fireflies_transcripts
           (id, meeting_id, title, meeting_date, duration_secs, transcript_text, summary_text, received_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (transcript_id, meeting_id, data["title"], str(data["meeting_date"]),
         data["duration_secs"], data["transcript_text"], data["summary_text"], received_at),
    )
    con.commit()
    con.close()

    return {"status": "stored", "id": transcript_id}


@app.get("/api/fireflies/transcripts")
async def list_fireflies_transcripts():
    """List stored Fireflies transcripts."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT id, title, meeting_date, duration_secs, received_at, workshop_id
           FROM fireflies_transcripts
           ORDER BY received_at DESC
           LIMIT 50"""
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.post("/api/workshops/{workshop_id}/use-transcript/{transcript_id}")
async def link_fireflies_transcript(workshop_id: str, transcript_id: str):
    """Link a Fireflies transcript to a workshop and return its text."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    t = con.execute(
        "SELECT transcript_text, summary_text FROM fireflies_transcripts WHERE id=?",
        (transcript_id,)
    ).fetchone()
    if not t:
        con.close()
        raise HTTPException(status_code=404, detail="Transkript hittades inte")

    con.execute(
        "UPDATE fireflies_transcripts SET workshop_id=? WHERE id=?",
        (workshop_id, transcript_id)
    )
    con.commit()
    con.close()

    parts = [t["transcript_text"]]
    if t["summary_text"]:
        parts.append(f"\n---\n{t['summary_text']}")
    return {"transcript": "\n".join(parts)}


# ══════════════════════════════════════════════════════════════════════════════
# Opportunity Graph API
# ══════════════════════════════════════════════════════════════════════════════

from opportunity_graph import (
    NODE_CLASSES, NodeType, OpportunityGraph, RelationType,
    Edge as GraphEdge,
)
from gap_hunter import summarize as gap_hunter_summarize
from playbook_generator import generate_playbook


def _graph_load(graph_id: str) -> OpportunityGraph:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT graph_json FROM opportunity_graphs WHERE id=?", (graph_id,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Graf hittades inte")
    return OpportunityGraph.from_json(row["graph_json"])


def _graph_save(graph: OpportunityGraph) -> None:
    now = datetime.now(timezone.utc).isoformat()
    opp = graph.get_opportunity()
    company = opp.company_name if opp else ""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO opportunity_graphs (id, company, graph_json, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             company=excluded.company,
             graph_json=excluded.graph_json,
             updated_at=excluded.updated_at""",
        (graph.opportunity_id, company, graph.to_json(), now, now),
    )
    con.commit()
    con.close()


# ── List / create ──────────────────────────────────────────────────────────────

@app.post("/api/graphs/backfill")
async def backfill_graphs():
    """
    Create Opportunity Graphs for all existing scans that don't have one yet.
    Also enriches graphs that have a completed workshop analysis but no Evidence nodes.
    """
    from graph_bootstrap import bootstrap_from_scan, graph_id_for_scan, update_from_analysis

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    scans = con.execute(
        "SELECT id, company_name, industry, revenue_msek, ebit_msek, ebit_margin_pct, "
        "years_analyzed, variation_score, e_msek, e_pct, l_msek, l_pct, i_msek, i_pct, "
        "r_msek, r_pct, total_potential_msek, confidence, workshop_hypotheses, report_markdown "
        "FROM scans ORDER BY id"
    ).fetchall()

    workshops = con.execute(
        "SELECT id, scan_id, session_json, analysis_markdown FROM workshop_sessions "
        "WHERE analysis_markdown IS NOT NULL"
    ).fetchall()
    con.close()

    ws_by_scan: dict[int, dict] = {}
    for ws in workshops:
        sid = ws["scan_id"]
        if sid not in ws_by_scan:
            ws_by_scan[sid] = {
                "id": ws["id"],
                "session": json.loads(ws["session_json"] or "{}"),
                "analysis": ws["analysis_markdown"] or "",
            }

    created, enriched = 0, 0
    errors: list[dict] = []

    for scan in scans:
        scan_id = scan["id"]
        gid = graph_id_for_scan(scan_id)

        # Bootstrap if graph doesn't exist or has no hypotheses yet
        try:
            graph = _graph_load(gid)
            needs_hypotheses = graph.get_opportunity() and not graph.find_nodes(NodeType.HYPOTHESIS)
        except HTTPException:
            graph = None
            needs_hypotheses = True

        if graph is None or needs_hypotheses:
            try:
                hyp_json = scan["workshop_hypotheses"]
                # Fall back to extracting hypotheses from report prose
                if not hyp_json and scan["report_markdown"]:
                    extracted = _extract_hypotheses_from_report(scan["report_markdown"])
                    hyp_json = json.dumps(extracted) if extracted else None
                graph = bootstrap_from_scan(scan_id, dict(scan), hyp_json)
                _graph_save(graph)
                created += 1
                graph = _graph_load(gid)
            except Exception as e:
                errors.append({"scan_id": scan_id, "phase": "bootstrap", "error": str(e)})
                continue

        # Enrich with analysis if available and graph has no Evidence nodes
        ws = ws_by_scan.get(scan_id)
        if ws and not graph.find_nodes(NodeType.EVIDENCE):
            try:
                client = get_client()
                graph = update_from_analysis(graph, ws["analysis"], ws["session"], client)
                _graph_save(graph)
                enriched += 1
            except Exception as e:
                errors.append({"scan_id": scan_id, "phase": "enrich", "error": str(e)})

    return {
        "total_scans": len(scans),
        "graphs_created": created,
        "graphs_enriched": enriched,
        "errors": errors,
    }


@app.get("/api/scans/{scan_id}/graph")
async def get_scan_graph(scan_id: int):
    """Shortcut: return the Opportunity Graph bootstrapped from this scan."""
    from graph_bootstrap import graph_id_for_scan
    return _graph_load(graph_id_for_scan(scan_id)).to_dict()


@app.get("/api/scans/{scan_id}/graph/summary")
async def get_scan_graph_summary(scan_id: int):
    from graph_bootstrap import graph_id_for_scan
    return _graph_load(graph_id_for_scan(scan_id)).summary()


@app.get("/api/scans/{scan_id}/graph/gaps")
async def get_scan_graph_gaps(scan_id: int):
    from graph_bootstrap import graph_id_for_scan
    return gap_hunter_summarize(_graph_load(graph_id_for_scan(scan_id)))


@app.post("/api/scans/{scan_id}/graph/playbook")
async def create_scan_playbook(scan_id: int):
    from graph_bootstrap import graph_id_for_scan
    return generate_playbook(_graph_load(graph_id_for_scan(scan_id)))


@app.get("/api/graphs")
async def list_graphs():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, company, created_at, updated_at FROM opportunity_graphs ORDER BY created_at DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


class CreateGraphRequest(BaseModel):
    opportunity_id: Optional[str] = None
    company_name: str
    industry: str = ""


@app.post("/api/graphs", status_code=201)
async def create_graph(req: CreateGraphRequest):
    gid = req.opportunity_id or str(uuid.uuid4())
    graph = OpportunityGraph(gid)
    from opportunity_graph import OpportunityNode
    graph.add_node(OpportunityNode(
        company_name=req.company_name,
        industry=req.industry,
    ))
    _graph_save(graph)
    return {"id": gid, "company": req.company_name}


@app.post("/api/graphs/seed/sweden-pelagic", status_code=201)
async def seed_sweden_pelagic():
    from sweden_pelagic_seed import build
    graph = build()
    _graph_save(graph)
    return {"id": graph.opportunity_id, "company": "Sweden Pelagic AB",
            "nodes": len(graph._nodes), "edges": len(graph._edges)}


# ── Read ───────────────────────────────────────────────────────────────────────

@app.get("/api/graphs/{graph_id}")
async def get_graph(graph_id: str):
    graph = _graph_load(graph_id)
    return graph.to_dict()


@app.get("/api/graphs/{graph_id}/summary")
async def get_graph_summary(graph_id: str):
    return _graph_load(graph_id).summary()


@app.get("/api/graphs/{graph_id}/validate")
async def validate_graph(graph_id: str):
    graph = _graph_load(graph_id)
    violations = graph.validate()
    return {
        "graph_id": graph_id,
        "violations": [v.model_dump() for v in violations],
        "valid": len(violations) == 0,
    }


# ── Gap Hunter ────────────────────────────────────────────────────────────────

@app.get("/api/graphs/{graph_id}/gaps")
async def get_gaps(graph_id: str):
    graph = _graph_load(graph_id)
    return gap_hunter_summarize(graph)


# ── Playbook ──────────────────────────────────────────────────────────────────

@app.post("/api/graphs/{graph_id}/playbook")
async def create_playbook(graph_id: str):
    graph = _graph_load(graph_id)
    return generate_playbook(graph)


# ── Node mutations ────────────────────────────────────────────────────────────

class AddNodeRequest(BaseModel):
    type: str
    data: dict


@app.post("/api/graphs/{graph_id}/nodes", status_code=201)
async def add_node(graph_id: str, req: AddNodeRequest):
    graph = _graph_load(graph_id)
    ntype = NodeType(req.type)
    node_data = {"type": ntype, **req.data}
    node = NODE_CLASSES[ntype](**node_data)
    graph.add_node(node)
    _graph_save(graph)
    return node.model_dump()


class UpdateNodeRequest(BaseModel):
    data: dict


@app.patch("/api/graphs/{graph_id}/nodes/{node_id}")
async def update_node(graph_id: str, node_id: str, req: UpdateNodeRequest):
    graph = _graph_load(graph_id)
    updated = graph.update_node(node_id, **req.data)
    if not updated:
        raise HTTPException(status_code=404, detail="Nod hittades inte")
    _graph_save(graph)
    return updated.model_dump()


@app.delete("/api/graphs/{graph_id}/nodes/{node_id}", status_code=204)
async def delete_node(graph_id: str, node_id: str):
    graph = _graph_load(graph_id)
    if not graph.remove_node(node_id):
        raise HTTPException(status_code=404, detail="Nod hittades inte")
    _graph_save(graph)


# ── Edge mutations ────────────────────────────────────────────────────────────

class AddEdgeRequest(BaseModel):
    from_id: str
    relation: str
    to_id: str
    metadata: dict = {}


@app.post("/api/graphs/{graph_id}/edges", status_code=201)
async def add_edge(graph_id: str, req: AddEdgeRequest):
    graph = _graph_load(graph_id)
    if not graph.get_node(req.from_id):
        raise HTTPException(status_code=404, detail=f"from_id {req.from_id} hittades inte")
    if not graph.get_node(req.to_id):
        raise HTTPException(status_code=404, detail=f"to_id {req.to_id} hittades inte")
    edge = graph.add_edge(req.from_id, RelationType(req.relation), req.to_id,
                          **req.metadata)
    _graph_save(graph)
    return edge.model_dump()


# ── Apply evidence ─────────────────────────────────────────────────────────────

@app.post("/api/graphs/{graph_id}/nodes/{node_id}/apply-evidence")
async def apply_evidence(graph_id: str, node_id: str):
    graph = _graph_load(graph_id)
    updated_ids = graph.apply_evidence(node_id)
    if not updated_ids and not graph.get_node(node_id):
        raise HTTPException(status_code=404, detail="Evidence-nod hittades inte")
    _graph_save(graph)
    return {"updated_node_ids": updated_ids}


# ── Live Listener WebSocket ───────────────────────────────────────────────────

def _build_listen_context(scan_id: int) -> str:
    """Build system prompt context for live hint generation."""
    try:
        from graph_bootstrap import graph_id_for_scan
        from opportunity_graph import NodeType
        graph = _graph_load(graph_id_for_scan(scan_id))
        opp = graph.get_opportunity()
        company = opp.company_name if opp else "kunden"
        hyps = graph.find_nodes(NodeType.HYPOTHESIS)
        hyp_lines = "\n".join(
            f"- {h.title} (status: {h.status}, confidence: {h.confidence:.0f}%)"
            for h in sorted(hyps, key=lambda x: x.priority, reverse=True)[:5]
        )
        return (
            f"Du är xZero:s AI-assistent under en discovery-workshop med {company}. "
            f"Workshopen validerar dessa hypoteser:\n{hyp_lines}\n\n"
            "När kunden säger något relevant — föreslå EN skarp följdfråga "
            "(max 20 ord, på svenska) som fördjupar förståelsen eller testar ett antagande. "
            "Bara frågan, inga inledningar eller förklaringar."
        )
    except Exception:
        return (
            "Du är xZero:s AI-assistent under en discovery-workshop. "
            "Föreslå EN skarp följdfråga (max 20 ord, svenska) baserat på vad kunden sa. "
            "Bara frågan."
        )


async def _generate_live_hint(fragment: str, context: str) -> str | None:
    """Call Bedrock in a thread to avoid blocking the event loop."""
    def _call():
        try:
            c = anthropic.AnthropicBedrock(
                aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_region="us-east-1",
            )
            msg = c.messages.create(
                model="us.anthropic.claude-sonnet-4-6",
                max_tokens=80,
                system=context,
                messages=[{"role": "user", "content": f'Kunden sa nyss: "{fragment}"'}],
            )
            return msg.content[0].text.strip().strip('"')
        except Exception:
            return None

    return await asyncio.to_thread(_call)


@app.websocket("/ws/scans/{scan_id}/listen")
async def ws_listen(websocket: WebSocket, scan_id: int):
    await websocket.accept()

    context = _build_listen_context(scan_id)

    try:
        from amazon_transcribe.client import TranscribeStreamingClient
        from amazon_transcribe.model import TranscriptEvent
    except ImportError:
        await websocket.send_json({"type": "error", "message": "amazon-transcribe inte installerat"})
        await websocket.close()
        return

    try:
        tc = TranscribeStreamingClient(region="eu-west-1")
        stream = await tc.start_stream_transcription(
            language_code="sv-SE",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

    buf: list[str] = []
    word_count = 0

    async def _feed():
        try:
            while True:
                data = await websocket.receive_bytes()
                await stream.input_stream.send_audio_event(audio_chunk=data)
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            try:
                await stream.input_stream.end_stream()
            except Exception:
                pass

    async def _read():
        nonlocal word_count
        try:
            async for event in stream.output_stream:
                if not isinstance(event, TranscriptEvent):
                    continue
                for result in event.transcript.results:
                    if result.is_partial:
                        continue
                    text = result.alternatives[0].transcript.strip()
                    if not text:
                        continue
                    buf.append(text)
                    word_count += len(text.split())
                    try:
                        await websocket.send_json({"type": "transcript", "text": text})
                    except Exception:
                        return
                    if word_count >= 70:
                        word_count = 0
                        fragment = " ".join(buf[-6:])
                        hint = await _generate_live_hint(fragment, context)
                        if hint:
                            try:
                                await websocket.send_json({"type": "hint", "question": hint})
                            except Exception:
                                return
        except Exception:
            pass

    await asyncio.gather(_feed(), _read())
