import json as _json
import os
import re
import uuid
import logging
import smtplib
import sqlite3
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Header
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent))
DB_PATH  = DATA_DIR / "scans.db"


def _get_db():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def _validate_orgnr(raw: str) -> str:
    clean = re.sub(r'[\s\-\.]', '', raw)
    if not re.match(r'^\d{10}$', clean):
        raise ValueError("Ange 10 siffror, t.ex. 5561234567")
    digits = [int(d) for d in clean]
    total = 0
    for i, d in enumerate(digits[:-1]):
        v = d * 2 if i % 2 == 0 else d
        total += v - 9 if v > 9 else v
    if (10 - total % 10) % 10 != digits[-1]:
        raise ValueError("Felaktigt kontrollnummer – kontrollera org.nr")
    return clean


def _send_email(to: str, subject: str, body: str):
    host = os.environ.get("SMTP_HOST", "")
    if not host:
        logging.info(f"[publ] EMAIL (no SMTP configured) to={to} subject={subject}")
        return
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    pw   = os.environ.get("SMTP_PASS", "")
    frm  = os.environ.get("SMTP_FROM", user)
    msg  = MIMEMultipart()
    msg["From"] = frm
    msg["To"]   = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    with smtplib.SMTP(host, port, timeout=15) as s:
        s.ehlo()
        s.starttls()
        s.login(user, pw)
        s.send_message(msg)


def _auto_scan_job(job_id: str, orgnr: str, contact_name: str, contact_email: str):
    """
    Bakgrundstask: hämtar iXBRL från Bolagsverket, kör Bedrock-scan,
    sparar resultatet och uppdaterar scan_job-status.

    Kräver BOLAGSVERKET_CLIENT_ID + BOLAGSVERKET_CLIENT_SECRET i env.
    Om kredentialerna saknas lämnas jobbet med status 'pending' (manuell hantering).
    """
    if not os.environ.get("BOLAGSVERKET_CLIENT_ID"):
        logging.info(f"[auto-scan] {job_id}: Bolagsverket-credentials saknas – manuell hantering")
        return

    def _update_job(status, error_msg=None, scan_id=None, msg=None):
        now = datetime.now(timezone.utc).isoformat()
        con = _get_db()
        con.execute(
            "UPDATE scan_jobs SET status=?, error_msg=?, scan_id=?, status_msg=?, updated_at=? WHERE id=?",
            (status, error_msg, scan_id, msg, now, job_id),
        )
        con.commit()
        con.close()

    import json as _json

    try:
        _update_job("processing", msg="Hämtar token från Bolagsverket...")

        # Lazy imports to avoid circular import (main.py imports publ.py at startup)
        from bolagsverket import get_annual_report_texts
        from main import _run_bedrock_from_texts, _parse_report_text, _db_save_scan

        logging.info(f"[auto-scan] {job_id}: hämtar årsredovisningar för {orgnr}")
        _update_job("processing", msg=f"Hämtar årsredovisningar för {orgnr}...")
        try:
            texts = get_annual_report_texts(orgnr)
        except RuntimeError as exc:
            if "Inga årsredovisningar" in str(exc):
                # Inget iXBRL — tyst fallback till manuell hantering
                logging.info(f"[auto-scan] {job_id}: ingen iXBRL – manuell hantering")
                _update_job("manual", msg="Ingen digital årsredovisning – hanteras manuellt")
                return
            raise

        logging.info(f"[auto-scan] {job_id}: {len(texts)} dokument, kör Bedrock")
        _update_job("processing", msg=f"Analyserar {len(texts)} årsredovisning{'ar' if len(texts) > 1 else ''} med AI...")
        report_text = _run_bedrock_from_texts(texts)
        logging.info(f"[auto-scan] {job_id}: rapport klar ({len(report_text):,} tecken), sparar")

        _update_job("processing", msg="Sparar rapport...")
        report_md, scan_json_str, hypotheses_json = _parse_report_text(report_text)
        scan_id = _db_save_scan(report_md, scan_json_str, hypotheses_json)

        # Link crm_lead to scan, fyll company_name från scan-JSON om det saknades
        now = datetime.now(timezone.utc).isoformat()
        try:
            sd_name = _json.loads(scan_json_str).get("company_name", "")
        except Exception:
            sd_name = ""
        con = _get_db()
        con.execute(
            """UPDATE crm_leads
               SET scan_id=?, status='Scan klar', updated_at=?, status_changed_at=?,
                   company_name=COALESCE(NULLIF(company_name,''), ?)
               WHERE scan_job_id=?""",
            (scan_id, now, now, sd_name, job_id),
        )
        con.commit()
        con.close()

        _update_job("complete", scan_id=scan_id, msg="Scan klar!")
        logging.info(f"[auto-scan] {job_id}: klar, scan_id={scan_id}")

        # Extrahera nyckeltal för e-post
        try:
            sd = _json.loads(scan_json_str)
            company  = sd.get("company_name", orgnr)
            revenue  = sd.get("revenue_msek")
            total    = sd.get("total_potential_msek")
            conf     = sd.get("confidence", "")
            metrics  = ""
            if revenue: metrics += f"\nOmsättning:       {revenue} MSEK"
            if total:   metrics += f"\nTotal potential:  {total} MSEK"
            if conf:    metrics += f"\nTillförlitlighet: {conf}"
        except Exception:
            company, metrics = orgnr, ""

        # Notifiera teamet
        team_email = os.environ.get("NOTIFY_EMAIL", "")
        base_url   = os.environ.get("APP_BASE_URL", "")
        if team_email:
            try:
                _send_email(
                    team_email,
                    f"[xZero Scan] Auto-scan klar – {company}",
                    f"Scan klar för {company} (org.nr {orgnr})\n"
                    f"Kontakt:  {contact_name} <{contact_email}>\n"
                    f"Scan-id:  {scan_id}\n"
                    f"Länk:     {base_url}/#scan-{scan_id}\n"
                    f"Ref:      {job_id}"
                    + (f"\n{metrics}" if metrics else ""),
                )
            except Exception as exc:
                logging.warning(f"[auto-scan] team email failed: {exc}")

        # Skicka rapporten till användaren
        first = contact_name.split()[0] if contact_name else ""
        try:
            _send_email(
                contact_email,
                f"Din Opportunity Scan är klar – {company}",
                f"Hej {first},\n\n"
                f"Din Opportunity Scan för {company} är nu klar!"
                + (f"\n{metrics}" if metrics else "")
                + f"\n\nVi återkommer inom kort med en genomgång av resultaten.\n\n"
                f"Referensnummer: {job_id}\n\n"
                f"Med vänliga hälsningar,\nxZero",
            )
        except Exception as exc:
            logging.warning(f"[auto-scan] user email failed: {exc}")

    except Exception as exc:
        logging.error(f"[auto-scan] {job_id}: fel: {exc}")
        _update_job("error", error_msg=str(exc)[:500], msg=str(exc)[:200])


@router.get("/publ", response_class=HTMLResponse)
async def publ_page():
    html_path = Path(__file__).parent / "publ.html"
    return html_path.read_text(encoding="utf-8")


class ScanRequest(BaseModel):
    orgnr:         str
    contact_name:  str
    contact_email: str
    company_name:  str = ""
    contact_phone: str = ""
    sni_codes:     list = []
    initial_status: str = "Lead"


@router.post("/publ/submit")
async def publ_submit(req: ScanRequest, background_tasks: BackgroundTasks):
    try:
        orgnr = _validate_orgnr(req.orgnr)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    contact_name  = req.contact_name.strip()
    contact_email = req.contact_email.strip()
    company_name  = req.company_name.strip()
    contact_phone = req.contact_phone.strip()

    if not contact_name:
        return JSONResponse({"error": "Namn saknas"}, status_code=422)
    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', contact_email):
        return JSONResponse({"error": "Ogiltig e-postadress"}, status_code=422)

    job_id  = str(uuid.uuid4())[:8].upper()
    lead_id = str(uuid.uuid4())[:8].upper()
    now     = datetime.now(timezone.utc).isoformat()

    con = _get_db()
    con.execute(
        """INSERT INTO scan_jobs (id, orgnr, contact_name, contact_email, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
        (job_id, orgnr, contact_name, contact_email, now, now)
    )
    initial_status = req.initial_status if req.initial_status in ("Prospekt", "Lead") else "Lead"
    con.execute(
        """INSERT INTO crm_leads
           (id, orgnr, company_name, contact_name, contact_email, contact_phone,
            status, scan_job_id, sni_codes, created_at, updated_at, status_changed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (lead_id, orgnr, company_name, contact_name, contact_email, contact_phone,
         initial_status, job_id, _json.dumps(req.sni_codes, ensure_ascii=False), now, now, now)
    )
    con.commit()
    con.close()

    # Notify team
    team_email = os.environ.get("NOTIFY_EMAIL", "")
    if team_email:
        try:
            _send_email(
                team_email,
                f"[xZero Scan] Ny förfrågan – {orgnr}",
                f"Org.nr:  {orgnr}\nNamn:    {contact_name}\nE-post:  {contact_email}\nRef:     {job_id}\nTid:     {now}"
            )
        except Exception as e:
            logging.warning(f"[publ] team email failed: {e}")

    # Confirm to user
    first = contact_name.split()[0] if contact_name else ""
    try:
        _send_email(
            contact_email,
            "Vi har tagit emot din förfrågan – xZero",
            f"Hej {first},\n\n"
            f"Tack för din förfrågan! Vi analyserar org.nr {orgnr} och återkommer "
            f"med din Opportunity Scan inom 24 timmar.\n\n"
            f"Referensnummer: {job_id}\n\n"
            f"Med vänliga hälsningar,\nxZero"
        )
    except Exception as e:
        logging.warning(f"[publ] user confirmation email failed: {e}")

    background_tasks.add_task(_auto_scan_job, job_id, orgnr, contact_name, contact_email)

    return {"job_id": job_id}


@router.get("/publ/status/{job_id}")
async def publ_status(job_id: str):
    con = _get_db()
    row = con.execute(
        "SELECT status, error_msg, scan_id, status_msg, created_at FROM scan_jobs WHERE id=?",
        (job_id,)
    ).fetchone()
    con.close()
    if not row:
        return JSONResponse({"error": "Hittades inte"}, status_code=404)
    return {
        "status":     row["status"],
        "error":      row["error_msg"],
        "scan_id":    row["scan_id"],
        "status_msg": row["status_msg"],
        "created_at": row["created_at"],
    }


# ── Admin: kör scan från förextraherade texter (t.ex. Textract-OCR) ─────────

class TextEntry(BaseModel):
    label: str
    text:  str

class GroupScanRequest(BaseModel):
    orgnr:         str
    company_name:  str = ""
    contact_name:  str = ""
    contact_email: str = ""
    contact_phone: str = ""
    texts:         List[TextEntry]   # [(label, text), ...]


def _run_group_scan_job(
    scan_id_holder: list,
    orgnr: str,
    company_name: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str,
    texts: list,
):
    """Bakgrundstask: kör Bedrock-scan på förextraherade texter och sparar resultatet."""
    import json as _json
    lead_id = str(uuid.uuid4())[:8].upper()
    now     = datetime.now(timezone.utc).isoformat()

    # Skapa crm_lead direkt
    con = _get_db()
    con.execute(
        """INSERT INTO crm_leads
           (id, orgnr, company_name, contact_name, contact_email, contact_phone,
            status, created_at, updated_at, status_changed_at)
           VALUES (?, ?, ?, ?, ?, ?, 'Lead', ?, ?, ?)""",
        (lead_id, orgnr, company_name, contact_name, contact_email, contact_phone,
         now, now, now),
    )
    con.commit()
    con.close()

    try:
        from main import _run_bedrock_from_texts, _parse_report_text, _db_save_scan

        text_tuples = [(t["label"], t["text"]) for t in texts]
        logging.info(f"[group-scan] {lead_id}: {len(text_tuples)} dokument, kör Bedrock")
        report_text = _run_bedrock_from_texts(text_tuples)

        report_md, scan_json_str, hypotheses_json = _parse_report_text(report_text)
        sid = _db_save_scan(report_md, scan_json_str, hypotheses_json)
        scan_id_holder.append(sid)

        # Fyll i bolagsnamn från scan-JSON om det saknades
        try:
            sd_name = _json.loads(scan_json_str).get("company_name", "") or company_name
        except Exception:
            sd_name = company_name

        now2 = datetime.now(timezone.utc).isoformat()
        con = _get_db()
        con.execute(
            "UPDATE crm_leads SET scan_id=?, status='Scan klar', company_name=?, updated_at=?, status_changed_at=? WHERE id=?",
            (sid, sd_name, now2, now2, lead_id),
        )
        con.commit()
        con.close()

        logging.info(f"[group-scan] {lead_id}: klar, scan_id={sid}")

        # Notifiera teamet
        team_email = os.environ.get("NOTIFY_EMAIL", "")
        base_url   = os.environ.get("APP_BASE_URL", "")
        if team_email:
            try:
                _send_email(
                    team_email,
                    f"[xZero Scan] Gruppscan klar – {sd_name or orgnr}",
                    f"Gruppscan klar för {sd_name or orgnr} (org.nr {orgnr})\n"
                    f"Kontakt:  {contact_name} <{contact_email}>\n"
                    f"Scan-id:  {sid}\n"
                    f"Länk:     {base_url}/#scan-{sid}\n"
                    f"Dokument: {len(text_tuples)} st",
                )
            except Exception as exc:
                logging.warning(f"[group-scan] team email failed: {exc}")

        # Bekräfta till kontaktpersonen
        if contact_email:
            first = contact_name.split()[0] if contact_name else ""
            try:
                _send_email(
                    contact_email,
                    f"Din Opportunity Scan är klar – {sd_name or orgnr}",
                    f"Hej {first},\n\n"
                    f"Din Opportunity Scan för {sd_name or orgnr} är nu klar!\n\n"
                    f"Vi återkommer inom kort med en genomgång av resultaten.\n\n"
                    f"Med vänliga hälsningar,\nxZero",
                )
            except Exception as exc:
                logging.warning(f"[group-scan] user email failed: {exc}")

    except Exception as exc:
        logging.error(f"[group-scan] {lead_id}: fel: {exc}")
        now2 = datetime.now(timezone.utc).isoformat()
        con = _get_db()
        con.execute(
            "UPDATE crm_leads SET status='Fel', notes=?, updated_at=?, status_changed_at=? WHERE id=?",
            (str(exc)[:500], now2, now2, lead_id),
        )
        con.commit()
        con.close()


@router.post("/admin/scan-from-texts")
async def admin_scan_from_texts(
    req: GroupScanRequest,
    background_tasks: BackgroundTasks,
    x_admin_token: Optional[str] = Header(None),
):
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if admin_token and x_admin_token != admin_token:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not req.texts:
        return JSONResponse({"error": "Inga texter skickades"}, status_code=422)

    try:
        orgnr = _validate_orgnr(req.orgnr)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    scan_id_holder: list = []
    background_tasks.add_task(
        _run_group_scan_job,
        scan_id_holder,
        orgnr,
        req.company_name.strip(),
        req.contact_name.strip(),
        req.contact_email.strip(),
        req.contact_phone.strip(),
        [t.dict() for t in req.texts],
    )

    logging.info(f"[admin] scan-from-texts queued: orgnr={orgnr}, docs={len(req.texts)}")
    return {"status": "queued", "orgnr": orgnr, "doc_count": len(req.texts)}


# ── Admin: konsolidera befintliga scanar till en koncernanalys ───────────────

_CONSOLIDATION_PROMPT = """\
Du är en affärsanalytiker som ska sammanställa en koncernanalys baserat på individuella \
Opportunity Scans för bolagen i en koncern.

Nedan följer ELIR-data och nyckeltal för varje bolag i koncernen.

{company_blocks}

## Din uppgift

Producera en strukturerad koncernanalys med följande delar:

### 1. Koncernöversikt
Sammanfatta koncernens samlade ELIR-potential i en tabell:
- Totalt E, L, I, R och Total i MSEK
- Varje bolags andel av totalpotentialen
- Viktat genomsnitt av EBIT-marginal

### 2. Dominerande mönster
Identifiera 3–5 mönster som återkommer i flera bolag (t.ex. gemensam kostnadsdrivare, \
strukturell ineffektivitet, gemensam marknadsmöjlighet).

### 3. Prioriterade möjligheter på koncernnivå
Lista de 3 mest prioriterade möjligheterna att adressera på koncernnivå, rangordnade efter \
förväntad effekt. För varje möjlighet: vilket/vilka bolag berörs, ELIR-kategori, \
uppskattad potential i MSEK.

### 4. Rekommenderad startpunkt
Vilket bolag och vilken hypotes ska koncernen adressera först, och varför? \
Referera till de individuella hypoteserna (H1/H2/H3) från respektive bolag.

Var konkret och kvantitativ. Använd siffrorna från ELIR-data konsekvent.
"""

_COMPANY_BLOCK_TEMPLATE = """\
---
## {company_name} (scan_id={scan_id})
Bransch: {industry}
Omsättning: {revenue_msek} MSEK  |  EBIT: {ebit_msek} MSEK  |  EBIT-marginal: {ebit_margin_pct}%
Potential: E={e_msek} MSEK ({e_pct}%)  L={l_msek} MSEK ({l_pct}%)  \
I={i_msek} MSEK ({i_pct}%)  R={r_msek} MSEK ({r_pct}%)
Total potential: {total_potential_msek} MSEK  |  Tillförlitlighet: {confidence}

Hypoteser:
{hypotheses}
"""


def _format_hypotheses(hyp_json: str) -> str:
    import json as _json
    try:
        hyps = _json.loads(hyp_json) if hyp_json else []
        if not hyps:
            return "  (inga hypoteser)"
        lines = []
        for h in hyps:
            elir = h.get("linked_elir", {})
            lines.append(
                f"  {h.get('hypothesis_id','?')}: {h.get('title','')} "
                f"[E:{elir.get('E_relevance','?')} L:{elir.get('L_relevance','?')} "
                f"I:{elir.get('I_relevance','?')} R:{elir.get('R_relevance','?')}] "
                f"prio={h.get('simulation_priority_rank','?')}"
            )
        return "\n".join(lines)
    except Exception:
        return "  (kunde inte tolka hypoteser)"


def _consolidation_job(
    group_name: str,
    scan_ids: list,
    contact_name: str,
    contact_email: str,
):
    import json as _json
    import anthropic

    logging.info(f"[consolidate] startar för {len(scan_ids)} scanar: {scan_ids}")

    # Hämta scan-data från DB
    con = _get_db()
    company_blocks = []
    missing = []
    for sid in scan_ids:
        row = con.execute(
            """SELECT id, company_name, industry, revenue_msek, ebit_msek, ebit_margin_pct,
                      e_msek, e_pct, l_msek, l_pct, i_msek, i_pct, r_msek, r_pct,
                      total_potential_msek, confidence, workshop_hypotheses
               FROM scans WHERE id=?""",
            (sid,)
        ).fetchone()
        if not row:
            logging.warning(f"[consolidate] scan_id={sid} hittades inte")
            missing.append(sid)
            continue

        # Hämta hypoteser – försök workshop_hypotheses i scans-tabellen, fallback till workshop_sessions
        hyp_json = row["workshop_hypotheses"]
        if not hyp_json or hyp_json == "[]":
            ws_row = con.execute(
                "SELECT session_json FROM workshop_sessions WHERE scan_id=? ORDER BY created_at DESC LIMIT 1",
                (sid,)
            ).fetchone()
            if ws_row:
                try:
                    ws = _json.loads(ws_row["session_json"])
                    hyp_list = ws.get("hypotheses") or []
                    if hyp_list:
                        hyp_json = _json.dumps(hyp_list, ensure_ascii=False)
                        logging.info(f"[consolidate] scan {sid}: hämtade {len(hyp_list)} hypoteser från workshop_sessions")
                except Exception:
                    pass

        block = _COMPANY_BLOCK_TEMPLATE.format(
            company_name      = row["company_name"] or f"Bolag {sid}",
            scan_id           = sid,
            industry          = row["industry"] or "okänd",
            revenue_msek      = row["revenue_msek"] or 0,
            ebit_msek         = row["ebit_msek"] or 0,
            ebit_margin_pct   = round(row["ebit_margin_pct"] or 0, 1),
            e_msek            = row["e_msek"] or 0,
            e_pct             = round(row["e_pct"] or 0, 1),
            l_msek            = row["l_msek"] or 0,
            l_pct             = round(row["l_pct"] or 0, 1),
            i_msek            = row["i_msek"] or 0,
            i_pct             = round(row["i_pct"] or 0, 1),
            r_msek            = row["r_msek"] or 0,
            r_pct             = round(row["r_pct"] or 0, 1),
            total_potential_msek = row["total_potential_msek"] or 0,
            confidence        = row["confidence"] or "okänd",
            hypotheses        = _format_hypotheses(hyp_json),
        )
        company_blocks.append(block)
    con.close()

    if not company_blocks:
        logging.error(f"[consolidate] inga giltiga scanar hittades")
        return

    prompt = _CONSOLIDATION_PROMPT.format(company_blocks="\n".join(company_blocks))

    # Kör Bedrock
    try:
        aws_key    = os.environ.get("AWS_ACCESS_KEY_ID", "")
        aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        aws_region = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")
        client = anthropic.AnthropicBedrock(
            aws_access_key=aws_key,
            aws_secret_key=aws_secret,
            aws_region=aws_region,
            max_retries=3,
        )
        msg = client.messages.create(
            model="us.anthropic.claude-sonnet-4-6",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        report_md = msg.content[0].text
        logging.info(f"[consolidate] rapport klar ({len(report_md):,} tecken)")
    except Exception as exc:
        logging.error(f"[consolidate] Bedrock-fel: {exc}")
        return

    # Spara som en ny scan med type='group'
    now = datetime.now(timezone.utc).isoformat()
    con = _get_db()
    try:
        cur = con.execute(
            """INSERT INTO scans
               (created_at, company_name, industry, revenue_msek, report_markdown, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now, group_name or "Koncernanalys",
             "Koncern",
             0,   # summeras i rapporten
             report_md,
             "group"),
        )
        group_scan_id = cur.lastrowid
        con.commit()
    except Exception as exc:
        logging.error(f"[consolidate] DB-fel: {exc}")
        con.close()
        return
    con.close()

    logging.info(f"[consolidate] sparat som scan_id={group_scan_id}")

    # Notifiera teamet
    team_email = os.environ.get("NOTIFY_EMAIL", "")
    base_url   = os.environ.get("APP_BASE_URL", "")
    if team_email:
        try:
            _send_email(
                team_email,
                f"[xZero] Koncernanalys klar – {group_name}",
                f"Koncernanalys klar för {group_name}\n"
                f"Ingående scanar: {scan_ids}\n"
                f"Scan-id: {group_scan_id}\n"
                f"Länk: {base_url}/#scan-{group_scan_id}\n"
                + (f"Missade scanar: {missing}" if missing else ""),
            )
        except Exception as exc:
            logging.warning(f"[consolidate] team email failed: {exc}")

    if contact_email:
        first = contact_name.split()[0] if contact_name else ""
        try:
            _send_email(
                contact_email,
                f"Er Opportunity Scan är klar – {group_name}",
                f"Hej {first},\n\n"
                f"Koncernanalysen för {group_name} är nu klar!\n\n"
                f"Vi återkommer inom kort med en genomgång av resultaten.\n\n"
                f"Med vänliga hälsningar,\nxZero",
            )
        except Exception as exc:
            logging.warning(f"[consolidate] user email failed: {exc}")

    return group_scan_id


class ConsolidateRequest(BaseModel):
    scan_ids:      List[int]
    group_name:    str = "Koncernanalys"
    contact_name:  str = ""
    contact_email: str = ""


class UpdateScanFinancialsRequest(BaseModel):
    scan_id:         int
    revenue_msek:    float
    ebit_msek:       float
    ebit_margin_pct: float = 0.0
    industry:        Optional[str] = None


@router.post("/admin/update-scan-financials")
async def admin_update_scan_financials(
    req: UpdateScanFinancialsRequest,
    x_admin_token: Optional[str] = Header(None),
):
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if admin_token and x_admin_token != admin_token:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    con = _get_db()
    row = con.execute("SELECT id FROM scans WHERE id=?", (req.scan_id,)).fetchone()
    if not row:
        con.close()
        return JSONResponse({"error": f"scan_id {req.scan_id} hittades inte"}, status_code=404)

    fields = {
        "revenue_msek":    req.revenue_msek,
        "ebit_msek":       req.ebit_msek,
        "ebit_margin_pct": req.ebit_margin_pct or (
            round(req.ebit_msek / req.revenue_msek * 100, 2) if req.revenue_msek else 0.0
        ),
    }
    if req.industry:
        fields["industry"] = req.industry

    set_clause = ", ".join(f"{k}=?" for k in fields)
    con.execute(
        f"UPDATE scans SET {set_clause} WHERE id=?",
        (*fields.values(), req.scan_id),
    )
    con.commit()
    con.close()

    logging.info(f"[admin] scan {req.scan_id} uppdaterad: {fields}")
    return {"updated": req.scan_id, "fields": fields}


@router.get("/admin/scans")
async def admin_list_scans(
    limit: int = 50,
    x_admin_token: Optional[str] = Header(None),
):
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if admin_token and x_admin_token != admin_token:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    con = _get_db()
    rows = con.execute(
        """SELECT s.id, s.company_name, s.industry, s.revenue_msek, s.total_potential_msek,
                  s.confidence, s.created_at,
                  CASE WHEN (s.workshop_hypotheses IS NOT NULL AND s.workshop_hypotheses != '[]')
                            OR EXISTS (
                                SELECT 1 FROM workshop_sessions ws
                                WHERE ws.scan_id = s.id
                                  AND ws.session_json LIKE '%"hypotheses"%'
                                  AND ws.session_json NOT LIKE '%"hypotheses": []%'
                                  AND ws.session_json NOT LIKE '%"hypotheses":[]%'
                            )
                       THEN 1 ELSE 0 END AS has_hypotheses
           FROM scans s ORDER BY s.id DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@router.post("/admin/consolidate")
async def admin_consolidate(
    req: ConsolidateRequest,
    background_tasks: BackgroundTasks,
    x_admin_token: Optional[str] = Header(None),
):
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if admin_token and x_admin_token != admin_token:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if len(req.scan_ids) < 2:
        return JSONResponse({"error": "Minst 2 scan_ids krävs"}, status_code=422)

    background_tasks.add_task(
        _consolidation_job,
        req.group_name,
        req.scan_ids,
        req.contact_name,
        req.contact_email,
    )

    logging.info(f"[admin] consolidate queued: {req.scan_ids}")
    return {"status": "queued", "scan_ids": req.scan_ids, "group_name": req.group_name}
