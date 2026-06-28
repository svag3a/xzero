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

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

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


@router.get("/publ", response_class=HTMLResponse)
async def publ_page():
    html_path = Path(__file__).parent / "publ.html"
    return html_path.read_text(encoding="utf-8")


class ScanRequest(BaseModel):
    orgnr:         str
    contact_name:  str
    contact_email: str


@router.post("/publ/submit")
async def publ_submit(req: ScanRequest):
    try:
        orgnr = _validate_orgnr(req.orgnr)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    contact_name  = req.contact_name.strip()
    contact_email = req.contact_email.strip()

    if not contact_name:
        return JSONResponse({"error": "Namn saknas"}, status_code=422)
    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', contact_email):
        return JSONResponse({"error": "Ogiltig e-postadress"}, status_code=422)

    job_id = str(uuid.uuid4())[:8].upper()
    now    = datetime.now(timezone.utc).isoformat()

    con = _get_db()
    con.execute(
        """INSERT INTO scan_jobs (id, orgnr, contact_name, contact_email, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
        (job_id, orgnr, contact_name, contact_email, now, now)
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

    return {"job_id": job_id}


@router.get("/publ/status/{job_id}")
async def publ_status(job_id: str):
    con = _get_db()
    row = con.execute(
        "SELECT status, error_msg, created_at FROM scan_jobs WHERE id=?",
        (job_id,)
    ).fetchone()
    con.close()
    if not row:
        return JSONResponse({"error": "Hittades inte"}, status_code=404)
    return {"status": row["status"], "error": row["error_msg"], "created_at": row["created_at"]}
