#!/usr/bin/env python3
"""
import_leads.py – Batch-import av bolag från Excel-kampanjfil.

Flöde per bolag:
  1. Filtrera på relevanta statuses (Unprocessed, Automatic redial, etc.)
  2. Om e-post saknas och --enrich är satt: Tavily-sökning → Claude extraherar adress
  3. POST till /publ/submit → appen kör scan i bakgrunden och mailar resultatet

Usage:
  python3 import_leads.py <excel-fil> [options]

Options:
  --api-url URL       App-URL, default: http://13.48.24.83
  --limit N           Max antal bolag (default: alla)
  --enrich            Försök hitta e-post via webben för bolag som saknar den
  --dry-run           Visa vad som skulle hända utan att skicka något
  --statuses S,S,...  Filtrera statuses (default: Unprocessed,Automatic redial,Shared callback,Success)
  --delay SECS        Sekunder mellan anrop (default: 2)

Miljövariabler som krävs för --enrich:
  TAVILY_API_KEY
  ANTHROPIC_API_KEY  (eller kör via Bedrock med AWS_*-variabler)
"""

import argparse
import os
import sys
import time
import json
import requests
import openpyxl
from datetime import datetime

# ── Kolumnindex (0-baserade) i Excel-filen ──────────────────────────────────
COL_CAMPAIGN  = 0
COL_STATUS    = 1
COL_NAME      = 2   # bolagsnamn
COL_ORGNR     = 3
COL_MOBILE    = 4
COL_PHONE     = 5
COL_EMAIL     = 6   # kontaktpersonens e-post

DEFAULT_STATUSES = {"Unprocessed", "Automatic redial", "Shared callback", "Success"}


def normalize_orgnr(raw):
    """Normaliserar org.nr till 10 siffror utan bindestreck."""
    s = str(raw).strip().replace("-", "").replace(" ", "").replace(".", "")
    # Excel sparar ibland som float: 5561948539.0
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 10 and s.isdigit():
        return s
    return None


def enrich_email(company_name: str, orgnr: str):
    """Försöker hitta e-postadress via Tavily + Claude."""
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        print("    [enrich] TAVILY_API_KEY saknas – hoppar över enrichment")
        return None

    # Tavily-sökning
    query = f'"{company_name}" kontakt e-post email'
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": tavily_key, "query": query, "max_results": 5},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        search_text = "\n\n".join(
            f"{r.get('title','')}\n{r.get('content','')}" for r in results
        )
    except Exception as e:
        print(f"    [enrich] Tavily-fel: {e}")
        return None

    # Claude extraherar e-post
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        print("    [enrich] ANTHROPIC_API_KEY saknas – hoppar över enrichment")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Bolag: {company_name} (org.nr {orgnr})\n\n"
                    f"Sökresultat:\n{search_text[:3000]}\n\n"
                    "Extrahera EN e-postadress (innehåller @) som troligen tillhör detta bolag. "
                    "Svara med BARA e-postadressen (t.ex. info@foretaget.se). "
                    "Om ingen e-postadress hittades, svara med exakt: NONE. "
                    "Telefonnummer, URL:er och annat är inte giltiga svar."
                ),
            }],
        )
        result = msg.content[0].text.strip()
        result = result.strip().strip('"').strip("'")
        if result.upper() == "NONE" or "@" not in result:
            return None
        # Sanity-check: måste se ut som en e-postadress
        parts = result.split("@")
        if len(parts) == 2 and "." in parts[1] and " " not in result and len(result) > 5:
            return result.lower()
    except Exception as e:
        print(f"    [enrich] Claude-fel: {e}")

    return None


def submit_scan(api_url: str, orgnr: str, company_name: str, email: str, phone: str = "") -> dict:
    """Skickar POST /publ/submit och returnerar svaret."""
    resp = requests.post(
        f"{api_url}/publ/submit",
        json={
            "orgnr":         orgnr,
            "company_name":  company_name,
            "contact_name":  company_name,  # används som avsändarnamn i mail
            "contact_email": email,
            "contact_phone": phone,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Batch-import av bolag till xZero Opportunity Scan")
    parser.add_argument("excel", help="Sökväg till Excel-filen")
    parser.add_argument("--api-url", default="http://13.48.24.83", help="App-URL")
    parser.add_argument("--limit", type=int, default=None, help="Max antal bolag")
    parser.add_argument("--enrich", action="store_true", help="Försök hitta e-post via webben")
    parser.add_argument("--dry-run", action="store_true", help="Simulera utan att skicka")
    parser.add_argument("--statuses", default=None, help="Kommaseparerade statuses att inkludera")
    parser.add_argument("--delay", type=float, default=2.0, help="Sekunder mellan anrop")
    args = parser.parse_args()

    target_statuses = (
        {s.strip() for s in args.statuses.split(",")}
        if args.statuses
        else DEFAULT_STATUSES
    )

    print(f"Läser {args.excel}...")
    wb = openpyxl.load_workbook(args.excel, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header, rows = rows[0], rows[1:]
    print(f"  {len(rows)} rader totalt")

    # Filter
    candidates = []
    for r in rows:
        status = r[COL_STATUS]
        if status not in target_statuses:
            continue
        orgnr = normalize_orgnr(r[COL_ORGNR])
        if not orgnr:
            continue
        candidates.append({
            "orgnr":        orgnr,
            "company_name": str(r[COL_NAME]   or "").strip(),
            "email":        str(r[COL_EMAIL]   or "").strip(),
            "mobile":       str(r[COL_MOBILE]  or "").strip(),
            "phone":        str(r[COL_PHONE]   or "").strip(),
            "status":       status,
        })

    print(f"  {len(candidates)} bolag matchar statuses: {', '.join(sorted(target_statuses))}")

    if args.limit:
        candidates = candidates[: args.limit]
        print(f"  Begränsar till {args.limit} bolag")

    has_email    = sum(1 for c in candidates if c["email"])
    missing_email = len(candidates) - has_email
    print(f"  Har e-post: {has_email}  |  Saknar e-post: {missing_email}")
    if missing_email and not args.enrich:
        print("  Tips: kör med --enrich för att söka efter saknade e-postadresser")

    print()

    stats = {"submitted": 0, "enriched": 0, "skipped_no_email": 0, "skipped_invalid": 0, "errors": 0}
    log_rows = []

    for i, c in enumerate(candidates, 1):
        orgnr        = c["orgnr"]
        company_name = c["company_name"] or orgnr
        email        = c["email"]
        phone        = c["mobile"] or c["phone"]
        status       = c["status"]

        prefix = f"[{i}/{len(candidates)}] {company_name} ({orgnr})"

        if not email:
            if args.enrich:
                print(f"{prefix} – söker e-post...")
                email = enrich_email(company_name, orgnr)
                if email:
                    print(f"    → hittade {email}")
                    stats["enriched"] += 1
                else:
                    print(f"    → ingen e-post hittad, hoppar över")
                    stats["skipped_no_email"] += 1
                    log_rows.append({"orgnr": orgnr, "company": company_name, "result": "no_email"})
                    continue
            else:
                print(f"{prefix} – saknar e-post, hoppar över")
                stats["skipped_no_email"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name, "result": "no_email"})
                continue

        if args.dry_run:
            print(f"{prefix} – [dry-run] skulle skicka till {email}")
            stats["submitted"] += 1
            log_rows.append({"orgnr": orgnr, "company": company_name, "email": email, "result": "dry_run"})
        else:
            try:
                result = submit_scan(args.api_url, orgnr, company_name, email, phone)
                job_id = result.get("job_id", "?")
                print(f"{prefix} – OK, job_id={job_id} → {email}")
                stats["submitted"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name, "email": email, "job_id": job_id, "result": "ok"})
            except requests.HTTPError as e:
                msg = ""
                try:
                    msg = e.response.json().get("error", "")
                except Exception:
                    pass
                print(f"{prefix} – FEL {e.response.status_code}: {msg}")
                stats["errors"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name, "email": email, "result": f"error_{e.response.status_code}", "detail": msg})
            except Exception as e:
                print(f"{prefix} – FEL: {e}")
                stats["errors"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name, "email": email, "result": "error", "detail": str(e)})

        if i < len(candidates):
            time.sleep(args.delay)

    # ── Sammanfattning ───────────────────────────────────────────────────────
    print()
    print("═" * 50)
    print(f"  Skickade:          {stats['submitted']}")
    print(f"  Berikade e-poster: {stats['enriched']}")
    print(f"  Hoppade (no mail): {stats['skipped_no_email']}")
    print(f"  Fel:               {stats['errors']}")
    print("═" * 50)

    # Skriv logg
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"import_log_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_rows, f, ensure_ascii=False, indent=2)
    print(f"  Logg sparad: {log_path}")


if __name__ == "__main__":
    main()
