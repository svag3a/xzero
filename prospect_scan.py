#!/usr/bin/env python3
"""
prospect_scan.py – Hitta prospektbolag via SCB-filen + Bolagsverket iXBRL.

Flöde:
  1. Läser SCB-bulkfil (scb_bulkfil.zip) → aktiva AB i målbranscher
  2. Hämtar senaste årsredovisning per bolag via Bolagsverket API
  3. Extraherar omsättning direkt från XBRL-taggar (ingen LLM behövs)
  4. Filtrerar på omsättning >= --min-revenue MSEK
  5. Berikar e-post via Tavily + Claude (om --enrich)
  6. Skickar till WasteZero-appen (om inte --dry-run)

Körs i omgångar – sparar vilka org.nr som kollerats i state-filen
(prospect_scan_state.json) och fortsätter där det slutade.

Usage:
  python3 prospect_scan.py --min-revenue 150 --enrich --api-url http://13.48.24.83
  python3 prospect_scan.py --min-revenue 150 --dry-run --batch-size 50
  python3 prospect_scan.py --list-candidates          # visa utan att köra

Miljövariabler:
  BOLAGSVERKET_CLIENT_ID      (krävs)
  BOLAGSVERKET_CLIENT_SECRET  (krävs)
  TAVILY_API_KEY              (krävs för --enrich)
  ANTHROPIC_API_KEY           (krävs för --enrich)
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Målbranscher (SNI 5-siffriga) ────────────────────────────────────────────
TARGET_SNI = {
    "49410",  # Godstransport på väg
    "52100",  # Lagring
    "52210",  # Stödtjänster till landtransport (terminaler m.m.)
    "52219",  # Annan stödtjänst till landtransport
    "52229",  # Annan stödtjänst till sjötransport
    "52240",  # Godshantering
    "52249",  # Annan stödtjänst till lufttransport
    "52290",  # Övrig stödtjänst till transport
    "46311",  # Partihandel med spannmål, råolja och råfett m.m.
    "46312",  # Partihandel med animala råvaror
    "46313",  # Partihandel med frukt och grönsaker
    "46320",  # Partihandel med kött och köttvaror
    "46330",  # Partihandel med mejeriprodukter, ägg och matolja
    "46340",  # Partihandel med drycker
    "46380",  # Partihandel med övriga livsmedel
    "46390",  # Partihandel med livsmedel, drycker och tobak i sortiment
    "10110",  # Charkuteri- och annan köttvaruframställning
    "10510",  # Framställning av mjölk och andra mjölkprodukter
    "10200",  # Beredning och konservering av fisk, kräftdjur och blötdjur
}

SCB_URL = "https://vardefulla-datamangder.bolagsverket.se/scb/scb_bulkfil.zip"

# Bolagsverket API
_TOKEN_URL  = "https://portal.api.bolagsverket.se/oauth2/token"
_BASE_URL   = "https://gw.api.bolagsverket.se/vardefulla-datamangder/v1"
_SCOPE      = "vardefulla-datamangder:read"

# XBRL-taggar för omsättning (svenska årsredovisningar)
_REVENUE_XBRL_NAMES = [
    "se-cd:NetSalesRevenues",
    "se-gen:NetSalesRevenues",
    "se-cd-base:NetSalesRevenues",
    "ifrs-full:Revenue",
    "ifrs:Revenue",
    "se:NetSalesRevenues",
    "bv:NetSalesRevenues",
    "bv-base:NetSalesRevenues",
]

# ── Bolagsverket API-hjälpare ─────────────────────────────────────────────────

_cached_token: Optional[str] = None
_token_expires_at: float = 0.0


def _get_token() -> str:
    global _cached_token, _token_expires_at
    if _cached_token and time.time() < _token_expires_at - 60:
        return _cached_token
    client_id     = os.environ["BOLAGSVERKET_CLIENT_ID"]
    client_secret = os.environ["BOLAGSVERKET_CLIENT_SECRET"]
    resp = requests.post(
        _TOKEN_URL,
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials", "scope": _SCOPE},
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f"Token-fel {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    _cached_token   = data["access_token"]
    _token_expires_at = time.time() + data.get("expires_in", 3600)
    return _cached_token


def _doc_list(orgnr: str) -> list:
    token = _get_token()
    resp  = requests.post(
        f"{_BASE_URL}/dokumentlista",
        headers={"Authorization": f"Bearer {token}"},
        json={"identitetsbeteckning": orgnr},
        timeout=30,
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("dokument", [])


def _fetch_ixbrl_zip(doc_id: str) -> bytes:
    token = _get_token()
    resp  = requests.get(
        f"{_BASE_URL}/dokument/{doc_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/zip"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


def _extract_revenue_from_ixbrl(zip_bytes: bytes) -> Optional[float]:
    """Returnerar omsättning i MSEK från iXBRL-ZIP, eller None."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            xhtml_files = [n for n in z.namelist() if n.lower().endswith(".xhtml")]
            if not xhtml_files:
                return None
            content = z.read(xhtml_files[0]).decode("utf-8", errors="replace")
    except Exception:
        return None

    # Sök efter XBRL-taggar för omsättning
    # Mönster: <ix:nonFraction name="se-cd:NetSalesRevenues" ... >12345678</ix:nonFraction>
    # Värdet är vanligen i SEK (heltal) eller tkr med scale-attribut
    pattern = re.compile(
        r'<ix:nonFraction\s[^>]*name="([^"]+)"[^>]*scale="([^"]*)"[^>]*>([^<]+)</ix:nonFraction>',
        re.IGNORECASE,
    )
    # Alternativt utan scale
    pattern2 = re.compile(
        r'<ix:nonFraction\s[^>]*name="([^"]+)"[^>]*>([^<]+)</ix:nonFraction>',
        re.IGNORECASE,
    )

    for m in pattern.finditer(content):
        tag_name, scale_str, raw_val = m.group(1), m.group(2), m.group(3)
        if tag_name not in _REVENUE_XBRL_NAMES:
            continue
        val = _parse_numeric(raw_val, scale_str)
        if val is not None:
            return val / 1_000_000  # SEK → MSEK

    # Fallback utan scale-attribut
    for m in pattern2.finditer(content):
        tag_name, raw_val = m.group(1), m.group(2)
        if tag_name not in _REVENUE_XBRL_NAMES:
            continue
        val = _parse_numeric(raw_val, "")
        if val is not None and val > 100_000:  # antar SEK om > 100k
            return val / 1_000_000

    return None


def _parse_numeric(raw: str, scale: str) -> Optional[float]:
    s = raw.strip().replace(" ", "").replace(" ", "").replace(",", ".")
    # ta bort ev. tusenavskiljare (punkt) om det finns decimaler
    try:
        val = float(s)
    except ValueError:
        return None
    try:
        exp = int(scale) if scale else 0
    except ValueError:
        exp = 0
    return val * (10 ** exp)


def lookup_revenue(orgnr: str) -> Optional[float]:
    """
    Returnerar omsättning i MSEK för ett bolag via Bolagsverkets iXBRL-API.
    Returnerar None om:
      - inga dokument finns
      - omsättningen inte kan parsas
    """
    docs = _doc_list(orgnr)
    if not docs:
        return None

    docs.sort(key=lambda d: d.get("rapporteringsperiodTom", ""), reverse=True)

    for doc in docs[:2]:   # prova senaste, sedan näst senaste
        doc_id = doc.get("dokumentId", "")
        if not doc_id:
            continue
        try:
            zip_bytes = _fetch_ixbrl_zip(doc_id)
            rev = _extract_revenue_from_ixbrl(zip_bytes)
            if rev is not None:
                return rev
        except Exception as exc:
            log.debug(f"  [{orgnr}] dok-fel {doc_id}: {exc}")

    return None


# ── SCB-fil ───────────────────────────────────────────────────────────────────

def load_scb_companies(scb_zip_path: Optional[str] = None) -> list[dict]:
    """
    Laddar aktiva AB i målbranscher från SCB-filen.
    Laddar ned filen om den inte finns (eller om --refresh-scb är satt).
    """
    cache_path = scb_zip_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scb_bulkfil.zip"
    )

    if not os.path.exists(cache_path):
        log.info(f"Laddar ned SCB-fil från {SCB_URL}...")
        resp = requests.get(SCB_URL, stream=True, timeout=300)
        resp.raise_for_status()
        total = 0
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)
        log.info(f"SCB-fil nedladdad: {total/1024/1024:.1f} MB")

    companies = []
    with zipfile.ZipFile(cache_path) as z:
        fname = z.namelist()[0]
        with z.open(fname) as f:
            header = f.readline().decode("latin-1", errors="replace").rstrip().split("\t")
            ng1_idx     = header.index("Ng1")
            jurform_idx = header.index("JurForm")
            orgnr_idx   = header.index("PeOrgNr")
            namn_idx    = header.index("Namn")
            ftgstat_idx = header.index("FtgStat")

            for line in f:
                row = line.decode("latin-1", errors="replace").rstrip().split("\t")
                if len(row) <= ng1_idx:
                    continue
                if row[ng1_idx] not in TARGET_SNI:
                    continue
                if row[jurform_idx] != "49":   # 49 = Aktiebolag i SCB:s kodning
                    continue
                if row[ftgstat_idx] != "1":    # 1 = aktivt driftsställe
                    continue

                raw = row[orgnr_idx]
                # SCB-filen har 12-siffriga PeOrgNr med "16"-prefix
                if len(raw) == 12 and raw.startswith("16"):
                    orgnr = raw[2:]
                elif len(raw) == 10 and raw.isdigit():
                    orgnr = raw
                else:
                    continue

                companies.append({
                    "orgnr": orgnr,
                    "company_name": row[namn_idx],
                    "sni": row[ng1_idx],
                })

    log.info(f"SCB: {len(companies)} aktiva AB i målbranscher")
    return companies


# ── State (vilka org.nr har kollerats) ───────────────────────────────────────

def load_state(state_path: str) -> dict:
    if os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}   # {orgnr: {"revenue_msek": float|None, "checked_at": str}}


def save_state(state: dict, state_path: str):
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


# ── Deduplication ─────────────────────────────────────────────────────────────

def load_already_sent(log_dir: str) -> set:
    sent = set()
    for fname in os.listdir(log_dir):
        if not (fname.startswith("import_log_") or fname.startswith("prospect_log_")):
            continue
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(log_dir, fname), encoding="utf-8") as f:
                for row in json.load(f):
                    if row.get("result") == "ok":
                        orgnr = row.get("orgnr", "")
                        if orgnr:
                            sent.add(orgnr)
        except Exception:
            pass
    return sent


# ── Enrichment ────────────────────────────────────────────────────────────────

def enrich_email(company_name: str, orgnr: str) -> Optional[str]:
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        return None
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_key,
                "query": f'"{company_name}" kontakt e-post email',
                "max_results": 5,
            },
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
        log.debug(f"  [{orgnr}] Tavily-fel: {e}")
        return None

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": (
                f"Bolag: {company_name} (org.nr {orgnr})\n\n"
                f"Sökresultat:\n{search_text[:3000]}\n\n"
                "Extrahera EN e-postadress (innehåller @) som troligen tillhör detta bolag. "
                "Svara med BARA e-postadressen (t.ex. info@foretaget.se). "
                "Om ingen hittades, svara med exakt: NONE."
            )}],
        )
        result = msg.content[0].text.strip().strip('"').strip("'")
        if result.upper() == "NONE" or "@" not in result:
            return None
        parts = result.split("@")
        if len(parts) == 2 and "." in parts[1] and " " not in result and len(result) > 5:
            return result.lower()
    except Exception as e:
        log.debug(f"  [{orgnr}] Claude-fel: {e}")
    return None


# ── Submission ────────────────────────────────────────────────────────────────

def submit_scan(api_url: str, orgnr: str, company_name: str, email: str, phone: str = "") -> dict:
    resp = requests.post(
        f"{api_url}/publ/submit",
        json={
            "orgnr":         orgnr,
            "company_name":  company_name,
            "contact_name":  company_name,
            "contact_email": email,
            "contact_phone": phone,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prospektera bolag via SCB + Bolagsverket iXBRL")
    parser.add_argument("--min-revenue",   type=float, default=150.0,
                        help="Lägsta omsättning i MSEK (default: 150)")
    parser.add_argument("--max-revenue",   type=float, default=None)
    parser.add_argument("--batch-size",    type=int,   default=30,
                        help="Antal bolag att kolla per körning (default: 30)")
    parser.add_argument("--enrich",        action="store_true",
                        help="Sök e-post via Tavily+Claude")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--delay",         type=float, default=1.5,
                        help="Sekunder mellan Bolagsverket-anrop (default: 1.5)")
    parser.add_argument("--api-url",       default="http://13.48.24.83")
    parser.add_argument("--scb-file",      default=None,
                        help="Sökväg till SCB-zip (default: scb_bulkfil.zip i skriptets katalog)")
    parser.add_argument("--state-file",    default=None,
                        help="Sökväg till state-JSON (default: prospect_scan_state.json)")
    parser.add_argument("--list-candidates", action="store_true",
                        help="Lista bolag som ännu inte kollerats och avsluta")
    parser.add_argument("--reset-state",   action="store_true",
                        help="Nollställ state-filen och börja om från början")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    state_path  = args.state_file  or os.path.join(script_dir, "prospect_scan_state.json")
    log_dir     = script_dir

    # ── Kontrollera Bolagsverket-nycklar ──────────────────────────────────────
    if not os.environ.get("BOLAGSVERKET_CLIENT_ID"):
        print("FEL: BOLAGSVERKET_CLIENT_ID saknas i miljön")
        sys.exit(1)
    if not os.environ.get("BOLAGSVERKET_CLIENT_SECRET"):
        print("FEL: BOLAGSVERKET_CLIENT_SECRET saknas i miljön")
        sys.exit(1)

    # ── Ladda data ────────────────────────────────────────────────────────────
    companies    = load_scb_companies(args.scb_file)
    already_sent = load_already_sent(log_dir)
    print(f"Dubblettskydd: {len(already_sent)} org.nr redan inskickade")

    if args.reset_state and os.path.exists(state_path):
        os.remove(state_path)
        print("State-fil nollställd")

    state = load_state(state_path)

    # Obearbetade bolag (ej inskickade, ej kollade)
    unchecked = [
        c for c in companies
        if c["orgnr"] not in state and c["orgnr"] not in already_sent
    ]
    already_qualified = [
        c for c in companies
        if c["orgnr"] in state
        and state[c["orgnr"]].get("revenue_msek") is not None
        and state[c["orgnr"]]["revenue_msek"] >= args.min_revenue
        and c["orgnr"] not in already_sent
    ]

    print(f"Bolag i SCB-fil:   {len(companies)}")
    print(f"Obearbetade:       {len(unchecked)}")
    print(f"Redo att skicka:   {len(already_qualified)} (redan kollade ≥{args.min_revenue} MSEK)")

    if args.list_candidates:
        print(f"\nObearbetade bolag (visar max 50):")
        for c in unchecked[:50]:
            print(f"  {c['orgnr']}  {c['company_name'][:50]}  SNI {c['sni']}")
        if already_qualified:
            print(f"\nRedo att skicka:")
            for c in already_qualified:
                rev = state[c["orgnr"]]["revenue_msek"]
                print(f"  {c['orgnr']}  {c['company_name'][:50]}  {rev:.0f} MSEK")
        return

    # ── Fas 1: Kontrollera omsättning för obearbetade ─────────────────────────
    batch = unchecked[:args.batch_size]
    if batch:
        print(f"\n{'─'*60}")
        print(f"Kontrollerar omsättning för {len(batch)} bolag (batch {args.batch_size})...")
        print(f"{'─'*60}")

    newly_qualified = []
    for i, c in enumerate(batch, 1):
        orgnr = c["orgnr"]
        name  = c["company_name"]
        print(f"[{i}/{len(batch)}] {name[:50]} ({orgnr})...", end=" ", flush=True)
        try:
            rev = lookup_revenue(orgnr)
        except Exception as exc:
            print(f"FEL: {exc}")
            state[orgnr] = {"revenue_msek": None, "checked_at": datetime.now().isoformat()}
            save_state(state, state_path)
            if i < len(batch):
                time.sleep(args.delay)
            continue

        state[orgnr] = {"revenue_msek": rev, "checked_at": datetime.now().isoformat()}
        save_state(state, state_path)

        if rev is None:
            print("ingen iXBRL")
        elif rev >= args.min_revenue and (args.max_revenue is None or rev <= args.max_revenue):
            print(f"✓ {rev:.0f} MSEK → kvalificerad!")
            newly_qualified.append({**c, "revenue_msek": rev})
        else:
            print(f"{rev:.0f} MSEK (under {args.min_revenue})")

        if i < len(batch):
            time.sleep(args.delay)

    if batch:
        print(f"\nBatch klar. {len(newly_qualified)} nya kvalificerade bolag.")
        print(f"Obearbetade kvar: {len(unchecked) - len(batch)}")

    # Kombinera kandidater: nykvalificerade + redan kvalificerade
    candidates = newly_qualified + already_qualified
    if not candidates:
        print("\nInga bolag att skicka just nu.")
        print(f"Kör igen för att bearbeta nästa batch av {args.batch_size} bolag.")
        return

    print(f"\n{'═'*60}")
    print(f"Totalt att skicka: {len(candidates)} bolag")
    print(f"{'═'*60}\n")

    # ── Fas 2: Berika och skicka ─────────────────────────────────────────────
    stats    = {"submitted": 0, "enriched": 0, "no_email": 0, "errors": 0}
    log_rows = []

    for i, c in enumerate(candidates, 1):
        orgnr        = c["orgnr"]
        company_name = c["company_name"]
        rev          = c.get("revenue_msek") or state.get(orgnr, {}).get("revenue_msek", 0)
        email        = ""

        prefix = f"[{i}/{len(candidates)}] {company_name[:45]} ({orgnr}) {rev:.0f} MSEK"

        if args.enrich:
            print(f"{prefix} – söker e-post...")
            email = enrich_email(company_name, orgnr) or ""
            if email:
                print(f"    → {email}")
                stats["enriched"] += 1
            else:
                print(f"    → ingen e-post, hoppar över")
                stats["no_email"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name,
                                 "revenue_msek": rev, "result": "no_email"})
                continue
        else:
            print(f"{prefix} – hoppar över (kör med --enrich för att söka e-post)")
            stats["no_email"] += 1
            log_rows.append({"orgnr": orgnr, "company": company_name,
                             "revenue_msek": rev, "result": "no_email"})
            continue

        if args.dry_run:
            print(f"    [dry-run] → {email}")
            stats["submitted"] += 1
            log_rows.append({"orgnr": orgnr, "company": company_name, "email": email,
                             "revenue_msek": rev, "sni": c["sni"], "result": "dry_run"})
        else:
            try:
                result = submit_scan(args.api_url, orgnr, company_name, email)
                job_id = result.get("job_id", "?")
                print(f"    → OK job_id={job_id}")
                stats["submitted"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name, "email": email,
                                 "revenue_msek": rev, "sni": c["sni"],
                                 "job_id": job_id, "result": "ok"})
            except requests.HTTPError as e:
                msg = ""
                try:
                    msg = e.response.json().get("error", "")
                except Exception:
                    pass
                print(f"    → FEL {e.response.status_code}: {msg}")
                stats["errors"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name,
                                 "revenue_msek": rev, "result": f"error_{e.response.status_code}",
                                 "detail": msg})
            except Exception as e:
                print(f"    → FEL: {e}")
                stats["errors"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name,
                                 "revenue_msek": rev, "result": "error", "detail": str(e)})

        if i < len(candidates):
            time.sleep(1.0)

    print()
    print("═" * 60)
    print(f"  Skickade:          {stats['submitted']}")
    print(f"  Berikade e-poster: {stats['enriched']}")
    print(f"  Hoppade (no mail): {stats['no_email']}")
    print(f"  Fel:               {stats['errors']}")
    remaining = len(unchecked) - len(batch)
    if remaining > 0:
        print(f"\n  Obearbetade kvar:  {remaining} bolag")
        print(f"  Kör igen för nästa batch.")
    print("═" * 60)

    if log_rows:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"prospect_log_{ts}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_rows, f, ensure_ascii=False, indent=2)
        print(f"  Logg sparad: {log_path}")


if __name__ == "__main__":
    main()
