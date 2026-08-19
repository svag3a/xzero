#!/usr/bin/env python3
"""
prospect_scan.py – Hitta prospektbolag via SCB-filen + allabolag.se.

Flöde:
  1. Läser SCB-bulkfil (scb_bulkfil.zip) → aktiva AB i målbranscher
  2. Hämtar omsättning per bolag via allabolag.se (per-bolagssida, ingen cap)
  3. Filtrerar på omsättning >= --min-revenue MSEK
  4. Skickar till WasteZero-appen som Prospekt-leads (om inte --dry-run)

Körs i omgångar – sparar vilka org.nr som kollerats i state-filen
(prospect_scan_state.json) och fortsätter där det slutade.

E-postberikelse hanteras separat av enrich_prospects.py efter att
alla bolag genomsökts.

Usage:
  python3 prospect_scan.py --min-revenue 150 --api-url http://13.48.24.83
  python3 prospect_scan.py --min-revenue 150 --dry-run --batch-size 50
  python3 prospect_scan.py --list-candidates
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import zipfile
from datetime import datetime
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
    "46311",  # Partihandel med spannmål m.m.
    "46312",  # Partihandel med animala råvaror
    "46313",  # Partihandel med frukt och grönsaker
    "46320",  # Partihandel med kött och köttvaror
    "46330",  # Partihandel med mejeriprodukter
    "46340",  # Partihandel med drycker
    "46380",  # Partihandel med övriga livsmedel
    "46390",  # Partihandel med livsmedel i sortiment
    "10110",  # Charkuteri- och annan köttvaruframställning
    "10510",  # Framställning av mjölkprodukter
    "10200",  # Beredning och konservering av fisk
}

SCB_URL = "https://vardefulla-datamangder.bolagsverket.se/scb/scb_bulkfil.zip"

_ALLABOLAG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(_ALLABOLAG_HEADERS)
    return _session


# ── allabolag.se per-bolag ────────────────────────────────────────────────────

def _fetch_allabolag(orgnr: str) -> Optional[dict]:
    """
    Hämtar bolagsdata från allabolag.se via orgnr-redirect.
    Returnerar company-dict ur __NEXT_DATA__, eller None vid fel.

    Revenue-fältet är i kSEK (tusen SEK), samma enhet som segmenterings-API:t.
    """
    sess = _get_session()
    try:
        resp = sess.get(
            f"https://www.allabolag.se/{orgnr}",
            timeout=20,
            allow_redirects=True,
        )
        if not resp.ok:
            return None
        m = re.search(
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            resp.text,
            re.DOTALL,
        )
        if not m:
            return None
        data   = json.loads(m.group(1))
        return data.get("props", {}).get("pageProps", {}).get("company")
    except Exception as exc:
        log.debug(f"  [{orgnr}] allabolag-fel: {exc}")
        return None


def lookup_company(orgnr: str) -> Optional[dict]:
    """
    Returnerar dict med revenue_msek, phone, email, homepage.
    revenue_msek = None om inte hittas eller omsättning saknas.
    """
    company = _fetch_allabolag(orgnr)
    if not company:
        return None

    rev_raw = company.get("revenue")
    try:
        rev_ksek = float(rev_raw)
        rev_msek = rev_ksek / 1000.0  # kSEK → MSEK
    except (TypeError, ValueError):
        rev_msek = None

    return {
        "revenue_msek": rev_msek,
        "phone":        str(company.get("phone") or company.get("legalPhone") or ""),
        "email":        (company.get("email") or "").lower().strip(),
        "homepage":     company.get("homePage") or "",
        "employees":    company.get("numberOfEmployees"),
    }


# ── SCB-fil ───────────────────────────────────────────────────────────────────

def load_scb_companies(scb_zip_path: Optional[str] = None) -> list:
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
            # Ng2–Ng5 är valfria kolumner
            ng_extra_idx = [header.index(f"Ng{i}") for i in range(2, 6) if f"Ng{i}" in header]

            for line in f:
                row = line.decode("latin-1", errors="replace").rstrip().split("\t")
                if len(row) <= ng1_idx:
                    continue
                if row[ng1_idx] not in TARGET_SNI:
                    continue
                if row[jurform_idx] != "49":   # 49 = AB i SCB:s kodning
                    continue
                if row[ftgstat_idx] != "1":    # 1 = aktivt
                    continue

                raw = row[orgnr_idx]
                if len(raw) == 12 and raw.startswith("16"):
                    orgnr = raw[2:]
                elif len(raw) == 10 and raw.isdigit():
                    orgnr = raw
                else:
                    continue

                sni_codes = [row[ng1_idx]]
                for idx in ng_extra_idx:
                    if idx < len(row) and row[idx] and row[idx] != row[ng1_idx]:
                        sni_codes.append(row[idx])

                companies.append({
                    "orgnr":        orgnr,
                    "company_name": row[namn_idx],
                    "sni":          row[ng1_idx],
                    "sni_codes":    sni_codes,
                })

    log.info(f"SCB: {len(companies)} aktiva AB i målbranscher")
    return companies


# ── State ─────────────────────────────────────────────────────────────────────

def load_state(state_path: str) -> dict:
    if os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


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


# ── Submission ────────────────────────────────────────────────────────────────

def submit_scan(api_url: str, orgnr: str, company_name: str, email: str,
                phone: str = "", sni_codes: list = []) -> dict:
    resp = requests.post(
        f"{api_url}/publ/submit",
        json={
            "orgnr":          orgnr,
            "company_name":   company_name,
            "contact_name":   company_name,
            "contact_email":  email,
            "contact_phone":  phone,
            "sni_codes":      sni_codes,
            "initial_status": "Prospekt",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prospektera bolag via SCB + allabolag.se")
    parser.add_argument("--min-revenue",     type=float, default=150.0,
                        help="Lägsta omsättning MSEK (default: 150)")
    parser.add_argument("--max-revenue",     type=float, default=None)
    parser.add_argument("--batch-size",      type=int,   default=30,
                        help="Bolag att kolla per körning (default: 30)")
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--delay",           type=float, default=2.0,
                        help="Sekunder mellan allabolag-anrop (default: 2.0)")
    parser.add_argument("--api-url",         default="http://13.48.24.83")
    parser.add_argument("--scb-file",        default=None)
    parser.add_argument("--state-file",      default=None)
    parser.add_argument("--list-candidates", action="store_true",
                        help="Lista bolag ej kollade och avsluta")
    parser.add_argument("--reset-state",     action="store_true",
                        help="Nollställ state och börja om")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    state_path = args.state_file or os.path.join(script_dir, "prospect_scan_state.json")
    log_dir    = script_dir

    companies    = load_scb_companies(args.scb_file)
    already_sent = load_already_sent(log_dir)
    print(f"Dubblettskydd: {len(already_sent)} org.nr redan inskickade")

    if args.reset_state and os.path.exists(state_path):
        os.remove(state_path)
        print("State-fil nollställd")

    state = load_state(state_path)

    unchecked = [
        c for c in companies
        if c["orgnr"] not in state and c["orgnr"] not in already_sent
    ]
    already_qualified = [
        c for c in companies
        if c["orgnr"] in state
        and state[c["orgnr"]].get("revenue_msek") is not None
        and state[c["orgnr"]]["revenue_msek"] >= args.min_revenue
        and (args.max_revenue is None or state[c["orgnr"]]["revenue_msek"] <= args.max_revenue)
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
            print(f"\nRedo att skicka ({len(already_qualified)} st):")
            for c in already_qualified:
                rev = state[c["orgnr"]]["revenue_msek"]
                email = state[c["orgnr"]].get("email", "")
                print(f"  {c['orgnr']}  {c['company_name'][:45]}  {rev:.0f} MSEK  {email}")
        return

    # ── Fas 1: Kolla omsättning för ny batch ──────────────────────────────────
    batch = unchecked[:args.batch_size]
    if batch:
        print(f"\n{'─'*60}")
        print(f"Kollar allabolag.se för {len(batch)} bolag...")
        print(f"{'─'*60}")

    newly_qualified = []
    for i, c in enumerate(batch, 1):
        orgnr = c["orgnr"]
        name  = c["company_name"]
        print(f"[{i}/{len(batch)}] {name[:50]} ({orgnr})...", end=" ", flush=True)

        info = lookup_company(orgnr)

        if info is None:
            print("ej hittad")
            state[orgnr] = {"revenue_msek": None, "checked_at": datetime.now().isoformat()}
        else:
            rev = info["revenue_msek"]
            state[orgnr] = {
                "revenue_msek": rev,
                "phone":        info["phone"],
                "email":        info["email"],
                "homepage":     info["homepage"],
                "employees":    info["employees"],
                "checked_at":   datetime.now().isoformat(),
            }
            if rev is None:
                print("ingen omsättning")
            elif rev >= args.min_revenue and (args.max_revenue is None or rev <= args.max_revenue):
                print(f"✓ {rev:.0f} MSEK → kvalificerad!")
                if info["email"]:
                    print(f"    e-post: {info['email']}")
                newly_qualified.append({**c,
                    "revenue_msek": rev,
                    "email": info["email"],
                    "phone": info["phone"],
                })
            else:
                print(f"{rev:.0f} MSEK")

        save_state(state, state_path)
        if i < len(batch):
            time.sleep(args.delay)

    if batch:
        print(f"\nBatch klar. {len(newly_qualified)} nya kvalificerade bolag.")
        remaining = len(unchecked) - len(batch)
        if remaining:
            print(f"Obearbetade kvar: {remaining}")

    # Kombinera kandidater
    candidates = newly_qualified[:]
    for c in already_qualified:
        s = state[c["orgnr"]]
        candidates.append({**c,
            "revenue_msek": s["revenue_msek"],
            "email":        s.get("email", ""),
            "phone":        s.get("phone", ""),
        })

    if not candidates:
        print("\nInga bolag att skicka just nu.")
        return

    print(f"\n{'═'*60}")
    print(f"Totalt att skicka: {len(candidates)} bolag")
    print(f"{'═'*60}\n")

    # ── Fas 2: Skicka ────────────────────────────────────────────────────────
    stats    = {"submitted": 0, "errors": 0}
    log_rows = []

    for i, c in enumerate(candidates, 1):
        orgnr        = c["orgnr"]
        company_name = c["company_name"]
        rev          = c.get("revenue_msek", 0) or 0
        email        = c.get("email", "")
        phone        = c.get("phone", "")

        prefix = f"[{i}/{len(candidates)}] {company_name[:45]} ({orgnr}) {rev:.0f} MSEK"

        if args.dry_run:
            print(f"{prefix} – [dry-run] → {email or '(ingen mail)'}")
            stats["submitted"] += 1
            log_rows.append({"orgnr": orgnr, "company": company_name, "email": email,
                             "revenue_msek": rev, "sni": c["sni"], "result": "dry_run"})
        else:
            try:
                result = submit_scan(args.api_url, orgnr, company_name, email, phone,
                                     sni_codes=c.get("sni_codes", []))
                job_id = result.get("job_id", "?")
                print(f"{prefix} – OK job_id={job_id} → {email or '(ingen mail)'}")
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
                print(f"{prefix} – FEL {e.response.status_code}: {msg}")
                stats["errors"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name,
                                 "revenue_msek": rev, "result": f"error_{e.response.status_code}",
                                 "detail": msg})
            except Exception as e:
                print(f"{prefix} – FEL: {e}")
                stats["errors"] += 1
                log_rows.append({"orgnr": orgnr, "company": company_name,
                                 "revenue_msek": rev, "result": "error", "detail": str(e)})

        if i < len(candidates):
            time.sleep(1.0)

    print()
    print("═" * 60)
    print(f"  Skickade:  {stats['submitted']}")
    print(f"  Fel:       {stats['errors']}")
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
