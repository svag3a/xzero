#!/usr/bin/env python3
"""
enrich_prospects.py – Berika Prospekt-leads med kontaktperson och e-post.

Flöde per bolag:
  1. Hämtar Prospekt-leads utan riktig kontaktperson från CRM-API:t
  2. Kör riktade Tavily-sökningar: VD/ledning/kontakt för bolaget
  3. Claude extraherar namn, titel och e-post ur sökresultaten
  4. Om namn hittas men ingen e-post → byggs förnamn.efternamn@domän
  5. PATCHar CRM-leadet med vad som hittades
  6. Sparar state så man kan köra i omgångar

Usage:
  python3 enrich_prospects.py --api-url http://13.48.24.83 --batch-size 10
  python3 enrich_prospects.py --dry-run --batch-size 10
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import anthropic
import requests
from tavily import TavilyClient


STATE_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrich_state.json")
SCAN_STATE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prospect_scan_state.json")
SNI_TITLES = {
    "49410": ["logistikchef", "transportchef", "VD"],
    "52100": ["lagerchef", "logistikchef", "VD"],
    "52210": ["terminalchef", "driftchef", "VD"],
    "52240": ["godshanteringschef", "driftchef", "VD"],
    "46311": ["inköpschef", "VD", "COO"],
    "46312": ["inköpschef", "VD", "COO"],
    "46313": ["inköpschef", "VD", "COO"],
    "46320": ["inköpschef", "VD", "COO"],
    "46330": ["inköpschef", "VD", "COO"],
    "46380": ["inköpschef", "VD", "COO"],
    "10110": ["produktionschef", "VD", "COO"],
    "10510": ["produktionschef", "VD", "COO"],
    "10200": ["produktionschef", "VD", "COO"],
}
DEFAULT_TITLES = ["VD", "logistikchef", "COO"]


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_session(api_url: str, password: str) -> requests.Session:
    sess = requests.Session()
    if not password:
        return sess
    resp = sess.post(
        f"{api_url}/login",
        data={"password": password, "next": "/"},
        allow_redirects=False,
        timeout=15,
    )
    if resp.status_code not in (302, 303):
        raise RuntimeError(f"Login misslyckades (HTTP {resp.status_code})")
    return sess


def fetch_prospects(api_url: str, sess: requests.Session) -> list:
    resp = sess.get(f"{api_url}/api/crm", timeout=30)
    if resp.status_code == 401 or (resp.status_code == 200 and not resp.text.strip()):
        raise RuntimeError("Ej autentiserad — ange --password")
    resp.raise_for_status()
    leads = resp.json()
    return [l for l in leads if l.get("status") == "Prospekt"]


def needs_enrichment(lead: dict) -> bool:
    name = (lead.get("contact_name") or "").strip()
    email = (lead.get("contact_email") or "").strip()
    # Skip if already has a real contact person (not just company name)
    if name and name != lead.get("company_name", "").strip() and "@" in email:
        return False
    return True


def _target_titles(sni_codes: list) -> list:
    titles = []
    for code in (sni_codes or []):
        titles.extend(SNI_TITLES.get(code, []))
    return list(dict.fromkeys(titles)) or DEFAULT_TITLES  # dedup, preserve order


def search_contact(company_name: str, titles: list) -> str:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        print("  TAVILY_API_KEY saknas")
        return ""
    client = TavilyClient(api_key=api_key)
    title_str = " OR ".join(titles[:3])
    queries = [
        f"{company_name} {titles[0]}",
        f"{company_name} kontakt {title_str}",
        f"{company_name} ledning styrelse",
    ]
    sections = []
    for q in queries:
        try:
            resp = client.search(q, max_results=3, search_depth="basic")
            hits = []
            for r in resp.get("results", []):
                title_txt = r.get("title", "").strip()
                url       = r.get("url", "").strip()
                content   = r.get("content", "").strip()[:500]
                hits.append(f"**{title_txt}** ({url})\n{content}")
            if hits:
                sections.append(f"### {q}\n\n" + "\n\n".join(hits))
        except Exception as e:
            print(f"  Tavily fel: {e}")
        time.sleep(0.5)
    return "\n\n".join(sections)


def extract_contact(company_name: str, company_domain: str, titles: list,
                    search_results: str) -> dict:
    """Kör Claude för att extrahera kontaktperson ur sökresultaten."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ANTHROPIC_API_KEY saknas")
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    title_str = ", ".join(titles[:3])

    prompt = f"""Du analyserar sökresultat för att hitta rätt kontaktperson på {company_name}.

Målroller (prioritetsordning): {title_str}

Sökresultat:
{search_results or "(inga sökresultat)"}

Extrahera den BÄSTA kontaktpersonen. Svara ENBART med JSON i detta format:
{{
  "found": true/false,
  "name": "Förnamn Efternamn",
  "title": "Exakt titel",
  "email": "epost@domän.se eller null",
  "confidence": "high/medium/low",
  "source": "URL eller beskrivning av källan"
}}

Regler:
- Om du hittar en e-post direkt i källorna, använd den
- Om du INTE hittar e-post men hittar namn: sätt email till null (ej konstruera)
- Om ingenting hittas: found=false, alla andra fält null
- Välj den person som är närmast operativt ansvarig för logistik/inköp/produktion
- Om bara VD hittas och inget bättre alternativ finns, ta VD"""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def build_email(name: str, domain: str) -> str:
    """Konstruera förnamn.efternamn@domän från namn och domän."""
    if not name or not domain:
        return ""
    parts = name.strip().split()
    if len(parts) < 2:
        return ""
    first = re.sub(r'[^a-zåäö]', '', parts[0].lower())
    last  = re.sub(r'[^a-zåäö]', '', parts[-1].lower())
    # Normalize Swedish chars for email
    for src, dst in [("å","a"),("ä","a"),("ö","o")]:
        first = first.replace(src, dst)
        last  = last.replace(src, dst)
    if not first or not last:
        return ""
    return f"{first}.{last}@{domain}"


def extract_domain(homepage: str) -> str:
    if not homepage:
        return ""
    m = re.search(r'https?://(?:www\.)?([^/\s]+)', homepage)
    return m.group(1).lower() if m else ""


def patch_lead(api_url: str, sess: requests.Session, lead_id: str,
               contact_name: str, contact_email: str):
    payload = {}
    if contact_name:
        payload["contact_name"] = contact_name
    if contact_email:
        payload["contact_email"] = contact_email
    if not payload:
        return
    resp = sess.patch(f"{api_url}/api/crm/{lead_id}", json=payload, timeout=30)
    resp.raise_for_status()


def main():
    parser = argparse.ArgumentParser(description="Berika Prospekt-leads med kontaktperson")
    parser.add_argument("--api-url",    default="http://13.48.24.83")
    parser.add_argument("--password",   default=os.environ.get("APP_PASSWORD", ""),
                        help="App-lösenord (eller sätt APP_PASSWORD i env)")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--reset",      action="store_true", help="Nollställ state")
    args = parser.parse_args()

    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("State nollställd")

    sess = get_session(args.api_url, args.password)
    state = load_state()

    # Läs prospect_scan_state för att få homepage per orgnr
    scan_state = {}
    if os.path.exists(SCAN_STATE_FILE):
        try:
            with open(SCAN_STATE_FILE, encoding="utf-8") as f:
                scan_state = json.load(f)
        except Exception:
            pass

    prospects = fetch_prospects(args.api_url, sess)
    print(f"Totalt Prospekt i CRM: {len(prospects)}")

    to_enrich = [
        l for l in prospects
        if needs_enrichment(l) and l["id"] not in state
    ]
    print(f"Behöver enrichment: {len(to_enrich)}")
    print(f"Redan behandlade:   {len([l for l in prospects if l['id'] in state])}")

    batch = to_enrich[:args.batch_size]
    if not batch:
        print("\nInget att berika just nu.")
        return

    print(f"\nKör batch: {len(batch)} bolag\n{'─'*60}")

    stats = {"found": 0, "email_direct": 0, "email_constructed": 0,
             "not_found": 0, "errors": 0}
    log_rows = []

    for i, lead in enumerate(batch, 1):
        lid          = lead["id"]
        orgnr        = lead.get("orgnr", "")
        company_name = lead.get("company_name") or orgnr or "?"
        sni_codes    = lead.get("sni_codes") or []
        homepage     = (scan_state.get(orgnr) or {}).get("homepage", "")
        domain       = extract_domain(homepage)
        titles       = _target_titles(sni_codes)

        print(f"[{i}/{len(batch)}] {company_name[:50]} ({lid})")
        print(f"  Söker: {', '.join(titles[:2])}…", end=" ", flush=True)

        try:
            search_results = search_contact(company_name, titles)
            result = extract_contact(company_name, domain, titles, search_results)
        except Exception as e:
            print(f"FEL: {e}")
            stats["errors"] += 1
            state[lid] = {"result": "error", "detail": str(e), "checked_at": datetime.now().isoformat()}
            save_state(state)
            continue

        if not result.get("found"):
            print("→ ingen kontakt hittad")
            stats["not_found"] += 1
            state[lid] = {"result": "not_found", "checked_at": datetime.now().isoformat()}
            save_state(state)
            log_rows.append({"id": lid, "company": company_name, "result": "not_found"})
            continue

        name  = result.get("name", "").strip()
        title = result.get("title", "").strip()
        email = (result.get("email") or "").strip()
        conf  = result.get("confidence", "")

        email_source = "direct"
        if not email and name and domain:
            email = build_email(name, domain)
            email_source = "constructed"

        print(f"→ {name} ({title}) {email or '(ingen mail)'} [{conf}]")
        stats["found"] += 1
        if email:
            if email_source == "direct":
                stats["email_direct"] += 1
            else:
                stats["email_constructed"] += 1

        if not args.dry_run:
            try:
                patch_lead(args.api_url, sess, lid, name, email)
            except Exception as e:
                print(f"  PATCH fel: {e}")
                stats["errors"] += 1
                state[lid] = {"result": "patch_error", "detail": str(e),
                               "checked_at": datetime.now().isoformat()}
                save_state(state)
                continue

        state[lid] = {
            "result": "enriched",
            "name": name,
            "title": title,
            "email": email,
            "email_source": email_source,
            "confidence": conf,
            "checked_at": datetime.now().isoformat(),
        }
        save_state(state)
        log_rows.append({"id": lid, "company": company_name, "name": name,
                         "title": title, "email": email, "email_source": email_source,
                         "confidence": conf, "result": "dry_run" if args.dry_run else "ok"})

        if i < len(batch):
            time.sleep(2.0)

    print(f"\n{'═'*60}")
    print(f"  Kontakt hittad:      {stats['found']}")
    print(f"  E-post direkt:       {stats['email_direct']}")
    print(f"  E-post konstruerad:  {stats['email_constructed']}")
    print(f"  Ingen kontakt:       {stats['not_found']}")
    print(f"  Fel:                 {stats['errors']}")
    remaining = len(to_enrich) - len(batch)
    if remaining:
        print(f"\n  Kvar att berika:     {remaining}")
        print(f"  Kör igen för nästa batch.")
    print("═" * 60)

    if log_rows:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"enrich_log_{ts}.json"
        )
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_rows, f, ensure_ascii=False, indent=2)
        print(f"  Logg sparad: {log_path}")


if __name__ == "__main__":
    main()
