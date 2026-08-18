#!/usr/bin/env python3
"""
consolidate.py – Slår ihop befintliga Opportunity Scans till en koncernanalys.

Förutsättning: alla individuella scanar är redan klara och har scan_ids.
Scan_ids hittar du i CRM-vyn eller i loggen efter varje scan.

Usage:
  python3 consolidate.py \\
    --scan-ids 12,17,23,24,25,26,27 \\
    --group-name "Koncernnamn AB" \\
    --contact-name "Anna Svensson" \\
    --contact-email anna@koncernen.se \\
    --api-url http://13.48.24.83

Miljövariabler:
  ADMIN_TOKEN  (om satt i appen)
"""

import argparse
import os
import sys
import requests


def list_scans(api_url: str, admin_token: str):
    headers = {}
    if admin_token:
        headers["X-Admin-Token"] = admin_token
    resp = requests.get(f"{api_url}/admin/scans", headers=headers, timeout=15)
    resp.raise_for_status()
    scans = resp.json()
    print(f"{'ID':>5}  {'Bolag':<35}  {'Oms MSEK':>9}  {'Potential':>9}  {'Hyp':>4}  Datum")
    print("─" * 80)
    for s in scans:
        hyp = "✓" if s.get("has_hypotheses") else "-"
        print(
            f"{s['id']:>5}  {(s['company_name'] or '')[:35]:<35}  "
            f"{str(s['revenue_msek'] or ''):>9}  "
            f"{str(s['total_potential_msek'] or ''):>9}  "
            f"{hyp:>4}  "
            f"{(s['created_at'] or '')[:10]}"
        )


def main():
    parser = argparse.ArgumentParser(description="Konsolidera scanar till koncernanalys")
    parser.add_argument("--scan-ids",      default="", help="Kommaseparerade scan_ids, t.ex. 12,17,23")
    parser.add_argument("--list",          action="store_true", help="Lista senaste scanar med ID")
    parser.add_argument("--group-name",    default="Koncernanalys", help="Koncernens namn")
    parser.add_argument("--contact-name",  default="")
    parser.add_argument("--contact-email", default="")
    parser.add_argument("--api-url",       default="http://13.48.24.83")
    args = parser.parse_args()

    admin_token = os.environ.get("ADMIN_TOKEN", "")

    if args.list:
        list_scans(args.api_url, admin_token)
        return

    if not args.scan_ids:
        print("Ange --scan-ids eller --list")
        sys.exit(1)

    try:
        scan_ids = [int(x.strip()) for x in args.scan_ids.split(",")]
    except ValueError:
        print("Fel: --scan-ids måste vara kommaseparerade heltal, t.ex. 12,17,23")
        sys.exit(1)

    if len(scan_ids) < 2:
        print("Fel: minst 2 scan_ids krävs")
        sys.exit(1)

    headers = {"Content-Type": "application/json"}
    if admin_token:
        headers["X-Admin-Token"] = admin_token

    print(f"Startar koncernanalys för scan_ids: {scan_ids}")
    print(f"Grupp: {args.group_name}")
    print()

    try:
        resp = requests.post(
            f"{args.api_url}/admin/consolidate",
            headers=headers,
            json={
                "scan_ids":      scan_ids,
                "group_name":    args.group_name,
                "contact_name":  args.contact_name,
                "contact_email": args.contact_email,
            },
            timeout=30,
        )
        resp.raise_for_status()
        print(f"OK: {resp.json()}")
        print()
        print("Analysen körs nu i bakgrunden på servern (~2–5 min).")
        print("Resultatet sparas i CRM:et automatiskt.")
        if args.contact_email:
            print(f"Bekräftelse skickas till {args.contact_email} när det är klart.")
    except requests.HTTPError as e:
        print(f"FEL {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"FEL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
